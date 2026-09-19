"""combat — target acquisition, shots, damage and deaths (task 8).

Direct-fire only. One tick of combat is, per squad in ascending id order:

1. `acquire_target` — keep the current target while it is still a legal one,
   otherwise pick the best visible enemy in range with LOS (or follow the
   squad's `Attack` order, walking toward the target until it is in range);
2. `fire_member` — per member, in loadout order, decide whether its weapon
   can fire this tick (ready, in its own range band, not moving with a
   moving-accuracy of 0, team weapon set up);
3. `resolve_bullets` — one bullet for a single-shot weapon, `rate_of_fire ×
   burst_duration` for a burst weapon, each rolled separately;
4. `apply_hit` / `add_suppression` — damage a randomly chosen living member
   (or the building), and accumulate suppression on the target regardless of
   whether the bullet hit;
5. `schedule_next_shot` — cooldown and reload bookkeeping.

Later tasks extend exactly these seams: task 9 consumes `squad.suppression`
and adds nearby spill, task 10 adds arcs / indirect fire / abandoned team
weapons, task 11 adds penetration to `apply_hit`, task 12 fills in
`on_building_destroyed`.

RNG discipline: every draw comes from `sim.state.rng`, in a fixed order
(squads ascending id -> members in loadout order -> burst length -> per
bullet: victim, hit roll -> cooldown -> reload). Bullets aimed at a
building draw nothing at all: they always hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from coh.data.schema import WeaponDef
from coh.maps.cover import cover_at
from coh.maps.format import cell_of, center_of
from coh.sim import orders as orders_mod
from coh.sim.constants import (
    ATTACK_ORDER_LOST_TARGET_S,
    ATTACK_ORDER_REPATH_S,
    BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT,
    CELL_M,
    FORMATION_OFFSETS,
    TICKS_PER_SECOND,
)
from coh.sim.state import Building, Event, Squad, SquadState
from coh.sim.systems import vision

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Member

__all__ = ["run", "acquire_target", "apply_hit", "on_building_destroyed"]

# States in which a squad neither acquires nor fires.
_NO_COMBAT_STATES = (SquadState.RETREATING, SquadState.CONSTRUCTING)

_LOST_TARGET_TICKS = int(round(ATTACK_ORDER_LOST_TARGET_S * TICKS_PER_SECOND))
_REPATH_TICKS = max(1, int(round(ATTACK_ORDER_REPATH_S * TICKS_PER_SECOND)))


# ---------------------------------------------------------------------------
# Per-tick context: entity positions as arrays, for cheap range rejection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Snapshot:
    """Positions of every targetable entity, taken once at the top of a tick.

    Used only to *reject* work cheaply; every candidate that survives the
    squared-distance filter is re-read from `state` before it is used, so a
    squad that dies mid-tick can never be shot at.
    """

    squad_ids: np.ndarray  # (n,) ascending
    squad_pos: np.ndarray  # (n, 2)
    squad_team: np.ndarray  # (n,)
    building_ids: tuple[int, ...]
    building_team: tuple[int, ...]


_NO_TEAM = -1


def _team_of(sim: "Sim", owner: int | None) -> int:
    if owner is None:
        return _NO_TEAM
    player = sim.state.players.get(owner)
    return _NO_TEAM if player is None else player.team


def _snapshot(sim: "Sim") -> _Snapshot:
    ids: list[int] = []
    pos: list[np.ndarray] = []
    teams: list[int] = []
    for sid in sorted(sim.state.squads):
        squad = sim.state.squads[sid]
        if not _is_targetable(squad):
            continue
        ids.append(sid)
        pos.append(squad.pos)
        teams.append(_team_of(sim, squad.owner))

    building_ids: list[int] = []
    building_team: list[int] = []
    for bid in sorted(sim.state.buildings):
        building = sim.state.buildings[bid]
        if building.neutral:
            continue  # never auto-targeted (and never Attack-able in task 8)
        building_ids.append(bid)
        building_team.append(_team_of(sim, building.owner))

    return _Snapshot(
        squad_ids=np.array(ids, dtype=np.int64),
        squad_pos=np.array(pos, dtype=float).reshape(len(ids), 2),
        squad_team=np.array(teams, dtype=np.int64),
        building_ids=tuple(building_ids),
        building_team=tuple(building_team),
    )


def _is_targetable(squad: Squad) -> bool:
    """Abandoned squads and squads with no members are never targets."""
    return not squad.abandoned and bool(squad.alive_members)


# ---------------------------------------------------------------------------
# Tick entry point
# ---------------------------------------------------------------------------


def run(sim: "Sim") -> None:
    """Advance the combat system by one tick."""
    snapshot = _snapshot(sim)
    for sid in sorted(sim.state.squads):
        squad = sim.state.squads.get(sid)
        if squad is None:
            continue  # destroyed earlier this tick
        if not _can_fight(sim, squad):
            squad.target_id = None
            continue
        acquire_target(sim, squad, snapshot)
        fire_squad(sim, squad)


def _can_fight(sim: "Sim", squad: Squad) -> bool:
    """Garrisoned squads fire (from the building); these ones never do."""
    if squad.abandoned or not squad.alive_members or squad.pinned:
        return False
    if squad.state in _NO_COMBAT_STATES:
        return False
    return sim.data.squads.get(squad.def_id) is not None


# ---------------------------------------------------------------------------
# Weapons / target descriptions
# ---------------------------------------------------------------------------


def _weapon_of(sim: "Sim", member: "Member") -> WeaponDef | None:
    """The member's direct-fire weapon, or None if unarmed / indirect.

    Indirect weapons (mortars) are task 10's; they never fire here.
    """
    if not member.weapon:
        return None
    weapon = sim.data.weapons.get(member.weapon)
    if weapon is None or weapon.indirect:
        return None
    return weapon


def _armed_members(sim: "Sim", squad: Squad) -> Iterable[tuple[int, "Member", WeaponDef]]:
    """`(slot, member, weapon)` for every living member with a usable weapon."""
    for slot, member in enumerate(squad.members):
        if member.hp <= 0:
            continue
        weapon = _weapon_of(sim, member)
        if weapon is not None:
            yield slot, member, weapon


def _max_range(sim: "Sim", squad: Squad) -> float:
    return max((w.ranges[2] for _, _, w in _armed_members(sim, squad)), default=0.0)


def _primary_weapon(sim: "Sim", squad: Squad) -> WeaponDef | None:
    """The squad's first usable weapon: its target table drives acquisition."""
    for _, _, weapon in _armed_members(sim, squad):
        return weapon
    return None


@dataclass(frozen=True)
class _Target:
    entity: Squad | Building
    target_type: str
    aim_pos: np.ndarray
    distance: float

    @property
    def id(self) -> int:
        return self.entity.id

    @property
    def is_squad(self) -> bool:
        return isinstance(self.entity, Squad)


def _footprint(sim: "Sim", def_id: str) -> tuple[int, int]:
    bdef = sim.data.buildings.get(def_id)
    if bdef is not None:
        return bdef.footprint
    ndef = sim.data.neutral.get(def_id)
    return ndef.footprint if ndef is not None else (1, 1)


def _building_target_type(sim: "Sim", building: Building) -> str | None:
    bdef = sim.data.buildings.get(building.def_id) or sim.data.neutral.get(building.def_id)
    return None if bdef is None else bdef.target_type


def _describe(sim: "Sim", from_pos: np.ndarray, entity: Squad | Building) -> _Target | None:
    """Aim point and distance from `from_pos` to `entity` (None if unknown def).

    A building is aimed at through its nearest footprint cell centre.
    """
    if isinstance(entity, Squad):
        sdef = sim.data.squads.get(entity.def_id)
        if sdef is None:
            return None
        aim = entity.pos
        return _Target(entity, sdef.target_type, aim, float(np.linalg.norm(aim - from_pos)))

    target_type = _building_target_type(sim, entity)
    if target_type is None:
        return None
    cells = _footprint_centres(sim, entity)
    d2 = ((cells - from_pos) ** 2).sum(axis=1)
    nearest = int(np.argmin(d2))  # ties -> lowest index: deterministic
    return _Target(entity, target_type, cells[nearest], math.sqrt(float(d2[nearest])))


def _footprint_centres(sim: "Sim", building: Building) -> np.ndarray:
    """`(k, 2)` world centres of the building's footprint cells (cached).

    A building never moves or resizes, so this is keyed by id alone and
    dropped when the building is destroyed.
    """
    cache = getattr(sim, "_combat_footprints", None)
    if cache is None:
        cache = {}
        sim._combat_footprints = cache
    cells = cache.get(building.id)
    if cells is None:
        width, height = _footprint(sim, building.def_id)
        cx0, cy0 = building.cell
        cells = np.array(
            [center_of((cx0 + dx, cy0 + dy), CELL_M) for dy in range(height) for dx in range(width)],
            dtype=float,
        )
        cache[building.id] = cells
    return cells


def _lookup(sim: "Sim", entity_id: int) -> Squad | Building | None:
    squad = sim.state.squads.get(entity_id)
    if squad is not None:
        return squad if _is_targetable(squad) else None
    return sim.state.buildings.get(entity_id)


# ---------------------------------------------------------------------------
# Target acquisition
# ---------------------------------------------------------------------------


def acquire_target(sim: "Sim", squad: Squad, snapshot: _Snapshot) -> None:
    """Set `squad.target_id` for this tick.

    An `Attack` order pins the target (and walks the squad toward it);
    otherwise the current target is kept while it stays legal, and a fresh
    one is picked when it does not.
    """
    team = _team_of(sim, squad.owner)
    reach = _max_range(sim, squad)
    if reach <= 0.0:
        squad.target_id = None
        return

    if isinstance(squad.order, orders_mod.Attack):
        _pursue_attack_order(sim, squad, squad.order, team, reach)
        return

    if squad.target_id is not None:
        current = _lookup(sim, squad.target_id)
        target = None if current is None else _describe(sim, squad.pos, current)
        if target is not None and _is_engageable(sim, squad, team, target, reach):
            return
        squad.target_id = None

    best = _choose_target(sim, squad, team, reach, snapshot)
    squad.target_id = None if best is None else best.id


def _is_engageable(sim: "Sim", squad: Squad, team: int, target: _Target, reach: float) -> bool:
    """Visible to the squad's team, hostile, within reach and in LOS."""
    if _team_of(sim, target.entity.owner) == team:
        return False
    if isinstance(target.entity, Building) and target.entity.neutral:
        return False
    if target.distance > reach:
        return False
    if not vision.is_visible(sim, team, target.entity):
        return False
    return vision.has_los(sim.map, squad.pos, target.aim_pos)


def _choose_target(
    sim: "Sim", squad: Squad, team: int, reach: float, snapshot: _Snapshot
) -> _Target | None:
    """Best `(priority desc, distance asc, id asc)` engageable enemy, or None."""
    primary = _primary_weapon(sim, squad)
    if primary is None:
        return None

    best: _Target | None = None
    best_key: tuple[float, float, int] | None = None

    for entity in _candidates(sim, squad, team, reach, snapshot):
        target = _describe(sim, squad.pos, entity)
        if target is None or not _is_engageable(sim, squad, team, target, reach):
            continue
        mods = primary.vs(target.target_type)
        if isinstance(entity, Building) and mods.damage < BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT:
            continue
        key = (-mods.priority, target.distance, target.id)
        if best_key is None or key < best_key:
            best, best_key = target, key
    return best


def _candidates(
    sim: "Sim", squad: Squad, team: int, reach: float, snapshot: _Snapshot
) -> Iterable[Squad | Building]:
    """Hostile entities that pass the cheap squared-distance filter."""
    if snapshot.squad_ids.size:
        delta = snapshot.squad_pos - squad.pos
        within = (delta * delta).sum(axis=1) <= reach * reach
        hostile = snapshot.squad_team != team
        for sid in snapshot.squad_ids[within & hostile].tolist():
            candidate = sim.state.squads.get(sid)
            if candidate is not None and _is_targetable(candidate):
                yield candidate

    for bid, bteam in zip(snapshot.building_ids, snapshot.building_team):
        if bteam == team or bteam == _NO_TEAM:
            continue
        building = sim.state.buildings.get(bid)
        if building is None:
            continue
        cells = _footprint_centres(sim, building)
        if ((cells - squad.pos) ** 2).sum(axis=1).min() <= reach * reach:
            yield building


# -- Attack order -----------------------------------------------------------


def _halt(sim: "Sim", squad: Squad) -> None:
    """Stop walking, the way movement's arrival does.

    A team weapon that stops deploys again, so that it can actually fire the
    target it just walked into range of.
    """
    squad.path = []
    if squad.state is not SquadState.MOVING:
        return
    from coh.sim.systems import movement

    sdef = sim.data.squads[squad.def_id]
    if sdef.kind == "team_weapon":
        squad.state = SquadState.SETTING_UP
        squad.facing = squad.heading
        squad.setup_done_tick = sim.state.tick + movement.setup_ticks(sim, sdef)
    else:
        squad.state = SquadState.IDLE


def _clear_attack_order(sim: "Sim", squad: Squad) -> None:
    squad.order = None
    squad.target_id = None
    squad.attack_last_seen_tick = -1
    squad.attack_last_repath_tick = -1
    _halt(sim, squad)


def _pursue_attack_order(
    sim: "Sim", squad: Squad, order: orders_mod.Attack, team: int, reach: float
) -> None:
    """Chase the `Attack` order's target: give up on it, stop in range, or walk.

    Pursuit memory lives on the squad (`attack_last_seen_tick` /
    `attack_last_repath_tick`), so it is part of `GameState` and the state
    hash. `orders.apply_attack` seeds them; a value of -1 means "no baseline
    yet", which this tick then becomes.
    """
    entity = _lookup(sim, order.target_id)
    if entity is None:
        _clear_attack_order(sim, squad)
        return

    if vision.is_visible(sim, team, entity) or squad.attack_last_seen_tick < 0:
        squad.attack_last_seen_tick = sim.state.tick
    elif sim.state.tick - squad.attack_last_seen_tick > _LOST_TARGET_TICKS:
        _clear_attack_order(sim, squad)
        return

    target = _describe(sim, squad.pos, entity)
    if target is not None and _is_engageable(sim, squad, team, target, reach):
        squad.target_id = target.id
        # Unconditionally: a squad that was MOVING when the order arrived has
        # no path left to clear but still has to settle out of MOVING (and a
        # team weapon has to redeploy) before it can fire. `_halt` is a no-op
        # for a squad that is already stopped, so a SET_UP team weapon given
        # an in-range Attack never needlessly tears down.
        _halt(sim, squad)
        return

    squad.target_id = None
    _repath_toward(sim, squad, order, entity)


def _repath_toward(sim: "Sim", squad: Squad, order: orders_mod.Attack, entity: Squad | Building) -> None:
    """Walk toward the target, re-planning at most once per `ATTACK_ORDER_REPATH_S`."""
    if squad.attack_last_repath_tick >= 0 and sim.state.tick - squad.attack_last_repath_tick < _REPATH_TICKS:
        return
    squad.attack_last_repath_tick = sim.state.tick

    from coh.sim.systems import movement

    if isinstance(entity, Squad):
        goal = cell_of(entity.pos, CELL_M)
    else:
        goal = _describe(sim, squad.pos, entity)
        goal = entity.cell if goal is None else cell_of(goal.aim_pos, CELL_M)
    movement.start_path(sim, squad, order, goal, SquadState.MOVING)


# ---------------------------------------------------------------------------
# Firing
# ---------------------------------------------------------------------------


def fire_squad(sim: "Sim", squad: Squad) -> None:
    """Give every ready member of `squad` its shot at `squad.target_id`."""
    if squad.target_id is None:
        return
    entity = _lookup(sim, squad.target_id)
    if entity is None:
        squad.target_id = None
        return
    target = _describe(sim, squad.pos, entity)
    if target is None:
        return

    sdef = sim.data.squads[squad.def_id]
    for slot, member, weapon in list(_armed_members(sim, squad)):
        if sim.state.squads.get(squad.id) is None or _lookup(sim, target.id) is None:
            return  # the shooter or the target died mid-volley
        fire_member(sim, squad, sdef, slot, member, weapon, target)


def fire_member(sim: "Sim", squad: Squad, sdef, slot: int, member: "Member", weapon: WeaponDef, target: _Target) -> None:
    """Decide whether this member fires this tick, and resolve the firing."""
    if sim.state.tick < member.next_ready_tick:
        return
    if sdef.kind == "team_weapon" and slot == 0 and squad.state is not SquadState.SET_UP:
        return  # the crewed weapon only fires deployed (arcs are task 10)
    if target.distance > weapon.ranges[2] or target.distance < weapon.min_range:
        return

    moving_mult = 1.0
    if squad.moving:
        if weapon.moving_accuracy <= 0.0:
            return
        moving_mult = weapon.moving_accuracy

    band = _range_band(weapon, target.distance)
    hit_any = resolve_bullets(sim, squad, member, weapon, band, moving_mult, target)
    sim.state.events.append(
        Event(
            kind="shot",
            tick=sim.state.tick,
            data={
                "src": squad.id,
                "dst": target.id,
                "hit": hit_any,
                "src_pos": [float(squad.pos[0]), float(squad.pos[1])],
                "dst_pos": [float(target.aim_pos[0]), float(target.aim_pos[1])],
            },
        )
    )
    schedule_next_shot(sim, squad, member, weapon, band)


def _range_band(weapon: WeaponDef, distance: float) -> int:
    short, medium, _ = weapon.ranges
    if distance <= short:
        return 0
    return 1 if distance <= medium else 2


def resolve_bullets(
    sim: "Sim",
    squad: Squad,
    member: "Member",
    weapon: WeaponDef,
    band: int,
    moving_mult: float,
    target: _Target,
) -> bool:
    """Resolve one firing (a shot or a whole burst). Returns whether any hit."""
    rng = sim.state.rng
    if weapon.burst is None:
        bullets = 1
    else:
        duration = float(rng.uniform(weapon.burst[0], weapon.burst[1]))
        bullets = max(1, int(round(weapon.rate_of_fire * duration)))

    hit_any = False
    for _ in range(bullets):
        entity = _lookup(sim, target.id)
        if entity is None:
            break  # target died mid-burst
        hit_any = _resolve_bullet(sim, squad, weapon, band, moving_mult, target) or hit_any
    return hit_any


def _resolve_bullet(
    sim: "Sim", squad: Squad, weapon: WeaponDef, band: int, moving_mult: float, target: _Target
) -> bool:
    rng = sim.state.rng
    mods = weapon.vs(target.target_type)

    if target.is_squad:
        victim_squad: Squad = target.entity  # type: ignore[assignment]
        living = [i for i, m in enumerate(victim_squad.members) if m.hp > 0]
        if not living:
            return False
        slot = living[int(rng.integers(len(living)))]
        victim = victim_squad.members[slot]
        cover = _cover_for(sim, victim_squad, slot, squad.pos)
        cover_mods = weapon.cover(cover)
        p_hit = (
            weapon.accuracy[band]
            * cover_mods.accuracy
            * moving_mult
            * mods.accuracy
            * (mods.moving if victim_squad.moving else 1.0)
            * _attacker_accuracy_mult(sim, squad)
            * _retreat_accuracy_mult(sim, victim_squad)
        )
        hit = bool(rng.random() < min(1.0, max(0.0, p_hit)))
        add_suppression(sim, squad, victim_squad, weapon, band, cover_mods.suppression * mods.suppression)
        if hit:
            apply_hit(sim, squad, weapon, target, victim=victim, cover_damage=cover_mods.damage)
        return hit

    # Buildings are large, static targets: shots at them always land. There
    # is no accuracy roll (and so no RNG draw), no cover, no movement mod and
    # no suppression; only `tt.damage` scales the hit.
    apply_hit(sim, squad, weapon, target, victim=None, cover_damage=1.0)
    return True


def _attacker_accuracy_mult(sim: "Sim", squad: Squad) -> float:
    return sim.data.economy.suppressed_accuracy_mult if squad.suppressed else 1.0


def _retreat_accuracy_mult(sim: "Sim", victim: Squad) -> float:
    if victim.state is not SquadState.RETREATING:
        return 1.0
    sdef = sim.data.squads.get(victim.def_id)
    return 1.0 if sdef is None else sdef.retreat_received_accuracy


# -- cover ------------------------------------------------------------------


def _cover_for(sim: "Sim", victim_squad: Squad, slot: int, from_pos: np.ndarray) -> str:
    if victim_squad.garrison_in is not None:
        return "garrison"
    return cover_at(sim.map, _member_cell(sim, victim_squad, slot), from_pos)


def _member_cell(sim: "Sim", squad: Squad, slot: int) -> tuple[int, int]:
    """Where the member in `slot` stands: its formation offset, or the squad
    cell if that offset lands off-map or somewhere infantry cannot stand."""
    squad_cell = cell_of(squad.pos, CELL_M)
    ox, oy = FORMATION_OFFSETS[slot % len(FORMATION_OFFSETS)]
    cos_h, sin_h = math.cos(squad.heading), math.sin(squad.heading)
    offset = np.array([ox * cos_h - oy * sin_h, ox * sin_h + oy * cos_h])
    cx, cy = cell_of(squad.pos + offset, CELL_M)
    if not (0 <= cx < sim.map.width and 0 <= cy < sim.map.height):
        return squad_cell
    if not sim.map.pass_inf[cy, cx]:
        return squad_cell
    return (cx, cy)


# -- suppression (accumulation only; task 9 owns thresholds and recovery) ----


def add_suppression(
    sim: "Sim", attacker: Squad, victim: Squad, weapon: WeaponDef, band: int, mult: float
) -> None:
    # `last_hit_tick` / `last_attacker_pos` mean "under fire", so they are set
    # per bullet whether or not it hit, and even for suppression-immune
    # squads: task 9's out-of-combat recovery timer and task 11's armour
    # facing both need them regardless of the suppression meter.
    victim.last_hit_tick = sim.state.tick
    victim.last_attacker_pos = (float(attacker.pos[0]), float(attacker.pos[1]))
    sdef = sim.data.squads.get(victim.def_id)
    if sdef is None or sdef.suppression is None:
        return  # vehicles and the like are immune
    victim.suppression = min(1.0, max(0.0, victim.suppression + weapon.suppression[band] * mult))


# -- damage -----------------------------------------------------------------


def apply_hit(
    sim: "Sim",
    attacker: Squad,
    weapon: WeaponDef,
    target: _Target,
    *,
    victim: "Member | None",
    cover_damage: float,
) -> None:
    """Land one bullet on `target`.

    Task 11 extends this with the penetration roll and deflection damage for
    vehicle targets; for now a vehicle simply takes `damage x tt.damage`.
    """
    mods = weapon.vs(target.target_type)

    if isinstance(target.entity, Building):
        building = target.entity
        building.hp -= weapon.damage * mods.damage
        if building.hp <= 0.0:
            _destroy_building(sim, building)
        return

    victim_squad: Squad = target.entity  # type: ignore[assignment]
    if victim is None:  # pragma: no cover - callers always pass a member
        return
    sdef = sim.data.squads.get(victim_squad.def_id)
    # A vehicle victim still gets `cover.accuracy` in `p_hit` (cover makes it
    # harder to hit) but not `cover.damage`: how much a hit hurts a vehicle is
    # task 11's penetration model, not the infantry cover table.
    is_vehicle = sdef is not None and sdef.kind == "vehicle"
    damage = weapon.damage * mods.damage * (1.0 if is_vehicle else cover_damage)
    victim.hp -= damage
    if victim.hp <= 0.0:
        _remove_member(sim, victim_squad, victim)


def _remove_member(sim: "Sim", squad: Squad, member: "Member") -> None:
    """A dead model leaves the squad, taking its weapon with it.

    (Task 10 keeps the team weapon itself on the field when its crew dies.)
    """
    squad.members = [m for m in squad.members if m is not member]
    if not squad.alive_members:
        _destroy_squad(sim, squad)


def _destroy_squad(sim: "Sim", squad: Squad) -> None:
    sim.state.squads.pop(squad.id, None)
    _forget_target(sim, squad.id)
    sim.state.events.append(
        Event(
            kind="squad_destroyed",
            tick=sim.state.tick,
            data={"id": squad.id, "owner": squad.owner, "def_id": squad.def_id,
                  "pos": [float(squad.pos[0]), float(squad.pos[1])]},
        )
    )


def _destroy_building(sim: "Sim", building: Building) -> None:
    sim.state.buildings.pop(building.id, None)
    getattr(sim, "_combat_footprints", {}).pop(building.id, None)
    sim.map.stamp_footprint(building.cell, _footprint(sim, building.def_id), blocked=False)
    _forget_target(sim, building.id)
    on_building_destroyed(sim, building)
    sim.state.events.append(
        Event(
            kind="building_destroyed",
            tick=sim.state.tick,
            data={"id": building.id, "owner": building.owner, "def_id": building.def_id,
                  "cell": list(building.cell)},
        )
    )


def on_building_destroyed(sim: "Sim", building: Building) -> None:
    """Hook for task 12: eject the garrison (with collapse damage).

    Called after the building leaves `state.buildings` and its footprint is
    freed, before the `building_destroyed` event is emitted.
    """


def _forget_target(sim: "Sim", entity_id: int) -> None:
    """Clear every dangling reference to a destroyed entity."""
    for sid in sorted(sim.state.squads):
        squad = sim.state.squads[sid]
        if squad.target_id == entity_id:
            squad.target_id = None
        if isinstance(squad.order, orders_mod.Attack) and squad.order.target_id == entity_id:
            _clear_attack_order(sim, squad)


# -- cooldown ---------------------------------------------------------------


def schedule_next_shot(sim: "Sim", squad: Squad, member: "Member", weapon: WeaponDef, band: int) -> None:
    rng = sim.state.rng
    seconds = float(rng.uniform(weapon.cooldown[0], weapon.cooldown[1]))
    seconds *= weapon.cooldown_range_mult[band]
    if squad.suppressed:
        seconds *= sim.data.economy.suppressed_cooldown_mult

    member.shots_since_reload += 1
    if weapon.reload_every > 0 and member.shots_since_reload >= weapon.reload_every:
        member.shots_since_reload = 0
        seconds += float(rng.uniform(weapon.reload[0], weapon.reload[1]))

    member.next_ready_tick = sim.state.tick + max(1, math.ceil(seconds * TICKS_PER_SECOND))
