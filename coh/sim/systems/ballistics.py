"""ballistics — target descriptions, per-member fire decisions, and bullets.

This is the shared base every other firing-related module builds on
(`destruction ← ballistics ← explosions/team_weapons ← combat`, per
`combat.py`'s module docstring): besides the actual gun mechanics --
`fire_member`'s per-member fire decision, `resolve_bullets` / `apply_hit`'s
p_hit composition and damage, `add_suppression` + its spill, and
`schedule_next_shot`'s cooldown/reload bookkeeping -- it also owns the
`_Target` description (aim point, distance, target type) and the small
"whose team is this" / "is this thing still alive" primitives
(`_lookup`, `_team_of`, `_is_targetable`, `_building_team`, `_weapon_of`,
...) that `combat.py`'s acquisition, `explosions.py`'s blast resolution and
`team_weapons.py`'s facing all need. They live here rather than in
`combat.py` (which uses them too, and always will) so that those downstream
modules can import them without importing `combat.py` itself and creating a
cycle -- `combat.py` re-exports everything for backward-compatible imports
and test access.

`fire_member` is the one place this module cannot stay one-directional on
its own: whether a member fires this tick depends on `team_weapons`'s firing
arc and indirect-weapon rules, and firing it dispatches to either this
module's own bullet path or `explosions.fire_shell`. Both of those modules
import `ballistics` at load time, so `fire_member` imports them back lazily,
inside the function, to break the cycle (documented at each call site).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from coh.data.schema import WeaponDef
from coh.maps.cover import cover_at
from coh.maps.format import cell_of
from coh.sim.constants import CELL_M, FORMATION_OFFSETS, TICKS_PER_SECOND
from coh.sim.state import Building, Event, Squad, SquadState
from coh.sim.systems import destruction, footprints, garrison, vehicle_combat, vision

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Member

__all__ = ["apply_hit", "add_suppression"]

_NO_TEAM = -1


# ---------------------------------------------------------------------------
# Teams / liveness
# ---------------------------------------------------------------------------


def _team_of(sim: "Sim", owner: int | None) -> int:
    if owner is None:
        return _NO_TEAM
    player = sim.state.players.get(owner)
    return _NO_TEAM if player is None else player.team


def _is_targetable(squad: Squad) -> bool:
    """Abandoned squads and squads with no members are never targets."""
    return not squad.abandoned and bool(squad.alive_members)


def _building_team(sim: "Sim", building: Building) -> int:
    """Whose building this is for targeting purposes.

    An *occupied* neutral building belongs to the team hiding inside it: a
    house full of riflemen is a legitimate thing to shell, while an empty one
    is map furniture (task 12).
    """
    if building.neutral:
        team = garrison.occupying_team(sim, building)
        return _NO_TEAM if team is None else team
    return _team_of(sim, building.owner)


def _lookup(sim: "Sim", entity_id: int) -> Squad | Building | None:
    squad = sim.state.squads.get(entity_id)
    if squad is not None:
        return squad if _is_targetable(squad) else None
    return sim.state.buildings.get(entity_id)


# ---------------------------------------------------------------------------
# Weapons / target descriptions
# ---------------------------------------------------------------------------


def _weapon_of(sim: "Sim", member: "Member") -> WeaponDef | None:
    """The member's weapon, or None if it is unarmed / carries an unknown id."""
    if not member.weapon:
        return None
    return sim.data.weapons.get(member.weapon)


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
    cells = footprints.centres(sim, entity)
    d2 = ((cells - from_pos) ** 2).sum(axis=1)
    nearest = int(np.argmin(d2))  # ties -> lowest index: deterministic
    return _Target(entity, target_type, cells[nearest], math.sqrt(float(d2[nearest])))


# ---------------------------------------------------------------------------
# Firing
# ---------------------------------------------------------------------------


def fire_member(sim: "Sim", squad: Squad, sdef, slot: int, member: "Member", weapon: WeaponDef, target: _Target) -> None:
    """Decide whether this member fires this tick, and resolve the firing."""
    if sim.state.tick < member.next_ready_tick:
        return
    # Deferred: `team_weapons.py` imports this module at load time (for
    # `_lookup` / `_describe`), so importing it back here at module level
    # would be circular. `_in_arc` (the firing-arc gate below) and
    # `_indirect_weapon` (the mortar-LOS check further down) are only needed
    # once fire_member actually runs, long after both modules are loaded.
    from coh.sim.systems import team_weapons

    if sdef.kind == "team_weapon" and slot == 0:
        # The crewed weapon fires only deployed, and only inside its arc; the
        # crew's own small arms (later slots) are not arc-limited.
        if squad.garrison_in is not None:
            # Inside a building the gun still has to be set up after moving
            # in, but it covers every window: no arc, and no lobbing mortar
            # bombs through the roof (task 12).
            if sim.state.tick < squad.setup_done_tick or weapon.indirect:
                return
        else:
            if squad.state is not SquadState.SET_UP:
                return
            if not team_weapons._in_arc(squad, weapon, target.aim_pos):
                return
    if not vehicle_combat.may_fire(squad, sdef, slot, weapon, target.aim_pos):
        return  # a vehicle's main gun is still traversing onto the target
    if target.distance > weapon.ranges[2] or target.distance < weapon.min_range:
        return

    moving_mult = 1.0
    if squad.moving:
        if weapon.moving_accuracy <= 0.0:
            return
        moving_mult = weapon.moving_accuracy

    band = _range_band(weapon, target.distance)
    if weapon.indirect:
        # Deferred: `explosions.py` imports this module at load time (for
        # `add_suppression`, `_cover_for`, ...), so importing it back here at
        # module level would be circular.
        from coh.sim.systems import explosions

        hit_any = explosions.fire_shell(sim, squad, weapon, band, target)
    elif team_weapons._indirect_weapon(sim, squad) is not None and not vision.has_los(
        sim.map, squad.pos, target.aim_pos, garrison.los_ignore(sim, squad, target.entity)
    ):
        # Acquisition dropped the LOS test for this squad's tube; the crew's
        # own small arms still cannot shoot through the hedgerow.
        return
    else:
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
            apply_hit(sim, squad, weapon, target, victim=victim, cover_damage=cover_mods.damage, band=band)
        return hit

    # Buildings are large, static targets: shots at them always land. There
    # is no accuracy roll (and so no RNG draw), no cover, no movement mod and
    # no suppression; only `tt.damage` scales the hit.
    apply_hit(sim, squad, weapon, target, victim=None, cover_damage=1.0, band=band)
    return True


def _attacker_accuracy_mult(sim: "Sim", squad: Squad) -> float:
    return sim.data.economy.suppressed_accuracy_mult if squad.suppressed else 1.0


def _retreat_accuracy_mult(sim: "Sim", victim: Squad) -> float:
    if victim.state is not SquadState.RETREATING:
        return 1.0
    sdef = sim.data.squads.get(victim.def_id)
    return 1.0 if sdef is None else sdef.retreat_received_accuracy


# -- cover --------------------------------------------------------------


def _cover_for(sim: "Sim", victim_squad: Squad, slot: int, from_pos: np.ndarray) -> str:
    if victim_squad.garrison_in is not None:
        return "garrison"
    return cover_at(sim.map, _member_cell(sim, victim_squad, slot), from_pos)


def _member_pos(sim: "Sim", squad: Squad, slot: int) -> np.ndarray:
    """World position of the member in `slot`: the squad position offset by
    its formation slot, rotated by the squad's heading."""
    ox, oy = FORMATION_OFFSETS[slot % len(FORMATION_OFFSETS)]
    cos_h, sin_h = math.cos(squad.heading), math.sin(squad.heading)
    return squad.pos + np.array([ox * cos_h - oy * sin_h, ox * sin_h + oy * cos_h])


def _member_cell(sim: "Sim", squad: Squad, slot: int) -> tuple[int, int]:
    """Where the member in `slot` stands: its formation offset, or the squad
    cell if that offset lands off-map or somewhere infantry cannot stand."""
    squad_cell = cell_of(squad.pos, CELL_M)
    cx, cy = cell_of(_member_pos(sim, squad, slot), CELL_M)
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
    s = weapon.suppression[band] * mult
    sdef = sim.data.squads.get(victim.def_id)
    # A squad that is retreating ignores suppression entirely (task 9): it is
    # cleared to 0 on `Retreat` and never accumulates again while en route.
    if sdef is not None and sdef.suppression is not None and victim.state is not SquadState.RETREATING:
        victim.suppression = min(1.0, max(0.0, victim.suppression + s))
    _spill_suppression(sim, victim, weapon, s)


def _spill_suppression(sim: "Sim", victim: Squad, weapon: WeaponDef, s: float) -> None:
    """Spill part of a bullet's suppression to squads near the victim.

    Every other squad on the victim's side (an enemy of the shooter) within
    `weapon.nearby_suppression_radius` of the victim's position gains
    `s * weapon.nearby_suppression_mult`. This only sets `last_hit_tick`
    (recovery's out-of-combat timer); it never touches `last_attacker_pos`,
    since the spilled-onto squad was not itself the bullet's aim point.
    """
    if weapon.nearby_suppression_radius <= 0.0 or s == 0.0:
        return
    victim_team = _team_of(sim, victim.owner)
    spill = s * weapon.nearby_suppression_mult
    radius2 = weapon.nearby_suppression_radius * weapon.nearby_suppression_radius
    for sid in sorted(sim.state.squads):
        if sid == victim.id:
            continue
        other = sim.state.squads[sid]
        if not other.alive_members or other.state is SquadState.RETREATING:
            continue
        if _team_of(sim, other.owner) != victim_team:
            continue
        other_sdef = sim.data.squads.get(other.def_id)
        if other_sdef is None or other_sdef.suppression is None:
            continue
        dist2 = float(((other.pos - victim.pos) ** 2).sum())
        if dist2 > radius2:
            continue
        other.last_hit_tick = sim.state.tick
        other.suppression = min(1.0, max(0.0, other.suppression + spill))


# -- damage -----------------------------------------------------------------


def apply_hit(
    sim: "Sim",
    attacker: Squad,
    weapon: WeaponDef,
    target: _Target,
    *,
    victim: "Member | None",
    cover_damage: float,
    band: int,
) -> None:
    """Land one bullet on `target`.

    A vehicle victim rolls `vehicle_combat.penetration_mult` for this hit
    instead of taking infantry cover damage (task 11).
    """
    mods = weapon.vs(target.target_type)

    if isinstance(target.entity, Building):
        building = target.entity
        building.hp -= weapon.damage * mods.damage
        if building.hp <= 0.0:
            destruction._destroy_building(sim, building)
        return

    victim_squad: Squad = target.entity  # type: ignore[assignment]
    if victim is None:  # pragma: no cover - callers always pass a member
        return
    sdef = sim.data.squads.get(victim_squad.def_id)
    # A vehicle victim still gets `cover.accuracy` in `p_hit` (cover makes it
    # harder to hit) but not `cover.damage`: how much a hit hurts a vehicle is
    # task 11's penetration model, not the infantry cover table.
    is_vehicle = sdef is not None and sdef.kind == "vehicle"
    if is_vehicle:
        armour = vehicle_combat.penetration_mult(sim, weapon, band, victim_squad, attacker.pos, mods)
        damage = weapon.damage * mods.damage * armour
    else:
        damage = weapon.damage * mods.damage * cover_damage
    victim.hp -= damage
    if victim.hp <= 0.0:
        destruction._remove_member(sim, victim_squad, victim)


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
