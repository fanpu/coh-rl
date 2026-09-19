"""combat — target acquisition, shots, damage and deaths (tasks 8, 10 and 11).

One tick of combat is, per squad in ascending id order:

1. `acquire_target` — keep the current target while it is still a legal one,
   otherwise pick the best visible enemy in range with LOS (or follow the
   squad's `Attack` order, walking toward the target until it is in range);
1b. a vehicle then aims (`vehicle_combat.aim`): turret traverse, or a hull
   swing for a hull-mounted gun;
2. `fire_member` — per member, in loadout order, decide whether its weapon
   can fire this tick (ready, in its own range band, not moving with a
   moving-accuracy of 0, team weapon set up and pointing the right way,
   vehicle main gun on target);
3. `resolve_bullets` — one bullet for a single-shot weapon, `rate_of_fire ×
   burst_duration` for a burst weapon, each rolled separately — or, for an
   indirect weapon, `fire_shell`: one scattered bomb, no hit roll;
4. `apply_hit` / `add_suppression` — damage a randomly chosen living member
   (or the building), and accumulate suppression on the target regardless of
   whether the bullet hit; an explosion instead damages and suppresses
   everything hostile inside `aoe_radius` (`_explode`);
5. `schedule_next_shot` — cooldown and reload bookkeeping.

Task 10 also owns team weapons as *objects* rather than squads: the crewed
weapon lives in `members[0]` and stays there as the crew dies
(`_promote_gunner`), outlives the crew entirely as an abandoned shell
(`_abandon_weapon`), and can be picked up again by any infantry squad walking
onto it (`try_recrew`, driven by `orders.apply_move` and `movement`'s
arrival). `set_facing` swings a gun onto a new bearing, whether an agent asked
for it (`SetFacing`) or acquisition decided the fight had moved (
`_auto_reface`, rate-limited by `Squad.reface_hold_tick`).

Task 9's `add_suppression` spills part of every bullet's suppression to squads
near the victim (`_spill_suppression`), while `coh/sim/systems/suppression.py`
owns the slower per-tick decay and `suppressed`/`pinned` thresholds.

Task 12's garrisons touch combat at four seams: `_building_team` makes an
*occupied* neutral building a target (and `_garrison_choice` decides between
the house and the men inside it), `garrison.los_ignore` lets both ends of a
line of fire see through the walls they are standing behind, `fire_member`
swaps a garrisoned team weapon's firing arc for a re-setup timer, and
`on_building_destroyed` hands the collapse to `systems/garrison.py`.

Task 11's vehicle rules — the penetration roll behind `apply_hit`'s vehicle
branch, the turret/hull arc gate in `fire_member`, and the wreck a destroyed
vehicle leaves in `_destroy_squad` — live in `systems/vehicle_combat.py`.

RNG discipline: every draw comes from `sim.state.rng`, in a fixed order
(squads ascending id -> members in loadout order -> burst length -> per
bullet: victim, hit roll, penetration if the victim is a vehicle -> cooldown
-> reload; an indirect firing draws its scatter x then y instead, then one
penetration per vehicle its blast catches, in ascending squad id). Bullets
aimed at a building draw nothing at all: they always hit, and so does every
explosion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from coh.data.schema import WeaponDef
from coh.maps.cover import cover_at
from coh.maps.format import cell_of
from coh.sim import orders as orders_mod
from coh.sim.constants import (
    AOE_EDGE_FALLOFF,
    ATTACK_ORDER_LOST_TARGET_S,
    ATTACK_ORDER_REPATH_S,
    AUTO_REFACE_HOLD_SETUPS,
    BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT,
    CELL_M,
    FORMATION_OFFSETS,
    RECREW_RANGE_M,
    TICKS_PER_SECOND,
)
from coh.sim.state import Building, Event, Squad, SquadState
from coh.sim.systems import footprints, garrison, vehicle_combat, vision

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Member

__all__ = [
    "run",
    "acquire_target",
    "apply_hit",
    "damage_member",
    "halt",
    "on_building_destroyed",
    "abandoned_weapon_near",
    "can_recrew",
    "try_recrew",
    "set_facing",
]

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
        team = _building_team(sim, building)
        if team == _NO_TEAM:
            continue  # unowned and unoccupied: never a target
        building_ids.append(bid)
        building_team.append(team)

    return _Snapshot(
        squad_ids=np.array(ids, dtype=np.int64),
        squad_pos=np.array(pos, dtype=float).reshape(len(ids), 2),
        squad_team=np.array(teams, dtype=np.int64),
        building_ids=tuple(building_ids),
        building_team=tuple(building_team),
    )


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
        if vehicle_combat.is_vehicle(sim, squad):
            # Turret traverse / hull swing happens before firing, so a gun
            # that comes on target this tick may take its shot this tick.
            vehicle_combat.aim(sim, squad, _target_aim_pos(sim, squad))
        fire_squad(sim, squad)


def _target_aim_pos(sim: "Sim", squad: Squad) -> np.ndarray | None:
    """Where `squad` is pointing its weapons this tick, if it has a target."""
    if squad.target_id is None:
        return None
    entity = _lookup(sim, squad.target_id)
    if entity is None:
        return None
    target = _describe(sim, squad.pos, entity)
    return None if target is None else target.aim_pos


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
# Team weapons: the crewed weapon in slot 0, its firing arc, and its bearings
# ---------------------------------------------------------------------------


def _team_weapon(sim: "Sim", squad: Squad) -> WeaponDef | None:
    """The crewed weapon `members[0]` carries, if this is a team weapon squad.

    Member 0 is always the gunner (`_promote_gunner` keeps it that way when
    the gunner dies), so the weapon is read off the model rather than off the
    squad def: a re-crewed shell has the same weapon in the same slot.
    """
    sdef = sim.data.squads.get(squad.def_id)
    if sdef is None or sdef.kind != "team_weapon" or not squad.members:
        return None
    gunner = squad.members[0]
    if gunner.hp <= 0:
        return None
    return _weapon_of(sim, gunner)


def _arc_weapon(sim: "Sim", squad: Squad) -> WeaponDef | None:
    """The team weapon, if it has a firing arc narrow enough to matter."""
    weapon = _team_weapon(sim, squad)
    return weapon if weapon is not None and weapon.arc_deg < 360.0 else None


def _indirect_weapon(sim: "Sim", squad: Squad) -> WeaponDef | None:
    """The team weapon, if it lobs its shells over the terrain (mortars).

    None for a garrisoned crew: a mortar cannot be dropped through a roof, so
    the tube never fires from inside (task 12) and the squad neither gains
    indirect fire's LOS exemption nor its minimum range.
    """
    if squad.garrison_in is not None:
        return None
    weapon = _team_weapon(sim, squad)
    return weapon if weapon is not None and weapon.indirect else None


def _bearing(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    delta = to_pos - from_pos
    return math.atan2(float(delta[1]), float(delta[0]))


def _in_arc(squad: Squad, weapon: WeaponDef, aim_pos: np.ndarray) -> bool:
    """Is `aim_pos` within `arc_deg / 2` of where the gun points?"""
    if weapon.arc_deg >= 360.0:
        return True
    from coh.sim.systems import movement

    facing = squad.heading if squad.facing is None else squad.facing
    offset = movement.angle_diff(_bearing(squad.pos, aim_pos), facing)
    return offset <= math.radians(weapon.arc_deg) / 2.0


def set_facing(sim: "Sim", squad: Squad, direction: float, *, auto: bool) -> None:
    """Swing the gun to `direction` (radians), paying `2 x setup_time`.

    The squad tears down (`setup_time`), then `movement` walks it through its
    ordinary arrival path and sets it up again on the new bearing (another
    `setup_time`). `heading` moves with `facing`: the crew turns bodily, and
    `_on_arrival` derives the new `facing` from the heading.

    `auto` marks an acquisition-driven re-face, which is rate-limited by
    `squad.reface_hold_tick` so that two enemies on opposite sides cannot
    make the crew spin in place forever without ever firing. An explicit
    `SetFacing` (`auto=False`) is always honoured and clears the hold.
    """
    from coh.sim.systems import movement

    sdef = sim.data.squads.get(squad.def_id)
    if sdef is None:
        return
    ticks = movement.setup_ticks(sim, sdef)
    squad.heading = direction
    squad.facing = direction
    squad.path = []
    if not auto:
        squad.reface_hold_tick = 0
    if squad.garrison_in is not None:
        # A gun in a building covers every window: the bearing is recorded
        # but there is no arc to swing and nothing to tear down (task 12).
        return
    if ticks <= 0:  # a "team weapon" with no setup time simply pivots
        return
    squad.state = SquadState.TEARING_DOWN
    squad.setup_done_tick = sim.state.tick + ticks
    if auto:
        squad.reface_hold_tick = sim.state.tick + AUTO_REFACE_HOLD_SETUPS * ticks


def _auto_reface(sim: "Sim", squad: Squad, aim_pos: np.ndarray) -> None:
    if sim.state.tick < squad.reface_hold_tick:
        return
    set_facing(sim, squad, _bearing(squad.pos, aim_pos), auto=True)


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

    A deployed team weapon with a firing arc prefers targets it can actually
    shoot at. Only when nothing is in the arc, and some enemy is otherwise
    engageable, does it swing the gun around (paying `2 x setup_time`).
    """
    team = _team_of(sim, squad.owner)
    reach = _max_range(sim, squad)
    if reach <= 0.0:
        squad.target_id = None
        return

    if isinstance(squad.order, orders_mod.Attack):
        _pursue_attack_order(sim, squad, squad.order, team, reach)
        return

    # Only a deployed gun is arc-limited: one that is still setting up (or
    # walking) cannot fire its team weapon at all, so restricting what it may
    # look at would only blind its crew's small arms.
    arc = _arc_weapon(sim, squad) if squad.state is SquadState.SET_UP else None

    if squad.target_id is not None:
        current = _lookup(sim, squad.target_id)
        target = None if current is None else _describe(sim, squad.pos, current)
        if (
            target is not None
            and _is_engageable(sim, squad, team, target, reach)
            and (arc is None or _in_arc(squad, arc, target.aim_pos))
        ):
            return
        squad.target_id = None

    best = _choose_target(sim, squad, team, reach, snapshot, arc)
    if best is not None:
        squad.target_id = best.id
        return

    squad.target_id = None
    if arc is None:
        return
    # Nothing in the arc: is anything worth turning the gun for?
    elsewhere = _choose_target(sim, squad, team, reach, snapshot, None)
    if elsewhere is None:
        return
    squad.target_id = elsewhere.id
    _auto_reface(sim, squad, elsewhere.aim_pos)


def _is_engageable(sim: "Sim", squad: Squad, team: int, target: _Target, reach: float) -> bool:
    """Visible to the squad's team, hostile, within reach and in LOS.

    An indirect weapon drops the LOS requirement -- a mortar shells anything
    its *team* can see -- and gains a minimum range in exchange.
    """
    if isinstance(target.entity, Building):
        if _building_team(sim, target.entity) in (team, _NO_TEAM):
            return False  # friendly, or an empty neutral house: not a target
    elif _team_of(sim, target.entity.owner) == team:
        return False
    if target.distance > reach:
        return False
    if not vision.is_visible(sim, team, target.entity):
        return False
    indirect = _indirect_weapon(sim, squad)
    if indirect is not None:
        return target.distance >= indirect.min_range
    return vision.has_los(sim.map, squad.pos, target.aim_pos, garrison.los_ignore(sim, squad, target.entity))


def _choose_target(
    sim: "Sim",
    squad: Squad,
    team: int,
    reach: float,
    snapshot: _Snapshot,
    arc_weapon: WeaponDef | None = None,
) -> _Target | None:
    """Best `(priority desc, distance asc, id asc)` engageable enemy, or None.

    With `arc_weapon`, only targets inside that weapon's firing arc count.
    """
    primary = _primary_weapon(sim, squad)
    if primary is None:
        return None

    best: _Target | None = None
    best_key: tuple[float, float, int] | None = None

    for entity in _candidates(sim, squad, team, reach, snapshot):
        target = _describe(sim, squad.pos, entity)
        if target is None or not _is_engageable(sim, squad, team, target, reach):
            continue
        if arc_weapon is not None and not _in_arc(squad, arc_weapon, target.aim_pos):
            continue
        mods = primary.vs(target.target_type)
        if isinstance(entity, Building) and mods.damage < BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT:
            continue
        if not _garrison_choice(sim, primary, entity):
            continue
        key = (-mods.priority, target.distance, target.id)
        if best_key is None or key < best_key:
            best, best_key = target, key
    return best


def _garrison_choice(sim: "Sim", weapon: WeaponDef, entity: Squad | Building) -> bool:
    """Should `weapon` *auto*-acquire this end of a garrison at all?

    An occupied house and the squad inside it sit on the same spot, so one of
    the two has to be dropped: a weapon goes for the building when it hurts
    the building at least as much as it hurts the occupants (and can hurt it
    at all), and for the squad otherwise. An explicit `Attack` order bypasses
    this — it never goes through `_choose_target`.
    """
    if isinstance(entity, Building):
        if not entity.neutral:
            return True  # a player's building is a target in its own right
        return _prefers_building(sim, weapon, entity)
    if entity.garrison_in is None:
        return True
    building = sim.state.buildings.get(entity.garrison_in)
    return building is None or not _prefers_building(sim, weapon, building)


def _prefers_building(sim: "Sim", weapon: WeaponDef, building: Building) -> bool:
    """Would `weapon` rather knock the house down than shoot the men in it?

    Compared against the lowest-id occupant's target type, so the answer is
    the same whichever end of the pair `_garrison_choice` is asked about.
    """
    inside = garrison.occupants(sim, building)
    if not inside:
        return False
    target_type = _building_target_type(sim, building)
    if target_type is None:
        return False
    vs_building = weapon.vs(target_type).damage
    if vs_building < BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT:
        return False
    sdef = sim.data.squads.get(inside[0].def_id)
    if sdef is None:
        return True
    return vs_building >= weapon.vs(sdef.target_type).damage


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
        cells = footprints.centres(sim, building)
        if ((cells - squad.pos) ** 2).sum(axis=1).min() <= reach * reach:
            yield building


# -- Attack order -----------------------------------------------------------


def halt(sim: "Sim", squad: Squad) -> None:
    """Stop walking, the way movement's arrival does.

    A team weapon that stops deploys again, so that it can actually fire the
    target it just walked into range of -- facing that target if it has one,
    otherwise its last travel direction.
    """
    squad.path = []
    if squad.state is not SquadState.MOVING:
        return
    from coh.sim.systems import movement

    sdef = sim.data.squads[squad.def_id]
    if sdef.kind == "team_weapon":
        squad.state = SquadState.SETTING_UP
        squad.facing = _stopping_facing(sim, squad)
        squad.setup_done_tick = sim.state.tick + movement.setup_ticks(sim, sdef)
    else:
        squad.state = SquadState.IDLE


def _stopping_facing(sim: "Sim", squad: Squad) -> float:
    """Where a team weapon that just stopped points its gun: at the target it
    stopped for, or failing that along its last travel direction."""
    if squad.target_id is not None:
        entity = _lookup(sim, squad.target_id)
        target = None if entity is None else _describe(sim, squad.pos, entity)
        if target is not None:
            return _bearing(squad.pos, target.aim_pos)
    return squad.heading


def _clear_attack_order(sim: "Sim", squad: Squad) -> None:
    squad.order = None
    squad.target_id = None
    squad.attack_last_seen_tick = -1
    squad.attack_last_repath_tick = -1
    halt(sim, squad)


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
        # team weapon has to redeploy) before it can fire. `halt` is a no-op
        # for a squad that is already stopped, so a SET_UP team weapon given
        # an in-range Attack never needlessly tears down.
        halt(sim, squad)
        return

    squad.target_id = None
    if squad.garrison_in is not None:
        return  # a garrisoned squad holds the building; it never walks out
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
            if not _in_arc(squad, weapon, target.aim_pos):
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
        hit_any = fire_shell(sim, squad, weapon, band, target)
    elif _indirect_weapon(sim, squad) is not None and not vision.has_los(
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


# ---------------------------------------------------------------------------
# Indirect fire: one scattered shell per firing, resolved as an explosion
# ---------------------------------------------------------------------------


def fire_shell(sim: "Sim", squad: Squad, weapon: WeaponDef, band: int, target: _Target) -> bool:
    """Lob one shell at `target` and resolve where it lands. Returns whether
    it hurt anything.

    There is no hit roll: the shell scatters instead. Each axis is offset by
    `N(0, scatter_m * d / max_range)`, drawn x then y so that the RNG stream
    stays fixed. Burst fields are ignored -- a mortar fires one bomb.
    """
    sigma = weapon.scatter_m * target.distance / weapon.ranges[2] if weapon.ranges[2] > 0 else 0.0
    rng = sim.state.rng
    offset_x = float(rng.normal(0.0, sigma))
    offset_y = float(rng.normal(0.0, sigma))
    impact = target.aim_pos + np.array([offset_x, offset_y])

    hurt = _explode(sim, squad, weapon, band, impact)
    sim.state.events.append(
        Event(
            kind="explosion",
            tick=sim.state.tick,
            data={
                "src": squad.id,
                "pos": [float(impact[0]), float(impact[1])],
                "radius": weapon.aoe_radius,
            },
        )
    )
    return hurt


def _falloff(distance: float, radius: float) -> float:
    """Linear from 1.0 at the impact point to `AOE_EDGE_FALLOFF` at `radius`."""
    return 1.0 - (1.0 - AOE_EDGE_FALLOFF) * (distance / radius)


def _explode(sim: "Sim", attacker: Squad, weapon: WeaponDef, band: int, impact: np.ndarray) -> bool:
    """Damage and suppress everything hostile inside `weapon.aoe_radius`.

    Only enemies of the shooter are caught: CoH mortars do hurt their own
    side, but M1's bots are not clever enough to be trusted with that.
    """
    radius = weapon.aoe_radius
    if radius <= 0.0:
        return False
    team = _team_of(sim, attacker.owner)
    hurt = False
    for sid in sorted(sim.state.squads):
        victim = sim.state.squads.get(sid)
        if victim is None or not _is_targetable(victim) or _team_of(sim, victim.owner) == team:
            continue
        hurt = _explode_on_squad(sim, attacker, weapon, band, impact, victim) or hurt
    for bid in sorted(sim.state.buildings):
        building = sim.state.buildings.get(bid)
        if building is None or _building_team(sim, building) in (team, _NO_TEAM):
            continue
        hurt = _explode_on_building(sim, weapon, impact, building) or hurt
    return hurt


def _explode_on_squad(
    sim: "Sim", attacker: Squad, weapon: WeaponDef, band: int, impact: np.ndarray, victim: Squad
) -> bool:
    sdef = sim.data.squads.get(victim.def_id)
    if sdef is None:
        return False
    mods = weapon.vs(sdef.target_type)
    is_vehicle = sdef.kind == "vehicle"

    # Snapshot the crew first: members are removed as they die, and every
    # model in the blast is meant to be caught where it stood when it went
    # off, not where the survivors shuffle to afterwards.
    caught = []
    for slot, member in enumerate(victim.members):
        if member.hp <= 0:
            continue
        distance = float(np.linalg.norm(_member_pos(sim, victim, slot) - impact))
        if distance > weapon.aoe_radius:
            continue
        cover = _cover_for(sim, victim, slot, impact)
        caught.append((member, distance, cover))
    if not caught:
        return False

    hurt = False
    for member, distance, cover in caught:
        # A vehicle caught in a blast rolls penetration (against the armour
        # facing the impact point) instead of taking infantry cover damage.
        if is_vehicle:
            cover_damage = vehicle_combat.penetration_mult(sim, weapon, band, victim, impact, mods)
        else:
            cover_damage = weapon.cover(cover).damage
        damage = weapon.damage * mods.damage * cover_damage * _falloff(distance, weapon.aoe_radius)
        if damage <= 0.0:
            continue
        hurt = True
        member.hp -= damage
        if member.hp <= 0.0:
            _remove_member(sim, victim, member)

    # One dose of suppression per shell, not per model, taken against the
    # cover of the model nearest the blast.
    _, _, best_cover = min(caught, key=lambda entry: entry[1])
    cover_mods = weapon.cover(best_cover)
    if _is_targetable(victim):  # it may have been wiped out by the blast
        add_suppression(sim, attacker, victim, weapon, band, cover_mods.suppression * mods.suppression)
    return hurt


def _explode_on_building(sim: "Sim", weapon: WeaponDef, impact: np.ndarray, building: Building) -> bool:
    cells = footprints.centres(sim, building)
    distance = math.sqrt(float(((cells - impact) ** 2).sum(axis=1).min()))
    if distance > weapon.aoe_radius:
        return False
    target_type = _building_target_type(sim, building)
    if target_type is None:
        return False
    damage = weapon.damage * weapon.vs(target_type).damage * _falloff(distance, weapon.aoe_radius)
    if damage <= 0.0:
        return False
    building.hp -= damage
    if building.hp <= 0.0:
        _destroy_building(sim, building)
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
    if is_vehicle:
        armour = vehicle_combat.penetration_mult(sim, weapon, band, victim_squad, attacker.pos, mods)
        damage = weapon.damage * mods.damage * armour
    else:
        damage = weapon.damage * mods.damage * cover_damage
    victim.hp -= damage
    if victim.hp <= 0.0:
        _remove_member(sim, victim_squad, victim)


def damage_member(sim: "Sim", squad: Squad, member: "Member", amount: float) -> None:
    """Hurt one model directly, retiring it (and its squad) if it dies.

    The raw damage path behind `apply_hit`, exposed for damage that isn't a
    bullet: task 12's garrison collapse.
    """
    member.hp -= amount
    if member.hp <= 0.0:
        _remove_member(sim, squad, member)


def _remove_member(sim: "Sim", squad: Squad, member: "Member") -> None:
    """A dead model leaves the squad, taking its weapon with it.

    A team weapon is the exception: it belongs to the gun, not to the man
    behind it. When the gunner falls the next crewman takes over
    (`_promote_gunner`), and when the last of them dies the weapon itself
    stays on the field as an abandoned shell (`_abandon_weapon`).
    """
    was_gunner = bool(squad.members) and squad.members[0] is member
    squad.members = [m for m in squad.members if m is not member]
    sdef = sim.data.squads.get(squad.def_id)
    is_team_weapon = sdef is not None and sdef.kind == "team_weapon"

    if not squad.alive_members:
        if is_team_weapon:
            _abandon_weapon(sim, squad)
        else:
            _destroy_squad(sim, squad)
        return

    if was_gunner and is_team_weapon and sdef.loadout:
        _promote_gunner(squad, sdef.loadout[0])


def _promote_gunner(squad: Squad, weapon_id: str) -> None:
    """The next crewman steps up to the gun, dropping his own weapon.

    His cooldown/reload bookkeeping belongs to the weapon he just let go of,
    so it resets with the hand-over.
    """
    gunner = squad.members[0]
    gunner.weapon = weapon_id
    gunner.next_ready_tick = 0
    gunner.shots_since_reload = 0
    gunner.burst_until_tick = 0


def _abandon_weapon(sim: "Sim", squad: Squad) -> None:
    """The last crewman died: leave the gun on the field, crewless.

    The `Squad` stays in `state.squads` as a shell so that it can be
    re-crewed. `owner` is left alone -- `abandoned` is what every other
    system reads -- but the shell is no longer visible, targetable, or
    counted towards its old owner's population and upkeep. A gun whose crew
    died inside a garrison is carried out of the building first (task 12),
    so that somebody can still walk up to it.
    """
    squad.members = []
    squad.abandoned = True
    squad.state = SquadState.IDLE
    squad.order = None
    squad.path = []
    squad.target_id = None
    squad.moving = False
    squad.suppression = 0.0
    squad.suppressed = False
    squad.pinned = False
    squad.reinforcing = False
    squad.recrew_target = None
    squad.reface_hold_tick = 0
    # A gun crewed inside a building is carried out to an exit cell rather
    # than left on the (impassable) footprint, where no squad could ever get
    # within `RECREW_RANGE_M` of it and the gun would be lost for good.
    # `settle=False`: the shell's own resting state is set above.
    garrison.leave(sim, squad, settle=False)
    _forget_target(sim, squad.id)
    sim.state.events.append(
        Event(
            kind="weapon_abandoned",
            tick=sim.state.tick,
            data={"id": squad.id, "owner": squad.owner, "def_id": squad.def_id,
                  "pos": [float(squad.pos[0]), float(squad.pos[1])]},
        )
    )


# ---------------------------------------------------------------------------
# Re-crewing an abandoned team weapon
# ---------------------------------------------------------------------------


def can_recrew(sim: "Sim", squad: Squad) -> bool:
    """Could `squad` man an abandoned weapon? Infantry, two models or more."""
    sdef = sim.data.squads.get(squad.def_id)
    if sdef is None or sdef.kind in ("vehicle", "team_weapon") or squad.abandoned:
        return False
    return len(squad.alive_members) >= 2


def abandoned_weapon_near(sim: "Sim", pos: np.ndarray) -> Squad | None:
    """The nearest abandoned team weapon within `RECREW_RANGE_M` of `pos`."""
    best: Squad | None = None
    best_distance = RECREW_RANGE_M
    for sid in sorted(sim.state.squads):
        shell = sim.state.squads[sid]
        if not shell.abandoned:
            continue
        distance = float(np.linalg.norm(shell.pos - np.asarray(pos, dtype=float)))
        if distance <= best_distance and (best is None or distance < best_distance):
            best, best_distance = shell, distance
    return best


def try_recrew(sim: "Sim", squad: Squad) -> bool:
    """`squad` has arrived: if it was sent to man an abandoned weapon and is
    standing next to it, hand the gun over and retire the squad.

    The shell takes the arriving squad's owner and models (capped at the
    weapon's own crew size -- surplus models are lost), the new member 0
    takes the team weapon, and the gun starts setting up on the arriving
    squad's heading. The arriving squad leaves the field like a destroyed
    one, but as a `weapon_recrewed` rather than a `squad_destroyed`.
    """
    shell_id = squad.recrew_target
    squad.recrew_target = None
    if shell_id is None:
        return False
    shell = sim.state.squads.get(shell_id)
    if shell is None or not shell.abandoned or not can_recrew(sim, squad):
        return False
    if float(np.linalg.norm(shell.pos - squad.pos)) > RECREW_RANGE_M:
        return False
    shell_def = sim.data.squads.get(shell.def_id)
    if shell_def is None or not shell_def.loadout:
        return False

    from coh.sim.systems import movement

    crew = squad.alive_members[: shell_def.members]
    shell.owner = squad.owner
    shell.members = crew
    _promote_gunner(shell, shell_def.loadout[0])
    shell.abandoned = False
    shell.heading = squad.heading
    shell.facing = squad.heading
    shell.state = SquadState.SETTING_UP
    shell.setup_done_tick = sim.state.tick + movement.setup_ticks(sim, shell_def)

    sim.state.squads.pop(squad.id, None)
    _forget_target(sim, squad.id)
    sim.state.events.append(
        Event(
            kind="weapon_recrewed",
            tick=sim.state.tick,
            data={"id": shell.id, "owner": shell.owner, "def_id": shell.def_id,
                  "by": squad.id, "pos": [float(shell.pos[0]), float(shell.pos[1])]},
        )
    )
    return True


def _destroy_squad(sim: "Sim", squad: Squad) -> None:
    """Take a squad off the field.

    A vehicle leaves a wreck where it died and announces itself as a
    `vehicle_destroyed` rather than a `squad_destroyed` -- one event per
    death, never both (task 11).
    """
    kind = "squad_destroyed"
    if vehicle_combat.is_vehicle(sim, squad):
        vehicle_combat.leave_wreck(sim, squad)
        kind = vehicle_combat.VEHICLE_DESTROYED
    garrison.detach(sim, squad)
    sim.state.squads.pop(squad.id, None)
    _forget_target(sim, squad.id)
    sim.state.events.append(
        Event(
            kind=kind,
            tick=sim.state.tick,
            data={"id": squad.id, "owner": squad.owner, "def_id": squad.def_id,
                  "pos": [float(squad.pos[0]), float(squad.pos[1])]},
        )
    )


def _destroy_building(sim: "Sim", building: Building) -> None:
    sim.state.buildings.pop(building.id, None)
    sim.map.stamp_footprint(building.cell, footprints.size(sim, building.def_id), blocked=False)
    _forget_target(sim, building.id)
    on_building_destroyed(sim, building)
    # After the hook: ejecting the garrison still needs the building's
    # geometry, and the geometry of a dead building can no longer change.
    footprints.forget(sim, building.id)
    sim.state.events.append(
        Event(
            kind="building_destroyed",
            tick=sim.state.tick,
            data={"id": building.id, "owner": building.owner, "def_id": building.def_id,
                  "cell": list(building.cell)},
        )
    )


def on_building_destroyed(sim: "Sim", building: Building) -> None:
    """Eject the garrison (with collapse damage) and clear up after the wreck.

    Called after the building leaves `state.buildings` and its footprint is
    freed, before the `building_destroyed` event is emitted. The rules live in
    `systems/garrison.py`.
    """
    garrison.on_building_destroyed(sim, building)


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
