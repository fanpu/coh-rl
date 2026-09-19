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

This module is the entry point (`run`) and owns target acquisition and
`Attack`-order pursuit; the mechanics behind each step now live in sibling
modules, one-directionally dependent so the split stays acyclic:

    destruction.py <- ballistics.py <- explosions.py, team_weapons.py <- combat.py

- `ballistics.py` — the `_Target` description and the "whose team, is it
  still alive" primitives every module needs, plus the per-member fire
  decision (`fire_member`), bullet resolution, `apply_hit`, suppression
  (`add_suppression` + its spill) and cooldown/reload bookkeeping.
- `explosions.py` — indirect fire: shell scatter and AOE resolution against
  squads and buildings.
- `team_weapons.py` — firing arcs, auto re-face + hysteresis, halting,
  gunner succession's caller-facing half, and re-crewing.
- `destruction.py` — squad/vehicle/building death paths, gunner succession,
  weapon abandonment, `on_building_destroyed` dispatch, dangling-target
  cleanup.

Everything those modules expose is re-exported here (imported at module
level, not just listed in `__all__`), so every existing `combat.foo` /
`combat._foo` import -- callers in `orders.py`, `garrison.py`, `movement.py`,
and every test that reaches into a "private" helper -- keeps working
unchanged. A couple of call sites *into* those modules had to become
function-local imports to avoid a cycle (`ballistics.fire_member` reaching
into `team_weapons` and `explosions`; `destruction._forget_target` reaching
into `team_weapons`); each is documented at its call site in the module that
contains it.

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from coh.data.schema import WeaponDef
from coh.maps.format import cell_of
from coh.sim import orders as orders_mod
from coh.sim.constants import (
    ATTACK_ORDER_LOST_TARGET_S,
    ATTACK_ORDER_REPATH_S,
    BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT,
    CELL_M,
    TICKS_PER_SECOND,
)
from coh.sim.state import Building, Squad, SquadState, entity_ids
from coh.sim.systems import footprints, garrison, vehicle_combat, vision

# Re-exported for backward compatibility: every name other modules and tests
# already reach through `combat.*` (see the module docstring's dependency
# list for who owns what now).
from coh.sim.systems.ballistics import (  # noqa: F401
    _NO_TEAM,
    _Target,
    _armed_members,
    _attacker_accuracy_mult,
    _building_target_type,
    _building_team,
    _cover_for,
    _describe,
    _is_targetable,
    _lookup,
    _max_range,
    _member_cell,
    _member_pos,
    _primary_weapon,
    _range_band,
    _resolve_bullet,
    _retreat_accuracy_mult,
    _spill_suppression,
    _team_of,
    _weapon_of,
    add_suppression,
    apply_hit,
    fire_member,
    resolve_bullets,
    schedule_next_shot,
)
from coh.sim.systems.destruction import (  # noqa: F401
    _abandon_weapon,
    _destroy_building,
    _destroy_squad,
    _forget_target,
    _promote_gunner,
    _remove_member,
    damage_member,
    on_building_destroyed,
)
from coh.sim.systems.explosions import (  # noqa: F401
    _explode,
    _explode_on_building,
    _explode_on_squad,
    _falloff,
    fire_shell,
)
from coh.sim.systems.team_weapons import (  # noqa: F401
    _arc_weapon,
    _auto_reface,
    _bearing,
    _clear_attack_order,
    _in_arc,
    _indirect_weapon,
    _stopping_facing,
    _team_weapon,
    abandoned_weapon_near,
    can_recrew,
    halt,
    set_facing,
    try_recrew,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim

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


def _snapshot(sim: "Sim") -> _Snapshot:
    ids: list[int] = []
    pos: list[np.ndarray] = []
    teams: list[int] = []
    for sid in entity_ids(sim.state.squads):
        squad = sim.state.squads[sid]
        if not _is_targetable(squad):
            continue
        ids.append(sid)
        pos.append(squad.pos)
        teams.append(_team_of(sim, squad.owner))

    building_ids: list[int] = []
    building_team: list[int] = []
    for bid in entity_ids(sim.state.buildings):
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


# ---------------------------------------------------------------------------
# Tick entry point
# ---------------------------------------------------------------------------


def run(sim: "Sim") -> None:
    """Advance the combat system by one tick."""
    snapshot = _snapshot(sim)
    for sid in entity_ids(sim.state.squads):
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
    if squad.abandoned or not squad.has_alive_members or squad.pinned:
        return False
    if squad.state in _NO_COMBAT_STATES:
        return False
    return sim.data.squads.get(squad.def_id) is not None


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
    if entity is None or not _still_hostile(sim, team, entity):
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


def _still_hostile(sim: "Sim", team: int, entity: Squad | Building) -> bool:
    """Could this target ever be engageable by `team` again?

    `_is_engageable`'s other refusals (out of reach, out of sight, no LOS)
    are all things pursuit exists to fix, but these two are permanent: a
    neutral house whose last occupant walked out is map furniture again, and
    an entity that changed hands (a re-crewed gun) is now a friend. Without
    this the squad would re-path at it forever.
    """
    if isinstance(entity, Building):
        return _building_team(sim, entity) not in (team, _NO_TEAM)
    return _team_of(sim, entity.owner) != team


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
