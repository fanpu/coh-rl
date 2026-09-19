"""Order types, their JSON form, and order validation / application.

Orders are frozen dataclasses naming a squad (`squad`) or a building
(`building`) by id, plus cells / ids / strings. They never raise: every order
goes through `validate_order`, which returns an `OrderResult`.

Validation is split in two layers so later tasks each extend exactly one
function:

- `validate_order` does the checks that are common to all orders (player and
  entity exist, entity belongs to the player, squad is not retreating, any
  `cell` is in bounds);
- per-order-type `validate_*` / `apply_*` functions, registered in
  `ORDER_HANDLERS` keyed by order class. In this task the specific validators
  accept everything and the appliers just store the order on the squad; the
  task that implements each order fills in its own pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, get_origin, get_type_hints

import numpy as np

from coh.data.schema import Cost
from coh.maps.format import cell_of, center_of
from coh.sim import pathfinding
from coh.sim.constants import CELL_M, MAX_QUEUE_LEN, TICKS_PER_SECOND
from coh.sim.state import SquadState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.data.schema import SquadDef
    from coh.sim.sim import Sim
    from coh.sim.state import Building, Player, Squad


@dataclass(frozen=True)
class Order:
    """Base class for all orders (no fields of its own)."""


@dataclass(frozen=True)
class Move(Order):
    squad: int
    cell: tuple[int, int]


@dataclass(frozen=True)
class AttackMove(Order):
    squad: int
    cell: tuple[int, int]


@dataclass(frozen=True)
class Attack(Order):
    squad: int
    target_id: int


@dataclass(frozen=True)
class Capture(Order):
    squad: int
    point_id: str


@dataclass(frozen=True)
class Garrison(Order):
    squad: int
    building_id: int


@dataclass(frozen=True)
class Ungarrison(Order):
    squad: int


@dataclass(frozen=True)
class Retreat(Order):
    squad: int


@dataclass(frozen=True)
class Reinforce(Order):
    squad: int


@dataclass(frozen=True)
class SetFacing(Order):
    squad: int
    direction_deg: float


@dataclass(frozen=True)
class Build(Order):
    squad: int
    structure: str
    cell: tuple[int, int]


@dataclass(frozen=True)
class Train(Order):
    building: int
    unit: str


@dataclass(frozen=True)
class Research(Order):
    building: int
    upgrade: str


@dataclass(frozen=True)
class BuyUpgrade(Order):
    squad: int
    upgrade: str


@dataclass(frozen=True)
class Stop(Order):
    squad: int


ORDER_TYPES: dict[str, type[Order]] = {
    cls.__name__: cls
    for cls in (
        Move,
        AttackMove,
        Attack,
        Capture,
        Garrison,
        Ungarrison,
        Retreat,
        Reinforce,
        SetFacing,
        Build,
        Train,
        Research,
        BuyUpgrade,
        Stop,
    )
}


# Fields declared as tuples, which `order_to_dict` writes out as JSON lists and
# `order_from_dict` must turn back into tuples (order equality depends on it).
TUPLE_FIELDS: dict[type[Order], frozenset[str]] = {
    cls: frozenset(
        f.name
        for f in fields(cls)
        if get_origin(get_type_hints(cls)[f.name]) is tuple
    )
    for cls in ORDER_TYPES.values()
}


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    reason: str = ""


OK = OrderResult(ok=True)


# ---------------------------------------------------------------------------
# JSON form
# ---------------------------------------------------------------------------


def order_to_dict(order: Order) -> dict[str, Any]:
    """`{"type": "Move", "squad": 3, "cell": [4, 5]}` — plain JSON types only."""
    out: dict[str, Any] = {"type": type(order).__name__}
    for f in fields(order):
        value = getattr(order, f.name)
        out[f.name] = list(value) if isinstance(value, tuple) else value
    return out


def order_from_dict(d: dict[str, Any]) -> Order:
    """Inverse of `order_to_dict`; raises `ValueError` on anything malformed."""
    if "type" not in d:
        raise ValueError(f"order dict is missing 'type': {d!r}")
    name = d["type"]
    cls = ORDER_TYPES.get(name)
    if cls is None:
        raise ValueError(f"unknown order type {name!r} (known: {sorted(ORDER_TYPES)})")
    expected = {f.name for f in fields(cls)}
    given = {k: v for k, v in d.items() if k != "type"}
    if set(given) != expected:
        raise ValueError(
            f"{name}: expected field(s) {sorted(expected)}, got {sorted(given)}"
        )
    tuple_fields = TUPLE_FIELDS[cls]
    kwargs = {key: tuple(value) if key in tuple_fields else value for key, value in given.items()}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Per-order validators (filled in by later tasks) and appliers
# ---------------------------------------------------------------------------


def _accept(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    """Placeholder validator: order-specific rules arrive with their task."""
    return OK


def apply_squad_order(sim: "Sim", order: Order) -> None:
    """Default application: the order becomes the squad's standing order."""
    sim.state.squads[order.squad].order = order  # type: ignore[attr-defined]


def apply_stop(sim: "Sim", order: Order) -> None:
    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    squad.order = None
    squad.path = []
    squad.target_id = None


# -- move-type orders (task 6): resolve to a path via coh/sim/systems/movement --


def _passable(sim: "Sim", squad: "Squad") -> Any:
    sdef = sim.data.squads[squad.def_id]
    return sim.map.pass_veh if sdef.kind == "vehicle" else sim.map.pass_inf


def apply_move(sim: "Sim", order: Order) -> None:
    """Walk to the cell -- and remember it if there is an abandoned team
    weapon there, so that arriving re-crews it (task 10)."""
    from coh.sim.systems import combat, movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    if combat.can_recrew(sim, squad):
        shell = combat.abandoned_weapon_near(sim, center_of(order.cell, CELL_M))  # type: ignore[attr-defined]
        squad.recrew_target = None if shell is None else shell.id
    movement.start_path(sim, squad, order, order.cell, SquadState.MOVING)  # type: ignore[attr-defined]


def apply_attack_move(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    movement.start_path(sim, squad, order, order.cell, SquadState.MOVING)  # type: ignore[attr-defined]


def validate_retreat(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    """Vehicles and abandoned squads cannot retreat; everything else can
    (including garrisoned squads and team weapons -- `apply_retreat` handles
    both specially)."""
    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    sdef = sim.data.squads[squad.def_id]
    if sdef.kind == "vehicle":
        return OrderResult(False, f"squad {squad.id} is a vehicle and cannot retreat")
    if squad.abandoned:
        return OrderResult(False, f"squad {squad.id} is abandoned and cannot retreat")
    return OK


def apply_retreat(sim: "Sim", order: Order) -> None:
    """Clear suppression and any target/pursuit state, exit a garrison if
    the squad is in one (task 12 owns garrison entry/exit proper; this just
    unblocks retreat), then path to the nearest cell adjacent to own HQ.

    Team weapons tear down instantly on retreat (no teardown delay) --
    `movement.start_path` special-cases `moving_state is RETREATING` for
    that.
    """
    from coh.sim.systems import movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]

    squad.suppression = 0.0
    squad.suppressed = False
    squad.pinned = False
    squad.target_id = None
    squad.attack_last_seen_tick = -1
    squad.attack_last_repath_tick = -1

    if squad.garrison_in is not None:
        _exit_garrison_for_retreat(sim, squad)

    player = sim.state.players[squad.owner]
    hq = sim.state.buildings[player.hq_id]
    bdef = sim.data.buildings[hq.def_id]
    start_cell = cell_of(squad.pos)
    goal_cell = pathfinding.nearest_adjacent_passable(_passable(sim, squad), hq.cell, bdef.footprint, start_cell)
    if goal_cell is None:
        goal_cell = hq.cell
    movement.start_path(sim, squad, order, goal_cell, SquadState.RETREATING)


def _exit_garrison_for_retreat(sim: "Sim", squad: "Squad") -> None:
    building = sim.state.buildings.get(squad.garrison_in)  # type: ignore[arg-type]
    squad.garrison_in = None
    if building is None:
        return
    bdef = sim.data.buildings.get(building.def_id) or sim.data.neutral.get(building.def_id)
    footprint = bdef.footprint if bdef is not None else (1, 1)
    start_cell = cell_of(squad.pos)
    exit_cell = pathfinding.nearest_adjacent_passable(_passable(sim, squad), building.cell, footprint, start_cell)
    if exit_cell is not None:
        squad.pos = np.asarray(center_of(exit_cell, CELL_M), dtype=float)


def validate_attack(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    """The target must exist, be an enemy squad/building, and be visible now.

    Neutral (enterable) buildings are not attackable in M1.
    """
    from coh.sim.systems import vision

    target_id = order.target_id  # type: ignore[attr-defined]
    entity = sim.state.squads.get(target_id) or sim.state.buildings.get(target_id)
    if entity is None:
        return OrderResult(False, f"no such target {target_id}")

    building = sim.state.buildings.get(target_id)
    if building is not None and building.neutral:
        return OrderResult(False, f"building {target_id} is neutral and cannot be attacked")

    owner = sim.state.players.get(entity.owner) if entity.owner is not None else None
    if owner is not None and owner.team == player.team:
        return OrderResult(False, f"target {target_id} is not an enemy of player {player.id}")

    if not vision.is_visible(sim, player.team, entity):
        return OrderResult(False, f"target {target_id} is not visible to team {player.team}")
    return OK


def apply_attack(sim: "Sim", order: Order) -> None:
    """Store the order; combat picks the target up on its next tick.

    Validation just proved the target visible, so that is the baseline for
    combat's "gave up: unseen too long" timer; `-1` for the re-path timer
    lets the squad path toward the target immediately. Any current movement
    is cancelled here; combat settles the squad's state (and redeploys a
    team weapon) once it knows whether the target is already in range.
    """
    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    squad.order = order
    squad.target_id = None
    squad.path = []
    squad.attack_last_seen_tick = sim.state.tick
    squad.attack_last_repath_tick = -1


def validate_set_facing(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    """Only a crewed team weapon has a facing worth setting."""
    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    sdef = sim.data.squads[squad.def_id]
    if sdef.kind != "team_weapon":
        return OrderResult(False, f"squad {squad.id} ({squad.def_id}) is not a team weapon")
    return OK


def apply_set_facing(sim: "Sim", order: Order) -> None:
    """Tear the gun down and set it up again on the new bearing.

    `direction_deg` is measured from due east (+x) and grows clockwise on
    screen. World `y` points down, so that is exactly the sense of
    `atan2(dy, dx)` and the conversion is a plain `radians()`.

    This is an immediate action rather than a standing order, so nothing is
    left on `squad.order` for later systems to trip over.
    """
    from coh.sim.systems import combat

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    squad.order = None
    combat.set_facing(sim, squad, math.radians(order.direction_deg), auto=False)  # type: ignore[attr-defined]


def validate_capture(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    from coh.sim.systems import territory

    if order.point_id not in sim.map.points:  # type: ignore[attr-defined]
        return OrderResult(False, f"no such point {order.point_id!r}")  # type: ignore[attr-defined]
    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    sdef = sim.data.squads[squad.def_id]
    if sdef.capture_rate <= 0:
        return OrderResult(False, f"squad {squad.id} cannot capture (capture_rate 0)")
    if territory.is_hq_point(sim, order.point_id):  # type: ignore[attr-defined]
        return OrderResult(False, f"point {order.point_id!r} is an HQ sector point and cannot be captured")  # type: ignore[attr-defined]
    point = sim.state.points[order.point_id]  # type: ignore[attr-defined]
    if point.owner_team == player.team and point.progress >= 1.0:
        return OrderResult(False, f"point {order.point_id!r} is already fully owned by team {player.team}")  # type: ignore[attr-defined]
    return OK


def apply_capture(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    point = sim.map.points[order.point_id]  # type: ignore[attr-defined]
    movement.start_path(sim, squad, order, point.cell, SquadState.MOVING)


def validate_garrison(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    """The building must exist and the squad must be able to garrison at all.

    `can_garrison` is false for vehicles (task 11); which *buildings* a squad
    that can garrison may enter is task 12's job.
    """
    if order.building_id not in sim.state.buildings:  # type: ignore[attr-defined]
        return OrderResult(False, f"no such building {order.building_id}")  # type: ignore[attr-defined]
    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    sdef = sim.data.squads[squad.def_id]
    if not sdef.can_garrison:
        return OrderResult(False, f"squad {squad.id} ({squad.def_id}) cannot garrison")
    return OK


def apply_garrison(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    building = sim.state.buildings[order.building_id]  # type: ignore[attr-defined]
    bdef = sim.data.buildings.get(building.def_id) or sim.data.neutral.get(building.def_id)
    # `building_id` existence is validated above; a def_id with no matching
    # data row shouldn't happen in practice, but garrison *legality* (can
    # this squad garrison this building at all) is Task 12's job, not ours.
    footprint = bdef.footprint if bdef is not None else (1, 1)
    start_cell = cell_of(squad.pos)
    goal_cell = pathfinding.nearest_adjacent_passable(_passable(sim, squad), building.cell, footprint, start_cell)
    if goal_cell is None:
        goal_cell = building.cell
    movement.start_path(sim, squad, order, goal_cell, SquadState.MOVING)


# -- production orders (task 14): see coh/sim/systems/production.py ------------


def validate_build(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    from coh.sim.systems import production

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    sdef = sim.data.squads[squad.def_id]
    structure = order.structure  # type: ignore[attr-defined]

    bdef = sim.data.buildings.get(structure)
    if bdef is None:
        return OrderResult(False, f"unknown building {structure!r}")
    if structure not in sdef.builds:
        return OrderResult(False, f"squad {squad.id} ({squad.def_id}) cannot build {structure!r}")
    if bdef.faction != player.faction:
        return OrderResult(False, f"{structure!r} is a {bdef.faction} building, player {player.id} is {player.faction}")

    missing = production.missing_requirements(sim, player.id, bdef.requires)
    if missing:
        return OrderResult(False, f"{structure!r} requires {missing}")

    # Assisting / resuming an unfinished site of the same structure is free.
    if production.unfinished_building_at(sim, player.id, structure, order.cell) is not None:  # type: ignore[attr-defined]
        return OK

    problem = production.build_site_problem(sim, player, bdef, order.cell)  # type: ignore[attr-defined]
    if problem:
        return OrderResult(False, problem)
    if not production.can_afford(player, bdef.cost):
        return OrderResult(False, f"player {player.id} cannot afford {structure!r}")
    return OK


def apply_build(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import movement, production

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    bdef = sim.data.buildings[order.structure]  # type: ignore[attr-defined]
    site = production.unfinished_building_at(sim, squad.owner, order.structure, order.cell)  # type: ignore[attr-defined]
    if site is None:
        site = production.start_construction(sim, squad.owner, bdef, order.cell)  # type: ignore[attr-defined]
    squad.build_target = site.id

    start_cell = cell_of(squad.pos)
    goal_cell = pathfinding.nearest_adjacent_passable(_passable(sim, squad), site.cell, bdef.footprint, start_cell)
    if goal_cell is None:
        goal_cell = site.cell
    movement.start_path(sim, squad, order, goal_cell, SquadState.MOVING)


def _producing_building(sim: "Sim", order: Order) -> tuple[Any, Any, OrderResult]:
    """The building an order targets and its def, or the reason it can't act."""
    building = sim.state.buildings[order.building]  # type: ignore[attr-defined]
    bdef = sim.data.buildings.get(building.def_id)
    if bdef is None:
        return building, None, OrderResult(False, f"building {building.id} cannot produce anything")
    if building.progress < 1.0:
        return building, bdef, OrderResult(False, f"building {building.id} is still under construction")
    if len(building.queue) >= MAX_QUEUE_LEN:
        return building, bdef, OrderResult(False, f"building {building.id}'s queue is full ({MAX_QUEUE_LEN})")
    return building, bdef, OK


def validate_train(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    from coh.sim.systems import economy, production

    building, bdef, result = _producing_building(sim, order)
    if not result.ok:
        return result

    unit = order.unit  # type: ignore[attr-defined]
    if unit not in bdef.produces:
        return OrderResult(False, f"building {building.def_id!r} does not produce {unit!r}")
    sdef = sim.data.squads[unit]

    if economy.pop_used(sim, player.id) + sdef.population > economy.pop_cap(sim, player.id):
        return OrderResult(False, f"population cap reached: {unit!r} needs {sdef.population}")
    if not production.can_afford(player, sdef.cost):
        return OrderResult(False, f"player {player.id} cannot afford {unit!r}")
    return OK


def apply_train(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import production

    building = sim.state.buildings[order.building]  # type: ignore[attr-defined]
    sdef = sim.data.squads[order.unit]  # type: ignore[attr-defined]
    production.enqueue(sim, building, "train", sdef.id, sdef.cost, sdef.build_time)


def validate_research(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    from coh.sim.systems import production

    building, bdef, result = _producing_building(sim, order)
    if not result.ok:
        return result

    upgrade_id = order.upgrade  # type: ignore[attr-defined]
    if upgrade_id not in bdef.researches:
        return OrderResult(False, f"building {building.def_id!r} does not research {upgrade_id!r}")
    udef = sim.data.upgrades[upgrade_id]

    if upgrade_id in player.upgrades:
        return OrderResult(False, f"player {player.id} already has {upgrade_id!r}")
    if production.research_queued(sim, player.id, upgrade_id):
        return OrderResult(False, f"{upgrade_id!r} is already queued")
    missing = production.missing_requirements(sim, player.id, udef.requires)
    if missing:
        return OrderResult(False, f"{upgrade_id!r} requires {missing}")
    if not production.can_afford(player, udef.cost):
        return OrderResult(False, f"player {player.id} cannot afford {upgrade_id!r}")
    return OK


def apply_research(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import production

    building = sim.state.buildings[order.building]  # type: ignore[attr-defined]
    udef = sim.data.upgrades[order.upgrade]  # type: ignore[attr-defined]
    production.enqueue(sim, building, "research", udef.id, udef.cost, udef.time)


def validate_buy_upgrade(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    from coh.sim.systems import production

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    sdef = sim.data.squads[squad.def_id]
    upgrade_id = order.upgrade  # type: ignore[attr-defined]

    udef = sim.data.squad_upgrades.get(upgrade_id)
    if udef is None:
        return OrderResult(False, f"unknown squad upgrade {upgrade_id!r}")
    if squad.def_id not in udef.applies_to:
        return OrderResult(False, f"{upgrade_id!r} does not apply to {squad.def_id!r}")
    if upgrade_id in squad.upgrades or squad.pending_upgrade == upgrade_id:
        return OrderResult(False, f"squad {squad.id} already has {upgrade_id!r}")

    missing = production.missing_requirements(sim, player.id, udef.requires)
    if missing:
        return OrderResult(False, f"{upgrade_id!r} requires {missing}")

    used_slots = len(squad.upgrades) + (1 if squad.pending_upgrade is not None else 0)
    if used_slots >= sdef.upgrade_slots:
        return OrderResult(False, f"squad {squad.id} has no free upgrade slot ({sdef.upgrade_slots})")

    cx, cy = cell_of(squad.pos)
    if int(sim.map.sector_id[cy, cx]) not in sim.state.connected.get(player.team, []):
        return OrderResult(False, f"squad {squad.id} is not in supplied friendly territory")
    if not production.can_afford(player, udef.cost):
        return OrderResult(False, f"player {player.id} cannot afford {upgrade_id!r}")
    return OK


def apply_buy_upgrade(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import production

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    production.start_squad_upgrade(sim, squad, sim.data.squad_upgrades[order.upgrade])  # type: ignore[attr-defined]


# -- reinforce (task 9): one model at a time, near a friendly building --------
#
# `reinforce_building` / `reinforce_model_cost` / `reinforce_model_time_s` are
# public (no leading underscore) because `coh/sim/systems/suppression.py`
# reuses them each tick to decide whether an in-progress `Reinforce` should
# keep going, stop for lack of range/funds, or complete its current model.


def reinforce_building(sim: "Sim", squad: "Squad") -> "Building | None":
    """The nearest complete building, owned by a player on `squad`'s team,
    with `reinforce_radius > 0` and within range of `squad` -- or `None`."""
    team = sim.state.players[squad.owner].team
    best: "Building | None" = None
    best_dist = math.inf
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        if building.owner is None or building.progress < 1.0:
            continue
        owner = sim.state.players.get(building.owner)
        if owner is None or owner.team != team:
            continue
        bdef = sim.data.buildings.get(building.def_id)
        if bdef is None or bdef.reinforce_radius <= 0.0:
            continue
        dist = _building_distance_m(building, bdef, squad.pos)
        if dist > bdef.reinforce_radius:
            continue
        if dist < best_dist:
            best, best_dist = building, dist
    return best


def _building_distance_m(building: "Building", bdef, pos: np.ndarray) -> float:
    """Distance in metres from `pos` to the nearest edge of `building`'s footprint."""
    cx0, cy0 = building.cell
    w, h = bdef.footprint
    x, y = pos[0] / CELL_M, pos[1] / CELL_M
    dx = max(cx0 - x, 0.0, x - (cx0 + w))
    dy = max(cy0 - y, 0.0, y - (cy0 + h))
    return math.hypot(dx, dy) * CELL_M


def reinforce_model_cost(sim: "Sim", sdef: "SquadDef") -> Cost:
    """Per-model reinforce cost: `squad.cost / members * reinforce_base_cost_frac
    * squad.reinforce_cost_mult`, applied to each resource."""
    frac = sim.data.economy.reinforce_base_cost_frac * sdef.reinforce_cost_mult / sdef.members
    cost = sdef.cost
    return Cost(manpower=cost.manpower * frac, munitions=cost.munitions * frac, fuel=cost.fuel * frac)


def reinforce_model_time_s(sdef: "SquadDef") -> float:
    return sdef.build_time / sdef.members * sdef.reinforce_time_mult


def validate_reinforce(sim: "Sim", player: "Player", order: Order) -> OrderResult:
    from coh.sim.systems import production

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    sdef = sim.data.squads[squad.def_id]

    if sdef.kind == "vehicle":
        return OrderResult(False, f"squad {squad.id} is a vehicle and cannot reinforce")
    if squad.abandoned:
        return OrderResult(False, f"squad {squad.id} is abandoned and cannot reinforce")
    if squad.garrison_in is not None:
        return OrderResult(False, f"squad {squad.id} is garrisoned and cannot reinforce")
    if len(squad.members) >= sdef.members:
        return OrderResult(False, f"squad {squad.id} is already at full strength")
    if reinforce_building(sim, squad) is None:
        return OrderResult(False, f"squad {squad.id} is not within reinforce range of a friendly building")
    if not production.can_afford(player, reinforce_model_cost(sim, sdef)):
        return OrderResult(False, f"player {player.id} cannot afford to reinforce squad {squad.id}")
    return OK


def apply_reinforce(sim: "Sim", order: Order) -> None:
    """Pay for and start timing the first model; `suppression.run` finishes
    it (and any further models the order implies) on later ticks."""
    from coh.sim.systems import production

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    sdef = sim.data.squads[squad.def_id]
    player = sim.state.players[squad.owner]

    squad.order = order
    production.pay(player, reinforce_model_cost(sim, sdef))
    squad.reinforcing = True
    seconds = reinforce_model_time_s(sdef)
    squad.reinforce_done_tick = sim.state.tick + max(1, math.ceil(seconds * TICKS_PER_SECOND))


@dataclass(frozen=True)
class OrderHandler:
    validate: Callable[["Sim", "Player", Order], OrderResult]
    apply: Callable[["Sim", Order], None]


# Later tasks replace the `_accept` validator of the order they implement with
# a `validate_<order>` of their own, and `apply_squad_order` where storing the
# order is not enough.
ORDER_HANDLERS: dict[type[Order], OrderHandler] = {
    Move: OrderHandler(_accept, apply_move),
    AttackMove: OrderHandler(_accept, apply_attack_move),
    Attack: OrderHandler(validate_attack, apply_attack),
    Capture: OrderHandler(validate_capture, apply_capture),
    Garrison: OrderHandler(validate_garrison, apply_garrison),
    Ungarrison: OrderHandler(_accept, apply_squad_order),
    Retreat: OrderHandler(validate_retreat, apply_retreat),
    Reinforce: OrderHandler(validate_reinforce, apply_reinforce),
    SetFacing: OrderHandler(validate_set_facing, apply_set_facing),
    Build: OrderHandler(validate_build, apply_build),
    Train: OrderHandler(validate_train, apply_train),
    Research: OrderHandler(validate_research, apply_research),
    BuyUpgrade: OrderHandler(validate_buy_upgrade, apply_buy_upgrade),
    Stop: OrderHandler(_accept, apply_stop),
}


# ---------------------------------------------------------------------------
# Common validation
# ---------------------------------------------------------------------------


def validate_order(sim: "Sim", player_id: int, order: Order) -> OrderResult:
    """Run the common checks, then the order type's own validator."""
    handler = ORDER_HANDLERS.get(type(order))
    if handler is None:
        return OrderResult(False, f"unknown order {type(order).__name__}")

    player = sim.state.players.get(player_id)
    if player is None:
        return OrderResult(False, f"no such player {player_id}")

    if hasattr(order, "squad"):
        squad = sim.state.squads.get(order.squad)  # type: ignore[attr-defined]
        if squad is None:
            return OrderResult(False, f"no such squad {order.squad}")  # type: ignore[attr-defined]
        if squad.owner != player_id:
            return OrderResult(False, f"squad {squad.id} owner is player {squad.owner}, not player {player_id}")
        if squad.state is SquadState.RETREATING:
            return OrderResult(False, f"squad {squad.id} is retreating and accepts no orders")
        # An abandoned team weapon keeps its old `owner` (nothing in the
        # state can express "unowned"), but it is a crewless object: it takes
        # no orders from anybody until somebody walks over and re-crews it.
        if squad.abandoned:
            return OrderResult(False, f"squad {squad.id} is abandoned and accepts no orders")

    if hasattr(order, "building"):
        building = sim.state.buildings.get(order.building)  # type: ignore[attr-defined]
        if building is None:
            return OrderResult(False, f"no such building {order.building}")  # type: ignore[attr-defined]
        if building.owner != player_id:
            return OrderResult(
                False, f"building {building.id} owner is {building.owner}, not player {player_id}"
            )

    if hasattr(order, "cell"):
        cx, cy = order.cell  # type: ignore[attr-defined]
        if not (0 <= cx < sim.map.width and 0 <= cy < sim.map.height):
            return OrderResult(False, f"cell {(cx, cy)} is out of bounds")

    return handler.validate(sim, player, order)


def apply_order(sim: "Sim", order: Order) -> None:
    """Apply an order that `validate_order` accepted.

    Any new order cancels a pending re-crew: only `apply_move` sets
    `recrew_target`, and only when the move is actually aimed at an
    abandoned team weapon.
    """
    squad_id = getattr(order, "squad", None)
    if squad_id is not None:
        squad = sim.state.squads.get(squad_id)
        if squad is not None:
            squad.recrew_target = None
    ORDER_HANDLERS[type(order)].apply(sim, order)
