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

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, get_origin, get_type_hints

from coh.maps.format import cell_of
from coh.sim import pathfinding
from coh.sim.state import SquadState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Player, Squad


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


def apply_building_order(sim: "Sim", order: Order) -> None:
    """Train/Research have no effect until task 14 implements queues."""


# -- move-type orders (task 6): resolve to a path via coh/sim/systems/movement --


def _passable(sim: "Sim", squad: "Squad") -> Any:
    sdef = sim.data.squads[squad.def_id]
    return sim.map.pass_veh if sdef.kind == "vehicle" else sim.map.pass_inf


def apply_move(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    movement.start_path(sim, squad, order, order.cell, SquadState.MOVING)  # type: ignore[attr-defined]


def apply_attack_move(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    movement.start_path(sim, squad, order, order.cell, SquadState.MOVING)  # type: ignore[attr-defined]


def apply_retreat(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    player = sim.state.players[squad.owner]
    hq = sim.state.buildings[player.hq_id]
    bdef = sim.data.buildings[hq.def_id]
    start_cell = cell_of(squad.pos)
    goal_cell = pathfinding.nearest_adjacent_passable(_passable(sim, squad), hq.cell, bdef.footprint, start_cell)
    if goal_cell is None:
        goal_cell = hq.cell
    movement.start_path(sim, squad, order, goal_cell, SquadState.RETREATING)


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
    """Store the order; combat picks the target up on its next tick."""
    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    squad.order = order
    squad.target_id = None
    squad.path = []


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
    if order.building_id not in sim.state.buildings:  # type: ignore[attr-defined]
        return OrderResult(False, f"no such building {order.building_id}")  # type: ignore[attr-defined]
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


def apply_build(sim: "Sim", order: Order) -> None:
    from coh.sim.systems import movement

    squad = sim.state.squads[order.squad]  # type: ignore[attr-defined]
    bdef = sim.data.buildings.get(order.structure)  # type: ignore[attr-defined]
    # `structure` naming a real, buildable-by-this-squad def is Task 14's
    # (production/construction) job to validate, not ours; we just need
    # *some* footprint to compute a walk-to cell.
    footprint = bdef.footprint if bdef is not None else (1, 1)
    start_cell = cell_of(squad.pos)
    goal_cell = pathfinding.nearest_adjacent_passable(
        _passable(sim, squad), order.cell, footprint, start_cell  # type: ignore[attr-defined]
    )
    if goal_cell is None:
        goal_cell = order.cell  # type: ignore[attr-defined]
    movement.start_path(sim, squad, order, goal_cell, SquadState.MOVING)


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
    Retreat: OrderHandler(_accept, apply_retreat),
    Reinforce: OrderHandler(_accept, apply_squad_order),
    SetFacing: OrderHandler(_accept, apply_squad_order),
    Build: OrderHandler(_accept, apply_build),
    Train: OrderHandler(_accept, apply_building_order),
    Research: OrderHandler(_accept, apply_building_order),
    BuyUpgrade: OrderHandler(_accept, apply_squad_order),
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
    """Apply an order that `validate_order` accepted."""
    ORDER_HANDLERS[type(order)].apply(sim, order)
