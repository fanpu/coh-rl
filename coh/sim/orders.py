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
from typing import TYPE_CHECKING, Any, Callable

from coh.sim.state import SquadState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Player


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
    expected = {f.name: f for f in fields(cls)}
    given = {k: v for k, v in d.items() if k != "type"}
    if set(given) != set(expected):
        raise ValueError(
            f"{name}: expected field(s) {sorted(expected)}, got {sorted(given)}"
        )
    kwargs = {
        key: tuple(value) if "tuple" in str(expected[key].type) else value
        for key, value in given.items()
    }
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


@dataclass(frozen=True)
class OrderHandler:
    validate: Callable[["Sim", "Player", Order], OrderResult]
    apply: Callable[["Sim", Order], None]


# Later tasks replace the `_accept` validator of the order they implement with
# a `validate_<order>` of their own, and `apply_squad_order` where storing the
# order is not enough.
ORDER_HANDLERS: dict[type[Order], OrderHandler] = {
    Move: OrderHandler(_accept, apply_squad_order),
    AttackMove: OrderHandler(_accept, apply_squad_order),
    Attack: OrderHandler(_accept, apply_squad_order),
    Capture: OrderHandler(_accept, apply_squad_order),
    Garrison: OrderHandler(_accept, apply_squad_order),
    Ungarrison: OrderHandler(_accept, apply_squad_order),
    Retreat: OrderHandler(_accept, apply_squad_order),
    Reinforce: OrderHandler(_accept, apply_squad_order),
    SetFacing: OrderHandler(_accept, apply_squad_order),
    Build: OrderHandler(_accept, apply_squad_order),
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
