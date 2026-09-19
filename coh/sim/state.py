"""Mutable simulation state plus its canonical serialization / hash.

`GameState` is a plain data structure: dataclasses holding scalars, lists,
dicts and numpy arrays. Systems in `coh/sim/systems/` mutate it in place; the
only randomness is `GameState.rng`.

Determinism rules that this module enforces:

- every dict of entities is keyed by id and iterated in ascending id order
  when serialized (`canonical_state`);
- `GameState.connected` stores sorted lists, never sets;
- `state_hash` covers everything that a system may depend on, and excludes the
  purely derived / presentational parts (`visible` grids, `events`) and the
  RNG object itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.orders import Order

_ROUND_DP = 6


class SquadState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    SETTING_UP = "setting_up"
    SET_UP = "set_up"
    TEARING_DOWN = "tearing_down"
    RETREATING = "retreating"
    GARRISONED = "garrisoned"
    CONSTRUCTING = "constructing"


@dataclass
class Member:
    """One model in a squad (for vehicles: the vehicle itself)."""

    hp: float
    weapon: str  # weapon id, "" = unarmed
    next_ready_tick: int = 0
    shots_since_reload: int = 0
    burst_until_tick: int = 0


@dataclass
class Squad:
    id: int
    owner: int
    def_id: str
    pos: np.ndarray  # world meters, x right / y down
    heading: float  # radians
    members: list[Member]
    state: SquadState = SquadState.IDLE
    order: "Order | None" = None
    path: list[tuple[int, int]] = field(default_factory=list)
    target_id: int | None = None
    suppression: float = 0.0
    suppressed: bool = False
    pinned: bool = False
    last_hit_tick: int = -1
    garrison_in: int | None = None
    facing: float | None = None  # team-weapon arc centre, radians
    setup_done_tick: int = 0
    reinforce_done_tick: int = 0
    upgrades: list[str] = field(default_factory=list)
    abandoned: bool = False  # team weapon whose crew died (task 10)
    turret_heading: float = 0.0  # vehicles (task 11)
    last_attacker_pos: tuple[float, float] | None = None  # armour facing (task 11)
    moving: bool = False  # set by movement, read by combat (moving accuracy)

    @property
    def alive_members(self) -> list[Member]:
        return [m for m in self.members if m.hp > 0]


@dataclass
class QueueItem:
    kind: str  # "train" | "research"
    item_id: str
    remaining_s: float


@dataclass
class Building:
    id: int
    owner: int | None  # None for neutral (enterable) buildings
    def_id: str
    cell: tuple[int, int]  # top-left cell of the footprint
    hp: float
    progress: float = 1.0  # 1.0 = complete
    queue: list[QueueItem] = field(default_factory=list)
    garrison: list[int] = field(default_factory=list)  # squad ids
    neutral: bool = False


@dataclass
class Player:
    id: int
    team: int
    faction: str
    manpower: float
    munitions: float
    fuel: float
    hq_id: int
    upgrades: list[str] = field(default_factory=list)
    invalid_orders: int = 0


@dataclass
class PointState:
    owner_team: int | None = None
    progress: float = 0.0  # 0..1 towards `capturing_team`'s ownership
    capturing_team: int | None = None
    op_building: int | None = None  # observation post building id


@dataclass
class Ghost:
    """Last-known state of an enemy building, kept after it leaves vision."""

    building_id: int
    def_id: str
    owner: int | None
    cell: tuple[int, int]
    hp_frac: float
    last_seen_tick: int


@dataclass
class Event:
    """A thing that happened this tick; consumed by the viewer/replay."""

    kind: str  # e.g. "shot", "death", "capture"
    tick: int
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    tick: int
    rng: np.random.Generator
    players: dict[int, Player]
    squads: dict[int, Squad]
    buildings: dict[int, Building]
    points: dict[str, PointState]
    connected: dict[int, list[int]]  # team -> sorted connected sector ids
    tickets: dict[int, float]  # team -> remaining tickets
    visible: dict[int, np.ndarray]  # team -> bool grid [cy, cx]
    ghosts: dict[int, dict[int, Ghost]]  # team -> building id -> ghost
    next_id: int  # shared id counter for squads and buildings
    winner: int | None = None  # winning team
    events: list[Event] = field(default_factory=list)
    # Append-only log of terrain edits a vehicle crushing cover makes, e.g.
    # (cx, cy, ".") for a fence cell cleared. Included in `state_hash` so two
    # sims that only diverge by a crushed fence hash differently.
    terrain_changes: list[tuple[int, int, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical serialization / hashing
# ---------------------------------------------------------------------------


def _r(x: float) -> float:
    """Round to 1e-6 and normalize -0.0, so hashes are platform-stable."""
    return round(float(x), _ROUND_DP) + 0.0


def _rs(values) -> list[float]:
    return [_r(v) for v in np.asarray(values).ravel().tolist()]


def _member(m: Member) -> list[Any]:
    return [_r(m.hp), m.weapon, m.next_ready_tick, m.shots_since_reload, m.burst_until_tick]


def _squad(s: Squad) -> dict[str, Any]:
    from coh.sim.orders import order_to_dict

    return {
        "id": s.id,
        "owner": s.owner,
        "def_id": s.def_id,
        "pos": _rs(s.pos),
        "heading": _r(s.heading),
        "members": [_member(m) for m in s.members],
        "state": s.state.name,
        "order": None if s.order is None else order_to_dict(s.order),
        "path": [list(c) for c in s.path],
        "target_id": s.target_id,
        "suppression": _r(s.suppression),
        "suppressed": s.suppressed,
        "pinned": s.pinned,
        "last_hit_tick": s.last_hit_tick,
        "garrison_in": s.garrison_in,
        "facing": None if s.facing is None else _r(s.facing),
        "setup_done_tick": s.setup_done_tick,
        "reinforce_done_tick": s.reinforce_done_tick,
        "upgrades": list(s.upgrades),
        "abandoned": s.abandoned,
        "turret_heading": _r(s.turret_heading),
        "last_attacker_pos": None if s.last_attacker_pos is None else _rs(s.last_attacker_pos),
        "moving": s.moving,
    }


def _building(b: Building) -> dict[str, Any]:
    return {
        "id": b.id,
        "owner": b.owner,
        "def_id": b.def_id,
        "cell": list(b.cell),
        "hp": _r(b.hp),
        "progress": _r(b.progress),
        "queue": [[q.kind, q.item_id, _r(q.remaining_s)] for q in b.queue],
        "garrison": list(b.garrison),
        "neutral": b.neutral,
    }


def _player(p: Player) -> dict[str, Any]:
    return {
        "id": p.id,
        "team": p.team,
        "faction": p.faction,
        "manpower": _r(p.manpower),
        "munitions": _r(p.munitions),
        "fuel": _r(p.fuel),
        "hq_id": p.hq_id,
        "upgrades": list(p.upgrades),
        "invalid_orders": p.invalid_orders,
    }


def _point(p: PointState) -> dict[str, Any]:
    return {
        "owner_team": p.owner_team,
        "progress": _r(p.progress),
        "capturing_team": p.capturing_team,
        "op_building": p.op_building,
    }


def _ghost(g: Ghost) -> dict[str, Any]:
    return {
        "building_id": g.building_id,
        "def_id": g.def_id,
        "owner": g.owner,
        "cell": list(g.cell),
        "hp_frac": _r(g.hp_frac),
        "last_seen_tick": g.last_seen_tick,
    }


def canonical_state(state: GameState) -> dict[str, Any]:
    """Plain-python, order-stable view of the state, used for hashing.

    Excludes the RNG object, `events` and the `visible` grids (derived each
    tick by the vision system).
    """
    return {
        "tick": state.tick,
        "winner": state.winner,
        "next_id": state.next_id,
        "players": [_player(state.players[pid]) for pid in sorted(state.players)],
        "squads": [_squad(state.squads[sid]) for sid in sorted(state.squads)],
        "buildings": [_building(state.buildings[bid]) for bid in sorted(state.buildings)],
        "points": [[pid, _point(state.points[pid])] for pid in sorted(state.points)],
        "connected": [[team, sorted(state.connected[team])] for team in sorted(state.connected)],
        "tickets": [[team, _r(state.tickets[team])] for team in sorted(state.tickets)],
        "ghosts": [
            [team, [_ghost(state.ghosts[team][bid]) for bid in sorted(state.ghosts[team])]]
            for team in sorted(state.ghosts)
        ],
        "terrain_changes": [[cx, cy, ch] for cx, cy, ch in state.terrain_changes],
    }


def state_hash(state: GameState) -> str:
    """sha256 hex digest of `canonical_state`; stable across processes."""
    blob = json.dumps(canonical_state(state), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()
