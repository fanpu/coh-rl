"""A synthetic mid/late-game state, and the periodic orders that keep it hot.

The T1 bots only ever field infantry, and a scripted match often ends by
annihilation well before the army sizes that matter for RL throughput, so
"how fast does a bot match run" is not the number to optimise against. This
module builds the state such a match would reach at its busiest — two full
armies of mixed arms inside each other's engagement range, a few squads
garrisoned in the map's houses — directly through `Sim.spawn_squad`, and
hands out periodic `Move` / `AttackMove` orders so movement, pathfinding,
vision, combat and suppression are all doing real work every tick.

It is deliberately *not* a sim feature: it lives outside `coh/sim`, uses only
public entry points (`Sim.spawn_squad`, `Sim.issue`, `garrison.enter`), and
carries its own unit table (`FIELD_ROSTER` / `GARRISON_UNIT`) so that
re-pointing it at the shipped stat tables is a matter of swapping def ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coh.data.schema import GameData
from coh.maps.format import GameMap, cell_of, center_of, load_map
from coh.sim.constants import CELL_M
from coh.sim.orders import AttackMove, Move, Order
from coh.sim.sim import PlayerSetup, Sim, SimConfig, neutral_footprints
from coh.sim.systems import garrison

MAP_NAME = "hedgerow_crossing"

# Cycled over each side's field squads, so every side fields the same mix of
# rifles, heavy weapons and armour. Fixture def ids; swap for the shipped
# roster once `coh/data/tables/` is populated.
FIELD_ROSTER = (
    "rifles",
    "rifles",
    "hmg_team",
    "rifles",
    "mortar_team",
    "rifles",
    "at_team",
    "tank",
    "rifles",
    "engineers",
)
GARRISON_UNIT = "rifles"

# Each side's front line, in cells: three ranks either side of the middle of
# the map, close enough that every weapon band is in use.
_TEAM0_ROWS = (38, 41, 44)
_TEAM1_ROWS = (52, 55, 58)
_COLUMN_RANGE = range(24, 73, 3)

# Where the periodic orders send squads: the far side of the enemy line, so
# they keep walking into each other rather than settling into a stalemate.
_TEAM0_OBJECTIVE_ROW = 56
_TEAM1_OBJECTIVE_ROW = 40


@dataclass(frozen=True)
class StressConfig:
    """Knobs for `build_stress_sim` / `orders_for_tick`."""

    squads_per_side: int = 30
    garrisoned_per_side: int = 4
    order_interval_s: float = 5.0
    squads_per_order: int = 6
    time_limit_s: float = 100_000.0  # never end on the clock: the bench sets the length


@dataclass
class StressState:
    """The built sim plus the bookkeeping `orders_for_tick` needs."""

    sim: Sim
    config: StressConfig
    squads_by_player: dict[int, list[int]] = field(default_factory=dict)


def load_stress_map(data: GameData) -> GameMap:
    return load_map(MAP_NAME, footprints=neutral_footprints(data))


def build_stress_sim(
    data: GameData,
    *,
    seed: int = 0,
    game_map: GameMap | None = None,
    config: StressConfig = StressConfig(),
) -> StressState:
    """Two armies of `config.squads_per_side` squads, deployed nose to nose."""
    game_map = game_map if game_map is not None else load_stress_map(data)
    sim = Sim(
        game_map=game_map,
        players=[
            PlayerSetup(faction="us", team=0, start_slot=0),
            PlayerSetup(faction="wehr", team=1, start_slot=1),
        ],
        data=data,
        seed=seed,
        config=SimConfig(time_limit_s=config.time_limit_s),
    )

    state = StressState(sim=sim, config=config, squads_by_player={0: [], 1: []})
    for player_id, rows in ((0, _TEAM0_ROWS), (1, _TEAM1_ROWS)):
        garrisoned = _garrison_squads(sim, player_id, config.garrisoned_per_side)
        wanted = max(0, config.squads_per_side - len(garrisoned))
        field_squads = _deploy_field_squads(sim, player_id, rows, wanted)
        state.squads_by_player[player_id] = sorted(garrisoned + field_squads)
    return state


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


def _passable_cells(sim: Sim, rows: tuple[int, ...]) -> list[tuple[int, int]]:
    cells = []
    for cy in rows:
        for cx in _COLUMN_RANGE:
            if 0 <= cy < sim.map.height and 0 <= cx < sim.map.width and sim.map.pass_inf[cy, cx]:
                cells.append((cx, cy))
    return cells


def _deploy_field_squads(sim: Sim, player_id: int, rows: tuple[int, ...], count: int) -> list[int]:
    cells = _passable_cells(sim, rows)
    if count > len(cells):  # pragma: no cover - only with a much larger army
        raise ValueError(f"stress: only {len(cells)} deployable cells for {count} squads")
    ids = []
    for index in range(count):
        def_id = FIELD_ROSTER[index % len(FIELD_ROSTER)]
        squad = sim.spawn_squad(player_id, def_id, center_of(cells[index]))
        # Face the enemy, so team weapons start with a useful arc.
        squad.heading = np.pi / 2 if player_id == 0 else -np.pi / 2
        squad.facing = squad.heading
        ids.append(squad.id)
    return ids


def _garrison_squads(sim: Sim, player_id: int, count: int) -> list[int]:
    """Fill the houses nearest this side's line, up to `count` squads."""
    if count <= 0:
        return []
    rows = _TEAM0_ROWS if player_id == 0 else _TEAM1_ROWS
    mid_row = rows[len(rows) // 2]
    houses = [
        sim.state.buildings[bid]
        for bid in sorted(sim.state.buildings)
        if sim.state.buildings[bid].neutral and garrison.capacity(sim, sim.state.buildings[bid]) > 0
    ]
    # Nearest houses first, ties by id: deterministic, and puts each side in
    # the buildings on its own half of the map.
    houses.sort(key=lambda b: (abs(b.cell[1] - mid_row), b.id))

    ids: list[int] = []
    for building in houses:
        if len(ids) >= count:
            break
        if garrison.occupying_team(sim, building) is not None:
            continue  # the other side got there first
        free = garrison.capacity(sim, building) - len(building.garrison)
        for _ in range(min(free, count - len(ids))):
            squad = sim.spawn_squad(player_id, GARRISON_UNIT, center_of(building.cell))
            garrison.enter(sim, squad, building)
            ids.append(squad.id)
    return ids


# ---------------------------------------------------------------------------
# Periodic orders
# ---------------------------------------------------------------------------


def orders_for_tick(state: StressState, tick: int, ticks_per_second: int) -> dict[int, list[Order]]:
    """Orders to issue on this tick (empty on every tick but the periodic one).

    Each order tick, a rotating slice of each side's squads is sent at the
    enemy: alternating `AttackMove` (fight on the way) and `Move` (run), so
    both movement paths stay exercised. Garrisoned squads are skipped — they
    are holding their house, and `Move` on them would only be rejected.
    """
    interval = max(1, round(state.config.order_interval_s * ticks_per_second))
    if tick % interval:
        return {}

    wave = tick // interval
    use_attack_move = wave % 2 == 0
    out: dict[int, list[Order]] = {}
    for player_id, squad_ids in sorted(state.squads_by_player.items()):
        movable = [
            sid
            for sid in squad_ids
            if sid in state.sim.state.squads and state.sim.state.squads[sid].garrison_in is None
        ]
        if not movable:
            continue
        size = state.config.squads_per_order
        start = (wave * size) % len(movable)
        chosen = [movable[(start + offset) % len(movable)] for offset in range(min(size, len(movable)))]

        goal_row = _TEAM0_OBJECTIVE_ROW if player_id == 0 else _TEAM1_OBJECTIVE_ROW
        orders: list[Order] = []
        for sid in chosen:
            squad = state.sim.state.squads[sid]
            column = min(max(cell_of(squad.pos, CELL_M)[0], 0), state.sim.map.width - 1)
            goal = (column, goal_row)
            orders.append(AttackMove(sid, goal) if use_attack_move else Move(sid, goal))
        out[player_id] = orders
    return out
