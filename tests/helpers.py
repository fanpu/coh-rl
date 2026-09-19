"""Shared test helpers: fixture game data, small inline maps, `Sim` factory.

Later tasks' tests build on these, so everything is overridable:

    make_sim()                                  # default 40x30, 3 sectors, 2 players
    make_sim(seed=7)
    make_sim(ascii_map="....\n....", starts=[...], sectors=[...], points=[...])
    make_sim(players=[PlayerSetup("us", 0, 0)])
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from coh.data.loader import load_game_data
from coh.data.schema import GameData
from coh.maps.format import GameMap, center_of, map_from_ascii
from coh.sim.sim import PlayerSetup, Sim, SimConfig, neutral_footprints
from coh.sim.state import Squad

FIXTURES_DIR = Path(__file__).parent / "data" / "fixtures"

# Default inline map: 40x30 open cells, three vertical sectors.
DEFAULT_WIDTH = 40
DEFAULT_HEIGHT = 30
_SIDE_SECTOR_FRAC = 0.35  # west/east sector width as a fraction of the map

# HQ footprints in the fixture data are 4x4; these leave room for the builder
# squad to spawn south of the footprint.
DEFAULT_STARTS = [
    {"slot": 0, "team": 0, "hq_cell": [2, 2], "sector": "a"},
    {"slot": 1, "team": 1, "hq_cell": [34, 22], "sector": "c"},
]
DEFAULT_POINTS = [
    {"id": "west", "name": "West Field", "type": "strategic", "cell": [6, 14]},
    {"id": "mid", "name": "Crossroads", "type": "victory", "cell": [20, 15]},
    {"id": "east", "name": "East Field", "type": "strategic", "cell": [33, 14]},
]


@lru_cache(maxsize=1)
def fixture_data() -> GameData:
    """The hand-written fixture tables under `tests/data/fixtures/` (cached)."""
    return load_game_data(FIXTURES_DIR)


def default_terrain(width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> list[str]:
    """All-open terrain rows."""
    return ["." * width] * height


def default_sectors(width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> list[str]:
    """Three vertical sectors (west / middle / east) over the default map."""
    side = int(width * _SIDE_SECTOR_FRAC)
    row = "a" * side + "b" * (width - 2 * side) + "c" * side
    return [row] * height


def make_map(
    ascii_map: str | list[str] | None = None,
    *,
    sectors: list[str] | None = None,
    points: list[dict[str, Any]] | None = None,
    starts: list[dict[str, Any]] | None = None,
    neutral_buildings: list[dict[str, Any]] = (),
    data: GameData | None = None,
    name: str = "inline",
) -> GameMap:
    """Build a small inline `GameMap`.

    With no arguments: the 40x30 default map with two opposite-corner HQ
    starts, a strategic point in each HQ sector and a victory point in the
    middle sector.

    With a custom `ascii_map` (newline-separated string or list of rows) and no
    other overrides: one sector covering the whole map with a single victory
    point at its centre, and HQ starts near the top-left / bottom-right
    corners. Pass `sectors` / `points` / `starts` to override any of these;
    every sector needs exactly one point and at least one point must be a
    victory point (see `coh.maps.format`).
    """
    if ascii_map is None:
        terrain = default_terrain()
        sectors = sectors if sectors is not None else default_sectors()
        points = points if points is not None else DEFAULT_POINTS
        starts = starts if starts is not None else DEFAULT_STARTS
    else:
        terrain = ascii_map.splitlines() if isinstance(ascii_map, str) else list(ascii_map)
        width, height = len(terrain[0]), len(terrain)
        if sectors is None:
            sectors = ["a" * width] * height
        if points is None:
            points = [{"id": "mid", "name": "Middle", "type": "victory", "cell": [width // 2, height // 2]}]
        if starts is None:
            starts = [
                {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
                {"slot": 1, "team": 1, "hq_cell": [width - 5, height - 5], "sector": "a"},
            ]

    return map_from_ascii(
        terrain_rows=terrain,
        sectors_rows=sectors,
        points=points,
        starts=starts,
        neutral_buildings=list(neutral_buildings),
        name=name,
        footprints=neutral_footprints(data or fixture_data()),
    )


def make_sim(
    ascii_map: str | list[str] | None = None,
    seed: int = 0,
    players: list[PlayerSetup] | None = None,
    *,
    data: GameData | None = None,
    config: SimConfig | None = None,
    game_map: GameMap | None = None,
    **map_kwargs: Any,
) -> Sim:
    """Build a `Sim` on fixture data and a small inline map.

    Default players are two US players on teams 0 and 1 using start slots 0
    and 1. Extra keyword arguments are forwarded to `make_map`.
    """
    data = data or fixture_data()
    if game_map is None:
        game_map = make_map(ascii_map, data=data, **map_kwargs)
    elif map_kwargs:
        raise TypeError("make_sim: pass either game_map= or map keyword arguments, not both")
    if players is None:
        players = [PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)]
    return Sim(game_map=game_map, players=players, data=data, seed=seed, config=config or SimConfig())


def spawn(sim: Sim, owner: int, def_id: str, cell: tuple[int, int]) -> Squad:
    """Spawn a squad at the centre of `cell` (faction is not enforced)."""
    return sim.spawn_squad(owner, def_id, center_of(cell))
