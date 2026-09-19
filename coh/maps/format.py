"""Map format: YAML/dict loading, passability/cover arrays, validation.

A `GameMap` is a mutable container of numpy layers derived from a terrain
character grid, plus sectors, points, neutral-building placements, and player
starts. See the terrain legend in the task brief / module docstring below for
what each character means.

`GameMap` does not import `coh.data` (kept dependency-light per the sim's
dependency rule); the default neutral-building footprints are duplicated here
as plain data and can be overridden by callers that already have `GameData`
loaded (e.g. the sim, later).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from coh.data.schema import COVER_TYPES, POINT_TYPES

LIBRARY_DIR = Path(__file__).parent / "library"

# Cell size in meters. This duplicates `coh.sim.constants.CELL_M` on purpose:
# `coh/maps` must not import `coh/sim` (the dependency runs the other way), so
# the two are kept equal by `test_map_cell_size_matches_the_sim_constant`.
CELL_M = 2.0

# COVER_TYPES = ("open", "light", "heavy", "negative", "garrison")
_COVER_INDEX = {name: i for i, name in enumerate(COVER_TYPES)}

# Default neutral-building footprints (width, height) in cells, referenced by
# `neutral_buildings[*].def` in map YAML. `load_map(..., footprints=...)` can
# override/extend this set.
DEFAULT_FOOTPRINTS: dict[str, tuple[int, int]] = {
    "house_small": (3, 3),
    "house_large": (4, 4),
    "barn": (5, 4),
}


class MapError(ValueError):
    """Raised for any malformed or invalid map definition."""


# ---------------------------------------------------------------------------
# Terrain legend
# ---------------------------------------------------------------------------

# ch -> (pass_inf, pass_veh, los_block, area_cover, cover_object)
# area_cover / cover_object are COVER_TYPES names or None (None => "open"/no
# object respectively).
_LEGEND: dict[str, tuple[bool, bool, bool, str, str | None]] = {
    ".": (True, True, False, "open", None),
    "r": (True, True, False, "negative", None),
    "c": (True, True, False, "heavy", None),
    "f": (True, True, False, "open", "light"),
    "w": (False, False, False, "open", "heavy"),
    "H": (False, False, True, "open", "heavy"),
    "T": (True, False, True, "light", None),
    "~": (False, False, False, "open", None),
}


@dataclass(frozen=True)
class SectorDef:
    id: int
    point_id: str | None
    neighbors: tuple[int, ...]


@dataclass(frozen=True)
class PointDef:
    id: str
    name: str
    type: str
    cell: tuple[int, int]
    sector: int


@dataclass(frozen=True)
class NeutralPlacement:
    def_id: str
    cell: tuple[int, int]


@dataclass(frozen=True)
class StartDef:
    slot: int
    team: int
    hq_cell: tuple[int, int]
    sector: int


@dataclass
class GameMap:
    name: str
    width: int
    height: int
    pass_inf: np.ndarray
    pass_veh: np.ndarray
    los_block: np.ndarray
    area_cover: np.ndarray
    cover_object: np.ndarray
    sector_id: np.ndarray
    sectors: dict[int, SectorDef]
    points: dict[str, PointDef]
    neutral_buildings: list[NeutralPlacement]
    starts: list[StartDef]
    terrain: np.ndarray  # dtype '<U1', original terrain chars (mutable ground truth)
    version: int = 0

    def set_terrain_cell(self, cell: tuple[int, int], ch: str) -> None:
        """Overwrite a single terrain cell and recompute its derived layers.

        Used by the sim for things like a vehicle crushing a fence (`f` ->
        `.`) or a wreck leaving a crater (`.` -> `c`).
        """
        cx, cy = cell
        if ch not in _LEGEND:
            raise MapError(f"set_terrain_cell: unknown terrain char {ch!r}")
        self.terrain[cy, cx] = ch
        pass_inf, pass_veh, los_block, area_cover, cover_object = _LEGEND[ch]
        self.pass_inf[cy, cx] = pass_inf
        self.pass_veh[cy, cx] = pass_veh
        self.los_block[cy, cx] = los_block
        self.area_cover[cy, cx] = _COVER_INDEX[area_cover]
        self.cover_object[cy, cx] = _COVER_INDEX[cover_object] if cover_object else 0
        self.version += 1

    def stamp_footprint(self, cell: tuple[int, int], size: tuple[int, int], blocked: bool) -> None:
        """Toggle impassable + LOS-blocking over a rectangle of cells.

        `cell` is the top-left corner, `size` is (w, h). When `blocked` is
        False, the rectangle's passability/LOS/cover is restored from the
        underlying `terrain` characters (does not touch `terrain` itself).
        """
        cx0, cy0 = cell
        w, h = size
        cx1 = min(cx0 + w, self.width)
        cy1 = min(cy0 + h, self.height)
        if blocked:
            self.pass_inf[cy0:cy1, cx0:cx1] = False
            self.pass_veh[cy0:cy1, cx0:cx1] = False
            self.los_block[cy0:cy1, cx0:cx1] = True
        else:
            for cy in range(cy0, cy1):
                for cx in range(cx0, cx1):
                    ch = str(self.terrain[cy, cx])
                    pass_inf, pass_veh, los_block, area_cover, cover_object = _LEGEND[ch]
                    self.pass_inf[cy, cx] = pass_inf
                    self.pass_veh[cy, cx] = pass_veh
                    self.los_block[cy, cx] = los_block
                    self.area_cover[cy, cx] = _COVER_INDEX[area_cover]
                    self.cover_object[cy, cx] = _COVER_INDEX[cover_object] if cover_object else 0
        self.version += 1

    def copy(self) -> "GameMap":
        """Return a deep-enough copy so mutation on one instance doesn't leak."""
        return GameMap(
            name=self.name,
            width=self.width,
            height=self.height,
            pass_inf=self.pass_inf.copy(),
            pass_veh=self.pass_veh.copy(),
            los_block=self.los_block.copy(),
            area_cover=self.area_cover.copy(),
            cover_object=self.cover_object.copy(),
            sector_id=self.sector_id.copy(),
            sectors=dict(self.sectors),
            points=dict(self.points),
            neutral_buildings=list(self.neutral_buildings),
            starts=list(self.starts),
            terrain=self.terrain.copy(),
            version=self.version,
        )


# ---------------------------------------------------------------------------
# cell / world helpers
# ---------------------------------------------------------------------------


def cell_of(pos: np.ndarray, cell_m: float = CELL_M) -> tuple[int, int]:
    """World position (meters, x right / y down) -> (cx, cy)."""
    return (int(pos[0] // cell_m), int(pos[1] // cell_m))


def center_of(cell: tuple[int, int], cell_m: float = CELL_M) -> np.ndarray:
    """(cx, cy) -> world position (meters) of the cell's center."""
    cx, cy = cell
    return np.array([(cx + 0.5) * cell_m, (cy + 0.5) * cell_m])


def distance(a, b) -> float:
    """Distance in meters between two world positions.

    Exactly `float(np.linalg.norm(a - b))` for the 2-D positions the sim uses
    -- `np.linalg.norm` of a 2-vector is `sqrt(x0*x0 + x1*x1)` over the same
    float64 values, so the two agree bit for bit -- without allocating the
    difference array. Worth having as its own function because range checks
    are the single most frequent numeric operation in a tick.
    """
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_map(
    name_or_path: str | Path | dict[str, Any],
    footprints: dict[str, tuple[int, int]] | None = None,
) -> GameMap:
    """Load a `GameMap` from a library name, a filesystem path, or a dict.

    - `str` that isn't an existing path: loaded from
      `coh/maps/library/<name>.yaml`.
    - `Path`, or a `str` that names an existing file: loaded from that file.
    - `dict`: already-parsed map data (used directly, e.g. by tests).
    """
    if isinstance(name_or_path, dict):
        data = name_or_path
    else:
        path = Path(name_or_path)
        if not path.exists():
            path = LIBRARY_DIR / f"{name_or_path}.yaml"
        if not path.exists():
            raise MapError(f"map not found: {name_or_path!r} (looked in {path})")
        try:
            with path.open() as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise MapError(f"{path}: invalid YAML: {e}") from e
        if not isinstance(data, dict):
            raise MapError(f"{path}: map file must contain a mapping at the top level")

    return _build_map(data, footprints or DEFAULT_FOOTPRINTS)


def map_from_ascii(
    terrain_rows: list[str],
    sectors_rows: list[str],
    points: list[dict[str, Any]],
    starts: list[dict[str, Any]],
    neutral_buildings: list[dict[str, Any]] = (),
    name: str = "inline",
    cell_m: float = CELL_M,
    footprints: dict[str, tuple[int, int]] | None = None,
) -> GameMap:
    """Convenience wrapper: build the dict form and load it.

    Lets tests (and later tasks) build small inline maps without writing
    YAML to disk.
    """
    data = {
        "name": name,
        "cell_m": cell_m,
        "terrain": list(terrain_rows),
        "sectors": list(sectors_rows),
        "points": list(points),
        "neutral_buildings": list(neutral_buildings),
        "starts": list(starts),
    }
    return _build_map(data, footprints or DEFAULT_FOOTPRINTS)


# ---------------------------------------------------------------------------
# Build + validate
# ---------------------------------------------------------------------------


def _build_map(data: dict[str, Any], footprints: dict[str, tuple[int, int]]) -> GameMap:
    name = data.get("name", "map")
    cell_m = float(data.get("cell_m", CELL_M))
    if cell_m != CELL_M:
        raise MapError(
            f"{name}: cell_m is {cell_m}, but the simulation grid is {CELL_M} m per cell; "
            "every distance in the stat tables is in meters on that grid"
        )

    terrain_rows = data.get("terrain")
    sectors_rows = data.get("sectors")
    if not terrain_rows or not isinstance(terrain_rows, list):
        raise MapError(f"{name}: 'terrain' must be a non-empty list of strings")
    if not sectors_rows or not isinstance(sectors_rows, list):
        raise MapError(f"{name}: 'sectors' must be a non-empty list of strings")

    height = len(terrain_rows)
    width = len(terrain_rows[0])
    for i, row in enumerate(terrain_rows):
        if len(row) != width:
            raise MapError(f"{name}: terrain row {i} has length {len(row)}, expected {width} (ragged grid)")
    if len(sectors_rows) != height:
        raise MapError(f"{name}: sectors has {len(sectors_rows)} rows, terrain has {height} (shape mismatch)")
    for i, row in enumerate(sectors_rows):
        if len(row) != width:
            raise MapError(f"{name}: sectors row {i} has length {len(row)}, expected {width} (shape mismatch)")
    for i, ch in enumerate(sorted({c for row in terrain_rows for c in row})):
        if ch not in _LEGEND:
            raise MapError(f"{name}: unknown terrain character {ch!r}")

    pass_inf = np.zeros((height, width), dtype=bool)
    pass_veh = np.zeros((height, width), dtype=bool)
    los_block = np.zeros((height, width), dtype=bool)
    area_cover = np.zeros((height, width), dtype=np.uint8)
    cover_object = np.zeros((height, width), dtype=np.uint8)
    terrain = np.empty((height, width), dtype="<U1")

    for cy, row in enumerate(terrain_rows):
        for cx, ch in enumerate(row):
            terrain[cy, cx] = ch
            p_inf, p_veh, los, area_c, obj_c = _LEGEND[ch]
            pass_inf[cy, cx] = p_inf
            pass_veh[cy, cx] = p_veh
            los_block[cy, cx] = los
            area_cover[cy, cx] = _COVER_INDEX[area_c]
            cover_object[cy, cx] = _COVER_INDEX[obj_c] if obj_c else 0

    # --- sectors ---------------------------------------------------------
    sector_chars = sorted({c for row in sectors_rows for c in row})
    char_to_id = {c: i for i, c in enumerate(sector_chars)}
    sector_id = np.zeros((height, width), dtype=np.int16)
    for cy, row in enumerate(sectors_rows):
        for cx, ch in enumerate(row):
            sector_id[cy, cx] = char_to_id[ch]

    neighbor_sets: dict[int, set[int]] = {i: set() for i in char_to_id.values()}
    for cy in range(height):
        for cx in range(width):
            sid = int(sector_id[cy, cx])
            if cx + 1 < width:
                other = int(sector_id[cy, cx + 1])
                if other != sid:
                    neighbor_sets[sid].add(other)
                    neighbor_sets[other].add(sid)
            if cy + 1 < height:
                other = int(sector_id[cy + 1, cx])
                if other != sid:
                    neighbor_sets[sid].add(other)
                    neighbor_sets[other].add(sid)

    # --- points ------------------------------------------------------------
    points_in: list[dict[str, Any]] = data.get("points") or []
    if not points_in:
        raise MapError(f"{name}: at least one point is required")

    points: dict[str, PointDef] = {}
    sector_point: dict[int, str] = {}
    for p in points_in:
        pid = p["id"]
        cx, cy = p["cell"]
        if not (0 <= cx < width and 0 <= cy < height):
            raise MapError(f"{name}: point {pid!r} cell {(cx, cy)} is out of bounds")
        if p["type"] not in POINT_TYPES:
            raise MapError(f"{name}: point {pid!r} has unknown type {p['type']!r}, expected one of {list(POINT_TYPES)}")
        sid = int(sector_id[cy, cx])
        if sid in sector_point:
            raise MapError(
                f"{name}: sector {sector_chars[sid]!r} has more than one point "
                f"({sector_point[sid]!r} and {pid!r})"
            )
        sector_point[sid] = pid
        points[pid] = PointDef(id=pid, name=p.get("name", pid), type=p["type"], cell=(cx, cy), sector=sid)

    missing = [sector_chars[sid] for sid in char_to_id.values() if sid not in sector_point]
    if missing:
        raise MapError(f"{name}: sector(s) {missing} have no point (each sector needs exactly one)")

    if not any(pt.type == "victory" for pt in points.values()):
        raise MapError(f"{name}: map must have at least one victory point")

    sectors: dict[int, SectorDef] = {
        sid: SectorDef(id=sid, point_id=sector_point.get(sid), neighbors=tuple(sorted(neighbor_sets[sid])))
        for sid in char_to_id.values()
    }

    if not _sectors_connected(sectors):
        raise MapError(f"{name}: sector graph is not connected")

    # --- neutral buildings ---------------------------------------------------
    neutral_buildings: list[NeutralPlacement] = []
    for nb in data.get("neutral_buildings") or []:
        def_id = nb["def"]
        cx, cy = nb["cell"]
        if def_id not in footprints:
            raise MapError(f"{name}: unknown neutral building def {def_id!r}")
        w, h = footprints[def_id]
        if not (0 <= cx and cx + w <= width and 0 <= cy and cy + h <= height):
            raise MapError(f"{name}: neutral building {def_id!r} at {(cx, cy)} footprint {(w, h)} out of bounds")
        neutral_buildings.append(NeutralPlacement(def_id=def_id, cell=(cx, cy)))

    # --- starts --------------------------------------------------------------
    starts_in = data.get("starts") or []
    if not starts_in:
        raise MapError(f"{name}: at least one start is required")
    starts: list[StartDef] = []
    seen_slots: set[int] = set()
    for s in starts_in:
        slot = s["slot"]
        if slot in seen_slots:
            raise MapError(f"{name}: start slot {slot} is defined more than once")
        seen_slots.add(slot)
        cx, cy = s["hq_cell"]
        if not (0 <= cx < width and 0 <= cy < height):
            raise MapError(f"{name}: start slot {slot} hq_cell {(cx, cy)} is out of bounds")
        if not pass_inf[cy, cx]:
            raise MapError(f"{name}: start slot {slot} hq_cell {(cx, cy)} is impassable to infantry")
        sector_ch = s["sector"]
        if sector_ch not in char_to_id:
            raise MapError(f"{name}: start slot {slot} references unknown sector {sector_ch!r}")
        actual_ch = sector_chars[int(sector_id[cy, cx])]
        if actual_ch != sector_ch:
            raise MapError(
                f"{name}: start slot {slot} claims sector {sector_ch!r} but its hq_cell {(cx, cy)} "
                f"lies in sector {actual_ch!r}"
            )
        starts.append(StartDef(slot=slot, team=s["team"], hq_cell=(cx, cy), sector=char_to_id[sector_ch]))

    game_map = GameMap(
        name=name,
        width=width,
        height=height,
        pass_inf=pass_inf,
        pass_veh=pass_veh,
        los_block=los_block,
        area_cover=area_cover,
        cover_object=cover_object,
        sector_id=sector_id,
        sectors=sectors,
        points=points,
        neutral_buildings=neutral_buildings,
        starts=starts,
        terrain=terrain,
    )

    # Stamp neutral-building footprints impassable + LOS-blocking at load.
    for placement in neutral_buildings:
        w, h = footprints[placement.def_id]
        game_map.stamp_footprint(placement.cell, (w, h), blocked=True)
    game_map.version = 0  # footprint stamping above doesn't count as post-load mutation

    _validate_reachability(game_map)

    return game_map


def _sectors_connected(sectors: dict[int, SectorDef]) -> bool:
    if not sectors:
        return True
    start = next(iter(sectors))
    seen = {start}
    queue = deque([start])
    while queue:
        sid = queue.popleft()
        for nb in sectors[sid].neighbors:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return seen == set(sectors)


def _validate_reachability(m: GameMap) -> None:
    """BFS from every HQ cell over `pass_inf`; every point must be reached
    from at least one HQ (players share a connected battlefield)."""
    reachable = np.zeros((m.height, m.width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for start in m.starts:
        cx, cy = start.hq_cell
        if not reachable[cy, cx]:
            reachable[cy, cx] = True
            queue.append((cx, cy))

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < m.width and 0 <= ny < m.height and not reachable[ny, nx] and m.pass_inf[ny, nx]:
                reachable[ny, nx] = True
                queue.append((nx, ny))

    for pt in m.points.values():
        px, py = pt.cell
        if not reachable[py, px]:
            raise MapError(f"{m.name}: point {pt.id!r} at {pt.cell} is not infantry-reachable from any HQ")
