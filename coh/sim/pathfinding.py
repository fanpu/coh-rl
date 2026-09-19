"""8-connected A* pathfinding over a passability grid, plus a per-`Sim` cache.

`find_path` is the pure function the brief specifies: given a boolean
passability grid (`[cy, cx]`) it returns a list of `(cx, cy)` cells from
`start` to `goal` inclusive, or `None` if no cell is reachable at all. If
`goal` itself is impassable, the search re-targets the nearest passable cell
(a plain 8-connected BFS "ring" search outward from `goal`).

`find_path_cached` adds the caching the sim wants: a path is only recomputed
when the (kind, start, goal) triple hasn't been seen since the map last
changed (`GameMap.version` bumps on any footprint stamp or terrain edit).

Implementation notes (perf): the open/closed sets are flat integer node
indices (`cy * width + cx`) rather than tuples, with `heapq` doing tie-break
on `(f, h, insertion_order, node)` so the search is both deterministic and
fast enough for a corner-to-corner search on a 96x96 map.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from coh.sim.constants import PATH_CACHE_MAX

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim

Cell = tuple[int, int]

_SQRT2 = math.sqrt(2.0)

# (dx, dy, step cost); straight moves first so plain-open-terrain paths prefer
# them on ties (lower h at equal f), diagonals after.
_MOVES: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (-1, -1, _SQRT2),
)

_RING_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
)


def _octile(ax: int, ay: int, bx: int, by: int) -> float:
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)


def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _nearest_passable(passable: np.ndarray, cell: Cell) -> Cell | None:
    """Nearest passable cell to `cell` (itself if already passable), via an
    8-connected BFS whose expansion order is fixed, so ties resolve the same
    way every run."""
    height, width = passable.shape
    cx, cy = cell
    cx = min(max(cx, 0), width - 1)
    cy = min(max(cy, 0), height - 1)
    if passable[cy, cx]:
        return (cx, cy)

    seen = {(cx, cy)}
    queue: deque[Cell] = deque([(cx, cy)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _RING_OFFSETS:
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, width, height) or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            if passable[ny, nx]:
                return (nx, ny)
            queue.append((nx, ny))
    return None


def find_path(passable: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """8-connected A* with octile heuristic, no corner cutting.

    Deterministic tie-break: lower f, then lower h, then insertion order.
    Returns the path including both `start` and the resolved goal, or `None`
    if nothing is reachable from `start`.
    """
    height, width = passable.shape
    sx, sy = start
    if not _in_bounds(sx, sy, width, height):
        return None

    resolved_goal = _nearest_passable(passable, goal)
    if resolved_goal is None:
        return None
    gx, gy = resolved_goal
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    n = width * height
    start_idx = sy * width + sx
    goal_idx = gy * width + gx

    g_score = np.full(n, np.inf, dtype=float)
    g_score[start_idx] = 0.0
    came_from = np.full(n, -1, dtype=np.int64)
    closed = np.zeros(n, dtype=bool)

    h0 = _octile(sx, sy, gx, gy)
    counter = 0
    heap: list[tuple[float, float, int, int]] = [(h0, h0, counter, start_idx)]

    while heap:
        _, _, _, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        if idx == goal_idx:
            return _reconstruct(came_from, start_idx, goal_idx, width)
        closed[idx] = True
        x, y = idx % width, idx // width

        for dx, dy, step in _MOVES:
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, width, height) or not passable[ny, nx]:
                continue
            if dx != 0 and dy != 0:
                # no corner cutting: both orthogonal neighbours must be open
                if not passable[y, nx] or not passable[ny, x]:
                    continue
            nidx = ny * width + nx
            if closed[nidx]:
                continue
            tentative = g_score[idx] + step
            if tentative < g_score[nidx]:
                g_score[nidx] = tentative
                came_from[nidx] = idx
                counter += 1
                h = _octile(nx, ny, gx, gy)
                heapq.heappush(heap, (tentative + h, h, counter, nidx))

    return None


def _reconstruct(came_from: np.ndarray, start_idx: int, goal_idx: int, width: int) -> list[Cell]:
    path_idx = [goal_idx]
    idx = goal_idx
    while idx != start_idx:
        idx = int(came_from[idx])
        path_idx.append(idx)
    path_idx.reverse()
    return [(i % width, i // width) for i in path_idx]


# ---------------------------------------------------------------------------
# Footprint adjacency (Retreat / Garrison / Build target a building/footprint,
# not a single cell)
# ---------------------------------------------------------------------------


def adjacent_cells(cell: Cell, size: tuple[int, int], width: int, height: int) -> list[Cell]:
    """The ring of cells directly bordering the `size` footprint at `cell`
    (top-left corner), clipped to the map bounds."""
    cx0, cy0 = cell
    w, h = size
    cx1, cy1 = cx0 + w - 1, cy0 + h - 1

    cells: list[Cell] = []
    for x in range(cx0 - 1, cx1 + 2):
        for y in (cy0 - 1, cy1 + 1):
            if _in_bounds(x, y, width, height):
                cells.append((x, y))
    for y in range(cy0, cy1 + 1):
        for x in (cx0 - 1, cx1 + 1):
            if _in_bounds(x, y, width, height):
                cells.append((x, y))
    return cells


def nearest_adjacent_passable(
    passable: np.ndarray, footprint_cell: Cell, size: tuple[int, int], from_cell: Cell
) -> Cell | None:
    """The footprint-adjacent cell closest to `from_cell` that's passable.

    Falls back to the nearest passable cell to the footprint's centre if the
    whole ring is blocked (e.g. a footprint hemmed in on every side).
    """
    height, width = passable.shape
    fx, fy = from_cell
    candidates = [c for c in adjacent_cells(footprint_cell, size, width, height) if passable[c[1], c[0]]]
    if candidates:
        return min(candidates, key=lambda c: (_octile(fx, fy, c[0], c[1]), c[1], c[0]))

    cx0, cy0 = footprint_cell
    w, h = size
    centre = (cx0 + (w - 1) // 2, cy0 + (h - 1) // 2)
    return _nearest_passable(passable, centre)


# ---------------------------------------------------------------------------
# Per-Sim path cache
# ---------------------------------------------------------------------------


def find_path_cached(sim: "Sim", is_vehicle: bool, start: Cell, goal: Cell) -> list[Cell] | None:
    """`find_path`, cached on `sim` by `(is_vehicle, start, goal)`.

    The whole cache is dropped whenever `sim.map.version` has moved on since
    it was last touched (any footprint stamp or terrain edit bumps it), so
    entries never need `map.version` in the key and never accumulate stale
    versions. Within one map version it is capped at `PATH_CACHE_MAX` entries
    with FIFO eviction — a pure cache, so eviction never changes an answer.
    """
    if sim._path_cache_version != sim.map.version:
        sim._path_cache.clear()
        sim._path_cache_version = sim.map.version

    key = (is_vehicle, start, goal)
    cache = sim._path_cache
    if key in cache:
        return cache[key]
    passable = sim.map.pass_veh if is_vehicle else sim.map.pass_inf
    result = find_path(passable, start, goal)
    while len(cache) >= PATH_CACHE_MAX:
        del cache[next(iter(cache))]  # FIFO eviction
    cache[key] = result
    return result
