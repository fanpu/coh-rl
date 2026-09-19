"""vision — per-team visibility grids and building ghosts (task 7).

Each tick, every squad and completed player-owned building casts vision from
its cell: a set of rays from the origin cell out to the boundary cells of a
disc of radius `sight / CELL_M` cells (a Bresenham circle), each ray traced
with a Bresenham line and stopped (after marking) at the first `los_block`
cell. A team's visible grid is the boolean OR of every friendly unit's mask.

Masks are pure functions of `(origin_cell, radius_cells, map.version)`, so
they're cached on the `Sim` instance (`sim._vision_cache`, keyed by
`(origin_cell, radius_cells)` and invalidated whole-sale whenever the map
version changes) alongside the per-radius ray table (`sim._vision_rays`).
Neither cache is a module global: two `Sim`s never share state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from coh.maps.format import GameMap, cell_of
from coh.sim.constants import CELL_M
from coh.sim.state import Building, Ghost, Squad

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim

__all__ = ["run", "is_visible", "has_los"]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Ordered cells from `(x0, y0)` to `(x1, y1)` inclusive."""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def _circle_perimeter(radius: int) -> list[tuple[int, int]]:
    """Offsets of a Bresenham (midpoint) circle's boundary, 8-way symmetric
    and gap-free (each consecutive point differs by at most one cell)."""
    if radius <= 0:
        return [(0, 0)]
    points: set[tuple[int, int]] = set()
    x, y, err = radius, 0, 0
    while x >= y:
        for px, py in ((x, y), (y, x), (-y, x), (-x, y), (-x, -y), (-y, -x), (y, -x), (x, -y)):
            points.add((px, py))
        y += 1
        if err <= 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1
    return sorted(points)


def has_los(m: GameMap, a_pos, b_pos) -> bool:
    """True if no `los_block` cell lies strictly between `a_pos` and `b_pos`
    (world meters; both endpoints excluded)."""
    ax, ay = cell_of(np.asarray(a_pos), CELL_M)
    bx, by = cell_of(np.asarray(b_pos), CELL_M)
    line = _bresenham(ax, ay, bx, by)
    for cx, cy in line[1:-1]:
        if m.los_block[cy, cx]:
            return False
    return True


# ---------------------------------------------------------------------------
# caches (owned by the Sim instance, never module-global)
# ---------------------------------------------------------------------------


def _radius_cells(sight_m: float) -> int:
    return max(0, round(sight_m / CELL_M))


def _rays_for_radius(sim: "Sim", radius_cells: int) -> list[list[tuple[int, int]]]:
    rays_cache = getattr(sim, "_vision_rays", None)
    if rays_cache is None:
        rays_cache = {}
        sim._vision_rays = rays_cache
    rays = rays_cache.get(radius_cells)
    if rays is None:
        rays = [_bresenham(0, 0, px, py) for px, py in _circle_perimeter(radius_cells)]
        rays_cache[radius_cells] = rays
    return rays


def _mask_cache(sim: "Sim") -> dict:
    """The per-`(origin_cell, radius_cells)` mask cache, reset whenever the
    map's `version` changes (footprint / terrain edits invalidate LOS)."""
    cache = getattr(sim, "_vision_cache", None)
    if cache is None or getattr(sim, "_vision_cache_version", None) != sim.map.version:
        cache = {}
        sim._vision_cache = cache
        sim._vision_cache_version = sim.map.version
    return cache


def _mask_for(
    sim: "Sim",
    origin_cell: tuple[int, int],
    radius_cells: int,
    ignore_block: frozenset[tuple[int, int]] = frozenset(),
) -> np.ndarray:
    """`ignore_block` is the source's own footprint (buildings, and squads
    garrisoned in one): those cells are marked visible but never stop a ray,
    since a unit isn't blind past its own building's walls."""
    cache = _mask_cache(sim)
    key = (origin_cell, radius_cells, ignore_block)
    mask = cache.get(key)
    if mask is not None:
        return mask

    width, height = sim.map.width, sim.map.height
    los_block = sim.map.los_block
    mask = np.zeros((height, width), dtype=bool)
    ox, oy = origin_cell
    if 0 <= ox < width and 0 <= oy < height:
        mask[oy, ox] = True
    for ray in _rays_for_radius(sim, radius_cells):
        for dx, dy in ray:
            cx, cy = ox + dx, oy + dy
            if not (0 <= cx < width and 0 <= cy < height):
                break
            mask[cy, cx] = True
            if los_block[cy, cx] and (cx, cy) not in ignore_block:
                break
    cache[key] = mask
    return mask


# ---------------------------------------------------------------------------
# footprints / visibility queries
# ---------------------------------------------------------------------------


def _footprint(sim: "Sim", def_id: str) -> tuple[int, int]:
    bdef = sim.data.buildings.get(def_id)
    if bdef is not None:
        return bdef.footprint
    ndef = sim.data.neutral.get(def_id)
    if ndef is not None:
        return ndef.footprint
    return (1, 1)


def _building_center_cell(sim: "Sim", cell: tuple[int, int], def_id: str) -> tuple[int, int]:
    w, h = _footprint(sim, def_id)
    cx0, cy0 = cell
    return (cx0 + w // 2, cy0 + h // 2)


def _footprint_cells(sim: "Sim", cell: tuple[int, int], def_id: str) -> frozenset[tuple[int, int]]:
    w, h = _footprint(sim, def_id)
    cx0, cy0 = cell
    return frozenset((cx0 + dx, cy0 + dy) for dx in range(w) for dy in range(h))


def _any_visible(grid: np.ndarray, cell: tuple[int, int], footprint: tuple[int, int]) -> bool:
    cx0, cy0 = cell
    w, h = footprint
    height, width = grid.shape
    cx0c, cy0c = max(cx0, 0), max(cy0, 0)
    cx1, cy1 = min(cx0 + w, width), min(cy0 + h, height)
    if cx0c >= cx1 or cy0c >= cy1:
        return False
    return bool(grid[cy0c:cy1, cx0c:cx1].any())


def is_visible(sim: "Sim", team: int, entity: Squad | Building) -> bool:
    """Is `entity` (a `Squad` or `Building`) visible to `team`?

    Own-team entities are always visible. A garrisoned squad is visible iff
    any footprint cell of the building it's in is visible.
    """
    if isinstance(entity, Squad):
        owner = sim.state.players.get(entity.owner)
        if owner is not None and owner.team == team:
            return True
        grid = sim.state.visible.get(team)
        if grid is None:
            return False
        if entity.garrison_in is not None:
            building = sim.state.buildings.get(entity.garrison_in)
            if building is None:
                return False
            return _any_visible(grid, building.cell, _footprint(sim, building.def_id))
        cx, cy = cell_of(entity.pos, CELL_M)
        height, width = grid.shape
        if not (0 <= cx < width and 0 <= cy < height):
            return False
        return bool(grid[cy, cx])

    if isinstance(entity, Building):
        if entity.owner is not None:
            owner = sim.state.players.get(entity.owner)
            if owner is not None and owner.team == team:
                return True
        grid = sim.state.visible.get(team)
        if grid is None:
            return False
        return _any_visible(grid, entity.cell, _footprint(sim, entity.def_id))

    raise TypeError(f"is_visible: unsupported entity type {type(entity)!r}")


# ---------------------------------------------------------------------------
# tick entry point
# ---------------------------------------------------------------------------


def _update_ghosts(sim: "Sim", grids: dict[int, np.ndarray]) -> None:
    for team, grid in grids.items():
        ghosts_for_team = sim.state.ghosts.setdefault(team, {})
        live_enemy_ids: set[int] = set()

        for bid in sorted(sim.state.buildings):
            building = sim.state.buildings[bid]
            if building.owner is None:
                continue
            owner = sim.state.players.get(building.owner)
            if owner is None or owner.team == team:
                continue
            live_enemy_ids.add(bid)
            footprint = _footprint(sim, building.def_id)
            if not _any_visible(grid, building.cell, footprint):
                continue
            bdef = sim.data.buildings.get(building.def_id)
            max_hp = bdef.hp if bdef is not None and bdef.hp else building.hp
            hp_frac = 0.0 if max_hp <= 0 else max(0.0, min(1.0, building.hp / max_hp))
            ghosts_for_team[bid] = Ghost(
                building_id=bid,
                def_id=building.def_id,
                owner=building.owner,
                cell=building.cell,
                hp_frac=hp_frac,
                last_seen_tick=sim.state.tick,
            )

        for bid in sorted(ghosts_for_team):
            if bid in live_enemy_ids:
                continue
            still_exists = bid in sim.state.buildings
            if still_exists:
                building = sim.state.buildings[bid]
                owner = sim.state.players.get(building.owner) if building.owner is not None else None
                if owner is not None and owner.team == team:
                    del ghosts_for_team[bid]  # captured by this team: no longer a ghost
                continue
            ghost = ghosts_for_team[bid]
            footprint = _footprint(sim, ghost.def_id)
            if _any_visible(grid, ghost.cell, footprint):
                del ghosts_for_team[bid]  # re-scouted: confirmed gone


def run(sim: "Sim") -> None:
    """Advance the vision system by one tick."""
    grids = {team: np.zeros((sim.map.height, sim.map.width), dtype=bool) for team in sim.state.visible}

    for sid in sorted(sim.state.squads):
        squad = sim.state.squads[sid]
        if squad.abandoned or not squad.alive_members:
            continue
        owner = sim.state.players.get(squad.owner)
        if owner is None or owner.team not in grids:
            continue
        sdef = sim.data.squads.get(squad.def_id)
        if sdef is None:
            continue
        radius_cells = _radius_cells(sdef.sight)
        if squad.garrison_in is not None:
            building = sim.state.buildings.get(squad.garrison_in)
            if building is None:
                continue
            origin = _building_center_cell(sim, building.cell, building.def_id)
            ignore_block = _footprint_cells(sim, building.cell, building.def_id)
        else:
            origin = cell_of(squad.pos, CELL_M)
            ignore_block = frozenset()
        grids[owner.team] |= _mask_for(sim, origin, radius_cells, ignore_block)

    for bid in sorted(sim.state.buildings):
        building = sim.state.buildings[bid]
        if building.neutral or building.owner is None or building.progress < 1.0:
            continue
        owner = sim.state.players.get(building.owner)
        if owner is None or owner.team not in grids:
            continue
        bdef = sim.data.buildings.get(building.def_id)
        if bdef is None:
            continue
        radius_cells = _radius_cells(bdef.sight)
        origin = _building_center_cell(sim, building.cell, building.def_id)
        ignore_block = _footprint_cells(sim, building.cell, building.def_id)
        grids[owner.team] |= _mask_for(sim, origin, radius_cells, ignore_block)

    for team, grid in grids.items():
        sim.state.visible[team] = grid

    _update_ghosts(sim, grids)
