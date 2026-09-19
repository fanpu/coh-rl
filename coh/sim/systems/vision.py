"""vision — per-team visibility grids and building ghosts (task 7).

Each tick, every squad and completed player-owned building casts vision from
its cell: a cell `C` within `sight / CELL_M` cells of the origin `O` is
visible iff no `los_block` cell lies strictly between `O` and `C` on the
Bresenham line `O -> C` (this is exactly `has_los`'s definition, and the two
are computed with the same `_bresenham` routine in the same direction so they
can never disagree; a footprint cell of the source's own building is exempt
via `ignore_block`, so a unit isn't blind past its own walls). A team's
visible grid is the boolean OR of every friendly unit's mask.

Per-cell LOS (rather than tracing rays only to the disc's *boundary*, which
leaves gaps in open terrain once the boundary's angular resolution falls
behind its circumference) is computed vectorized: per radius, the disc's
cell offsets and each one's interior-line offsets are precomputed once
(`sim._vision_disc`); per origin, `los_block` is gathered at every interior
cell of every disc offset in one shot and AND-reduced to a per-target
visibility vector.

Masks are pure functions of `(origin_cell, radius_cells, ignore_block,
map.version)`, so they're cached on the `Sim` instance (`sim._vision_cache`,
reset whole-sale whenever the map version changes, and capped at
`VISION_MASK_CACHE_MAX` entries with FIFO eviction — a pure cache, so
eviction never affects results/determinism). Neither cache is a module
global: two `Sim`s never share state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from coh.maps.format import GameMap, cell_of
from coh.sim.constants import CELL_M, VISION_MASK_CACHE_MAX
from coh.sim.state import Building, Ghost, Squad

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim

__all__ = ["run", "is_visible", "has_los", "building_center_cell"]


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


def has_los(m: GameMap, a_pos, b_pos, ignore: frozenset[tuple[int, int]] = frozenset()) -> bool:
    """True if no `los_block` cell lies strictly between `a_pos` and `b_pos`
    (world meters; both endpoints excluded) on the Bresenham line `a -> b`.

    Uses the same `_bresenham` routine, in the same direction, as the mask
    computation below, so the two agree by construction. `ignore` is the
    line-of-sight equivalent of `_mask_for`'s `ignore_block`: cells that are
    there but must not block, namely the footprint of a building one of the
    two ends is garrisoned in (task 12) — a squad is not blind past its own
    walls, and is not safe behind them either.
    """
    ax, ay = cell_of(np.asarray(a_pos), CELL_M)
    bx, by = cell_of(np.asarray(b_pos), CELL_M)
    line = _bresenham(ax, ay, bx, by)
    for cx, cy in line[1:-1]:
        if m.los_block[cy, cx] and (cx, cy) not in ignore:
            return False
    return True


# ---------------------------------------------------------------------------
# per-radius disc precompute (cached on the Sim instance)
# ---------------------------------------------------------------------------


def _radius_cells(sight_m: float) -> int:
    return max(0, round(sight_m / CELL_M))


def _disc_offsets(radius: int) -> list[tuple[int, int]]:
    """Every cell offset `(dx, dy) != (0, 0)` with `dx**2 + dy**2 <= radius**2`
    (a filled disc, not just its boundary — every disc cell gets its own
    dedicated LOS check, so there are no angular-resolution gaps)."""
    r2 = radius * radius
    return [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if (dx, dy) != (0, 0) and dx * dx + dy * dy <= r2
    ]


def _disc_for(sim: "Sim", radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(offsets[n, 2], interior[n, max_len, 2], valid[n, max_len])`: for each
    of the `n` disc-offset targets, the offsets of the cells strictly between
    the origin and that target on the Bresenham line to it, padded to
    `max_len` with `valid` marking the real (non-padding) entries."""
    disc_cache = getattr(sim, "_vision_disc", None)
    if disc_cache is None:
        disc_cache = {}
        sim._vision_disc = disc_cache
    entry = disc_cache.get(radius)
    if entry is not None:
        return entry

    offsets = _disc_offsets(radius)
    lines = [_bresenham(0, 0, dx, dy)[1:-1] for dx, dy in offsets]
    n = len(offsets)
    max_len = max((len(line) for line in lines), default=0)

    off_arr = np.array(offsets, dtype=np.int64).reshape(n, 2)
    interior = np.zeros((n, max_len, 2), dtype=np.int64)
    valid = np.zeros((n, max_len), dtype=bool)
    for i, line in enumerate(lines):
        for j, (x, y) in enumerate(line):
            interior[i, j] = (x, y)
            valid[i, j] = True

    entry = (off_arr, interior, valid)
    disc_cache[radius] = entry
    return entry


# ---------------------------------------------------------------------------
# per-origin mask cache (owned by the Sim instance, never module-global)
# ---------------------------------------------------------------------------


def _mask_cache(sim: "Sim") -> dict:
    """The per-`(origin_cell, radius_cells, ignore_block)` mask cache, reset
    whenever the map's `version` changes (footprint / terrain edits
    invalidate LOS) and capped at `VISION_MASK_CACHE_MAX` entries (FIFO
    eviction; a pure cache, so this never affects results)."""
    cache = getattr(sim, "_vision_cache", None)
    if cache is None or getattr(sim, "_vision_cache_version", None) != sim.map.version:
        cache = {}
        sim._vision_cache = cache
        sim._vision_cache_version = sim.map.version
    return cache


def _compute_mask(
    sim: "Sim",
    origin_cell: tuple[int, int],
    radius_cells: int,
    ignore_block: frozenset[tuple[int, int]],
) -> np.ndarray:
    width, height = sim.map.width, sim.map.height
    mask = np.zeros((height, width), dtype=bool)
    ox, oy = origin_cell
    if not (0 <= ox < width and 0 <= oy < height):
        return mask
    mask[oy, ox] = True
    if radius_cells <= 0:
        return mask

    off_arr, interior, valid = _disc_for(sim, radius_cells)
    if off_arr.shape[0] == 0:
        return mask

    tx = ox + off_arr[:, 0]
    ty = oy + off_arr[:, 1]
    in_bounds = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)

    # Interior cells of an in-bounds target are always in-bounds too (they
    # lie within the O/target bounding box), so clipping here is only to
    # make the gather safe for out-of-bounds *targets*, whose interior rows
    # get masked out below via `in_bounds` regardless of the clipped value.
    ix = ox + interior[:, :, 0]
    iy = oy + interior[:, :, 1]
    ix_c = np.clip(ix, 0, width - 1)
    iy_c = np.clip(iy, 0, height - 1)

    blocked = sim.map.los_block[iy_c, ix_c] & valid
    if ignore_block:
        for gx, gy in ignore_block:
            blocked &= ~((ix == gx) & (iy == gy))

    visible = in_bounds & ~blocked.any(axis=1)
    mask[ty[visible], tx[visible]] = True
    return mask


def _mask_for(
    sim: "Sim",
    origin_cell: tuple[int, int],
    radius_cells: int,
    ignore_block: frozenset[tuple[int, int]] = frozenset(),
) -> np.ndarray:
    """`ignore_block` is the source's own footprint (buildings, and squads
    garrisoned in one): those cells are marked visible but never block LOS,
    since a unit isn't blind past its own building's walls."""
    cache = _mask_cache(sim)
    key = (origin_cell, radius_cells, ignore_block)
    mask = cache.get(key)
    if mask is not None:
        return mask

    mask = _compute_mask(sim, origin_cell, radius_cells, ignore_block)
    if len(cache) >= VISION_MASK_CACHE_MAX:
        del cache[next(iter(cache))]  # FIFO eviction
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


def building_center_cell(sim: "Sim", cell: tuple[int, int], def_id: str) -> tuple[int, int]:
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
            if building.neutral:
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
            origin = building_center_cell(sim, building.cell, building.def_id)
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
        origin = building_center_cell(sim, building.cell, building.def_id)
        ignore_block = _footprint_cells(sim, building.cell, building.def_id)
        grids[owner.team] |= _mask_for(sim, origin, radius_cells, ignore_block)

    for team, grid in grids.items():
        sim.state.visible[team] = grid

    _update_ghosts(sim, grids)
