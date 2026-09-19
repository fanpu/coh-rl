"""Tests for the `hedgerow_crossing` library map: loads, validates, and is
180-degree rotationally symmetric in passability and point types."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from coh.maps.format import load_map

# fuel_high/munitions_high are an intentional swap pair: the two central
# "high" economy points sit at rotationally-mirrored positions but hand out
# different resources, so both sides get one high-value fuel *and* one
# high-value munitions point without either side having a positional
# advantage. Every other point type is required to match exactly at its
# rotated position.
_TYPE_EQUIVALENCE = {
    "fuel_high": "munitions_high",
    "munitions_high": "fuel_high",
}


def _bfs_reachable(passable: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    height, width = passable.shape
    reached = np.zeros_like(passable, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x, y in starts:
        if passable[y, x] and not reached[y, x]:
            reached[y, x] = True
            queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and passable[ny, nx] and not reached[ny, nx]:
                reached[ny, nx] = True
                queue.append((nx, ny))
    return reached


@pytest.fixture(scope="module")
def hedgerow_map():
    return load_map("hedgerow_crossing")


def test_loads_and_validates(hedgerow_map):
    m = hedgerow_map
    assert m.width == 96
    assert m.height == 96
    assert len(m.sectors) == 13
    assert len(m.points) == 13
    assert len(m.neutral_buildings) == 4
    assert sum(1 for p in m.points.values() if p.type == "victory") == 3


def test_pass_inf_180_rotationally_symmetric(hedgerow_map):
    m = hedgerow_map
    assert np.array_equal(m.pass_inf, m.pass_inf[::-1, ::-1])


def test_pass_veh_180_rotationally_symmetric(hedgerow_map):
    m = hedgerow_map
    assert np.array_equal(m.pass_veh, m.pass_veh[::-1, ::-1])


def test_point_types_180_rotationally_symmetric(hedgerow_map):
    m = hedgerow_map
    width, height = m.width, m.height
    for p in m.points.values():
        rx, ry = width - 1 - p.cell[0], height - 1 - p.cell[1]
        rotated_sector = int(m.sector_id[ry, rx])
        rotated_point_id = m.sectors[rotated_sector].point_id
        rotated_type = m.points[rotated_point_id].type
        expected = _TYPE_EQUIVALENCE.get(p.type, p.type)
        assert rotated_type == expected, (
            f"{p.id} ({p.type}) at {p.cell} rotates to sector "
            f"{rotated_sector} whose point {rotated_point_id!r} has type "
            f"{rotated_type!r}, expected {expected!r}"
        )


def test_every_point_reachable_by_infantry_from_both_hqs(hedgerow_map):
    m = hedgerow_map
    starts = [tuple(s.hq_cell) for s in m.starts]
    reached = _bfs_reachable(m.pass_inf, starts)
    for p in m.points.values():
        x, y = p.cell
        assert reached[y, x], f"point {p.id!r} at {p.cell} unreachable by infantry"


def test_every_point_reachable_by_vehicle_from_both_hqs(hedgerow_map):
    m = hedgerow_map
    starts = [tuple(s.hq_cell) for s in m.starts]
    reached = _bfs_reachable(m.pass_veh, starts)
    for p in m.points.values():
        x, y = p.cell
        assert reached[y, x], f"point {p.id!r} at {p.cell} unreachable by vehicle"


def test_hq_areas_have_12x12_open_buildable_ground(hedgerow_map):
    m = hedgerow_map
    for start in m.starts:
        hx, hy = start.hq_cell
        # search for a 12x12 window, fully passable to vehicles and open
        # (no cover object / non-open area cover), containing or near the HQ.
        found = False
        for y0 in range(max(0, hy - 20), min(m.height - 12, hy + 8) + 1):
            for x0 in range(max(0, hx - 20), min(m.width - 12, hx + 8) + 1):
                window_pass = m.pass_veh[y0 : y0 + 12, x0 : x0 + 12]
                window_area = m.area_cover[y0 : y0 + 12, x0 : x0 + 12]
                window_obj = m.cover_object[y0 : y0 + 12, x0 : x0 + 12]
                if window_pass.all() and (window_area == 0).all() and (window_obj == 0).all():
                    found = True
                    break
            if found:
                break
        assert found, f"no 12x12 open buildable window found near HQ slot {start.slot}"
