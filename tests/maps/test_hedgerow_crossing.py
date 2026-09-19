"""Tests for the `hedgerow_crossing` library map.

The map is a 1v1 benchmark arena, so the bar is not "roughly mirrored" but
*exactly* seat-symmetric: rotating the whole map 180 degrees, (x, y) ->
(W-1-x, H-1-y), must map seat 0's world onto seat 1's cell for cell. That
covers terrain-derived layers, the sector grid, every point's cell and type,
and the stamped HQ footprints; and it is checked in the currency that decides
matches, infantry path distance, by BFS from each HQ footprint.

`victory_mid` is the one point that cannot be its own rotational image (an
even grid has no centre cell), so it is held to the weaker promise the
symmetry is there for: equal infantry path distance from both HQs.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from coh.maps.format import GameMap, load_map
from coh.sim.sim import PlayerSetup, Sim, neutral_footprints
from tests.helpers import fixture_data

# The mid victory point's rotational image is a different cell, so it is the
# one point exempt from cell-for-cell mirroring (see the module docstring).
MID_VP = "victory_mid"


def _bfs(passable: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    """4-neighbour BFS distance in cells; -1 where unreachable.

    `starts` are seeded at distance 0 whether or not they are passable, so a
    whole HQ footprint (which the sim stamps impassable) can be the source.
    """
    height, width = passable.shape
    dist = np.full((height, width), -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()
    for x, y in starts:
        if dist[y, x] < 0:
            dist[y, x] = 0
            queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and passable[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((nx, ny))
    return dist


def _rotate(m: GameMap, cell: tuple[int, int]) -> tuple[int, int]:
    return (m.width - 1 - cell[0], m.height - 1 - cell[1])


def _partner(m: GameMap, point_id: str) -> str:
    """The point of the sector that `point_id`'s sector rotates onto."""
    rx, ry = _rotate(m, m.points[point_id].cell)
    partner_id = m.sectors[int(m.sector_id[ry, rx])].point_id
    assert partner_id is not None
    return partner_id


def _footprint_cells(cell: tuple[int, int], size: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = cell
    w, h = size
    return [(x, y) for y in range(y0, y0 + h) for x in range(x0, x0 + w)]


@pytest.fixture(scope="module")
def data():
    return fixture_data()


@pytest.fixture(scope="module")
def hedgerow_map(data):
    return load_map("hedgerow_crossing", footprints=neutral_footprints(data))


@pytest.fixture(scope="module")
def hq_footprints(hedgerow_map, data) -> dict[int, list[tuple[int, int]]]:
    """Each seat's HQ footprint cells, exactly as the sim stamps them."""
    sim = Sim(
        game_map=hedgerow_map,
        players=[PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)],
        data=data,
        seed=0,
    )
    out: dict[int, list[tuple[int, int]]] = {}
    for player_id, player in sorted(sim.state.players.items()):
        hq = sim.state.buildings[player.hq_id]
        out[player_id] = _footprint_cells(hq.cell, data.buildings[hq.def_id].footprint)
    return out


@pytest.fixture(scope="module")
def hq_distances(hedgerow_map, hq_footprints) -> dict[int, np.ndarray]:
    return {pid: _bfs(hedgerow_map.pass_inf, cells) for pid, cells in hq_footprints.items()}


def test_loads_and_validates(hedgerow_map):
    m = hedgerow_map
    assert m.width == 96
    assert m.height == 96
    assert len(m.sectors) == 13
    assert len(m.points) == 13
    assert len(m.neutral_buildings) == 4
    assert sum(1 for p in m.points.values() if p.type == "victory") == 3


def test_terrain_layers_180_rotationally_symmetric(hedgerow_map):
    m = hedgerow_map
    for name in ("pass_inf", "pass_veh", "los_block", "area_cover", "cover_object", "terrain"):
        layer = getattr(m, name)
        assert np.array_equal(layer, layer[::-1, ::-1]), f"{name} is not 180-rotationally symmetric"


def test_sector_grid_180_rotationally_symmetric(hedgerow_map):
    """Cells of one sector must rotate onto cells of a single other sector."""
    m = hedgerow_map
    rotated = m.sector_id[::-1, ::-1]
    for sid in m.sectors:
        images = set(np.unique(rotated[m.sector_id == sid]).tolist())
        assert len(images) == 1, f"sector {sid} rotates onto sectors {sorted(images)}"


def test_neutral_buildings_are_rotational_image_pairs(hedgerow_map, data):
    m = hedgerow_map
    footprints = neutral_footprints(data)
    stamped = set()
    for placement in m.neutral_buildings:
        stamped.update(_footprint_cells(placement.cell, footprints[placement.def_id]))
    assert {_rotate(m, cell) for cell in stamped} == stamped


def test_point_cells_are_exact_rotational_images(hedgerow_map):
    m = hedgerow_map
    for point_id, point in m.points.items():
        if point_id == MID_VP:
            continue
        partner = m.points[_partner(m, point_id)]
        assert partner.cell == _rotate(m, point.cell), (
            f"{point_id} at {point.cell} rotates to {_rotate(m, point.cell)}, "
            f"but its partner {partner.id} sits at {partner.cell}"
        )


def test_point_types_are_identical_across_the_rotation(hedgerow_map):
    """Strict type symmetry: no 'swap pairs' of unequal-value resources."""
    m = hedgerow_map
    for point_id, point in m.points.items():
        partner = m.points[_partner(m, point_id)]
        assert partner.type == point.type, (
            f"{point_id} is {point.type!r} but its rotational partner "
            f"{partner.id} is {partner.type!r}"
        )


def test_both_seats_get_the_same_point_type_mix(hedgerow_map, hq_distances):
    """Each seat's nearer half of every point pair carries the same types."""
    m = hedgerow_map
    mix: dict[int, list[str]] = {0: [], 1: []}
    for point_id, point in m.points.items():
        if point_id == MID_VP:
            continue
        x, y = point.cell
        nearer = 0 if hq_distances[0][y, x] <= hq_distances[1][y, x] else 1
        mix[nearer].append(point.type)
    assert sorted(mix[0]) == sorted(mix[1]), f"seat 0 gets {sorted(mix[0])}, seat 1 gets {sorted(mix[1])}"


def test_hq_footprints_are_exact_rotational_images(hedgerow_map, hq_footprints):
    """The footprints the *sim* stamps -- not just the `hq_cell` anchors.

    Fails loudly if `Sim.spawn_building`'s footprint anchoring ever changes
    without `starts[*].hq_cell` being re-derived for the new rule.
    """
    m = hedgerow_map
    assert {_rotate(m, cell) for cell in hq_footprints[0]} == set(hq_footprints[1])


def test_each_point_is_the_same_path_distance_from_its_owner_as_its_partner(hedgerow_map, hq_distances):
    m = hedgerow_map
    for point_id, point in m.points.items():
        if point_id == MID_VP:
            continue
        x, y = point.cell
        rx, ry = _rotate(m, point.cell)
        assert hq_distances[0][y, x] == hq_distances[1][ry, rx] >= 0, (
            f"{point_id} is {hq_distances[0][y, x]} cells from HQ 0 but its partner "
            f"{_partner(m, point_id)} is {hq_distances[1][ry, rx]} cells from HQ 1"
        )


def test_mid_victory_point_is_path_equidistant_from_both_hqs(hedgerow_map, hq_distances):
    m = hedgerow_map
    x, y = m.points[MID_VP].cell
    d0, d1 = int(hq_distances[0][y, x]), int(hq_distances[1][y, x])
    assert d0 > 0 and d1 > 0
    assert abs(d0 - d1) <= 1, f"{MID_VP} is {d0} cells from HQ 0 and {d1} from HQ 1"


def test_every_point_reachable_by_infantry_from_both_hqs(hedgerow_map, hq_footprints):
    m = hedgerow_map
    for player_id, cells in hq_footprints.items():
        reached = _bfs(m.pass_inf, cells)
        for p in m.points.values():
            x, y = p.cell
            assert reached[y, x] >= 0, f"point {p.id!r} at {p.cell} unreachable by infantry from HQ {player_id}"


def test_every_point_reachable_by_vehicle_from_both_hqs(hedgerow_map, hq_footprints):
    m = hedgerow_map
    for player_id, cells in hq_footprints.items():
        reached = _bfs(m.pass_veh, cells)
        for p in m.points.values():
            x, y = p.cell
            assert reached[y, x] >= 0, f"point {p.id!r} at {p.cell} unreachable by vehicle from HQ {player_id}"


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
