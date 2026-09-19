"""Tests for coh.maps.format: map loading, arrays, and validation."""

from __future__ import annotations

import numpy as np
import pytest

from coh.maps.format import (
    GameMap,
    MapError,
    NeutralPlacement,
    PointDef,
    SectorDef,
    StartDef,
    cell_of,
    center_of,
    load_map,
    map_from_ascii,
)

# A 12x8 inline map exercising every terrain legend character.
TERRAIN = [
    "............",
    "..r.........",
    "..........~.",
    "....c.......",
    "....f.......",
    "....w.......",
    "....H.......",
    "....T.......",
]
SECTORS = [
    "AAAAAABBBBBB",
    "AAAAAABBBBBB",
    "AAAAAABBBBBB",
    "AAAAAABBBBBB",
    "AAAAAABBBBBB",
    "AAAAAABBBBBB",
    "AAAAAABBBBBB",
    "AAAAAABBBBBB",
]
POINTS = [
    {"id": "victory_a", "name": "Victory A", "type": "victory", "cell": [1, 1]},
    {"id": "strategic_b", "name": "Strategic B", "type": "strategic", "cell": [8, 1]},
]
STARTS = [
    {"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "A"},
    {"slot": 1, "team": 1, "hq_cell": [11, 0], "sector": "B"},
]


def make_map(**overrides) -> GameMap:
    kwargs = dict(
        terrain_rows=TERRAIN,
        sectors_rows=SECTORS,
        points=POINTS,
        starts=STARTS,
        neutral_buildings=(),
        name="test12x8",
    )
    kwargs.update(overrides)
    return map_from_ascii(**kwargs)


# ---------------------------------------------------------------------------
# Basic shape / legend correctness
# ---------------------------------------------------------------------------


def test_dimensions():
    m = make_map()
    assert m.width == 12
    assert m.height == 8
    assert m.pass_inf.shape == (8, 12)
    assert m.pass_veh.shape == (8, 12)
    assert m.los_block.shape == (8, 12)
    assert m.area_cover.shape == (8, 12)
    assert m.cover_object.shape == (8, 12)
    assert m.sector_id.shape == (8, 12)


def test_open_ground():
    m = make_map()
    assert m.pass_inf[0, 0] and m.pass_veh[0, 0]
    assert not m.los_block[0, 0]
    assert m.area_cover[0, 0] == 0  # open


def test_road():
    m = make_map()
    assert m.pass_inf[1, 2] and m.pass_veh[1, 2]
    assert not m.los_block[1, 2]
    assert m.area_cover[1, 2] == 3  # negative


def test_water():
    m = make_map()
    assert not m.pass_inf[2, 10]
    assert not m.pass_veh[2, 10]
    assert not m.los_block[2, 10]


def test_crater():
    m = make_map()
    assert m.pass_inf[3, 4] and m.pass_veh[3, 4]
    assert not m.los_block[3, 4]
    assert m.area_cover[3, 4] == 2  # heavy


def test_fence():
    m = make_map()
    assert m.pass_inf[4, 4] and m.pass_veh[4, 4]
    assert not m.los_block[4, 4]
    assert m.cover_object[4, 4] == 1  # light


def test_wall():
    m = make_map()
    assert not m.pass_inf[5, 4]
    assert not m.pass_veh[5, 4]
    assert not m.los_block[5, 4]
    assert m.cover_object[5, 4] == 2  # heavy


def test_bocage():
    m = make_map()
    assert not m.pass_inf[6, 4]
    assert not m.pass_veh[6, 4]
    assert m.los_block[6, 4]
    assert m.cover_object[6, 4] == 2  # heavy


def test_trees():
    m = make_map()
    assert m.pass_inf[7, 4]
    assert not m.pass_veh[7, 4]
    assert m.los_block[7, 4]
    assert m.area_cover[7, 4] == 1  # light


# ---------------------------------------------------------------------------
# Sectors / points / adjacency
# ---------------------------------------------------------------------------


def test_sector_ids_and_adjacency():
    m = make_map()
    a_id = m.sector_id[0, 0]
    b_id = m.sector_id[0, 11]
    assert a_id != b_id
    assert len(m.sectors) == 2
    assert b_id in m.sectors[a_id].neighbors
    assert a_id in m.sectors[b_id].neighbors


def test_points_resolved():
    m = make_map()
    assert set(m.points) == {"victory_a", "strategic_b"}
    va = m.points["victory_a"]
    assert isinstance(va, PointDef)
    assert va.type == "victory"
    assert tuple(va.cell) == (1, 1)
    a_id = m.sector_id[0, 0]
    assert va.sector == a_id


def test_starts():
    m = make_map()
    assert len(m.starts) == 2
    s0 = m.starts[0]
    assert isinstance(s0, StartDef)
    assert s0.team == 0
    assert tuple(s0.hq_cell) == (0, 0)


# ---------------------------------------------------------------------------
# cell_of / center_of
# ---------------------------------------------------------------------------


def test_cell_of():
    assert cell_of(np.array([0.0, 0.0])) == (0, 0)
    assert cell_of(np.array([3.9, 5.9])) == (1, 2)
    assert cell_of(np.array([4.0, 6.0])) == (2, 3)


def test_center_of():
    c = center_of((0, 0))
    assert np.allclose(c, [1.0, 1.0])
    c = center_of((2, 3))
    assert np.allclose(c, [5.0, 7.0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_ragged_terrain_raises():
    bad_terrain = list(TERRAIN)
    bad_terrain[0] = bad_terrain[0] + "."  # one char too long
    with pytest.raises(MapError):
        make_map(terrain_rows=bad_terrain)


def test_terrain_sectors_shape_mismatch_raises():
    bad_sectors = SECTORS[:-1]  # one row short
    with pytest.raises(MapError):
        make_map(sectors_rows=bad_sectors)


def test_point_outside_its_sector_raises():
    bad_points = [
        {"id": "victory_a", "name": "Victory A", "type": "victory", "cell": [8, 1]},  # in sector B
        {"id": "strategic_b", "name": "Strategic B", "type": "strategic", "cell": [8, 1]},
    ]
    with pytest.raises(MapError):
        make_map(points=bad_points)


def test_sector_without_point_raises():
    bad_points = [
        {"id": "victory_a", "name": "Victory A", "type": "victory", "cell": [1, 1]},
    ]
    with pytest.raises(MapError):
        make_map(points=bad_points)


def test_sector_with_two_points_raises():
    bad_points = POINTS + [
        {"id": "extra", "name": "Extra", "type": "strategic", "cell": [2, 2]},
    ]
    with pytest.raises(MapError):
        make_map(points=bad_points)


def test_three_sector_grid_is_connected():
    # Sanity check: a fully-tiled rectangular grid's sector adjacency graph
    # is always connected by construction (every cell belongs to some
    # sector, and the grid itself is simply connected), so this must load.
    sectors = [
        "AAAABBBBCCCC",
        "AAAABBBBCCCC",
        "AAAABBBBCCCC",
        "AAAABBBBCCCC",
        "AAAABBBBCCCC",
        "AAAABBBBCCCC",
        "AAAABBBBCCCC",
        "AAAABBBBCCCC",
    ]
    points = [
        {"id": "victory_a", "name": "Victory A", "type": "victory", "cell": [1, 1]},
        {"id": "strategic_b", "name": "Strategic B", "type": "strategic", "cell": [5, 1]},
        {"id": "muni_c", "name": "Muni C", "type": "munitions_low", "cell": [9, 1]},
    ]
    starts = [
        {"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "A"},
        {"slot": 1, "team": 1, "hq_cell": [11, 0], "sector": "C"},
    ]
    m = map_from_ascii(
        terrain_rows=TERRAIN,
        sectors_rows=sectors,
        points=points,
        starts=starts,
        neutral_buildings=(),
        name="three_sector",
    )
    assert len(m.sectors) == 3


def test_sectors_connected_helper_detects_disconnection():
    # A rectangular grid's sector graph can never actually be disconnected
    # (it's a connected planar subdivision by construction), so we exercise
    # the connectivity validator directly against a synthetic graph to prove
    # the "sector graph connected" rule is real, defensive validation.
    from coh.maps.format import _sectors_connected

    connected = {
        0: SectorDef(id=0, point_id="p0", neighbors=(1,)),
        1: SectorDef(id=1, point_id="p1", neighbors=(0,)),
    }
    assert _sectors_connected(connected)

    disconnected = {
        0: SectorDef(id=0, point_id="p0", neighbors=(1,)),
        1: SectorDef(id=1, point_id="p1", neighbors=(0,)),
        2: SectorDef(id=2, point_id="p2", neighbors=()),
    }
    assert not _sectors_connected(disconnected)


def test_start_references_unknown_sector_raises():
    bad_starts = [
        {"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "Z"},
        {"slot": 1, "team": 1, "hq_cell": [11, 0], "sector": "B"},
    ]
    with pytest.raises(MapError):
        make_map(starts=bad_starts)


def test_hq_cell_impassable_raises():
    bad_starts = [
        {"slot": 0, "team": 0, "hq_cell": [4, 5], "sector": "A"},  # wall cell
        {"slot": 1, "team": 1, "hq_cell": [11, 0], "sector": "B"},
    ]
    with pytest.raises(MapError):
        make_map(starts=bad_starts)


def test_hq_cell_out_of_bounds_raises():
    bad_starts = [
        {"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "A"},
        {"slot": 1, "team": 1, "hq_cell": [12, 0], "sector": "B"},  # width is 12, cols 0-11
    ]
    with pytest.raises(MapError):
        make_map(starts=bad_starts)


def test_unknown_terrain_character_raises():
    bad_terrain = list(TERRAIN)
    bad_terrain[0] = "X" + bad_terrain[0][1:]
    with pytest.raises(MapError):
        make_map(terrain_rows=bad_terrain)


def test_point_unreachable_from_hq_raises():
    # Wall off the victory point entirely with impassable 'w' cells.
    terrain = [
        "............",
        "..r.........",
        "..........~.",
        "....c.......",
        "....f.......",
        "....w.......",
        "....H.......",
        "....T.......",
    ]
    # Build a small map where the point sits in a sector separated from HQ
    # by a solid wall ring, so it's unreachable to infantry despite the
    # sector graph itself being adjacent.
    terrain2 = [
        "www.........",
        "w.w.........",
        "www.........",
        "............",
        "............",
        "............",
        "............",
        "............",
    ]
    points = [
        {"id": "victory_a", "name": "Victory A", "type": "victory", "cell": [1, 1]},
        {"id": "strategic_b", "name": "Strategic B", "type": "strategic", "cell": [8, 1]},
    ]
    starts = [
        {"slot": 0, "team": 0, "hq_cell": [6, 6], "sector": "A"},
        {"slot": 1, "team": 1, "hq_cell": [11, 0], "sector": "B"},
    ]
    with pytest.raises(MapError):
        map_from_ascii(
            terrain_rows=terrain2,
            sectors_rows=SECTORS,
            points=points,
            starts=starts,
            neutral_buildings=(),
            name="unreachable",
        )


def test_no_victory_point_raises():
    bad_points = [
        {"id": "strategic_a", "name": "Strategic A", "type": "strategic", "cell": [1, 1]},
        {"id": "strategic_b", "name": "Strategic B", "type": "strategic", "cell": [8, 1]},
    ]
    with pytest.raises(MapError):
        make_map(points=bad_points)


# ---------------------------------------------------------------------------
# Neutral buildings / footprints
# ---------------------------------------------------------------------------


def test_neutral_building_stamps_footprint_impassable():
    neutral = [{"def": "house_small", "cell": [8, 3]}]  # 3x3, well inside sector B open ground
    m = make_map(neutral_buildings=neutral)
    assert len(m.neutral_buildings) == 1
    placement = m.neutral_buildings[0]
    assert isinstance(placement, NeutralPlacement)
    assert placement.def_id == "house_small"
    assert tuple(placement.cell) == (8, 3)
    for dy in range(3):
        for dx in range(3):
            assert not m.pass_inf[3 + dy, 8 + dx]
            assert not m.pass_veh[3 + dy, 8 + dx]
            assert m.los_block[3 + dy, 8 + dx]


def test_default_footprints_used_without_override():
    m = make_map(neutral_buildings=[{"def": "barn", "cell": [1, 2]}])
    # barn is 5x4 by default
    assert not m.pass_inf[2, 1]
    assert not m.pass_inf[5, 5]


# ---------------------------------------------------------------------------
# Mutation API
# ---------------------------------------------------------------------------


def test_set_terrain_cell_updates_layers_and_version():
    m = make_map()
    v0 = m.version
    assert m.pass_veh[4, 4]  # fence, passable
    m.set_terrain_cell((4, 4), ".")  # crush the fence
    assert m.cover_object[4, 4] == 0
    assert m.version == v0 + 1


def test_stamp_footprint_toggle_restores_terrain():
    m = make_map()
    assert m.pass_inf[0, 0]
    m.stamp_footprint((0, 0), (2, 2), blocked=True)
    assert not m.pass_inf[0, 0]
    assert not m.pass_inf[1, 1]
    m.stamp_footprint((0, 0), (2, 2), blocked=False)
    assert m.pass_inf[0, 0]
    assert m.pass_inf[1, 1]


def test_copy_is_independent():
    m = make_map()
    m2 = m.copy()
    m2.set_terrain_cell((0, 0), "w")
    assert m.pass_inf[0, 0]
    assert not m2.pass_inf[0, 0]


# ---------------------------------------------------------------------------
# load_map: dict / path / library
# ---------------------------------------------------------------------------


def test_load_map_from_dict():
    d = {
        "name": "dict_map",
        "cell_m": 2.0,
        "terrain": TERRAIN,
        "sectors": SECTORS,
        "points": POINTS,
        "neutral_buildings": [],
        "starts": STARTS,
    }
    m = load_map(d)
    assert m.name == "dict_map"
    assert m.width == 12 and m.height == 8


def test_load_map_from_path(tmp_path):
    import yaml

    d = {
        "name": "path_map",
        "cell_m": 2.0,
        "terrain": TERRAIN,
        "sectors": SECTORS,
        "points": POINTS,
        "neutral_buildings": [],
        "starts": STARTS,
    }
    p = tmp_path / "path_map.yaml"
    p.write_text(yaml.safe_dump(d))
    m = load_map(p)
    assert m.name == "path_map"


def test_load_map_unknown_library_name_raises():
    with pytest.raises(MapError):
        load_map("does_not_exist")
