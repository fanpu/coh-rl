"""Tests for coh.maps.cover: directional and area cover lookup."""

from __future__ import annotations

import numpy as np

from coh.maps.cover import cover_at
from coh.maps.format import center_of, map_from_ascii

# 12x8 map: wall at (4,5); fence at (4,4); open unit cell east of both at
# (5,5); crater at (4,3); road at (2,1).
TERRAIN = [
    "............",
    "..r.........",
    "............",
    "....c.......",
    "....f.......",
    "....w.......",
    "............",
    "............",
]
SECTORS = ["A" * 12 for _ in range(8)]
POINTS = [{"id": "v", "name": "V", "type": "victory", "cell": [1, 1]}]
STARTS = [{"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "A"}]


def make_map():
    return map_from_ascii(
        terrain_rows=TERRAIN,
        sectors_rows=SECTORS,
        points=POINTS,
        starts=STARTS,
        neutral_buildings=(),
        name="cover_test",
    )


def test_wall_cover_from_the_side_it_faces():
    m = make_map()
    unit_cell = (5, 5)  # open ground, directly east of the wall at (4,5)
    shooter_west = center_of((0, 5))
    assert cover_at(m, unit_cell, shooter_west) == "heavy"


def test_wall_no_cover_from_opposite_side():
    m = make_map()
    unit_cell = (5, 5)
    shooter_east = center_of((11, 5))
    assert cover_at(m, unit_cell, shooter_east) == "open"


def test_heavy_beats_light_when_both_in_cone():
    # unit_cell (5,5) has a heavy wall due west at (4,5) (0 degrees off the
    # shooter direction) and a light fence northwest at (4,4) (45 degrees
    # off) -- both qualify (<=60 degrees), heavy must win.
    m = make_map()
    unit_cell = (5, 5)
    shooter_west = center_of((0, 5))
    assert cover_at(m, unit_cell, shooter_west) == "heavy"


def test_crater_cover_any_direction():
    m = make_map()
    crater_cell = (4, 3)
    for shooter_cell in [(0, 3), (11, 3), (4, 0), (4, 7), (0, 0), (11, 7)]:
        shooter = center_of(shooter_cell)
        assert cover_at(m, crater_cell, shooter) == "heavy"


def test_road_cover_negative():
    m = make_map()
    road_cell = (2, 1)
    for shooter_cell in [(0, 1), (11, 1), (2, 0), (2, 7)]:
        shooter = center_of(shooter_cell)
        assert cover_at(m, road_cell, shooter) == "negative"


def test_open_ground_default():
    m = make_map()
    open_cell = (1, 1)
    shooter = center_of((10, 6))
    assert cover_at(m, open_cell, shooter) == "open"
