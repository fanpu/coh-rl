"""Tests for the vision system: LOS, per-team visible grids, is_visible, ghosts."""

from __future__ import annotations

import time

import numpy as np
import pytest

from coh.maps.format import cell_of, center_of, load_map
from coh.sim.constants import CELL_M
from coh.sim.sim import PlayerSetup
from coh.sim.state import Ghost, SquadState
from coh.sim.systems import vision
from tests.helpers import fixture_data, make_map, make_sim, spawn

WIDTH, HEIGHT = 60, 20
HEDGE_CELL = (33, 10)
SQUAD_CELL = (30, 10)
NEAR_CELL = (32, 10)  # between the squad and the hedge: same side, in range
FAR_CELL = (36, 10)  # past the hedge: blocked


def _hedge_map():
    """A wide-open map with a single hedgerow cell far from both HQs, so HQ /
    builder sight (radius ~15 / ~10 cells) never reaches the test cells."""
    rows = ["." * WIDTH for _ in range(HEIGHT)]
    row = list(rows[HEDGE_CELL[1]])
    row[HEDGE_CELL[0]] = "H"
    rows[HEDGE_CELL[1]] = "".join(row)
    return make_map(
        rows,
        starts=[
            {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
            {"slot": 1, "team": 1, "hq_cell": [WIDTH - 5, HEIGHT - 5], "sector": "a"},
        ],
    )


# ---------------------------------------------------------------------------
# has_los
# ---------------------------------------------------------------------------


def test_has_los_true_with_no_blocker_between():
    m = _hedge_map()
    a = center_of(SQUAD_CELL)
    b = center_of(NEAR_CELL)
    assert vision.has_los(m, a, b)


def test_has_los_false_through_hedgerow():
    m = _hedge_map()
    a = center_of(SQUAD_CELL)
    b = center_of(FAR_CELL)
    assert not vision.has_los(m, a, b)


def test_has_los_excludes_endpoints():
    m = _hedge_map()
    # the hedge cell itself as an endpoint doesn't count as "between".
    a = center_of(SQUAD_CELL)
    b = center_of(HEDGE_CELL)
    assert vision.has_los(m, a, b)


# ---------------------------------------------------------------------------
# integration: run() populates state.visible
# ---------------------------------------------------------------------------


def test_cell_in_front_of_hedgerow_visible_cell_behind_is_not():
    sim = make_sim(game_map=_hedge_map())
    spawn(sim, 0, "rifles", SQUAD_CELL)
    sim.tick()

    grid = sim.state.visible[0]
    nx, ny = NEAR_CELL
    fx, fy = FAR_CELL
    assert grid[ny, nx]
    assert not grid[fy, fx]


def test_hedge_cell_itself_is_visible_as_the_blocker():
    sim = make_sim(game_map=_hedge_map())
    spawn(sim, 0, "rifles", SQUAD_CELL)
    sim.tick()

    hx, hy = HEDGE_CELL
    assert sim.state.visible[0][hy, hx]


def test_teammates_share_vision():
    sim = make_sim(
        game_map=_hedge_map(),
        players=[
            PlayerSetup(faction="us", team=0, start_slot=0),
            PlayerSetup(faction="us", team=0, start_slot=1),
        ],
    )
    cell_a = SQUAD_CELL
    cell_b = (WIDTH - 10, HEIGHT - 10)
    spawn(sim, 0, "rifles", cell_a)
    spawn(sim, 1, "rifles", cell_b)
    sim.tick()

    grid = sim.state.visible[0]
    ax, ay = cell_a
    bx, by = cell_b
    assert grid[ay, ax]
    assert grid[by, bx]


def test_enemy_squad_outside_sight_not_visible():
    sim = make_sim(game_map=_hedge_map())
    enemy = spawn(sim, 1, "rifles", (WIDTH - 3, HEIGHT - 3))
    spawn(sim, 0, "rifles", SQUAD_CELL)
    sim.tick()

    assert not vision.is_visible(sim, 0, enemy)


def test_own_squad_always_visible_to_own_team():
    sim = make_sim(game_map=_hedge_map())
    own = spawn(sim, 0, "rifles", (WIDTH - 3, HEIGHT - 3))  # far from any friendly vision
    sim.tick()

    assert vision.is_visible(sim, 0, own)


def test_abandoned_squad_gives_no_vision():
    sim = make_sim(game_map=_hedge_map())
    squad = spawn(sim, 0, "rifles", SQUAD_CELL)
    squad.abandoned = True
    sim.tick()

    fx, fy = FAR_CELL  # would be in range if the squad still gave vision
    nx, ny = NEAR_CELL
    assert not sim.state.visible[0][ny, nx]
    assert not sim.state.visible[0][fy, fx]


BUILDING_CELL = (28, 14)  # far from both HQs / their builder squads' sight


def test_building_under_construction_gives_no_vision():
    sim = make_sim(game_map=_hedge_map())
    building = sim.spawn_building(0, "barracks", BUILDING_CELL, complete=False)
    assert building.progress < 1.0
    sim.tick()

    cx, cy = building.cell
    # the barracks' own footprint is impassable/LOS-blocking, not "seen".
    assert not sim.state.visible[0][cy, cx]


def test_garrisoned_squad_sees_from_building_centre():
    sim = make_sim(game_map=_hedge_map())
    building = sim.spawn_building(0, "barracks", BUILDING_CELL)
    squad = spawn(sim, 0, "rifles", BUILDING_CELL)
    squad.garrison_in = building.id
    squad.state = SquadState.GARRISONED
    sim.tick()

    # sight from the barracks' centre should reach well past its own footprint.
    cx, cy = building.cell
    far_x, far_y = cx + 8, cy
    assert sim.state.visible[0][far_y, far_x]


# ---------------------------------------------------------------------------
# ghosts
# ---------------------------------------------------------------------------


def test_ghost_persists_after_losing_sight_and_disappears_after_rescout():
    sim = make_sim()  # default 40x30 map, opposite-corner HQs
    enemy_hq = sim.state.buildings[sim.state.players[1].hq_id]
    ex, ey = enemy_hq.cell
    scout_cell = (ex - 4, ey)  # within rifles' 25m sight, open terrain
    scout = spawn(sim, 0, "rifles", scout_cell)

    sim.tick()
    assert enemy_hq.id in sim.state.ghosts[0]
    ghost = sim.state.ghosts[0][enemy_hq.id]
    assert ghost.hp_frac == 1.0
    assert ghost.owner == 1
    assert ghost.cell == enemy_hq.cell

    # move the scout far away: vision is lost, but the ghost persists.
    scout.pos = center_of((1, 1))
    sim.tick()
    assert enemy_hq.id in sim.state.ghosts[0]
    assert not vision.is_visible(sim, 0, enemy_hq)

    # the building is destroyed while out of sight: ghost still persists.
    del sim.state.buildings[enemy_hq.id]
    sim.tick()
    assert enemy_hq.id in sim.state.ghosts[0]

    # re-scouting the (now-empty) cell clears the ghost.
    scout.pos = center_of(scout_cell)
    sim.tick()
    assert enemy_hq.id not in sim.state.ghosts[0]


def test_ghost_not_created_for_neutral_building():
    sim = make_sim(neutral_buildings=[{"def": "house", "cell": (18, 14)}])
    spawn(sim, 0, "rifles", (18, 14))
    sim.tick()

    assert sim.state.ghosts[0] == {}


# ---------------------------------------------------------------------------
# performance: cache-hit path
# ---------------------------------------------------------------------------


def test_vision_tick_is_fast_with_warm_cache_on_hedgerow_crossing():
    data = fixture_data()
    # `hedgerow_crossing`'s neutral buildings use defs the fixture data doesn't
    # define (it's a minimal test fixture, not the real content tables); the
    # footprints still get stamped impassable/LOS-blocking by `load_map`, but
    # the placements themselves aren't needed for this perf test, so drop
    # them rather than spawn `Building` entities the fixture data can't back.
    game_map = load_map("hedgerow_crossing")
    game_map.neutral_buildings = []
    sim = make_sim(game_map=game_map, data=data)

    passable = np.argwhere(game_map.pass_inf)
    rng = np.random.default_rng(0)
    chosen = passable[rng.choice(len(passable), size=40, replace=False)]
    for i, (cy, cx) in enumerate(chosen):
        spawn(sim, i % 2, "rifles", (int(cx), int(cy)))

    sim.tick()  # warm-up: populates the mask cache

    n = 20
    start = time.perf_counter()
    for _ in range(n):
        sim.tick()
    elapsed_per_tick_ms = (time.perf_counter() - start) / n * 1000
    assert elapsed_per_tick_ms < 20.0, f"vision tick took {elapsed_per_tick_ms:.2f}ms (warm cache)"
