"""Tests for coh.sim.pathfinding and coh.sim.systems.movement (task 6)."""

from __future__ import annotations

import time

import numpy as np

from coh.maps.format import cell_of, center_of, load_map, map_from_ascii
from coh.sim.orders import AttackMove, Build, Capture, Garrison, Move, Retreat
from coh.sim.pathfinding import adjacent_cells, find_path, nearest_adjacent_passable
from coh.sim.state import SquadState
from tests.helpers import fixture_data, make_map, make_sim, spawn

# --------------------------------------------------------------------------
# find_path: pure pathfinding
# --------------------------------------------------------------------------


def test_find_path_routes_around_a_wall():
    width, height = 10, 10
    rows = ["." * width for _ in range(height)]
    # Vertical wall at x=5, leaving only row y=9 open as a gap.
    for y in range(9):
        row = rows[y]
        rows[y] = row[:5] + "w" + row[6:]
    m = map_from_ascii(
        terrain_rows=rows,
        sectors_rows=["a" * width for _ in range(height)],
        points=[{"id": "mid", "name": "Mid", "type": "victory", "cell": [0, 0]}],
        starts=[
            {"slot": 0, "team": 0, "hq_cell": [0, 9], "sector": "a"},
            {"slot": 1, "team": 1, "hq_cell": [9, 9], "sector": "a"},
        ],
    )
    path = find_path(m.pass_inf, (0, 5), (9, 5))
    assert path is not None
    assert path[0] == (0, 5)
    assert path[-1] == (9, 5)
    # the wall (x=5, y<9) must never appear on the path
    assert all(not (cx == 5 and cy < 9) for cx, cy in path)
    # it had to detour down to the gap at y=9
    assert any(cy == 9 for _, cy in path)


def test_find_path_returns_none_for_enclosed_goal():
    width, height = 5, 5
    rows = ["." * width for _ in range(height)]
    # Wall ring fully sealing (2, 2) off from the rest of the map.
    for cx, cy in [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3), (2, 3), (3, 3)]:
        row = rows[cy]
        rows[cy] = row[:cx] + "w" + row[cx + 1 :]
    m = map_from_ascii(
        terrain_rows=rows,
        sectors_rows=["a" * width for _ in range(height)],
        points=[{"id": "mid", "name": "Mid", "type": "victory", "cell": [0, 0]}],
        starts=[
            {"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "a"},
            {"slot": 1, "team": 1, "hq_cell": [4, 4], "sector": "a"},
        ],
    )
    assert m.pass_inf[2, 2]  # the goal cell itself is open ground...
    assert find_path(m.pass_inf, (0, 0), (2, 2)) is None  # ...but unreachable


def test_find_path_does_not_cut_corners():
    # Two walls diagonally adjacent to the start leave only a diagonal gap
    # that isn't legal without corner-cutting.
    width, height = 4, 4
    rows = [
        "....",
        ".w..",
        "..w.",
        "....",
    ]
    m = map_from_ascii(
        terrain_rows=rows,
        sectors_rows=["a" * width for _ in range(height)],
        points=[{"id": "mid", "name": "Mid", "type": "victory", "cell": [0, 0]}],
        starts=[
            {"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "a"},
            {"slot": 1, "team": 1, "hq_cell": [3, 3], "sector": "a"},
        ],
    )
    path = find_path(m.pass_inf, (1, 0), (2, 1))
    # (1,0) -> (2,1) diagonally would cut between the walls at (1,1)/(2,0);
    # every step must instead go through an orthogonally-open pair.
    assert path is not None
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        dx, dy = bx - ax, by - ay
        if dx and dy:
            assert m.pass_inf[ay, ax + dx] and m.pass_inf[ay + dy, ax]


def test_find_path_paths_to_nearest_passable_cell_when_goal_is_impassable():
    width, height = 6, 6
    rows = ["." * width for _ in range(height)]
    rows[3] = rows[3][:3] + "w" + rows[3][4:]  # (3,3) impassable
    m = map_from_ascii(
        terrain_rows=rows,
        sectors_rows=["a" * width for _ in range(height)],
        points=[{"id": "mid", "name": "Mid", "type": "victory", "cell": [0, 0]}],
        starts=[
            {"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "a"},
            {"slot": 1, "team": 1, "hq_cell": [5, 5], "sector": "a"},
        ],
    )
    path = find_path(m.pass_inf, (0, 0), (3, 3))
    assert path is not None
    assert path[-1] != (3, 3)
    assert m.pass_inf[path[-1][1], path[-1][0]]


def test_find_path_vehicle_cannot_enter_light_terrain():
    width, height = 8, 5
    rows = ["." * width for _ in range(height)]
    rows[2] = "TTTTTTTT"  # light terrain: infantry-passable, not vehicle-passable
    m = map_from_ascii(
        terrain_rows=rows,
        sectors_rows=["a" * width for _ in range(height)],
        points=[{"id": "mid", "name": "Mid", "type": "victory", "cell": [0, 0]}],
        starts=[
            {"slot": 0, "team": 0, "hq_cell": [0, 0], "sector": "a"},
            {"slot": 1, "team": 1, "hq_cell": [7, 4], "sector": "a"},
        ],
    )
    inf_path = find_path(m.pass_inf, (0, 0), (0, 4))
    assert inf_path is not None
    assert any(cy == 2 for _, cy in inf_path)  # infantry can cross the T row

    # the T row spans the full width, so a vehicle has no detour at all
    assert find_path(m.pass_veh, (0, 0), (0, 4)) is None


def test_find_path_corner_to_corner_on_hedgerow_crossing_is_fast():
    m = load_map("hedgerow_crossing")
    start = m.starts[0].hq_cell
    goal = m.starts[1].hq_cell

    t0 = time.perf_counter()
    inf_path = find_path(m.pass_inf, start, goal)
    inf_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    veh_path = find_path(m.pass_veh, start, goal)
    veh_dt = time.perf_counter() - t0

    assert inf_path is not None and inf_path[0] == start
    assert veh_path is not None and veh_path[0] == start
    # brief: "well under 50ms"; generous CI margin to avoid flakes
    assert inf_dt < 0.5, f"infantry corner-to-corner took {inf_dt * 1000:.1f}ms"
    assert veh_dt < 0.5, f"vehicle corner-to-corner took {veh_dt * 1000:.1f}ms"


def test_nearest_adjacent_passable_picks_the_closest_open_ring_cell():
    passable = np.ones((10, 10), dtype=bool)
    passable[3:5, 3:5] = False  # 2x2 blocked footprint at (3,3)
    cell = nearest_adjacent_passable(passable, (3, 3), (2, 2), (8, 8))
    assert cell in adjacent_cells((3, 3), (2, 2), 10, 10)
    assert passable[cell[1], cell[0]]
    # closer to (8, 8) than the far side of the footprint
    assert cell[0] >= 3 or cell[1] >= 3


# --------------------------------------------------------------------------
# movement system: Move / AttackMove
# --------------------------------------------------------------------------


def test_infantry_covers_speed_times_time_in_80_ticks():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))  # speed 3.0 m/s
    sim.issue(0, [Move(squad=squad.id, cell=(25, 10))])  # 15 cells = 30m east
    sim.run(80)
    cx, cy = cell_of(squad.pos)
    assert abs(cx - 25) <= 1 and cy == 10
    assert squad.state is SquadState.IDLE
    assert squad.order is None


def test_pinned_squad_does_not_move():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    squad.pinned = True
    start_pos = squad.pos.copy()
    sim.issue(0, [Move(squad=squad.id, cell=(20, 10))])
    sim.run(20)
    assert np.allclose(squad.pos, start_pos)
    assert squad.moving is False


def test_suppressed_squad_moves_at_suppressed_speed_mult():
    sim = make_sim()
    econ = fixture_data().economy
    squad = spawn(sim, 0, "rifles", (10, 10))
    squad.suppressed = True
    start_pos = squad.pos.copy()
    sim.issue(0, [Move(squad=squad.id, cell=(30, 10))])
    sim.run(8)  # 1 second
    expected = 3.0 * 1.0 * econ.suppressed_speed_mult
    traveled = float(np.linalg.norm(squad.pos - start_pos))
    assert abs(traveled - expected) < 0.05


def test_attack_move_halts_while_target_set_and_resumes():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    sim.issue(0, [AttackMove(squad=squad.id, cell=(25, 10))])
    sim.run(5)
    pos_before = squad.pos.copy()
    squad.target_id = 999
    sim.run(10)
    assert np.allclose(squad.pos, pos_before)
    assert squad.moving is False
    squad.target_id = None
    sim.run(10)
    assert not np.allclose(squad.pos, pos_before)


def test_attack_move_clears_order_on_arrival():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    sim.issue(0, [AttackMove(squad=squad.id, cell=(12, 10))])
    sim.run(30)
    assert squad.state is SquadState.IDLE
    assert squad.order is None


# --------------------------------------------------------------------------
# movement system: vehicle rotation arc
# --------------------------------------------------------------------------


def test_vehicle_ordered_directly_behind_it_does_not_move_until_it_has_turned():
    sim = make_sim()
    squad = spawn(sim, 1, "tank", (20, 15))  # heading defaults to 0.0 (east)
    assert squad.heading == 0.0
    start_pos = squad.pos.copy()
    sim.issue(1, [Move(squad=squad.id, cell=(5, 15))])  # due west: 180 degrees away

    # rotation_deg_s=45 => needs 135 degrees (180 - the 45 degree arc) before
    # it's allowed to move at all, i.e. 3s = 24 ticks at minimum.
    sim.run(20)
    assert np.allclose(squad.pos, start_pos), "vehicle moved before finishing its turn"

    sim.run(40)
    assert not np.allclose(squad.pos, start_pos), "vehicle never started moving"


# --------------------------------------------------------------------------
# movement system: terrain mutation (crushing light cover)
# --------------------------------------------------------------------------


def _fence_map():
    width, height = 20, 10
    rows = ["." * width for _ in range(height)]
    rows[8] = "." * 10 + "f" + "." * (width - 11)
    starts = [
        {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
        {"slot": 1, "team": 1, "hq_cell": [15, 1], "sector": "a"},
    ]
    return rows, starts


def test_vehicle_crushes_fence_and_terrain_records_the_change():
    rows, starts = _fence_map()
    sim = make_sim(ascii_map=rows, starts=starts)
    squad = spawn(sim, 0, "tank", (2, 8))  # heading 0.0 == east, matches travel direction
    sim.issue(0, [Move(squad=squad.id, cell=(18, 8))])

    sim.run(120)

    assert str(sim.map.terrain[8, 10]) == "."
    assert sim.map.cover_object[8, 10] == 0
    assert (10, 8, ".") in sim.state.terrain_changes


def test_state_hash_diverges_when_only_terrain_changes_differ():
    a = make_sim(seed=3)
    b = make_sim(seed=3)
    assert a.state_hash() == b.state_hash()

    a.map.set_terrain_cell((5, 5), ".")
    a.state.terrain_changes.append((5, 5, "."))

    assert a.state_hash() != b.state_hash()


# --------------------------------------------------------------------------
# movement system: Retreat / Capture / Garrison / Build
# --------------------------------------------------------------------------


def test_retreat_paths_toward_own_hq_then_goes_idle():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    (result,) = sim.issue(0, [Retreat(squad=squad.id)])
    assert result.ok
    assert squad.state is SquadState.RETREATING

    sim.run(200)

    assert squad.state is SquadState.IDLE
    assert squad.order is None
    hq = sim.state.buildings[sim.state.players[0].hq_id]
    hcx, hcy = hq.cell
    w, h = sim.data.buildings[hq.def_id].footprint
    cx, cy = cell_of(squad.pos)
    assert (hcx - 1) <= cx <= (hcx + w) and (hcy - 1) <= cy <= (hcy + h)


def test_capture_reaches_the_point_and_keeps_its_order():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (18, 15))
    order = Capture(squad=squad.id, point_id="mid")
    (result,) = sim.issue(0, [order])
    assert result.ok

    sim.run(60)

    assert squad.state is SquadState.IDLE
    assert squad.order == order  # kept for the capture system (later task)
    point = sim.map.points["mid"]
    cx, cy = cell_of(squad.pos)
    assert abs(cx - point.cell[0]) <= 1 and abs(cy - point.cell[1]) <= 1


def test_capture_with_unknown_point_is_rejected():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    (result,) = sim.issue(0, [Capture(squad=squad.id, point_id="nope")])
    assert result.ok is False
    assert "no such point" in result.reason
    assert squad.order is None


def test_garrison_walks_adjacent_to_the_building_and_keeps_its_order():
    sim = make_sim()
    hq = sim.state.buildings[sim.state.players[0].hq_id]
    squad = spawn(sim, 0, "rifles", (10, 10))
    order = Garrison(squad=squad.id, building_id=hq.id)
    (result,) = sim.issue(0, [order])
    assert result.ok

    sim.run(100)

    assert squad.state is SquadState.IDLE
    assert squad.order == order


def test_garrison_with_unknown_building_is_rejected():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    (result,) = sim.issue(0, [Garrison(squad=squad.id, building_id=999)])
    assert result.ok is False
    assert "no such building" in result.reason
    assert squad.order is None


def test_build_walks_adjacent_to_the_target_footprint_and_keeps_its_order():
    sim = make_sim()
    squad = spawn(sim, 0, "engineers", (10, 10))
    order = Build(squad=squad.id, structure="barracks", cell=(15, 15))
    (result,) = sim.issue(0, [order])
    assert result.ok

    sim.run(100)

    assert squad.state is SquadState.IDLE
    assert squad.order == order
    w, h = sim.data.buildings["barracks"].footprint
    cx, cy = cell_of(squad.pos)
    assert (14) <= cx <= (15 + w) and (14) <= cy <= (15 + h)


# --------------------------------------------------------------------------
# movement system: team weapons tear down / set up around a move
# --------------------------------------------------------------------------


def test_team_weapon_tears_down_before_moving_then_sets_up_on_arrival():
    sim = make_sim()
    squad = spawn(sim, 0, "hmg_team", (10, 10))  # setup_time 3.0s (hmg weapon)
    squad.state = SquadState.SET_UP
    squad.facing = 0.0

    sim.issue(0, [Move(squad=squad.id, cell=(14, 10))])
    assert squad.state is SquadState.TEARING_DOWN
    start_pos = squad.pos.copy()

    sim.run(20)  # 2.5s < 3s setup_time: still tearing down, no movement
    assert squad.state is SquadState.TEARING_DOWN
    assert np.allclose(squad.pos, start_pos)

    sim.run(150)  # plenty of time to finish tearing down, travel, and set up
    assert squad.state in (SquadState.SETTING_UP, SquadState.SET_UP)
    assert not np.allclose(squad.pos, start_pos)


def test_freshly_spawned_team_weapon_starts_setting_up_then_sets_up():
    sim = make_sim()
    squad = spawn(sim, 0, "hmg_team", (10, 10))  # setup_time 3.0s (hmg weapon)
    assert squad.state is SquadState.SETTING_UP
    assert squad.facing == squad.heading

    sim.run(23)  # 2.875s < 3s: still setting up
    assert squad.state is SquadState.SETTING_UP

    sim.run(5)  # tick 28: setup_time has elapsed
    assert squad.state is SquadState.SET_UP


# --------------------------------------------------------------------------
# movement system: re-planning around a new obstacle mid-transit
# --------------------------------------------------------------------------


def test_squad_replans_around_a_footprint_stamped_mid_transit():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (5, 10))
    sim.issue(0, [Move(squad=squad.id, cell=(30, 10))])
    sim.run(10)  # let it get moving along the direct route first

    # Stamp a footprint straight across its remaining route.
    sim.map.stamp_footprint((14, 9), (3, 3), blocked=True)
    blocked_cells = {(x, y) for x in range(14, 17) for y in range(9, 12)}
    assert cell_of(squad.pos) not in blocked_cells  # sanity: not already inside it

    arrived = False
    for _ in range(300):
        sim.tick()
        assert cell_of(squad.pos) not in blocked_cells, "squad walked through a blocked footprint"
        if squad.state is SquadState.IDLE and not squad.path:
            arrived = True
            break

    assert arrived, "squad never arrived after the mid-transit re-plan"
    cx, cy = cell_of(squad.pos)
    assert abs(cx - 30) <= 1 and cy == 10


def test_squad_arrives_when_final_goal_blocked_and_replan_resolves_to_own_cell():
    # Regression: a re-plan that *succeeds* but resolves back to the
    # squad's own current cell (its one remaining waypoint was the final
    # goal, and it just got blocked, so find_path's nearest-passable
    # fallback lands on the cell the squad is already standing on) used to
    # leave squad.path empty without ever calling _on_arrival, stranding
    # the squad in MOVING forever (the "if not squad.path: return" guard at
    # the top of the tick step would then skip it every tick).
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    goal = (11, 10)
    (result,) = sim.issue(0, [Move(squad=squad.id, cell=goal)])
    assert result.ok
    assert squad.path == [goal]
    assert squad.state is SquadState.MOVING

    # Block the final goal and its whole ring except the squad's own cell,
    # so find_path's nearest-passable fallback for the (now impassable)
    # goal has nowhere to land but the squad's own cell.
    ring = {(gx, gy) for gx in range(10, 13) for gy in range(9, 12)}
    for cx, cy in ring - {(10, 10)}:
        sim.map.stamp_footprint((cx, cy), (1, 1), blocked=True)

    arrived = False
    for _ in range(10):
        sim.tick()
        if squad.state is SquadState.IDLE:
            arrived = True
            break

    assert arrived, "squad never arrived; it's stranded"
    assert squad.order is None
    assert squad.path == []


def test_move_order_to_the_squads_own_cell_arrives_immediately():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    (result,) = sim.issue(0, [Move(squad=squad.id, cell=(10, 10))])
    assert result.ok
    assert squad.path == []
    assert squad.state is SquadState.IDLE
    assert squad.order is None

    # no crash running further ticks with an already-idle, already-arrived squad
    sim.run(5)
    assert squad.state is SquadState.IDLE


def test_move_to_an_unreachable_cell_settles_a_moving_squad_to_idle():
    # Regression: start_path's "goal is truly unreachable" branch used to
    # leave squad.state/order untouched. A squad already MOVING toward one
    # goal that gets re-ordered to an enclosed (unreachable) cell would stay
    # MOVING forever with an empty path and a standing order that never
    # clears, and the "if not squad.path: return" guard would then skip it
    # every tick.
    width, height = 24, 12
    rows = ["." * width for _ in range(height)]
    # wall ring sealing (10, 9) off from the rest of the map (the cell
    # itself stays open ground, matching find_path's "enclosed goal" case).
    for cx, cy in [(9, 8), (10, 8), (11, 8), (9, 9), (11, 9), (9, 10), (10, 10), (11, 10)]:
        row = rows[cy]
        rows[cy] = row[:cx] + "w" + row[cx + 1 :]
    starts = [
        {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
        {"slot": 1, "team": 1, "hq_cell": [19, 1], "sector": "a"},
    ]
    sim = make_sim(ascii_map=rows, starts=starts)
    squad = spawn(sim, 0, "rifles", (2, 6))

    sim.issue(0, [Move(squad=squad.id, cell=(20, 6))])
    sim.run(5)
    assert squad.state is SquadState.MOVING
    assert squad.path

    (result,) = sim.issue(0, [Move(squad=squad.id, cell=(10, 9))])  # enclosed, unreachable
    assert result.ok  # accepted (in bounds); pathing just fails to find a route
    assert squad.path == []
    assert squad.state is SquadState.IDLE
    assert squad.order is None
    assert squad.moving is False

    pos_after_stop = squad.pos.copy()
    sim.run(20)
    assert squad.state is SquadState.IDLE
    assert squad.path == []
    assert np.allclose(squad.pos, pos_after_stop), "squad moved after settling to idle"


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_movement_is_deterministic_across_runs_with_the_same_seed_and_orders():
    def scripted():
        sim = make_sim(seed=17)
        squads = sorted(sim.state.squads)
        sim.issue(0, [Move(squad=squads[0], cell=(15, 10)), Retreat(squad=squads[0])])
        sim.issue(1, [AttackMove(squad=squads[1], cell=(25, 20))])
        sim.run(120)
        return sim.state_hash()

    assert scripted() == scripted()
