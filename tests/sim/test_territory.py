"""Tests for coh.sim.systems.territory (task 13): capture, neutralize,
connectivity and observation-post bookkeeping."""

from __future__ import annotations

from coh.sim.orders import Capture, Stop
from coh.sim.sim import PlayerSetup
from coh.sim.state import PointState, SquadState
from coh.sim.systems import territory
from tests.helpers import fixture_data, make_map, make_sim, spawn

# capture_time_s=8, neutralize_time_s=4, capture_radius=10 in the fixture
# economy; at TICKS_PER_SECOND=8 and rate=1.0 that's 64 / 32 ticks.
CAPTURE_TICKS = 64
NEUTRALIZE_TICKS = 32


# ---------------------------------------------------------------------------
# Basic capture / neutralize / stall / decay (default 3-sector map: 'west' is
# team 0's HQ point, 'east' is team 1's, 'mid' is the only contestable one)
# ---------------------------------------------------------------------------


def test_capture_takes_capture_time_over_rate():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (20, 15))  # 'mid' point cell
    result = sim.issue(0, [Capture(squad.id, "mid")])[0]
    assert result.ok

    sim.run(CAPTURE_TICKS - 1)
    point = sim.state.points["mid"]
    assert point.owner_team is None
    assert 0.0 < point.progress < 1.0

    sim.tick()
    point = sim.state.points["mid"]
    assert point.owner_team == 0
    assert point.progress == 1.0
    assert point.capturing_team is None
    # the squad's Capture order completes and is cleared
    assert sim.state.squads[squad.id].order is None


def test_contested_point_stalls(no_combat):
    sim = make_sim()
    s0 = spawn(sim, 0, "rifles", (20, 15))
    s1 = spawn(sim, 1, "rifles", (20, 15))
    sim.issue(0, [Capture(s0.id, "mid")])
    sim.issue(1, [Capture(s1.id, "mid")])

    sim.run(CAPTURE_TICKS)  # would be enough to finish if uncontested

    point = sim.state.points["mid"]
    assert point.owner_team is None
    assert point.progress == 0.0
    assert point.capturing_team is None


def test_enemy_point_needs_neutralize_then_capture():
    sim = make_sim()
    # Team 0 captures 'mid' outright first.
    sim.state.points["mid"] = PointState(owner_team=0, progress=1.0)

    squad = spawn(sim, 1, "rifles", (20, 15))
    sim.issue(1, [Capture(squad.id, "mid")])

    sim.run(NEUTRALIZE_TICKS - 1)
    point = sim.state.points["mid"]
    assert point.owner_team == 0
    assert point.progress > 0.0

    sim.tick()
    point = sim.state.points["mid"]
    assert point.owner_team is None
    assert point.progress == 0.0
    assert any(e.kind == "point_neutralized" and e.data["point_id"] == "mid" for e in sim.state.events)

    # Now the same team 1 squad continues on to capture it outright.
    sim.run(CAPTURE_TICKS - 1)
    assert sim.state.points["mid"].owner_team is None

    sim.tick()
    point = sim.state.points["mid"]
    assert point.owner_team == 1
    assert point.progress == 1.0
    assert any(e.kind == "point_captured" and e.data["point_id"] == "mid" and e.data["team"] == 1 for e in sim.state.events)


def test_vehicle_cannot_capture():
    sim = make_sim()
    squad = spawn(sim, 1, "tank", (20, 15))
    result = sim.issue(1, [Capture(squad.id, "mid")])[0]
    assert not result.ok
    assert sim.state.squads[squad.id].order is None


def test_progress_decays_to_resting_value_without_capturers():
    sim = make_sim()
    sim.state.points["mid"] = PointState(owner_team=0, progress=1.0)
    squad = spawn(sim, 1, "rifles", (20, 15))
    sim.issue(1, [Capture(squad.id, "mid")])
    sim.run(10)
    progress_after_push = sim.state.points["mid"].progress
    assert progress_after_push < 1.0

    # Stop the attacker: progress should decay back up toward 1.0 (owned).
    sim.issue(1, [Stop(squad.id)])
    sim.run(20)
    point = sim.state.points["mid"]
    assert point.owner_team == 0
    assert point.progress > progress_after_push
    assert point.capturing_team is None


# ---------------------------------------------------------------------------
# Capturer eligibility
# ---------------------------------------------------------------------------


def test_pinned_squad_does_not_capture(monkeypatch):
    # The suppression system (task 9) would otherwise decay a bare `pinned`
    # flag with no backing `squad.suppression` value straight back off well
    # before `CAPTURE_TICKS` elapses; this test only cares about capture
    # eligibility reading the flag, so it is stubbed out (as `no_combat`
    # does for combat elsewhere).
    from coh.sim.systems import suppression

    monkeypatch.setattr(suppression, "run", lambda sim: None)

    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (20, 15))
    squad.pinned = True
    sim.issue(0, [Capture(squad.id, "mid")])
    sim.run(CAPTURE_TICKS)
    assert sim.state.points["mid"].owner_team is None
    assert sim.state.points["mid"].progress == 0.0


def test_retreating_squad_does_not_capture():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (20, 15))
    sim.issue(0, [Capture(squad.id, "mid")])
    squad.state = SquadState.RETREATING
    sim.run(CAPTURE_TICKS)
    assert sim.state.points["mid"].owner_team is None


def test_garrisoned_squad_does_not_capture():
    from coh.sim.systems import garrison

    # A house 6 m from 'mid': well inside the 10 m capture radius, so being
    # garrisoned is the only thing that can stop the squad capturing.
    sim = make_sim(neutral_buildings=[{"def": "house", "cell": [22, 14]}])
    house = next(b for b in sim.state.buildings.values() if b.neutral)
    squad = spawn(sim, 0, "rifles", (20, 15))
    sim.issue(0, [Capture(squad.id, "mid")])
    garrison.enter(sim, squad, house)
    sim.run(CAPTURE_TICKS)
    assert sim.state.points["mid"].owner_team is None


def test_squad_out_of_capture_radius_does_not_capture():
    sim = make_sim()
    # 'mid' is at cell (20, 15); capture_radius is 10m = 5 cells.
    squad = spawn(sim, 0, "rifles", (20 + 20, 15))
    sim.issue(0, [Capture(squad.id, "mid")])
    sim.run(CAPTURE_TICKS)
    assert sim.state.points["mid"].owner_team is None


# ---------------------------------------------------------------------------
# HQ sector points
# ---------------------------------------------------------------------------


def test_hq_sector_point_uncapturable():
    sim = make_sim()
    assert territory.is_hq_point(sim, "west")
    assert territory.is_hq_point(sim, "east")
    assert sim.state.points["west"].owner_team == 0

    squad = spawn(sim, 1, "rifles", (6, 14))  # 'west' point cell
    result = sim.issue(1, [Capture(squad.id, "west")])[0]
    assert not result.ok

    sim.run(CAPTURE_TICKS * 2)
    assert sim.state.points["west"].owner_team == 0
    assert sim.state.points["west"].progress == 1.0


def test_already_owned_full_progress_capture_rejected():
    sim = make_sim()
    sim.state.points["mid"] = PointState(owner_team=0, progress=1.0)
    squad = spawn(sim, 0, "rifles", (20, 15))
    result = sim.issue(0, [Capture(squad.id, "mid")])[0]
    assert not result.ok


# ---------------------------------------------------------------------------
# Observation posts
# ---------------------------------------------------------------------------


def test_op_prevents_neutralize_until_destroyed(no_combat):
    sim = make_sim()
    sim.state.points["mid"] = PointState(owner_team=0, progress=1.0)
    building = sim.spawn_building(0, "op_us", (20, 15))
    sim.state.points["mid"].op_building = building.id

    squad = spawn(sim, 1, "rifles", (20, 15))
    sim.issue(1, [Capture(squad.id, "mid")])

    sim.run(NEUTRALIZE_TICKS * 3)
    point = sim.state.points["mid"]
    assert point.owner_team == 0
    assert point.progress == 0.0

    del sim.state.buildings[building.id]
    sim.tick()
    point = sim.state.points["mid"]
    assert point.op_building is None
    assert point.owner_team is None


def test_op_building_reference_cleared_when_building_dies():
    sim = make_sim()
    building = sim.spawn_building(0, "op_us", (20, 15))
    sim.state.points["mid"].op_building = building.id
    del sim.state.buildings[building.id]
    sim.tick()
    assert sim.state.points["mid"].op_building is None


# ---------------------------------------------------------------------------
# Supply connectivity (needs a 4-sector chain so cutting the middle sector
# strands the far one)
# ---------------------------------------------------------------------------


def _chain_map():
    width, height = 24, 6
    terrain = ["." * width for _ in range(height)]
    sectors = ["a" * 6 + "b" * 6 + "c" * 6 + "d" * 6] * height
    points = [
        {"id": "hq0", "name": "HQ0", "type": "strategic", "cell": [2, 2]},
        {"id": "midb", "name": "Mid B", "type": "victory", "cell": [8, 2]},
        {"id": "midc", "name": "Mid C", "type": "strategic", "cell": [14, 2]},
        {"id": "hq1", "name": "HQ1", "type": "strategic", "cell": [20, 2]},
    ]
    starts = [
        {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
        {"slot": 1, "team": 1, "hq_cell": [19, 1], "sector": "d"},
    ]
    return make_map(terrain, sectors=sectors, points=points, starts=starts, data=fixture_data())


def _chain_sim():
    return make_sim(
        game_map=_chain_map(),
        players=[PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)],
    )


def test_connectivity_recomputes_over_owned_chain():
    sim = _chain_sim()
    a, b, c, d = 0, 1, 2, 3  # sorted(char_to_id) assigns ids in 'a'..'d' order
    assert sim.state.connected[0] == [a]
    assert sim.state.connected[1] == [d]

    sim.state.points["midb"] = PointState(owner_team=0, progress=1.0)
    territory.recompute_connectivity(sim)
    assert sim.state.connected[0] == [a, b]

    sim.state.points["midc"] = PointState(owner_team=0, progress=1.0)
    territory.recompute_connectivity(sim)
    assert sim.state.connected[0] == [a, b, c]


def test_cutting_middle_sector_strands_far_sector_and_restores(no_combat):
    sim = _chain_sim()
    a, b, c, _d = 0, 1, 2, 3

    squad_b = spawn(sim, 0, "rifles", (8, 2))  # 'midb' cell
    sim.issue(0, [Capture(squad_b.id, "midb")])
    sim.run(CAPTURE_TICKS)
    assert sim.state.points["midb"].owner_team == 0

    squad_c = spawn(sim, 0, "rifles", (14, 2))  # 'midc' cell
    sim.issue(0, [Capture(squad_c.id, "midc")])
    sim.run(CAPTURE_TICKS)
    assert sim.state.points["midc"].owner_team == 0

    assert sim.state.connected[0] == [a, b, c]

    # Enemy neutralizes the middle sector's point: 'c' becomes unreachable
    # even though team 0 still owns it.
    enemy = spawn(sim, 1, "rifles", (8, 2))
    sim.issue(1, [Capture(enemy.id, "midb")])
    sim.run(NEUTRALIZE_TICKS)
    assert sim.state.points["midb"].owner_team is None
    assert sim.state.connected[0] == [a]

    # Retake it: stop the enemy, resume team 0's capture from scratch.
    sim.issue(1, [Stop(enemy.id)])
    sim.issue(0, [Capture(squad_b.id, "midb")])
    sim.run(CAPTURE_TICKS)
    assert sim.state.points["midb"].owner_team == 0
    assert sim.state.connected[0] == [a, b, c]
