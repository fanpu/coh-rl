"""Tests for coh.sim.systems.economy (task 14): income, upkeep, population.

Fixture economy: base income 240 mp/min, `strategic` points pay 8 mp/min and
2 pop, `victory` pays nothing, `op_income_mult` 1.5, upkeep floor 25 % of base
manpower (60/min), base pop 20.

On the default map team 0 owns 'west' (strategic, sector 'a') from the start
and begins with a single `engineers` squad (2 mp/min upkeep), so player 0's
baseline manpower income is 240 + 8 - 2 = 246/min.
"""

from __future__ import annotations

import pytest

from coh.sim.orders import Train
from coh.sim.state import PointState
from coh.sim.systems import economy, territory
from tests.helpers import fixture_data, make_sim, spawn

BASELINE_MP_PER_MIN = 246.0
TICKS_PER_MINUTE = 8 * 60


# ---------------------------------------------------------------------------
# Income
# ---------------------------------------------------------------------------


def test_income_accrues_at_the_table_rate_over_a_minute():
    sim = make_sim()
    assert economy.income_per_min(sim, 0).manpower == pytest.approx(BASELINE_MP_PER_MIN)

    start = sim.state.players[0].manpower
    sim.run(TICKS_PER_MINUTE)

    assert sim.state.players[0].manpower - start == pytest.approx(BASELINE_MP_PER_MIN)


def test_every_player_on_a_team_receives_the_full_team_income():
    from coh.sim.sim import PlayerSetup

    sim = make_sim(
        players=[
            PlayerSetup(faction="us", team=0, start_slot=0),
            PlayerSetup(faction="us", team=0, start_slot=1),
        ],
        starts=[
            {"slot": 0, "team": 0, "hq_cell": [2, 2], "sector": "a"},
            {"slot": 1, "team": 0, "hq_cell": [34, 22], "sector": "c"},
        ],
    )
    # Team 0 owns both HQ strategic points; each player gets both, not a share.
    for player_id in (0, 1):
        assert economy.income_per_min(sim, player_id).manpower == pytest.approx(240 + 8 + 8 - 2)


def test_disconnected_sector_stops_paying_income_and_population():
    sim = make_sim()
    # Team 0 owns everything: 'mid' (victory, no income) links 'east' to base.
    sim.state.points["mid"] = PointState(owner_team=0, progress=1.0)
    sim.state.points["east"] = PointState(owner_team=0, progress=1.0)
    territory.recompute_connectivity(sim)

    assert economy.income_per_min(sim, 0).manpower == pytest.approx(240 + 8 + 8 - 2)
    assert economy.pop_cap(sim, 0) == 20 + 2 + 2

    # Losing the middle link cuts 'east' off: no income, no pop, still owned.
    sim.state.points["mid"] = PointState(owner_team=None, progress=0.0)
    territory.recompute_connectivity(sim)

    assert sim.state.points["east"].owner_team == 0
    assert economy.income_per_min(sim, 0).manpower == pytest.approx(BASELINE_MP_PER_MIN)
    assert economy.pop_cap(sim, 0) == 20 + 2


def test_observation_post_multiplies_its_points_income():
    sim = make_sim()
    op = sim.spawn_building(0, "op_us", (6, 14), complete=False)
    sim.state.points["west"].op_building = op.id

    # Still under construction: no bonus yet.
    assert economy.income_per_min(sim, 0).manpower == pytest.approx(BASELINE_MP_PER_MIN)

    op.progress = 1.0
    assert economy.income_per_min(sim, 0).manpower == pytest.approx(240 + 8 * 1.5 - 2)


# ---------------------------------------------------------------------------
# Upkeep
# ---------------------------------------------------------------------------


def test_upkeep_lowers_manpower_income_but_not_below_the_floor():
    sim = make_sim()
    for _ in range(10):
        spawn(sim, 0, "rifles", (10, 10))  # 3 mp/min each

    assert economy.upkeep_per_min(sim, 0) == pytest.approx(2 + 10 * 3)
    assert economy.income_per_min(sim, 0).manpower == pytest.approx(248 - 32)

    # Enough upkeep to overshoot the gross income: floored at 25 % of base.
    for _ in range(100):
        spawn(sim, 0, "rifles", (10, 10))
    assert economy.income_per_min(sim, 0).manpower == pytest.approx(240 * 0.25)


def test_upkeep_counts_only_living_own_squads():
    sim = make_sim()
    dead = spawn(sim, 0, "rifles", (10, 10))
    for member in dead.members:
        member.hp = 0.0
    shell = spawn(sim, 0, "hmg_team", (10, 10))
    shell.abandoned = True
    spawn(sim, 1, "rifles", (30, 10))  # the enemy's upkeep is not ours

    assert economy.upkeep_per_min(sim, 0) == pytest.approx(2)


def test_supply_yard_upgrade_halves_upkeep():
    sim = make_sim()
    for _ in range(10):
        spawn(sim, 0, "rifles", (10, 10))
    assert economy.upkeep_per_min(sim, 0) == pytest.approx(32)

    sim.state.players[0].upgrades.append("supply_yard")  # upkeep_mult 0.5

    assert economy.upkeep_per_min(sim, 0) == pytest.approx(16)
    assert economy.income_per_min(sim, 0).manpower == pytest.approx(248 - 16)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def test_pop_used_counts_living_squads_and_queued_trains():
    sim = make_sim()
    assert economy.pop_used(sim, 0) == 4  # the starting engineers squad
    assert economy.pop_cap(sim, 0) == 22  # 20 base + 2 from 'west'

    rifles = spawn(sim, 0, "rifles", (10, 10))
    assert economy.pop_used(sim, 0) == 4 + 5

    for member in rifles.members:
        member.hp = 0.0
    assert economy.pop_used(sim, 0) == 4

    shell = spawn(sim, 0, "hmg_team", (10, 10))
    shell.abandoned = True
    assert economy.pop_used(sim, 0) == 4

    sim.state.players[0].manpower = 5000
    hq = sim.state.buildings[sim.state.players[0].hq_id]
    assert sim.issue(0, [Train(building=hq.id, unit="engineers")])[0].ok
    assert economy.pop_used(sim, 0) == 4 + 4


def test_pop_cap_is_clamped_to_max_population():
    from dataclasses import replace

    base = fixture_data()
    data = replace(base, economy=replace(base.economy, max_population=21))
    sim = make_sim(data=data)

    assert economy.pop_cap(sim, 0) == 21  # 20 base + 2 from 'west', clamped


def test_income_is_deterministic_across_two_runs():
    a, b = make_sim(seed=3), make_sim(seed=3)
    a.run(200)
    b.run(200)
    assert a.state_hash() == b.state_hash()
