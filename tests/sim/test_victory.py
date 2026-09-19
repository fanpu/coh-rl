"""Tests for the victory system: VP ticket drain, annihilation, time limit."""

from __future__ import annotations

from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.sim import SimConfig
from tests.helpers import DEFAULT_POINTS, fixture_data, make_sim

# Three victory points (instead of the default strategic/victory/strategic
# mix) sharing the default map's three-sector layout, so tests can assign VP
# ownership per team via `sim.state.points[pid].owner_team`.
THREE_VP_POINTS = [{**p, "type": "victory"} for p in DEFAULT_POINTS]

ECON = fixture_data().economy
INTERVAL_TICKS = round(ECON.ticket_interval_s * TICKS_PER_SECOND)


def _vp_sim(**kwargs):
    return make_sim(points=THREE_VP_POINTS, **kwargs)


def test_vp_lead_drains_trailing_team_only():
    sim = _vp_sim()
    sim.state.points["west"].owner_team = 0
    sim.state.points["mid"].owner_team = 0
    sim.state.points["east"].owner_team = 1

    start_a, start_b = sim.state.tickets[0], sim.state.tickets[1]
    sim.run(INTERVAL_TICKS)

    assert sim.state.tickets[0] == start_a
    assert sim.state.tickets[1] == start_b - ECON.tickets_per_vp_lead
    assert sim.state.winner is None


def test_equal_vp_drains_nothing():
    sim = _vp_sim()
    sim.state.points["west"].owner_team = 0
    sim.state.points["east"].owner_team = 1
    # "mid" left unowned: both teams hold 1 VP each.

    start_a, start_b = sim.state.tickets[0], sim.state.tickets[1]
    sim.run(INTERVAL_TICKS)

    assert sim.state.tickets[0] == start_a
    assert sim.state.tickets[1] == start_b
    assert sim.state.winner is None


def test_zero_tickets_ends_the_game():
    sim = _vp_sim()
    sim.state.tickets[1] = 0.0

    sim.tick()

    assert sim.state.winner == 0
    assert any(e.kind == "game_over" and e.data.get("winner") == 0 for e in sim.state.events)


def test_zero_tickets_both_teams_ties_is_a_draw():
    sim = _vp_sim()
    sim.state.tickets[0] = 0.0
    sim.state.tickets[1] = 0.0

    sim.tick()

    assert sim.state.winner == -1


def test_hq_kill_ends_the_game():
    sim = _vp_sim()
    loser = sim.state.players[1]
    del sim.state.buildings[loser.hq_id]

    sim.tick()

    assert sim.state.winner == 0
    assert any(e.kind == "game_over" and e.data.get("winner") == 0 for e in sim.state.events)


def test_annihilation_takes_precedence_over_tickets():
    """Both HQ-destroyed and tickets-depleted the same tick: annihilation
    wins, since it is checked before the tickets rule."""
    sim = _vp_sim()
    loser = sim.state.players[1]
    del sim.state.buildings[loser.hq_id]
    sim.state.tickets[0] = 0.0  # would otherwise force a draw via tickets

    sim.tick()

    assert sim.state.winner == 0


def test_time_limit_picks_the_ticket_leader():
    sim = _vp_sim(config=SimConfig(time_limit_s=1.0))
    sim.state.tickets[0] = 250.0
    sim.state.tickets[1] = 200.0

    ticks_to_limit = round(1.0 * TICKS_PER_SECOND)
    sim.run(ticks_to_limit)

    assert sim.state.winner == 0
    assert any(e.kind == "game_over" and e.data.get("winner") == 0 for e in sim.state.events)


def test_time_limit_tie_is_a_draw():
    sim = _vp_sim(config=SimConfig(time_limit_s=1.0))

    ticks_to_limit = round(1.0 * TICKS_PER_SECOND)
    sim.run(ticks_to_limit)

    assert sim.state.winner == -1


def test_ticking_after_the_end_changes_nothing_but_drops_stale_events():
    sim = _vp_sim()
    sim.state.tickets[1] = 0.0
    sim.tick()
    assert sim.state.winner == 0

    tick_after_end = sim.state.tick
    tickets_after_end = dict(sim.state.tickets)
    hash_after_end = sim.state_hash()
    assert any(e.kind == "game_over" for e in sim.state.events)

    for _ in range(5):
        sim.tick()

    assert sim.state.tick == tick_after_end
    assert sim.state.tickets == tickets_after_end
    assert sim.state_hash() == hash_after_end
    # The `game_over` event belongs to the tick it happened on; a no-op tick
    # must not leave it looking like it just happened again.
    assert sim.state.events == []
