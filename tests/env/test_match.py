"""Tests for `coh.env.run_match`, the shared whole-match loop."""

from __future__ import annotations

import pytest

from coh.agents import AGENTS
from coh.env import run_match
from coh.sim.sim import PlayerSetup, SimConfig
from tests.helpers import fixture_data, make_map

PLAYERS = [PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)]


class _Spy:
    """An agent that does nothing but remember what it was handed."""

    def __init__(self) -> None:
        self.map = None
        self.data = None

    def reset(self, player_id, map, data):  # noqa: A002 - matches the Agent protocol
        self.map = map
        self.data = data

    def act(self, obs):
        return []


@pytest.fixture
def inline_map():
    return make_map(data=fixture_data())


def test_every_agent_gets_its_own_map_and_tables(inline_map):
    data = fixture_data()
    spies = [_Spy(), _Spy()]
    result = run_match(
        "inline", list(PLAYERS), spies, data=data, game_map=inline_map, config=SimConfig(time_limit_s=1.0)
    )

    assert spies[0].map is not spies[1].map
    assert spies[0].map is not inline_map
    assert spies[0].data is not spies[1].data
    assert spies[0].data is not data
    assert result.env.sim is not None

    # An agent scribbling on its copy reaches neither the sim nor its opponent.
    spies[0].map.set_terrain_cell((20, 10), "w")
    spies[0].data.squads["rifles"].loadout  # tables are real objects, not stubs
    assert inline_map.pass_inf[10, 20]
    assert result.env.sim.map.pass_inf[10, 20]
    assert spies[1].map.pass_inf[10, 20]


def test_run_match_reports_the_result(inline_map):
    data = fixture_data()
    result = run_match(
        "inline",
        list(PLAYERS),
        [AGENTS["t1"](), AGENTS["t0"]()],
        data=data,
        game_map=inline_map,
        config=SimConfig(time_limit_s=120.0),
    )
    assert result.winner in (0, 1, -1)
    assert result.ticks > 0
    assert result.duration_s == pytest.approx(result.ticks * 0.125)
    assert set(result.orders_issued) == {0, 1}
    assert result.invalid_order_rate[0] <= 1.0
    # The terminal reward is paid once, so the return is the match result.
    assert sorted(result.rewards.values()) in ([-1.0, 1.0], [0.0, 0.0])


def test_agent_count_must_match_the_player_count(inline_map):
    with pytest.raises(ValueError):
        run_match("inline", list(PLAYERS), [AGENTS["t0"]()], data=fixture_data(), game_map=inline_map)
