"""Golden-master state hashes for the whole simulation.

These play a full T1-vs-T1 mirror match on `hedgerow_crossing` (the same
machinery `tests/agents/test_bots.py::test_t1_mirror_matches_finish_cleanly`
uses) for seeds 0, 1 and 2, capped at 1200 game-seconds, and assert the final
tick and `state_hash()`.

Purpose: refactors of the sim (the `combat.py` split this file was written
for, the perf work after it) must not change behaviour — RNG draw order,
event order and public signatures all have to stay identical. If any of
these hashes moves and you did not deliberately change a *rule*, behaviour
changed by accident: go find it and undo it, never update the values here to
match new output. When a rule really does change, re-record in a commit that
says which ruling did it (see `GOLDEN` below).
"""

from __future__ import annotations

import pytest

from coh.agents import AGENTS
from coh.env import CohEnv
from coh.maps.format import load_map
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints
from tests.helpers import fixture_data

MAP_NAME = "hedgerow_crossing"
PLAYERS = [PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)]
TIME_LIMIT_S = 1200.0

# (final tick, final state_hash).
#
# Re-recorded in the post-review fix wave, which deliberately changed four
# simulation rules (and only then): capture contesting is decided by eligible
# *presence* rather than by `Capture` orders; `Sim.__init__` runs the vision
# system, so the opening state is lit; an `Attack` order on a target that can
# never be engageable again is dropped instead of pursued forever; a trained
# unit with nowhere to stand waits at the head of its queue. The same wave
# centred the HQ footprint on `StartDef.hq_cell` and sent both the starting
# builder and every rally point out on the side facing the map centre, to
# remove a measured seat-0 bias. Anything that moves these values *without*
# such a ruling behind it is a regression: go find it, don't re-record.
GOLDEN = {
    0: (5439, "d6b89f91c9939f6ced0c2f14da6a85b20448c392a4b9b578b075945d219a987e"),
    1: (9600, "6d6f60dcd4027b70dc00343c999dbc12683e4ca371e781b3e547dfe87851aaca"),
    2: (9600, "a5b9d29701f90a070b3b2eb158a6bfb63d7129519042cab577becd7e07084558"),
}


@pytest.fixture(scope="module")
def match_map():
    return load_map(MAP_NAME, footprints=neutral_footprints(fixture_data()))


def _play(match_map, seed: int) -> tuple[int, str]:
    data = fixture_data()
    env = CohEnv(
        map_name=MAP_NAME,
        players=list(PLAYERS),
        seed=seed,
        config=SimConfig(time_limit_s=TIME_LIMIT_S),
        data=data,
        game_map=match_map,
    )
    agents = [AGENTS["t1"](), AGENTS["t1"]()]
    obs = env.reset()
    for player_id, agent in enumerate(agents):
        agent.reset(player_id, match_map, data)

    done = False
    while not done:
        orders = {pid: agents[pid].act(obs[pid]) for pid in env.player_ids}
        obs, rewards, done, infos = env.step(orders)

    return env.sim.state.tick, env.sim.state_hash()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t1_mirror_match_golden_hash(match_map, seed):
    tick, state_hash = _play(match_map, seed)
    expected_tick, expected_hash = GOLDEN[seed]
    assert tick == expected_tick
    assert state_hash == expected_hash
