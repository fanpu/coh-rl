"""Golden-master state hashes for the combat.py split refactor.

These play a full T1-vs-T1 mirror match on `hedgerow_crossing` (the same
machinery `tests/agents/test_bots.py::test_t1_mirror_matches_finish_cleanly`
uses) for seeds 0, 1 and 2, capped at 1200 game-seconds, and assert the final
tick and `state_hash()` against values recorded from the pre-split code.

Purpose: `coh/sim/systems/combat.py` is about to be split into several
focused modules with no intended behaviour change (RNG draw order, event
order, and public signatures must all stay identical). If any of these
hashes change after the split, behaviour changed -- go find it and undo it,
never update the values here to match new output.
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

# (final tick, final state_hash), recorded from the code as it stood before
# the combat.py split (coh/sim/systems/combat.py, single ~1350-line module).
GOLDEN = {
    0: (3679, "89f4cb987936f6591554249379b4dfcdc2e0967b015819ddb8f4b4d8945946f9"),
    1: (4423, "9344f5b6c7a30174ebb20accf07400c3a62405a14b6a8e8bdcf34e5d69b9ea0a"),
    2: (9600, "15f817820a03f1530d24876cea65a6161ac36ef9af470c58a455cbcfe4da328a"),
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
