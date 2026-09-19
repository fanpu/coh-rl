"""Ladder sanity: the bot that plays wins, whichever seat and faction it takes.

`tests/agents/test_bots.py` already checks that T1 beats T0 once. This is the
ladder-shaped version of that claim: ten matches, alternating which *side* T1
plays (so a map asymmetry cannot be doing the work) and which *faction* it
fields (so a faction asymmetry in the tables cannot be either), and T1 has to
take at least nine of them.

Runs on the fixture tables, like everything else in M1; nothing here depends
on which stat tables are loaded beyond `data`.
"""

from __future__ import annotations

import pytest

from coh.agents import AGENTS
from coh.env import CohEnv
from coh.maps.format import load_map
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints
from tests.helpers import fixture_data

MAP_NAME = "hedgerow_crossing"
TIME_LIMIT_S = 3600.0
SEEDS = range(10)
MIN_WINS = 9
FACTIONS = ("us", "wehr")


@pytest.fixture(scope="module")
def data():
    return fixture_data()


@pytest.fixture(scope="module")
def match_map(data):
    return load_map(MAP_NAME, footprints=neutral_footprints(data))


def _play(match_map, data, *, t1_slot: int, t1_faction: str, seed: int) -> int | None:
    """One T1-vs-T0 match; returns the winning team (`-1` for a draw)."""
    factions = [None, None]
    factions[t1_slot] = t1_faction
    factions[1 - t1_slot] = FACTIONS[(FACTIONS.index(t1_faction) + 1) % len(FACTIONS)]

    players = [PlayerSetup(faction=factions[slot], team=slot, start_slot=slot) for slot in (0, 1)]
    env = CohEnv(
        map_name=MAP_NAME,
        players=players,
        seed=seed,
        config=SimConfig(time_limit_s=TIME_LIMIT_S),
        data=data,
        game_map=match_map,
    )
    agent_names = ["t0", "t0"]
    agent_names[t1_slot] = "t1"
    agents = [AGENTS[name]() for name in agent_names]

    obs = env.reset()
    for player_id, agent in enumerate(agents):
        agent.reset(player_id, match_map, data)

    done = False
    while not done:
        orders = {pid: agents[pid].act(obs[pid]) for pid in env.player_ids}
        obs, _rewards, done, _infos = env.step(orders)

    assert env.sim is not None
    return env.sim.state.winner


@pytest.mark.slow
def test_t1_beats_t0_from_either_side_and_either_faction(match_map, data):
    results = []
    for seed in SEEDS:
        t1_slot = seed % 2
        t1_faction = FACTIONS[(seed // 2) % len(FACTIONS)]
        winner = _play(match_map, data, t1_slot=t1_slot, t1_faction=t1_faction, seed=seed)
        results.append((seed, t1_slot, t1_faction, winner))

    wins = [entry for entry in results if entry[3] == entry[1]]  # T1's team == its slot
    assert len(wins) >= MIN_WINS, f"T1 won {len(wins)}/{len(results)}: {results}"
