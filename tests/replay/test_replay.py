"""Tests for replay save / load / re-simulation (task 16).

A replay is only worth anything if it is *exact*: the whole point of storing
orders instead of frames is that re-running them reproduces the match
bit-for-bit. Every test here is ultimately about `state_hash`.
"""

from __future__ import annotations

import json

import pytest

from coh.agents import T1Capper
from coh.env import CohEnv, load_match_map
from coh.replay import Replay, load, replay_final_hash, resimulate, save
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import Move
from coh.sim.sim import PlayerSetup, SimConfig
from tests.helpers import FIXTURES_DIR, fixture_data

MAP_NAME = "hedgerow_crossing"
PLAYERS = [PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)]
MATCH_SECONDS = 120.0


@pytest.fixture(scope="module")
def match_map():
    return load_match_map(MAP_NAME, fixture_data(), skip_unknown_neutrals=True)


def play(match_map, seconds: float, seed: int = 0) -> CohEnv:
    """Play `seconds` of T1 vs T1 and hand back the finished env."""
    data = fixture_data()
    env = CohEnv(
        map_name=MAP_NAME,
        players=list(PLAYERS),
        seed=seed,
        config=SimConfig(time_limit_s=3600.0),
        data=data,
        game_map=match_map,
    )
    agents = [T1Capper(), T1Capper()]
    obs = env.reset()
    for player_id, agent in enumerate(agents):
        agent.reset(player_id, match_map, data)

    done = False
    while not done and env.sim.state.tick < seconds * TICKS_PER_SECOND:
        obs, _rewards, done, _infos = env.step({pid: agents[pid].act(obs[pid]) for pid in env.player_ids})
    return env


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def test_save_load_resimulate_reproduces_the_final_hash(tmp_path, match_map):
    env = play(match_map, MATCH_SECONDS)
    replay = env.to_replay()
    assert replay.orders, "the bots should have issued orders in 120 s"
    assert replay.final_tick == env.sim.state.tick

    path = tmp_path / "match.replay.json"
    save(replay, path)
    restored = load(path)

    assert restored.map_name == replay.map_name
    assert restored.players == replay.players
    assert restored.orders == replay.orders
    assert restored.final_hash == replay.final_hash
    assert replay_final_hash(restored, fixture_data(), match_map) == replay.final_hash


def test_replay_json_is_plain_and_versioned(tmp_path, match_map):
    env = play(match_map, 10.0)
    path = tmp_path / "match.replay.json"
    save(env.to_replay(data_dir="tests/data/fixtures"), path)

    blob = json.loads(path.read_text())
    assert blob["version"] == 1
    assert blob["map_name"] == MAP_NAME
    assert blob["seed"] == 0
    assert blob["decision_interval_s"] == 2.0
    assert blob["config"] == {"time_limit_s": 3600.0}
    assert blob["data_dir"] == "tests/data/fixtures"
    assert blob["players"][0] == {"faction": "us", "team": 0, "start_slot": 0}
    tick, player_id, order = blob["orders"][0]
    assert isinstance(tick, int) and isinstance(player_id, int) and "type" in order


def test_loading_an_unknown_version_is_refused(tmp_path, match_map):
    path = tmp_path / "match.replay.json"
    save(play(match_map, 4.0).to_replay(), path)
    blob = json.loads(path.read_text())
    blob["version"] = 99
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="unsupported replay version"):
        load(path)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_tampering_with_one_order_changes_the_final_hash(tmp_path, match_map):
    env = play(match_map, MATCH_SECONDS)
    replay = env.to_replay()
    assert replay_final_hash(replay, fixture_data(), match_map) == replay.final_hash

    tick, player_id, order = replay.orders[-1]
    tampered = Replay(
        map_name=replay.map_name,
        players=replay.players,
        seed=replay.seed,
        decision_interval_s=replay.decision_interval_s,
        config=replay.config,
        orders=replay.orders[:-1] + [(tick, player_id, {"type": "Stop", "squad": order.get("squad", 1)})],
        final_hash=replay.final_hash,
        final_tick=replay.final_tick,
    )
    assert replay_final_hash(tampered, fixture_data(), match_map) != replay.final_hash


def test_resimulate_yields_one_sim_per_tick(match_map):
    env = play(match_map, 10.0)
    replay = env.to_replay()
    ticks = [sim.state.tick for sim in resimulate(replay, fixture_data(), match_map)]
    assert ticks == list(range(1, replay.final_tick + 1))


def test_resimulate_reproduces_a_hand_driven_match(match_map):
    """Orders issued by hand, not by a bot, replay just the same."""
    data = fixture_data()
    env = CohEnv(
        map_name=MAP_NAME,
        players=list(PLAYERS),
        seed=3,
        config=SimConfig(time_limit_s=3600.0),
        data=data,
        game_map=match_map,
    )
    env.reset()
    squad = next(s for s in env.sim.state.squads.values() if s.owner == 0)
    env.step({0: [Move(squad=squad.id, cell=(20, 30))]})
    env.step({})
    live_hashes = [env.sim.state_hash()]

    replay = env.to_replay()
    replayed = [sim.state_hash() for sim in resimulate(replay, data, match_map)]
    assert replayed[-1] == live_hashes[-1] == replay.final_hash
    assert len(replayed) == replay.final_tick
    # ... and re-running it again walks the identical trajectory.
    assert [sim.state_hash() for sim in resimulate(replay, data, match_map)] == replayed


def test_an_empty_replay_is_the_opening_state(match_map):
    data = fixture_data()
    env = CohEnv(map_name=MAP_NAME, players=list(PLAYERS), data=data, game_map=match_map)
    env.reset()
    replay = env.to_replay()
    assert replay.final_tick == 0
    assert replay_final_hash(replay, data, match_map) == replay.final_hash


def test_replay_falls_back_to_its_recorded_data_dir(tmp_path, match_map):
    env = play(match_map, 10.0)
    path = tmp_path / "match.replay.json"
    save(env.to_replay(data_dir=str(FIXTURES_DIR)), path)
    restored = load(path)
    # No `data` argument: the replay has to find the tables on its own.
    assert replay_final_hash(restored, game_map=match_map) == restored.final_hash
