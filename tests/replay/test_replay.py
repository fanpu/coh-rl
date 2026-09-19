"""Tests for replay save / load / re-simulation (task 16).

A replay is only worth anything if it is *exact*: the whole point of storing
orders instead of frames is that re-running them reproduces the match
bit-for-bit. Every test here is ultimately about `state_hash`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from coh.agents import T1Capper
from coh.data.hashing import data_hash
from coh.env import CohEnv
from coh.maps.hashing import map_hash
from coh.replay import (
    Replay,
    ReplayError,
    ReplayMismatchError,
    load,
    replay_final_hash,
    resimulate,
    save,
)
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import AttackMove, Move, SetFacing, Train
from coh.maps.format import load_map
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints
from tests.helpers import FIXTURES_DIR, fixture_data, make_map

MAP_NAME = "hedgerow_crossing"
PLAYERS = [PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)]
MATCH_SECONDS = 120.0


@pytest.fixture(scope="module")
def match_map():
    return load_map(MAP_NAME, footprints=neutral_footprints(fixture_data()))


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


class _NumpyfyingAgent:
    """Wraps an agent, converting its `Move` / `AttackMove` / `SetFacing` /
    `Train` orders to numpy-typed fields -- exactly what an RL policy hands
    back. `env.order_log` (and therefore a saved replay) must stay plain
    Python / JSON-safe regardless."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def reset(self, player_id, match_map, data) -> None:
        self._inner.reset(player_id, match_map, data)

    def act(self, obs):
        import numpy as np

        out = []
        for order in self._inner.act(obs):
            if isinstance(order, Move):
                order = replace(order, squad=np.int64(order.squad), cell=(np.int64(order.cell[0]), np.int64(order.cell[1])))
            elif isinstance(order, AttackMove):
                order = replace(order, squad=np.int64(order.squad), cell=np.array(order.cell))
            elif isinstance(order, SetFacing):
                order = replace(order, squad=np.int64(order.squad), direction_deg=np.float32(order.direction_deg))
            elif isinstance(order, Train):
                order = replace(order, building=np.int64(order.building))
            out.append(order)
        return out


def test_numpy_typed_orders_from_an_rl_policy_still_replay(tmp_path, match_map):
    """A policy that hands back numpy scalars must not poison the replay:
    `to_replay()` -> `save()` -> `load()` -> `resimulate(verify=True)` still
    reproduces the final hash, and the saved JSON contains no numpy types."""
    pytest.importorskip("numpy")
    data = fixture_data()
    env = CohEnv(
        map_name=MAP_NAME,
        players=list(PLAYERS),
        seed=0,
        config=SimConfig(time_limit_s=3600.0),
        data=data,
        game_map=match_map,
    )
    agents = [_NumpyfyingAgent(T1Capper()), T1Capper()]
    obs = env.reset()
    for player_id, agent in enumerate(agents):
        agent.reset(player_id, match_map, data)

    done = False
    while not done and env.sim.state.tick < 30.0 * TICKS_PER_SECOND:
        obs, _rewards, done, _infos = env.step({pid: agents[pid].act(obs[pid]) for pid in env.player_ids})

    assert env.order_log, "the bots should have issued orders in 30 s"
    json.dumps(env.order_log)  # the defect under test: must not raise

    replay = env.to_replay()
    path = tmp_path / "numpy.replay.json"
    save(replay, path)
    restored = load(path)

    replayed = list(resimulate(restored, data, match_map, verify=True))
    assert replayed[-1].state_hash() == env.sim.state_hash() == replay.final_hash


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


# ---------------------------------------------------------------------------
# identity: which tables, which map
# ---------------------------------------------------------------------------


def test_replay_records_the_data_and_map_it_was_played_on(tmp_path, match_map):
    env = play(match_map, 10.0)
    replay = env.to_replay()
    assert replay.data_hash == data_hash(fixture_data())
    assert replay.map_hash == map_hash(match_map)

    path = tmp_path / "match.replay.json"
    save(replay, path)
    blob = json.loads(path.read_text())
    assert blob["data_hash"] == replay.data_hash
    assert blob["map_hash"] == replay.map_hash
    assert load(path).data_hash == replay.data_hash


def test_map_hash_survives_a_match(match_map):
    """The sim mutates its own copy, so the caller's map still hashes the same."""
    before = map_hash(match_map)
    play(match_map, 10.0)
    assert map_hash(match_map) == before
    assert map_hash(load_map(MAP_NAME, footprints=neutral_footprints(fixture_data()))) == before


def test_resimulating_against_different_tables_is_refused(match_map):
    replay = play(match_map, 10.0).to_replay()
    other = copy.deepcopy(fixture_data())
    other.weapons["rifle"] = replace(other.weapons["rifle"], damage=999.0)

    with pytest.raises(ReplayMismatchError, match="data"):
        resimulate(replay, other, match_map)


def test_resimulating_against_a_different_map_is_refused(match_map):
    replay = play(match_map, 10.0).to_replay()
    with pytest.raises(ReplayMismatchError, match="map"):
        resimulate(replay, fixture_data(), make_map(data=fixture_data()))


def test_a_diverging_final_hash_is_reported(match_map):
    replay = play(match_map, 10.0).to_replay()
    replay.final_hash = "0" * 64
    with pytest.raises(ReplayMismatchError, match="final state"):
        list(resimulate(replay, fixture_data(), match_map))


def test_verification_can_be_switched_off(match_map):
    replay = play(match_map, 10.0).to_replay()
    replay.final_hash = "0" * 64
    replay.map_hash = "nonsense"
    assert list(resimulate(replay, fixture_data(), match_map, verify=False))


def test_a_v1_replay_without_hashes_still_replays(tmp_path, match_map):
    """Identity fields are optional, so replays recorded before them still run."""
    path = tmp_path / "match.replay.json"
    save(play(match_map, 10.0).to_replay(), path)
    blob = json.loads(path.read_text())
    del blob["data_hash"]
    del blob["map_hash"]
    path.write_text(json.dumps(blob))

    restored = load(path)
    assert restored.data_hash is None and restored.map_hash is None
    assert list(resimulate(restored, fixture_data(), match_map))


# ---------------------------------------------------------------------------
# hostile files
# ---------------------------------------------------------------------------


HOSTILE_REPLAYS = [
    ("not_a_mapping", []),
    ("missing_map_name", {"version": 1, "players": [], "seed": 0}),
    (
        "player_missing_a_field",
        {
            "version": 1,
            "map_name": MAP_NAME,
            "players": [{"faction": "us", "team": 0}],
            "seed": 0,
            "decision_interval_s": 2.0,
            "orders": [],
            "final_hash": "x",
        },
    ),
    (
        "order_entry_is_not_a_triple",
        {
            "version": 1,
            "map_name": MAP_NAME,
            "players": [],
            "seed": 0,
            "decision_interval_s": 2.0,
            "orders": [[0, 0]],
            "final_hash": "x",
        },
    ),
    (
        "order_tick_is_not_a_number",
        {
            "version": 1,
            "map_name": MAP_NAME,
            "players": [],
            "seed": 0,
            "decision_interval_s": 2.0,
            "orders": [["soon", 0, {"type": "Stop", "squad": 1}]],
            "final_hash": "x",
        },
    ),
    (
        "config_has_an_unknown_key",
        {
            "version": 1,
            "map_name": MAP_NAME,
            "players": [],
            "seed": 0,
            "decision_interval_s": 2.0,
            "config": {"nonsense": 1},
            "orders": [],
            "final_hash": "x",
        },
    ),
]


@pytest.mark.parametrize(("case_id", "blob"), HOSTILE_REPLAYS, ids=[c for c, _ in HOSTILE_REPLAYS])
def test_malformed_replay_files_raise_replay_error(tmp_path, case_id, blob):
    path = tmp_path / "hostile.replay.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(ReplayError):
        load(path)


def test_truncated_replay_file_raises_replay_error(tmp_path):
    path = tmp_path / "truncated.replay.json"
    path.write_text('{"version": 1, "map_name": "hedge')
    with pytest.raises(ReplayError):
        load(path)


def test_an_unplayable_order_names_the_offending_entry(match_map):
    replay = play(match_map, 10.0).to_replay()
    tick, player_id, _order = replay.orders[0]
    replay.orders[0] = (tick, player_id, {"type": "Teleport", "squad": 1})
    with pytest.raises(ReplayError) as exc:
        list(resimulate(replay, fixture_data(), match_map, verify=False))
    assert "Teleport" in str(exc.value)
