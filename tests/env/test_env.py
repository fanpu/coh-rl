"""Tests for `CohEnv` and the fog-filtered `Observation` (task 16)."""

from __future__ import annotations

import json

import pytest

from coh.env import CohEnv, build_observation
from coh.maps.format import center_of
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import Capture, Move, Train
from coh.sim.sim import PlayerSetup, SimConfig
from tests.helpers import fixture_data, make_map

PLAYERS = [PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)]


def make_env(**kwargs) -> CohEnv:
    data = fixture_data()
    return CohEnv(
        map_name="inline",
        players=list(PLAYERS),
        data=data,
        game_map=make_map(data=data),
        **kwargs,
    )


@pytest.fixture
def env() -> CohEnv:
    return make_env()


def _point(obs, point_id: str):
    return next(p for p in obs.points if p.id == point_id)


def _neutral(obs, building_id: int):
    return next(b for b in obs.neutral_buildings if b.id == building_id)


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_reset_returns_one_observation_per_player(env):
    obs = env.reset()
    assert sorted(obs) == [0, 1]
    assert obs[0].player_id == 0 and obs[0].team == 0
    assert obs[1].player_id == 1 and obs[1].team == 1
    assert obs[0].time_s == 0.0


def test_step_advances_exactly_the_decision_interval(env):
    env.reset()
    env.step({})
    assert env.sim.state.tick == 16 == round(2.0 * TICKS_PER_SECOND)
    env.step({})
    assert env.sim.state.tick == 32


def test_decision_interval_sets_ticks_per_step():
    env = make_env(decision_interval_s=0.5)
    env.reset()
    env.step({})
    assert env.ticks_per_step == 4
    assert env.sim.state.tick == 4


def test_step_before_reset_raises(env):
    with pytest.raises(RuntimeError):
        env.step({})


def test_nonpositive_decision_interval_rejected():
    with pytest.raises(ValueError):
        make_env(decision_interval_s=0.0)


# ---------------------------------------------------------------------------
# fog of war
# ---------------------------------------------------------------------------


def test_enemy_squads_outside_vision_are_absent_and_appear_when_scouted(env):
    obs = env.reset()
    # Opposite corners of a 40x30 map: neither side can see the other.
    assert obs[0].enemy_squads == []
    assert obs[1].enemy_squads == []

    # A player-1 squad walks right up to player 0's HQ.
    intruder = env.sim.spawn_squad(1, "rifles", center_of((4, 8)))
    obs, _rewards, _done, _infos = env.step({})

    seen = [s for s in obs[0].enemy_squads if s.id == intruder.id]
    assert len(seen) == 1
    assert all(s.owner == 1 for s in obs[0].enemy_squads)

    # A squad that stays in its own corner is still invisible to player 0.
    hidden = env.sim.spawn_squad(1, "rifles", center_of((36, 26)))
    obs, _rewards, _done, _infos = env.step({})
    assert hidden.id not in {s.id for s in obs[0].enemy_squads}


def test_enemy_squad_view_hides_order_cover_and_reinforce_range(env):
    env.reset()
    intruder = env.sim.spawn_squad(1, "rifles", center_of((4, 8)))
    env.sim.issue(1, [Move(squad=intruder.id, cell=(10, 10))])
    obs, _r, _d, _i = env.step({})

    enemy = next(s for s in obs[0].enemy_squads if s.id == intruder.id)
    assert enemy.order is None
    assert enemy.cover is None
    assert enemy.in_reinforce_range is None

    own = obs[1].own_squads
    mine = next(s for s in own if s.id == intruder.id)
    assert mine.order == {"type": "Move", "squad": intruder.id, "cell": [10, 10]}
    assert mine.cover is not None
    assert mine.in_reinforce_range in (True, False)


def test_enemy_building_becomes_a_ghost_when_vision_is_lost(env):
    env.reset()
    enemy_hq = env.sim.state.buildings[env.sim.state.players[1].hq_id]
    scout = env.sim.spawn_squad(0, "rifles", center_of((34, 27)))

    obs, _r, _d, _i = env.step({})
    assert [b.id for b in obs[0].enemy_buildings] == [enemy_hq.id]
    assert obs[0].ghosts == []

    del env.sim.state.squads[scout.id]  # the scout dies; the memory remains
    obs, _r, _d, _i = env.step({})
    assert obs[0].enemy_buildings == []
    ghost = next(g for g in obs[0].ghosts if g.id == enemy_hq.id)
    assert ghost.def_id == enemy_hq.def_id
    assert ghost.cell == enemy_hq.cell
    assert ghost.last_seen_s > 0


def test_own_and_ally_squads_are_always_visible(env):
    obs = env.reset()
    own_ids = {s.id for s in obs[0].own_squads}
    assert own_ids == {s.id for s in env.sim.state.squads.values() if s.owner == 0}
    assert obs[0].ally_squads == []  # 1v1: no allies


def test_enemy_observation_post_is_hidden_until_it_is_scouted(env):
    """`has_op` is a building on the ground, so it is fogged like one."""
    env.reset()
    op = env.sim.spawn_building(1, "op_us", (33, 16))
    env.sim.state.points["east"].op_building = op.id

    obs, _r, _d, _i = env.step({})
    assert _point(obs[1], "east").has_op is True  # its owner knows
    assert _point(obs[0], "east").has_op is False  # nobody from team 0 has looked

    scout = env.sim.spawn_squad(0, "rifles", center_of((33, 18)))
    obs, _r, _d, _i = env.step({})
    assert _point(obs[0], "east").has_op is True

    del env.sim.state.squads[scout.id]  # eyes gone, memory stays
    env.sim.state.points["east"].op_building = None
    obs, _r, _d, _i = env.step({})
    assert _point(obs[0], "east").has_op is True


def test_neutral_building_damage_outside_vision_is_not_reported(env):
    env.reset()
    house = env.sim.spawn_building(None, "house", (24, 2))
    obs, _r, _d, _i = env.step({})
    assert _neutral(obs[0], house.id).hp_frac == 1.0

    house.hp *= 0.5  # shelled while nobody from team 0 is watching
    obs, _r, _d, _i = env.step({})
    assert _neutral(obs[0], house.id).hp_frac == 1.0

    scout = env.sim.spawn_squad(0, "rifles", center_of((24, 5)))
    obs, _r, _d, _i = env.step({})
    assert _neutral(obs[0], house.id).hp_frac == pytest.approx(0.5)

    del env.sim.state.squads[scout.id]
    house.hp *= 0.5
    obs, _r, _d, _i = env.step({})
    assert _neutral(obs[0], house.id).hp_frac == pytest.approx(0.5)  # last known


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------


def test_point_ownership_is_public_but_enemy_supply_is_not(env):
    obs = env.reset()
    by_id = {p.id: p for p in obs[0].points}
    assert set(by_id) == set(env.sim.map.points)

    own_hq_point = next(p for p in obs[0].points if p.owner_team == 0)
    enemy_hq_point = next(p for p in obs[0].points if p.owner_team == 1)
    assert own_hq_point.connected is True
    assert enemy_hq_point.connected is None  # supply is the enemy's business

    neutral = next(p for p in obs[0].points if p.owner_team is None)
    assert neutral.connected is False
    assert neutral.progress == 0.0


def test_capture_progress_is_reported(env):
    env.reset()
    squad = next(s for s in env.sim.state.squads.values() if s.owner == 0)
    env.sim.issue(0, [Capture(squad=squad.id, point_id="west")])
    for _ in range(20):
        env.step({})
        obs = {pid: build_observation(env.sim, pid) for pid in (0, 1)}
        west = next(p for p in obs[1].points if p.id == "west")
        if west.progress > 0.0:
            return
    pytest.fail("capture progress never showed up in the observation")


# ---------------------------------------------------------------------------
# available orders
# ---------------------------------------------------------------------------


def test_available_reflects_affordability(env):
    obs = env.reset()
    hq = next(b for b in obs[0].own_buildings if b.def_id == "hq_us")
    assert obs[0].available["train"][hq.id] == ["engineers"]
    assert "barracks" in obs[0].available["build"]

    env.sim.state.players[0].manpower = 10.0
    obs = {0: build_observation(env.sim, 0)}
    assert obs[0].available["train"] == {}
    # The free observation post is still placeable; the 200mp barracks is not.
    assert "barracks" not in obs[0].available["build"]


def test_available_build_needs_a_builder(env):
    env.reset()
    for squad_id in [s.id for s in env.sim.state.squads.values() if s.owner == 0]:
        del env.sim.state.squads[squad_id]
    assert build_observation(env.sim, 0).available["build"] == []


def test_available_train_empties_when_the_queue_is_full(env):
    env.reset()
    hq_id = env.sim.state.players[0].hq_id
    env.sim.state.players[0].manpower = 10_000.0
    for _ in range(5):
        env.sim.issue(0, [Train(building=hq_id, unit="engineers")])
    assert hq_id not in build_observation(env.sim, 0).available["train"]


# ---------------------------------------------------------------------------
# orders, rewards, order log
# ---------------------------------------------------------------------------


def test_order_log_records_valid_and_invalid_orders(env):
    env.reset()
    squad = next(s for s in env.sim.state.squads.values() if s.owner == 0)
    _obs, _rewards, _done, infos = env.step(
        {0: [Move(squad=squad.id, cell=(10, 10)), Move(squad=9999, cell=(10, 10))]}
    )
    assert infos[0]["invalid_orders"] == 1
    assert [r.ok for r in infos[0]["results"]] == [True, False]
    assert [entry[0] for entry in env.order_log] == [0, 0]
    assert [entry[1] for entry in env.order_log] == [0, 0]
    assert env.order_log[0][2] == {"type": "Move", "squad": squad.id, "cell": [10, 10]}


def test_rewards_are_zero_until_the_game_ends(env):
    env.reset()
    _obs, rewards, done, _infos = env.step({})
    assert done is False
    assert rewards == {0: 0.0, 1: 0.0}


def test_rewards_pay_the_winner_at_the_end(env):
    env.reset()
    del env.sim.state.buildings[env.sim.state.players[1].hq_id]  # team 1 annihilated
    obs, rewards, done, _infos = env.step({})
    assert done is True
    assert env.sim.state.winner == 0
    assert rewards == {0: 1.0, 1: -1.0}
    assert obs[0].time_s > 0


def test_a_draw_pays_nobody():
    env = make_env(config=SimConfig(time_limit_s=1.0))
    env.reset()
    _obs, rewards, done, _infos = env.step({})
    assert done is True
    assert env.sim.state.winner == -1
    assert rewards == {0: 0.0, 1: 0.0}


def test_steps_after_the_game_ends_are_no_ops_and_pay_nothing(env):
    """The terminal reward is paid once, so the episode return is the result."""
    env.reset()
    del env.sim.state.buildings[env.sim.state.players[1].hq_id]
    _obs, first_rewards, _done, _infos = env.step({})
    assert first_rewards == {0: 1.0, 1: -1.0}
    ticks, log_len = env.sim.state.tick, len(env.order_log)

    squad = next(s for s in env.sim.state.squads.values() if s.owner == 0)
    obs, rewards, done, infos = env.step({0: [Move(squad=squad.id, cell=(10, 10))]})
    assert done is True
    assert env.sim.state.tick == ticks
    assert len(env.order_log) == log_len
    assert infos[0]["invalid_orders"] == 0
    assert rewards == {0: 0.0, 1: 0.0}
    assert set(obs) == {0, 1}


def test_the_order_issue_sequence_rotates_across_players(env):
    """Player 0 does not get first call on a contested resource every step."""
    env.reset()
    assert env._issue_sequence() == [0, 1]
    env.step({})
    assert env._issue_sequence() == [1, 0]
    env.step({})
    assert env._issue_sequence() == [0, 1]


def test_orders_are_issued_and_logged_in_the_rotated_sequence(env):
    env.reset()
    env.step({})  # step 1: player 1 now goes first
    squads = {pid: next(s.id for s in env.sim.state.squads.values() if s.owner == pid) for pid in (0, 1)}
    env.step(
        {
            0: [Move(squad=squads[0], cell=(10, 10))],
            1: [Move(squad=squads[1], cell=(20, 20))],
        }
    )
    assert [player_id for _tick, player_id, _order in env.order_log] == [1, 0]


def test_junk_in_an_agents_order_list_counts_as_invalid_and_is_skipped(env):
    env.reset()
    squad = next(s for s in env.sim.state.squads.values() if s.owner == 0)
    good = Move(squad=squad.id, cell=(10, 10))
    _obs, _r, _d, infos = env.step(
        {0: ["fire everything", good, {"type": "Stop", "squad": squad.id}, {"type": "Teleport"}, None]}
    )

    results = infos[0]["results"]
    assert [r.ok for r in results] == [False, True, True, False, False]
    assert infos[0]["invalid_orders"] == 3
    # Only the two real orders reached the sim, so only they were logged.
    assert [entry[2]["type"] for entry in env.order_log] == ["Move", "Stop"]
    assert env.sim.state.players[0].invalid_orders == 0


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_observation_to_dict_is_json_serializable(env):
    obs = env.reset()
    blob = json.dumps(obs[0].to_dict())
    restored = json.loads(blob)
    assert restored["player_id"] == 0
    assert restored["own_squads"][0]["cell"] == list(obs[0].own_squads[0].cell)
    assert set(restored["available"]) == {"train", "build", "research", "squad_upgrades"}
    assert set(restored["income"]) == {"manpower", "munitions", "fuel"}
