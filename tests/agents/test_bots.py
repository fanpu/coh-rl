"""Tests for the scripted ladder bots (task 16).

These are whole-match tests on the shipped `hedgerow_crossing` map. The stat
tables they run on are the hand-written fixtures (the real CoH1 tables are a
later task), so the thresholds below are calibrated against fixture numbers —
see `tests/data/fixtures/economy.yaml`.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import coh.agents
from coh.agents import AGENTS, T0Idle, T1Capper
from coh.env import CohEnv
from coh.maps.format import load_map
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints
from tests.helpers import FIXTURES_DIR, fixture_data

MAP_NAME = "hedgerow_crossing"
PLAYERS = [PlayerSetup(faction="us", team=0, start_slot=0), PlayerSetup(faction="us", team=1, start_slot=1)]

# Fixture economy: 300 tickets, 1 ticket per VP lead every 30 s, so a pure
# ticket win needs ~50 minutes. A bot that actually fights finishes far sooner
# (T1 vs T0 annihilates in ~3.5 minutes), and this limit leaves room for both.
TIME_LIMIT_S = 3600.0
MAX_INVALID_ORDER_RATE = 0.05
MIN_POINTS_BY_MINUTE_8 = 6


@pytest.fixture(scope="module")
def match_map():
    return load_map(MAP_NAME, footprints=neutral_footprints(fixture_data()))


def play_match(match_map, p0: str, p1: str, seed: int = 0) -> dict:
    """Run a full match and report the numbers the assertions below need."""
    data = fixture_data()
    env = CohEnv(
        map_name=MAP_NAME,
        players=list(PLAYERS),
        seed=seed,
        config=SimConfig(time_limit_s=TIME_LIMIT_S),
        data=data,
        game_map=match_map,
    )
    agents = [AGENTS[p0](), AGENTS[p1]()]
    obs = env.reset()
    for player_id, agent in enumerate(agents):
        agent.reset(player_id, match_map, data)

    issued = {pid: 0 for pid in env.player_ids}
    invalid = {pid: 0 for pid in env.player_ids}
    points_by_time: list[tuple[float, dict[int, int]]] = []
    # Orders sent to a squad that was already reinforcing. Reinforce restores
    # one model after another by itself and *any* other order cancels it, so
    # a sound bot never touches such a squad.
    reinforce_interruptions: list[tuple[int, int]] = []

    done = False
    while not done:
        orders = {pid: agents[pid].act(obs[pid]) for pid in env.player_ids}
        for pid, player_orders in orders.items():
            issued[pid] += len(player_orders)
            reinforcing = {
                squad.id
                for squad in obs[pid].own_squads
                if squad.order is not None and squad.order["type"] == "Reinforce"
            }
            reinforce_interruptions += [
                (pid, order.squad)
                for order in player_orders
                if getattr(order, "squad", None) in reinforcing
            ]
        obs, rewards, done, infos = env.step(orders)
        for pid, info in infos.items():
            invalid[pid] += info["invalid_orders"]
        points_by_time.append(
            (
                obs[0].time_s,
                {team: sum(1 for p in obs[0].points if p.owner_team == team) for team in (0, 1)},
            )
        )

    return {
        "env": env,
        "winner": env.sim.state.winner,
        "duration_s": obs[0].time_s,
        "rewards": rewards,
        "issued": issued,
        "invalid": invalid,
        "points_by_time": points_by_time,
        "reinforce_interruptions": reinforce_interruptions,
    }


def invalid_rate(result: dict, player_id: int) -> float:
    issued = result["issued"][player_id]
    return result["invalid"][player_id] / issued if issued else 0.0


# ---------------------------------------------------------------------------
# T0
# ---------------------------------------------------------------------------


def test_t0_issues_nothing(match_map):
    data = fixture_data()
    env = CohEnv(map_name=MAP_NAME, players=list(PLAYERS), data=data, game_map=match_map)
    obs = env.reset()
    bot = T0Idle()
    bot.reset(1, match_map, data)
    assert bot.act(obs[1]) == []


def test_agent_registry_covers_the_ladder():
    assert AGENTS == {"t0": T0Idle, "t1": T1Capper}


# ---------------------------------------------------------------------------
# T1 vs T0
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t1_vs_t0(match_map):
    return play_match(match_map, "t1", "t0")


def test_t1_beats_t0_before_the_time_limit(t1_vs_t0):
    assert t1_vs_t0["winner"] == 0
    assert t1_vs_t0["rewards"] == {0: 1.0, 1: -1.0}
    assert t1_vs_t0["duration_s"] < TIME_LIMIT_S


def test_t1_holds_six_points_by_minute_eight(t1_vs_t0):
    early = [held for time_s, held in t1_vs_t0["points_by_time"] if time_s <= 8 * 60]
    assert max(held[0] for held in early) >= MIN_POINTS_BY_MINUTE_8


def test_t1_issues_orders_and_almost_none_are_invalid(t1_vs_t0):
    assert t1_vs_t0["issued"][0] > 0
    assert invalid_rate(t1_vs_t0, 0) <= MAX_INVALID_ORDER_RATE
    assert t1_vs_t0["issued"][1] == 0  # T0 never orders anything


def test_t1_builds_an_army_and_takes_ground(t1_vs_t0):
    env = t1_vs_t0["env"]
    squads = [s for s in env.sim.state.squads.values() if s.owner == 0 and s.alive_members]
    assert len(squads) > 1, "T1 should have trained reinforcements"
    held = t1_vs_t0["points_by_time"][-1][1]
    assert held[0] > held[1]


# ---------------------------------------------------------------------------
# T1 vs T1
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t1_mirror_matches_finish_cleanly(match_map, seed):
    result = play_match(match_map, "t1", "t1", seed=seed)
    assert result["winner"] is not None
    assert result["duration_s"] <= TIME_LIMIT_S
    for player_id in (0, 1):
        assert result["issued"][player_id] > 0
        assert invalid_rate(result, player_id) <= MAX_INVALID_ORDER_RATE


def test_t1_never_interrupts_a_reinforcing_squad(match_map):
    result = play_match(match_map, "t1", "t1", seed=2)
    assert result["reinforce_interruptions"] == []


def test_t1_is_deterministic_for_a_given_seed(match_map):
    first = play_match(match_map, "t1", "t1", seed=5)
    second = play_match(match_map, "t1", "t1", seed=5)
    assert first["env"].sim.state_hash() == second["env"].sim.state_hash()
    assert first["env"].order_log == second["env"].order_log


# ---------------------------------------------------------------------------
# dependency rule
# ---------------------------------------------------------------------------


def test_agents_never_import_the_sim():
    """`coh/agents` may only see `coh.env` (plus the public map/data types)."""
    allowed_roots = {"coh.env", "coh.agents", "coh.data.schema", "coh.maps.format"}
    offenders: list[str] = []
    for path in sorted(Path(coh.agents.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                root = ".".join(module.split(".")[:2])
                if module.startswith("coh.") and module not in allowed_roots and root != "coh.agents":
                    offenders.append(f"{path}: {module}")
    assert offenders == []


# ---------------------------------------------------------------------------
# the match script
# ---------------------------------------------------------------------------


def test_play_match_script_runs_and_writes_a_replay(tmp_path):
    repo_root = Path(coh.agents.__file__).parents[2]
    out = tmp_path / "match.replay.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "play_match.py"),
            "--map", MAP_NAME,
            "--p0", "t1",
            "--p1", "t0",
            "--seed", "0",
            "--time-limit", "60",
            "--data-dir", str(FIXTURES_DIR),
            "--out", str(out),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "winner" in proc.stdout
    assert "duration" in proc.stdout
    assert out.exists()
