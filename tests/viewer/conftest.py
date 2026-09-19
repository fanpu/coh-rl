"""Replays the viewer tests build frames from, plus a hard per-test deadline."""

from __future__ import annotations

import contextlib
import signal
import threading

import pytest

from coh.agents import AGENTS
from coh.env import CohEnv
from coh.maps.format import load_map
from coh.replay.replay import Replay
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints
from tests.helpers import fixture_data

MAP_NAME = "hedgerow_crossing"
SHORT_MATCH_S = 30.0


def play(seconds: float, agent: str = "t1", seed: int = 0) -> Replay:
    """Play `seconds` of `hedgerow_crossing` on fixture data and return the replay."""
    data = fixture_data()
    game_map = load_map(MAP_NAME, footprints=neutral_footprints(data))
    players = [PlayerSetup("us", 0, 0), PlayerSetup("us", 1, 1)]
    env = CohEnv(
        map_name=MAP_NAME,
        players=players,
        seed=seed,
        decision_interval_s=2.0,
        config=SimConfig(time_limit_s=seconds),
        data=data,
        game_map=game_map,
    )
    obs = env.reset()
    agents = [AGENTS[agent](), AGENTS[agent]()]
    for player_id, bot in enumerate(agents):
        bot.reset(player_id, game_map, data)
    done = False
    while not done:
        orders = {pid: agents[pid].act(obs[pid]) for pid in env.player_ids}
        obs, _rewards, done, _infos = env.step(orders)
    return env.to_replay(data_dir="tests/data/fixtures")


@pytest.fixture(scope="session")
def short_replay() -> Replay:
    """A 30 s scripted match — long enough for captures, shots and training."""
    return play(SHORT_MATCH_S)


# ---------------------------------------------------------------------------
# Hard deadline
# ---------------------------------------------------------------------------

# Playwright's sync `evaluate()` has no timeout of its own: if the page's main
# thread wedges, the call blocks forever and takes the whole suite with it.
# This is the backstop — a test that blows its budget fails loudly instead of
# stalling. SIGALRM only fires on the main thread of a POSIX process, so on
# anything else the guard is a no-op and the per-call Playwright timeouts are
# the only protection.
_CAN_ALARM = hasattr(signal, "SIGALRM")


class DeadlineExceeded(AssertionError):
    pass


@contextlib.contextmanager
def deadline(seconds: float, what: str):
    """Fail the test if the body has not finished within `seconds`."""
    if not _CAN_ALARM or threading.current_thread() is not threading.main_thread():
        yield
        return

    def fire(signum, frame):
        raise DeadlineExceeded(f"{what} exceeded its {seconds:g}s deadline")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
