"""Seat fairness: the same bot, the same faction, both seats.

This is the benchmark's fairness floor. If one seat of `hedgerow_crossing`
wins the mirror far more often than the other, a score on this environment
measures which chair an agent sat in as much as how it played.

Sixteen matches (two factions x eight seeds) of T1 against itself. The bound
is deliberately generous -- 16 coin flips land 12-4 or worse about 8% of the
time by chance alone -- so the test fails on a structural advantage, not on
variance.

T1 is deterministic and both seats see a rotationally identical map, so the
mirror is nearly degenerate: all eight Wehrmacht seeds used to produce
byte-identical matches. `_DelayedStart` breaks that by holding each bot's
first orders for a seeded 0-10 s. That is a property of this *harness*, not
of the bot, and `test_the_mirror_is_not_degenerate` checks it still bites --
without it, this file would be asserting the same match sixteen times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from coh.agents import AGENTS
from coh.env import run_match
from coh.maps.format import load_map
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints
from tests.helpers import fixture_data

MAP_NAME = "hedgerow_crossing"
TIME_LIMIT_S = 1200.0
SEEDS = range(8)
FACTIONS = ("us", "wehr")
MAX_DELAY_S = 10.0
MAX_WINS_PER_SEAT = 12  # out of 16


class _DelayedStart:
    """Wraps an agent so its first orders come after a fixed pause."""

    def __init__(self, inner, delay_s: float) -> None:
        self.inner = inner
        self.delay_s = delay_s

    def reset(self, player_id, map, data):  # noqa: A002 - matches the Agent protocol
        self.inner.reset(player_id, map, data)

    def act(self, obs):
        return [] if obs.time_s < self.delay_s else self.inner.act(obs)


@dataclass(frozen=True)
class _Mirror:
    faction: str
    seed: int
    winner: int | None  # team id, -1 for a draw
    final_hash: str


@pytest.fixture(scope="module")
def data():
    return fixture_data()


@pytest.fixture(scope="module")
def match_map(data):
    return load_map(MAP_NAME, footprints=neutral_footprints(data))


def _mirror_match(match_map, data, *, faction: str, seed: int) -> _Mirror:
    rng = np.random.default_rng(seed)
    delays = [float(rng.uniform(0.0, MAX_DELAY_S)) for _ in (0, 1)]
    result = run_match(
        MAP_NAME,
        [PlayerSetup(faction=faction, team=slot, start_slot=slot) for slot in (0, 1)],
        [_DelayedStart(AGENTS["t1"](), delays[slot]) for slot in (0, 1)],
        data=data,
        game_map=match_map,
        seed=seed,
        config=SimConfig(time_limit_s=TIME_LIMIT_S),
    )
    assert result.env.sim is not None
    return _Mirror(faction, seed, result.winner, result.env.sim.state_hash())


@pytest.fixture(scope="module")
def mirrors(match_map, data) -> list[_Mirror]:
    """The whole study, run once: 2 factions x 8 seeds of T1 against itself."""
    return [
        _mirror_match(match_map, data, faction=faction, seed=seed)
        for faction in FACTIONS
        for seed in SEEDS
    ]


@pytest.mark.slow
def test_neither_seat_wins_the_mirror_much_more_often(mirrors):
    wins = {0: 0, 1: 0}
    for mirror in mirrors:
        if mirror.winner in wins:
            wins[mirror.winner] += 1
    draws = len(mirrors) - wins[0] - wins[1]

    split = f"seat 0 {wins[0]}, seat 1 {wins[1]}, draws {draws} (of {len(mirrors)})"
    assert max(wins.values()) <= MAX_WINS_PER_SEAT, f"{split}: {mirrors}"


@pytest.mark.slow
def test_the_mirror_is_not_degenerate(mirrors):
    """Different seeds must actually produce different matches."""
    for faction in FACTIONS:
        finals = {m.final_hash for m in mirrors if m.faction == faction}
        assert len(finals) > 1, f"every {faction} mirror ended in the identical state"
