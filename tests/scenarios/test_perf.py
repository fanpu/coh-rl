"""A wall-clock floor under simulation throughput.

Runs the same state `scripts/bench_sim.py --stress` builds (via
`coh.bench.stress`, so the two can never drift apart), shortened to a few
game-seconds, and asserts a *loose* floor: the bench measures ~1000 ticks/s
on a developer machine with the full 60-squad army, so 300 ticks/s leaves
plenty of room for a loaded CI box while still failing loudly if someone
puts an order of magnitude back into the tick loop.

Marked `perf` and `slow`: it asserts on wall-clock, and the setup alone
(loading the map and deploying 60 squads) is not millisecond work.
"""

from __future__ import annotations

import time

import pytest

from coh.bench import stress
from coh.sim.constants import TICKS_PER_SECOND
from tests.helpers import fixture_data

MIN_TICKS_PER_S = 300.0
WARMUP_TICKS = TICKS_PER_SECOND  # one game-second: fills the vision mask cache
MEASURED_TICKS = TICKS_PER_SECOND * 6


@pytest.mark.perf
@pytest.mark.slow
def test_stress_state_sustains_the_throughput_floor():
    state = stress.build_stress_sim(fixture_data(), seed=0)
    sim = state.sim

    def step() -> None:
        for player_id, orders in sorted(stress.orders_for_tick(state, sim.state.tick, TICKS_PER_SECOND).items()):
            sim.issue(player_id, orders)
        sim.tick()

    for _ in range(WARMUP_TICKS):
        step()

    started = time.perf_counter()
    for _ in range(MEASURED_TICKS):
        step()
    elapsed = time.perf_counter() - started

    ticks_per_s = MEASURED_TICKS / elapsed
    # Both sides should still be fighting: a floor measured on an empty map
    # would not be measuring anything.
    alive = sum(1 for squad in sim.state.squads.values() if squad.has_alive_members)
    assert alive >= 40, f"only {alive} squads left; the stress state is not loaded"
    assert ticks_per_s >= MIN_TICKS_PER_S, f"{ticks_per_s:.0f} ticks/s < {MIN_TICKS_PER_S:.0f}"
