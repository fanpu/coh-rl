"""Benchmark support: synthetic sim states that are too expensive to reach by playing.

Nothing under `coh/sim` may import this package: it exists purely so
`scripts/bench_sim.py` and `tests/scenarios/test_perf.py` can build the *same*
mid/late-game state without duplicating it.
"""
