#!/usr/bin/env python
"""Measure simulation throughput, overall and per system.

Two modes:

    # a scripted T1-vs-T1 match on hedgerow_crossing (what `play_match.py` runs)
    uv run python scripts/bench_sim.py --data-dir tests/data/fixtures --seconds 600

    # the number that matters for RL: a mid/late-game army fighting flat out
    uv run python scripts/bench_sim.py --stress --data-dir tests/data/fixtures --seconds 120

Reports ticks/s, game-seconds per wall-second, and a per-system breakdown.
The breakdown is taken by wrapping each system module's `run` with
`perf_counter` from *here* -- `Sim.tick` looks `run` up on the module object
at call time, so the wrapper installed below is genuinely what the tick loop
calls, and no timing code ever lands inside `coh/sim`. `build_observation`
and the agents' `act` are timed the same way, separately, so the sim's own
cost can be read apart from the harness around it.

`--profile N` additionally runs the whole loop under cProfile and prints the
top N functions by cumulative time.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__ is None and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coh.agents import AGENTS  # noqa: E402
from coh.bench import stress  # noqa: E402
from coh.data.loader import load_game_data  # noqa: E402
from coh.env import observation as observation_mod  # noqa: E402
from coh.maps.format import load_map  # noqa: E402
from coh.sim import systems  # noqa: E402
from coh.sim.constants import DT, TICKS_PER_SECOND  # noqa: E402
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints  # noqa: E402

MAP_NAME = stress.MAP_NAME


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stress", action="store_true", help="bench the synthetic mid/late-game army instead of a bot match")
    parser.add_argument("--seconds", type=float, default=600.0, help="game seconds to simulate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=None, help="stat tables directory (default: packaged tables)")
    parser.add_argument("--squads", type=int, default=stress.StressConfig().squads_per_side, help="--stress: squads per side")
    parser.add_argument("--decision-interval", type=float, default=2.0, help="match mode: seconds of game time per agent decision")
    parser.add_argument("--profile", type=int, default=0, metavar="N", help="also run under cProfile and print the top N functions")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


class Timers:
    """Named wall-clock accumulators, in first-seen order."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - started)

    def wrap(self, name: str, func):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - started)

        return timed


@contextmanager
def instrumented(timers: Timers):
    """Wrap every tick system's `run` (and `build_observation`) with a timer.

    `Sim.tick` resolves `system.run` on the module each tick, so replacing the
    module attribute is enough; everything is restored on the way out.
    """
    patched: list[tuple[object, str, object]] = []
    for module in systems.SYSTEM_ORDER:
        name = module.__name__.rsplit(".", 1)[-1]
        patched.append((module, "run", module.run))
        module.run = timers.wrap(f"system:{name}", module.run)

    from coh.env import env as env_mod

    patched.append((observation_mod, "build_observation", observation_mod.build_observation))
    patched.append((env_mod, "build_observation", env_mod.build_observation))
    wrapped_obs = timers.wrap("build_observation", observation_mod.build_observation)
    observation_mod.build_observation = wrapped_obs
    env_mod.build_observation = wrapped_obs

    try:
        yield
    finally:
        for target, attr, original in patched:
            setattr(target, attr, original)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_match(args: argparse.Namespace, data, timers: Timers) -> dict:
    """T1 vs T1 (us vs wehr) on hedgerow_crossing, capped at `--seconds`."""
    from coh.env import CohEnv

    game_map = load_map(MAP_NAME, footprints=neutral_footprints(data))
    env = CohEnv(
        map_name=MAP_NAME,
        players=[
            PlayerSetup(faction="us", team=0, start_slot=0),
            PlayerSetup(faction="wehr", team=1, start_slot=1),
        ],
        seed=args.seed,
        decision_interval_s=args.decision_interval,
        config=SimConfig(time_limit_s=args.seconds),
        data=data,
        game_map=game_map,
    )
    agents = [AGENTS["t1"](), AGENTS["t1"]()]
    obs = env.reset()
    for player_id, agent in enumerate(agents):
        agent.reset(player_id, game_map, data)

    started = time.perf_counter()
    done = False
    while not done:
        with timers.measure("agents"):
            orders = {pid: agents[pid].act(obs[pid]) for pid in env.player_ids}
        obs, _rewards, done, _infos = env.step(orders)
    wall_s = time.perf_counter() - started

    assert env.sim is not None
    return {
        "label": f"match (T1 vs T1, {MAP_NAME}, seed {args.seed})",
        "ticks": env.sim.state.tick,
        "wall_s": wall_s,
        "squads": len(env.sim.state.squads),
        "winner": env.sim.state.winner,
    }


def run_stress(args: argparse.Namespace, data, timers: Timers) -> dict:
    """The synthetic army, ticked for `--seconds` with periodic orders."""
    config = stress.StressConfig(squads_per_side=args.squads)
    state = stress.build_stress_sim(data, seed=args.seed, config=config)
    sim = state.sim
    ticks = int(round(args.seconds * TICKS_PER_SECOND))

    started = time.perf_counter()
    for _ in range(ticks):
        orders = stress.orders_for_tick(state, sim.state.tick, TICKS_PER_SECOND)
        for player_id, player_orders in sorted(orders.items()):
            with timers.measure("orders"):
                sim.issue(player_id, player_orders)
        sim.tick()
    wall_s = time.perf_counter() - started

    return {
        "label": f"stress ({config.squads_per_side} squads/side, {MAP_NAME}, seed {args.seed})",
        "ticks": sim.state.tick,
        "wall_s": wall_s,
        "squads": sum(1 for s in sim.state.squads.values() if s.alive_members),
        "winner": sim.state.winner,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(result: dict, timers: Timers) -> None:
    ticks, wall_s = result["ticks"], result["wall_s"]
    ticks_per_s = ticks / wall_s if wall_s > 0 else float("inf")
    print(result["label"])
    print(f"  ticks          {ticks} ({ticks * DT:.1f}s of game time)")
    print(f"  wall clock     {wall_s:.2f}s")
    print(f"  throughput     {ticks_per_s:.0f} ticks/s   ({ticks_per_s * DT:.1f} game-s per wall-s)")
    print(f"  live squads    {result['squads']} at the end")
    if result["winner"] is not None:
        print(f"  winner         {result['winner']} (the match ended early)")

    if not timers.totals:
        return
    print("\n  breakdown                 wall s     %     us/tick")
    accounted = 0.0
    for name, seconds in sorted(timers.totals.items(), key=lambda kv: -kv[1]):
        accounted += seconds
        share = 100.0 * seconds / wall_s if wall_s > 0 else 0.0
        print(f"  {name:<24}{seconds:7.2f}{share:7.1f}{1e6 * seconds / ticks:11.1f}")
    other = wall_s - accounted
    share = 100.0 * other / wall_s if wall_s > 0 else 0.0
    print(f"  {'(unaccounted)':<24}{other:7.2f}{share:7.1f}{1e6 * other / ticks:11.1f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_game_data(Path(args.data_dir)) if args.data_dir else load_game_data()
    mode = run_stress if args.stress else run_match

    timers = Timers()
    with instrumented(timers):
        result = mode(args, data, timers)
    report(result, timers)

    if args.profile:
        print(f"\ncProfile: top {args.profile} by cumulative time")
        profiler = cProfile.Profile()
        profiler.enable()
        mode(args, data, Timers())
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(args.profile)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
