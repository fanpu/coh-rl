#!/usr/bin/env python
"""Play one scripted match and (optionally) write its replay.

    uv run python scripts/play_match.py --map hedgerow_crossing \
        --p0 t1 --p1 t0 --seed 0 --data-dir tests/data/fixtures \
        --out match.replay.json

Prints the winner, the ticket counts, the duration and the per-player
invalid-order rate. `--data-dir` defaults to the packaged stat tables under
`coh/data/tables/`, which a later task populates; until then point it at a
fixture directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coh.agents import AGENTS  # noqa: E402
from coh.data.loader import load_game_data  # noqa: E402
from coh.env import CohEnv  # noqa: E402
from coh.maps.format import load_map  # noqa: E402
from coh.replay.replay import save  # noqa: E402
from coh.sim.constants import DT  # noqa: E402
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", default="hedgerow_crossing", help="map name or path")
    parser.add_argument("--p0", default="t1", choices=sorted(AGENTS), help="player 0's agent")
    parser.add_argument("--p1", default="t0", choices=sorted(AGENTS), help="player 1's agent")
    parser.add_argument("--f0", default="us", help="player 0's faction")
    parser.add_argument("--f1", default="wehr", help="player 1's faction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decision-interval", type=float, default=2.0, help="seconds of game time per step")
    parser.add_argument("--time-limit", type=float, default=SimConfig().time_limit_s, help="seconds")
    parser.add_argument("--data-dir", default=None, help="stat tables directory (default: packaged tables)")
    parser.add_argument("--out", default=None, help="write the replay JSON here")
    return parser.parse_args(argv)


def run_match(args: argparse.Namespace) -> dict:
    """Play the match and return a summary dict (also used by the tests)."""
    data = load_game_data(Path(args.data_dir)) if args.data_dir else load_game_data()
    game_map = load_map(args.map, footprints=neutral_footprints(data))
    players = [
        PlayerSetup(faction=args.f0, team=0, start_slot=0),
        PlayerSetup(faction=args.f1, team=1, start_slot=1),
    ]
    env = CohEnv(
        map_name=args.map,
        players=players,
        seed=args.seed,
        decision_interval_s=args.decision_interval,
        config=SimConfig(time_limit_s=args.time_limit),
        data=data,
        game_map=game_map,
    )
    agents = [AGENTS[args.p0](), AGENTS[args.p1]()]

    obs = env.reset()
    for player_id, agent in enumerate(agents):
        agent.reset(player_id, game_map, data)

    issued = {pid: 0 for pid in env.player_ids}
    invalid = {pid: 0 for pid in env.player_ids}
    started = time.perf_counter()
    done = False
    while not done:
        orders = {pid: agents[pid].act(obs[pid]) for pid in env.player_ids}
        for pid, player_orders in orders.items():
            issued[pid] += len(player_orders)
        obs, rewards, done, infos = env.step(orders)
        for pid, info in infos.items():
            invalid[pid] += info["invalid_orders"]
    wall_s = time.perf_counter() - started

    assert env.sim is not None
    summary = {
        "winner_team": env.sim.state.winner,
        "tickets": dict(env.sim.state.tickets),
        "duration_s": env.sim.state.tick * DT,
        "ticks": env.sim.state.tick,
        "wall_s": wall_s,
        "ticks_per_s": env.sim.state.tick / wall_s if wall_s > 0 else float("inf"),
        "rewards": rewards,
        "orders_issued": issued,
        "invalid_orders": invalid,
        "points_held": {
            team: sum(1 for p in env.sim.state.points.values() if p.owner_team == team)
            for team in sorted(env.sim.state.tickets)
        },
        "env": env,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_match(args)
    env = summary["env"]

    winner = summary["winner_team"]
    print(f"map          {args.map}  seed {args.seed}")
    print(f"agents       p0={args.p0} ({args.f0})  p1={args.p1} ({args.f1})")
    print(f"winner       {'draw' if winner == -1 else f'team {winner}' if winner is not None else 'unfinished'}")
    print(f"tickets      {summary['tickets']}")
    print(f"points held  {summary['points_held']}")
    print(f"duration     {summary['duration_s']:.1f}s of game time ({summary['ticks']} ticks)")
    print(f"wall clock   {summary['wall_s']:.2f}s ({summary['ticks_per_s']:.0f} ticks/s)")
    for pid in env.player_ids:
        total = summary["orders_issued"][pid]
        bad = summary["invalid_orders"][pid]
        rate = 100.0 * bad / total if total else 0.0
        print(f"player {pid}     {total} orders, {bad} invalid ({rate:.1f}%)")

    if args.out:
        save(env.to_replay(data_dir=args.data_dir), args.out)
        print(f"replay       {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
