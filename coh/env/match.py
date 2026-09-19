"""Play one whole match between scripted agents and report the result.

`scripts/play_match.py`, the ladder tests and the seat-fairness study all
need the same loop, and all of them need the same guarantee: **an agent may
not share mutable state with the simulation or with its opponent**. An agent
that stashed a reference to the `GameMap` it was handed could otherwise stamp
footprints into the very array the sim paths on, and one that mutated a
`SquadDef` would change the game for both sides. `run_match` therefore hands
every agent its own `GameMap.copy()` and its own deep copy of the stat tables.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

from coh.data.schema import GameData
from coh.env.env import CohEnv
from coh.maps.format import GameMap, load_map
from coh.sim.constants import DT
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints


class MatchAgent(Protocol):  # pragma: no cover - structural typing only
    def reset(self, player_id: int, map: GameMap, data: GameData) -> None: ...

    def act(self, obs: Any) -> list[Any]: ...


@dataclass(frozen=True)
class MatchResult:
    """Everything a caller has wanted to know about a finished match so far."""

    winner: int | None  # team id, -1 for a draw, None if the match never ended
    rewards: dict[int, float]
    tickets: dict[int, float]
    ticks: int
    duration_s: float
    orders_issued: dict[int, int]
    invalid_orders: dict[int, int]
    points_held: dict[int, int]
    env: CohEnv

    @property
    def invalid_order_rate(self) -> dict[int, float]:
        return {
            pid: (self.invalid_orders[pid] / self.orders_issued[pid] if self.orders_issued[pid] else 0.0)
            for pid in self.orders_issued
        }


def run_match(
    map_name: str,
    players: list[PlayerSetup],
    agents: list[MatchAgent],
    *,
    data: GameData,
    game_map: GameMap | None = None,
    seed: int = 0,
    decision_interval_s: float = 2.0,
    config: SimConfig = SimConfig(),
) -> MatchResult:
    """Play `agents` against each other to the end of the match."""
    if len(agents) != len(players):
        raise ValueError(f"run_match: {len(agents)} agents for {len(players)} players")
    if game_map is None:
        game_map = load_map(map_name, footprints=neutral_footprints(data))

    env = CohEnv(
        map_name=map_name,
        players=players,
        seed=seed,
        decision_interval_s=decision_interval_s,
        config=config,
        data=data,
        game_map=game_map,
    )
    obs = env.reset()
    for player_id, agent in enumerate(agents):
        agent.reset(player_id, game_map.copy(), copy.deepcopy(data))

    issued = {pid: 0 for pid in env.player_ids}
    invalid = {pid: 0 for pid in env.player_ids}
    rewards = {pid: 0.0 for pid in env.player_ids}
    done = False
    while not done:
        orders = {pid: agents[pid].act(obs[pid]) for pid in env.player_ids}
        for pid, player_orders in orders.items():
            issued[pid] += len(player_orders)
        obs, step_rewards, done, infos = env.step(orders)
        for pid, info in infos.items():
            invalid[pid] += info["invalid_orders"]
        for pid, reward in step_rewards.items():
            rewards[pid] += reward

    assert env.sim is not None
    return MatchResult(
        winner=env.sim.state.winner,
        rewards=rewards,
        tickets=dict(env.sim.state.tickets),
        ticks=env.sim.state.tick,
        duration_s=env.sim.state.tick * DT,
        orders_issued=issued,
        invalid_orders=invalid,
        points_held={
            team: sum(1 for p in env.sim.state.points.values() if p.owner_team == team)
            for team in sorted(env.sim.state.tickets)
        },
        env=env,
    )
