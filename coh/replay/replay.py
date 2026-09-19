"""Replays: the order log plus everything needed to replay it bit-for-bit.

A replay stores no frames. The `Sim` is deterministic given
`(map, players, data, seed)` and the sequence of `issue()` / `tick()` calls,
so the map name, the player setups, the seed, the config and the timestamped
order log are enough to reproduce the match exactly — which is what
`resimulate` does, yielding the `Sim` after every tick so a viewer (or a
test) can walk the whole game.

JSON shape (`version` 1)::

    {"version": 1, "map_name": "hedgerow_crossing",
     "players": [{"faction": "us", "team": 0, "start_slot": 0}, ...],
     "seed": 0, "decision_interval_s": 2.0, "config": {"time_limit_s": 600.0},
     "final_tick": 4800,
     "orders": [[tick, player_id, {"type": "Move", ...}], ...],
     "final_hash": "…", "data_dir": "tests/data/fixtures"}

`final_tick` records where the recording stopped: a match cut short before a
winner emerged has no other way to say how far to replay. `data_dir` is
optional and only a hint — it names the stat tables the match was played
with so a replay of a fixture match can find them again; `resimulate(data=…)`
always wins over it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from coh.data.loader import load_game_data
from coh.data.schema import GameData
from coh.maps.format import GameMap, load_map
from coh.sim.orders import order_from_dict
from coh.sim.sim import PlayerSetup, Sim, SimConfig, neutral_footprints

REPLAY_VERSION = 1


@dataclass
class Replay:
    map_name: str
    players: list[PlayerSetup]
    seed: int
    decision_interval_s: float
    config: SimConfig
    orders: list[tuple[int, int, dict]]
    final_hash: str
    final_tick: int = 0
    data_dir: str | None = None
    version: int = field(default=REPLAY_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "map_name": self.map_name,
            "players": [
                {"faction": p.faction, "team": p.team, "start_slot": p.start_slot} for p in self.players
            ],
            "seed": self.seed,
            "decision_interval_s": self.decision_interval_s,
            "config": {"time_limit_s": self.config.time_limit_s},
            "final_tick": self.final_tick,
            "orders": [[tick, player_id, order] for tick, player_id, order in self.orders],
            "final_hash": self.final_hash,
            "data_dir": self.data_dir,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Replay":
        version = d.get("version")
        if version != REPLAY_VERSION:
            raise ValueError(f"unsupported replay version {version!r} (this build reads {REPLAY_VERSION})")
        return cls(
            map_name=d["map_name"],
            players=[
                PlayerSetup(faction=p["faction"], team=p["team"], start_slot=p["start_slot"])
                for p in d["players"]
            ],
            seed=d["seed"],
            decision_interval_s=d["decision_interval_s"],
            config=SimConfig(**d.get("config", {})),
            orders=[(int(tick), int(player_id), order) for tick, player_id, order in d["orders"]],
            final_hash=d["final_hash"],
            final_tick=int(d.get("final_tick", 0)),
            data_dir=d.get("data_dir"),
        )


def save(replay: Replay, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(replay.to_dict(), f, indent=1, sort_keys=False)
        f.write("\n")


def load(path: str | Path) -> Replay:
    with Path(path).open() as f:
        return Replay.from_dict(json.load(f))


def _game_data(replay: Replay, data: GameData | None) -> GameData:
    if data is not None:
        return data
    if replay.data_dir:
        return load_game_data(Path(replay.data_dir))
    return load_game_data()


def resimulate(
    replay: Replay, data: GameData | None = None, game_map: GameMap | None = None
) -> Iterator[Sim]:
    """Replay the match, yielding the `Sim` after every tick.

    Orders are issued at the tick they were logged at, in log order (which is
    the order `CohEnv` issued them in), before that tick runs. The generator
    stops when the game ends or `replay.final_tick` is reached.
    """
    data = _game_data(replay, data)
    if game_map is None:
        game_map = load_map(replay.map_name, footprints=neutral_footprints(data))

    by_tick: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for tick, player_id, order in replay.orders:
        by_tick[tick].append((player_id, order))

    sim = Sim(
        game_map=game_map,
        players=replay.players,
        data=data,
        seed=replay.seed,
        config=replay.config,
    )
    while sim.state.winner is None and sim.state.tick < replay.final_tick:
        for player_id, order in by_tick.get(sim.state.tick, ()):
            sim.issue(player_id, [order_from_dict(order)])
        sim.tick()
        yield sim


def replay_final_hash(
    replay: Replay, data: GameData | None = None, game_map: GameMap | None = None
) -> str:
    """Run `resimulate` to the end and return the resulting `state_hash`.

    With an empty order log and a zero `final_tick` the replay is the opening
    state, so the sim is built once and hashed directly.
    """
    sim = None
    for sim in resimulate(replay, data, game_map):
        pass
    if sim is not None:
        return sim.state_hash()

    data = _game_data(replay, data)
    if game_map is None:
        game_map = load_map(replay.map_name, footprints=neutral_footprints(data))
    return Sim(
        game_map=game_map,
        players=replay.players,
        data=data,
        seed=replay.seed,
        config=replay.config,
    ).state_hash()
