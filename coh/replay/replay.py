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
     "final_hash": "…", "data_dir": "tests/data/fixtures",
     "data_hash": "…", "map_hash": "…"}

`final_tick` records where the recording stopped: a match cut short before a
winner emerged has no other way to say how far to replay. `data_dir` is
optional and only a hint — it names the stat tables the match was played
with so a replay of a fixture match can find them again; `resimulate(data=…)`
always wins over it.

`data_hash` / `map_hash` pin the *identity* of those inputs: the order log
only reproduces the match against the stat tables and the map it was recorded
on, and silently replaying it against a re-scraped table set would produce a
plausible-looking lie. `resimulate(..., verify=True)` (the default) checks
them before the first tick and checks `final_hash` after the last one, and
raises `ReplayMismatchError` on either. Both fields are optional, so replays
recorded before they existed still load and still replay — unverified.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from coh.data.hashing import data_hash
from coh.data.loader import load_game_data
from coh.data.schema import GameData
from coh.maps.format import GameMap, load_map
from coh.maps.hashing import map_hash
from coh.sim.orders import order_from_dict
from coh.sim.sim import PlayerSetup, Sim, SimConfig, neutral_footprints

REPLAY_VERSION = 1


class ReplayError(ValueError):
    """A replay file (or one of its entries) is malformed or unplayable."""


class ReplayMismatchError(ReplayError):
    """The replay does not reproduce: wrong tables, wrong map, or divergence."""


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
    data_hash: str | None = None
    map_hash: str | None = None
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
            "config": asdict(self.config),
            "final_tick": self.final_tick,
            "orders": [[tick, player_id, order] for tick, player_id, order in self.orders],
            "final_hash": self.final_hash,
            "data_dir": self.data_dir,
            "data_hash": self.data_hash,
            "map_hash": self.map_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Replay":
        if not isinstance(d, dict):
            raise ReplayError(f"replay must be a JSON object, got {type(d).__name__}")
        version = d.get("version")
        if version != REPLAY_VERSION:
            raise ReplayError(f"unsupported replay version {version!r} (this build reads {REPLAY_VERSION})")
        return cls(
            map_name=_require(d, "map_name", str),
            players=[_player(entry, index) for index, entry in enumerate(_require(d, "players", list))],
            seed=_require(d, "seed", int),
            decision_interval_s=float(_require(d, "decision_interval_s", (int, float))),
            config=_config(d.get("config")),
            orders=[_order_entry(entry, index) for index, entry in enumerate(_require(d, "orders", list))],
            final_hash=_require(d, "final_hash", str),
            final_tick=int(d.get("final_tick", 0)),
            data_dir=d.get("data_dir"),
            data_hash=d.get("data_hash"),
            map_hash=d.get("map_hash"),
        )


def _require(d: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in d:
        raise ReplayError(f"replay is missing required key {key!r}")
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        expected = kind.__name__ if isinstance(kind, type) else " or ".join(k.__name__ for k in kind)
        raise ReplayError(f"replay key {key!r} must be {expected}, got {value!r}")
    return value


def _player(entry: Any, index: int) -> PlayerSetup:
    if not isinstance(entry, dict):
        raise ReplayError(f"replay players[{index}] must be an object, got {entry!r}")
    try:
        return PlayerSetup(
            faction=str(entry["faction"]), team=int(entry["team"]), start_slot=int(entry["start_slot"])
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ReplayError(f"replay players[{index}] is malformed ({entry!r}): {e}") from e


def _config(raw: Any) -> SimConfig:
    if raw is None:
        return SimConfig()
    if not isinstance(raw, dict):
        raise ReplayError(f"replay key 'config' must be an object, got {raw!r}")
    try:
        return SimConfig(**raw)
    except TypeError as e:
        raise ReplayError(f"replay key 'config' is malformed ({raw!r}): {e}") from e


def _order_entry(entry: Any, index: int) -> tuple[int, int, dict]:
    if not isinstance(entry, (list, tuple)) or len(entry) != 3:
        raise ReplayError(f"replay orders[{index}] must be [tick, player_id, order], got {entry!r}")
    tick, player_id, order = entry
    if not isinstance(order, dict):
        raise ReplayError(f"replay orders[{index}] order must be an object, got {order!r}")
    try:
        return (int(tick), int(player_id), order)
    except (TypeError, ValueError) as e:
        raise ReplayError(f"replay orders[{index}] has a non-numeric tick/player id ({entry!r}): {e}") from e


def save(replay: Replay, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(replay.to_dict(), f, indent=1, sort_keys=False)
        f.write("\n")


def load(path: str | Path) -> Replay:
    path = Path(path)
    try:
        with path.open() as f:
            blob = json.load(f)
    except json.JSONDecodeError as e:
        raise ReplayError(f"{path}: not valid JSON: {e}") from e
    except OSError as e:
        raise ReplayError(f"{path}: cannot be read: {e}") from e
    try:
        return Replay.from_dict(blob)
    except ReplayError as e:
        raise ReplayError(f"{path}: {e}") from e


def _game_data(replay: Replay, data: GameData | None) -> GameData:
    if data is not None:
        return data
    if replay.data_dir:
        return load_game_data(Path(replay.data_dir))
    return load_game_data()


def _check_inputs(replay: Replay, data: GameData, game_map: GameMap) -> None:
    """Refuse to replay against tables or a map the match was not played on."""
    if replay.data_hash is not None:
        actual = data_hash(data)
        if actual != replay.data_hash:
            raise ReplayMismatchError(
                f"replay was recorded on different data tables: recorded data_hash "
                f"{replay.data_hash[:12]}…, supplied {actual[:12]}…"
            )
    if replay.map_hash is not None:
        actual = map_hash(game_map)
        if actual != replay.map_hash:
            raise ReplayMismatchError(
                f"replay was recorded on a different map ({replay.map_name!r}): recorded map_hash "
                f"{replay.map_hash[:12]}…, supplied {actual[:12]}…"
            )


def resimulate(
    replay: Replay,
    data: GameData | None = None,
    game_map: GameMap | None = None,
    verify: bool = True,
) -> Iterator[Sim]:
    """Replay the match, yielding the `Sim` after every tick.

    Orders are issued at the tick they were logged at, in log order (which is
    the order `CohEnv` issued them in), before that tick runs. The generator
    stops when the game ends or `replay.final_tick` is reached.

    With `verify` (the default), the recorded `data_hash` / `map_hash` are
    checked *before* the first tick -- so a mismatch raises from the call
    itself, not halfway through the stream -- and `final_hash` is checked
    after the last one. Either failure raises `ReplayMismatchError`.
    """
    data = _game_data(replay, data)
    if game_map is None:
        game_map = load_map(replay.map_name, footprints=neutral_footprints(data))
    if verify:
        _check_inputs(replay, data, game_map)
    return _resimulate(replay, data, game_map, verify)


def _resimulate(replay: Replay, data: GameData, game_map: GameMap, verify: bool) -> Iterator[Sim]:
    by_tick: dict[int, list[tuple[int, int, dict]]] = defaultdict(list)
    for index, (tick, player_id, order) in enumerate(replay.orders):
        by_tick[tick].append((index, player_id, order))

    sim = Sim(
        game_map=game_map,
        players=replay.players,
        data=data,
        seed=replay.seed,
        config=replay.config,
    )
    while sim.state.winner is None and sim.state.tick < replay.final_tick:
        for index, player_id, order in by_tick.get(sim.state.tick, ()):
            try:
                parsed = order_from_dict(order)
            except (ValueError, TypeError, KeyError) as e:
                raise ReplayError(f"replay orders[{index}] is not a playable order ({order!r}): {e}") from e
            sim.issue(player_id, [parsed])
        sim.tick()
        yield sim

    if verify:
        actual = sim.state_hash()
        if actual != replay.final_hash:
            raise ReplayMismatchError(
                f"replay diverged: final state hash {actual[:12]}… does not match the "
                f"recorded {replay.final_hash[:12]}… at tick {sim.state.tick}"
            )


def replay_final_hash(
    replay: Replay,
    data: GameData | None = None,
    game_map: GameMap | None = None,
    verify: bool = False,
) -> str:
    """Run `resimulate` to the end and return the resulting `state_hash`.

    Verification is *off* by default here: this function's job is to report
    the hash the replay actually produces, which is exactly what a caller
    comparing it against `replay.final_hash` wants to see.

    With an empty order log and a zero `final_tick` the replay is the opening
    state, so the sim is built once and hashed directly.
    """
    sim = None
    for sim in resimulate(replay, data, game_map, verify=verify):
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
