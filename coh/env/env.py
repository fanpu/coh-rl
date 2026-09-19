"""`CohEnv` — a PettingZoo-style parallel environment around `Sim`.

One `step` issues every player's orders (players in ascending id order, so
two players racing for the same manpower resolve deterministically), then
advances `decision_interval_s` of game time.

The env also *is* the replay recorder: every order handed to `Sim.issue` is
appended to `order_log` as `(tick, player_id, order_dict)` — including the
invalid ones, so a replay reproduces the invalid-order counts exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coh.data.loader import load_game_data
from coh.data.schema import GameData
from coh.env.observation import Observation, build_observation
from coh.maps.format import GameMap, load_map
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import Order, OrderResult, order_to_dict
from coh.sim.sim import PlayerSetup, Sim, SimConfig, neutral_footprints

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.replay.replay import Replay

# A draw (`state.winner == -1`) pays no one.
DRAW = -1


class CohEnv:
    """A multi-agent environment over one `Sim`.

    `data=None` loads the packaged stat tables lazily on `reset()`; pass a
    `GameData` (and optionally a pre-loaded `GameMap`) to run against fixture
    tables instead.
    """

    def __init__(
        self,
        map_name: str,
        players: list[PlayerSetup],
        seed: int = 0,
        decision_interval_s: float = 2.0,
        config: SimConfig = SimConfig(),
        data: GameData | None = None,
        game_map: GameMap | None = None,
    ) -> None:
        if decision_interval_s <= 0:
            raise ValueError(f"decision_interval_s must be positive, got {decision_interval_s}")
        self.map_name = map_name
        self.players = list(players)
        self.seed = seed
        self.decision_interval_s = decision_interval_s
        self.config = config
        self._data = data
        self._map = game_map
        self.sim: Sim | None = None
        self.order_log: list[tuple[int, int, dict]] = []

    # -- properties -------------------------------------------------------

    @property
    def ticks_per_step(self) -> int:
        return round(self.decision_interval_s * TICKS_PER_SECOND)

    @property
    def player_ids(self) -> list[int]:
        return list(range(len(self.players)))

    @property
    def done(self) -> bool:
        return self.sim is not None and self.sim.state.winner is not None

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> dict[int, Observation]:
        """Build a fresh `Sim` and return the opening observations."""
        if self._data is None:
            self._data = load_game_data()
        data = self._data
        if self._map is None:
            self._map = load_map(self.map_name, footprints=neutral_footprints(data))
        self.sim = Sim(
            game_map=self._map,
            players=self.players,
            data=data,
            seed=self.seed,
            config=self.config,
        )
        self.order_log = []
        return self._observations()

    def step(
        self, orders: dict[int, list[Order]]
    ) -> tuple[dict[int, Observation], dict[int, float], bool, dict[int, dict]]:
        """Issue `orders`, advance one decision interval, and report back.

        Rewards are 0 until the game ends, then +1 / -1 / 0 for the winning /
        losing / drawing side. Once the game is over further steps are no-ops
        that still return the final observations and `done=True`.
        """
        if self.sim is None:
            raise RuntimeError("CohEnv.step called before reset()")

        infos: dict[int, dict] = {pid: {"invalid_orders": 0, "results": []} for pid in self.player_ids}
        if self.done:
            return self._observations(), self._rewards(), True, infos

        tick = self.sim.state.tick
        for player_id in sorted(orders):
            player_orders = orders[player_id]
            if not player_orders:
                continue
            results: list[OrderResult] = self.sim.issue(player_id, player_orders)
            for order in player_orders:
                self.order_log.append((tick, player_id, order_to_dict(order)))
            info = infos.setdefault(player_id, {"invalid_orders": 0, "results": []})
            info["results"] = results
            info["invalid_orders"] = sum(1 for result in results if not result.ok)

        for _ in range(self.ticks_per_step):
            if self.sim.state.winner is not None:
                break
            self.sim.tick()

        return self._observations(), self._rewards(), self.done, infos

    # -- replay -----------------------------------------------------------

    def to_replay(self, data_dir: str | None = None) -> "Replay":
        """Snapshot this env's match as a `Replay`.

        `data_dir` is recorded as a hint so a replay played on non-packaged
        stat tables (fixtures, a scraped set) can find them again.
        """
        from coh.replay.replay import Replay

        if self.sim is None:
            raise RuntimeError("CohEnv.to_replay called before reset()")
        return Replay(
            map_name=self.map_name,
            players=list(self.players),
            seed=self.seed,
            decision_interval_s=self.decision_interval_s,
            config=self.config,
            orders=list(self.order_log),
            final_hash=self.sim.state_hash(),
            final_tick=self.sim.state.tick,
            data_dir=data_dir,
        )

    # -- internals --------------------------------------------------------

    def _observations(self) -> dict[int, Observation]:
        assert self.sim is not None
        return {pid: build_observation(self.sim, pid) for pid in self.player_ids}

    def _rewards(self) -> dict[int, float]:
        assert self.sim is not None
        winner = self.sim.state.winner
        if winner is None:
            return {pid: 0.0 for pid in self.player_ids}
        if winner == DRAW:
            return {pid: 0.0 for pid in self.player_ids}
        return {pid: (1.0 if self.players[pid].team == winner else -1.0) for pid in self.player_ids}
