"""`CohEnv` — a PettingZoo-style parallel environment around `Sim`.

One `step` issues every player's orders, then advances `decision_interval_s`
of game time. Orders within a step are issued one player at a time — two
players racing for the last of a shared resource cannot both win — and *who
goes first rotates by step index*, so the same seat does not take every race
in a match. The rotation is a pure function of the step count, so it stays
deterministic.

The env also *is* the replay recorder: every order handed to `Sim.issue` is
appended to `order_log` as `(tick, player_id, order_dict)` in the order it
was actually issued — including the invalid ones, so a replay reproduces the
invalid-order counts exactly.

Rewards are terminal and paid once: the step on which the game ends pays
+1 / -1 / 0, and every later (no-op) step pays 0. An episode's return is
therefore exactly the match result, however long the caller keeps stepping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coh.data.hashing import data_hash
from coh.data.loader import load_game_data
from coh.data.schema import GameData
from coh.env.observation import Observation, ObservationMemory, build_observation
from coh.maps.format import GameMap, load_map
from coh.maps.hashing import map_hash
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import Order, OrderResult, normalized, order_from_dict, order_to_dict, payload_problem
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
        self.observation_memory = ObservationMemory()
        self.step_count = 0
        self._terminal_reward_paid = False

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
        self.observation_memory = ObservationMemory()
        self.step_count = 0
        self._terminal_reward_paid = False
        return self._observations()

    def step(
        self, orders: dict[int, list[Order]]
    ) -> tuple[dict[int, Observation], dict[int, float], bool, dict[int, dict]]:
        """Issue `orders`, advance one decision interval, and report back.

        Rewards are 0 until the game ends; the step that ends it pays
        +1 / -1 / 0 for the winning / losing / drawing side, and steps after
        that are no-ops paying 0 while still returning the final observations
        and `done=True`.

        An agent's list may contain anything: entries that are not `Order`s
        (a dict is parsed, anything else is not) are counted as invalid orders
        and skipped rather than raising.
        """
        if self.sim is None:
            raise RuntimeError("CohEnv.step called before reset()")

        infos: dict[int, dict] = {pid: {"invalid_orders": 0, "results": []} for pid in self.player_ids}
        if self.done:
            return self._observations(), self._zero_rewards(), True, infos

        tick = self.sim.state.tick
        for player_id in self._issue_sequence():
            player_orders = orders.get(player_id) or []
            if not player_orders:
                continue
            parsed = [_as_order(entry) for entry in player_orders]
            issuable = [order for order in parsed if order is not None]
            issued = iter(self.sim.issue(player_id, issuable))

            results: list[OrderResult] = []
            for entry, order in zip(player_orders, parsed):
                if order is None:
                    results.append(OrderResult(ok=False, reason=f"not an order: {entry!r}"))
                    continue
                results.append(next(issued))
                # Log the normalized order (plain Python types: a numpy-typed
                # Move(cell=(np.int64(4), np.int64(5))) must not put an
                # np.int64 in a JSON-bound log). An order whose payload does
                # not even have the right shape after normalizing (e.g. a
                # non-numeric squad id) is not JSON-safe either way, so it is
                # counted invalid above but left out of the log rather than
                # poisoning it.
                canonical = normalized(order)
                if not payload_problem(canonical):
                    self.order_log.append((tick, player_id, order_to_dict(canonical)))

            info = infos.setdefault(player_id, {"invalid_orders": 0, "results": []})
            info["results"] = results
            info["invalid_orders"] = sum(1 for result in results if not result.ok)

        for _ in range(self.ticks_per_step):
            if self.sim.state.winner is not None:
                break
            self.sim.tick()

        self.step_count += 1
        rewards = self._zero_rewards()
        if self.done and not self._terminal_reward_paid:
            rewards = self._rewards()
            self._terminal_reward_paid = True
        return self._observations(), rewards, self.done, infos

    def _issue_sequence(self) -> list[int]:
        """Player ids in this step's issue order.

        Starts at `step_count % n_players` and runs cyclically upwards, so no
        seat is permanently first in line for a contested resource.
        """
        ids = self.player_ids
        if not ids:
            return ids
        start = self.step_count % len(ids)
        return [ids[(start + offset) % len(ids)] for offset in range(len(ids))]

    # -- replay -----------------------------------------------------------

    def to_replay(self, data_dir: str | None = None) -> "Replay":
        """Snapshot this env's match as a `Replay`.

        `data_dir` is recorded as a hint so a replay played on non-packaged
        stat tables (fixtures, a scraped set) can find them again; the
        recorded `data_hash` / `map_hash` then pin down *which* tables and
        map, so a re-simulation on the wrong ones fails loudly.
        """
        from coh.replay.replay import Replay

        if self.sim is None:
            raise RuntimeError("CohEnv.to_replay called before reset()")
        assert self._data is not None and self._map is not None  # set by reset()
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
            data_hash=data_hash(self._data),
            map_hash=map_hash(self._map),
        )

    # -- internals --------------------------------------------------------

    def _observations(self) -> dict[int, Observation]:
        assert self.sim is not None
        return {pid: build_observation(self.sim, pid, self.observation_memory) for pid in self.player_ids}

    def _zero_rewards(self) -> dict[int, float]:
        return {pid: 0.0 for pid in self.player_ids}

    def _rewards(self) -> dict[int, float]:
        assert self.sim is not None
        winner = self.sim.state.winner
        if winner is None or winner == DRAW:
            return self._zero_rewards()
        return {pid: (1.0 if self.players[pid].team == winner else -1.0) for pid in self.player_ids}


def _as_order(entry: Any) -> Order | None:
    """An agent's list entry as an `Order`, or None if it is not one.

    Agent output is untrusted: a dict is given the benefit of the doubt (it is
    what `order_to_dict` produces, and what a text/LLM adapter emits), and
    anything else -- or a dict that will not parse -- is simply not an order.
    """
    if isinstance(entry, Order):
        return entry
    if isinstance(entry, dict):
        try:
            return order_from_dict(entry)
        except (ValueError, TypeError, KeyError):
            return None
    return None
