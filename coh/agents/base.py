"""The agent protocol.

An agent sees exactly what a human player would: the static map and the stat
tables (both public, handed over once at `reset`), then one fog-filtered
`Observation` per decision step. It never touches the `Sim` — everything it
can know is in the `Observation`, and everything it can do is an `Order`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from coh.data.schema import GameData
from coh.env import Observation, Order
from coh.maps.format import GameMap


@runtime_checkable
class Agent(Protocol):
    """What `scripts/play_match.py` (and the M2 benchmark runner) expects."""

    def reset(self, player_id: int, map: GameMap, data: GameData) -> None:
        """Start a new match as `player_id`. The map and stat tables are public."""

    def act(self, obs: Observation) -> list[Order]:
        """The orders to issue this decision step; an empty list is valid."""
