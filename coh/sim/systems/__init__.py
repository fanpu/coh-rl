"""Tick systems, each a module exposing `run(sim) -> None`.

`SYSTEM_ORDER` is the fixed per-tick order from the design spec §2.3 (orders
themselves are applied when issued, not as a system). `Sim.tick` looks `run`
up on the module at call time, so tests can monkeypatch a single system.
"""

from __future__ import annotations

from coh.sim.systems import (
    combat,
    economy,
    garrison,
    movement,
    production,
    suppression,
    territory,
    victory,
    vision,
)

SYSTEM_ORDER = (
    production,
    movement,
    vision,
    combat,
    suppression,
    garrison,
    territory,
    economy,
    victory,
)

__all__ = ["SYSTEM_ORDER"]
