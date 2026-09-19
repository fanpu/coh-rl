"""footprints — the cells a building covers, cached per building.

A building never moves or resizes, so its footprint is a pure function of its
id; `cells` (which cells it occupies, for LOS exemptions) and `centres` (their
world centres, for aiming) are computed together the first time either is
asked for and dropped by `forget` when the building is destroyed.

This lives in its own module because both `combat` (aiming, blast radii) and
`garrison` (own-wall LOS exemptions, exit rings) need it, and it must not
import either of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from coh.maps.format import center_of
from coh.sim.constants import CELL_M

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Building

__all__ = ["size", "cells", "centres", "forget"]


def size(sim: "Sim", def_id: str) -> tuple[int, int]:
    """`(width, height)` in cells, for a player or a neutral building def."""
    bdef = sim.data.buildings.get(def_id) or sim.data.neutral.get(def_id)
    return bdef.footprint if bdef is not None else (1, 1)


def cells(sim: "Sim", building: "Building") -> frozenset[tuple[int, int]]:
    """The cells `building` occupies.

    A `frozenset`, so callers can only ever test membership or union it —
    never iterate it in a way that could depend on ordering.
    """
    return _entry(sim, building)[0]


def centres(sim: "Sim", building: "Building") -> np.ndarray:
    """`(k, 2)` world centres of `building`'s cells, in row-major cell order.

    The order is part of the contract: `combat` breaks aim-point ties with
    `argmin`, which resolves to the lowest index.
    """
    return _entry(sim, building)[1]


def forget(sim: "Sim", building_id: int) -> None:
    """Drop a destroyed building's entry."""
    _cache(sim).pop(building_id, None)


def _cache(sim: "Sim") -> dict[int, tuple[frozenset[tuple[int, int]], np.ndarray]]:
    cache = getattr(sim, "_footprint_cache", None)
    if cache is None:
        cache = {}
        sim._footprint_cache = cache
    return cache


def _entry(sim: "Sim", building: "Building") -> tuple[frozenset[tuple[int, int]], np.ndarray]:
    cache = _cache(sim)
    entry = cache.get(building.id)
    if entry is None:
        width, height = size(sim, building.def_id)
        cx0, cy0 = building.cell
        grid = [(cx0 + dx, cy0 + dy) for dy in range(height) for dx in range(width)]
        entry = (frozenset(grid), np.array([center_of(cell, CELL_M) for cell in grid], dtype=float))
        cache[building.id] = entry
    return entry
