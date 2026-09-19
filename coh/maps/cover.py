"""Directional + area cover lookup for a target cell."""

from __future__ import annotations

import math

import numpy as np

from coh.data.schema import COVER_TYPES
from coh.maps.format import GameMap, center_of

_LIGHT = COVER_TYPES.index("light")
_HEAVY = COVER_TYPES.index("heavy")
_NEGATIVE = COVER_TYPES.index("negative")

_NEIGHBOR_OFFSETS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]

_ANGLE_THRESHOLD_DEG = 60.0


def cover_at(m: GameMap, cell: tuple[int, int], from_pos: np.ndarray) -> str:
    """Cover a unit standing at `cell` gets from fire originating at `from_pos`.

    Order: area cover on the cell itself (light/heavy, non-directional) wins
    first; otherwise the best qualifying directional object cover among the
    8 neighbours (an object counts if it's roughly between the cell and the
    shooter -- within 60 degrees of the shooter's direction); otherwise
    `negative` if the cell is a road, else `open`.
    """
    cx, cy = cell
    area = int(m.area_cover[cy, cx])
    if area == _LIGHT:
        return "light"
    if area == _HEAVY:
        return "heavy"

    to_shooter = from_pos - center_of(cell)
    shooter_norm = np.linalg.norm(to_shooter)

    best = None  # None, "light", or "heavy"
    if shooter_norm > 1e-9:
        for dx, dy in _NEIGHBOR_OFFSETS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < m.width and 0 <= ny < m.height):
                continue
            obj = int(m.cover_object[ny, nx])
            if obj not in (_LIGHT, _HEAVY):
                continue
            to_neighbor = np.array([dx, dy], dtype=float)
            cos_angle = np.dot(to_neighbor, to_shooter) / (np.linalg.norm(to_neighbor) * shooter_norm)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.degrees(math.acos(cos_angle))
            if angle <= _ANGLE_THRESHOLD_DEG:
                kind = "heavy" if obj == _HEAVY else "light"
                if kind == "heavy":
                    best = "heavy"
                elif best is None:
                    best = "light"

    if best is not None:
        return best

    if area == _NEGATIVE:
        return "negative"
    return "open"
