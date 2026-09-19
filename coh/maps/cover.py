"""Directional + area cover lookup for a target cell."""

from __future__ import annotations

import math

import numpy as np

from coh.data.schema import COVER_TYPES
from coh.maps.format import GameMap, center_of

_LIGHT = COVER_TYPES.index("light")
_HEAVY = COVER_TYPES.index("heavy")
_NEGATIVE = COVER_TYPES.index("negative")

# `(dx, dy, |(dx, dy)|)` for the 8 neighbours. The length is a constant of the
# offset, so it is computed once here rather than per lookup; `math.sqrt` of
# the same integer sum is bit-identical to the `np.linalg.norm` this replaced.
_NEIGHBORS = tuple(
    (dx, dy, math.sqrt(dx * dx + dy * dy))
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)

_ANGLE_THRESHOLD_DEG = 60.0


def cover_at(m: GameMap, cell: tuple[int, int], from_pos: np.ndarray) -> str:
    """Cover a unit standing at `cell` gets from fire originating at `from_pos`.

    Order: area cover on the cell itself (light/heavy, non-directional) wins
    first; otherwise the best qualifying directional object cover among the
    8 neighbours (an object counts if it's roughly between the cell and the
    shooter -- within 60 degrees of the shooter's direction); otherwise
    `negative` if the cell is a road, else `open`.

    The angle test is done in plain floats: `cos_angle` is the same dot
    product over the same two quantities numpy computed, so the comparison
    against the threshold is unchanged, but a lookup no longer allocates ten
    small arrays.
    """
    cx, cy = cell
    area = int(m.area_cover[cy, cx])
    if area == _LIGHT:
        return "light"
    if area == _HEAVY:
        return "heavy"

    center = center_of(cell)
    to_shooter_x = float(from_pos[0]) - float(center[0])
    to_shooter_y = float(from_pos[1]) - float(center[1])
    shooter_norm = math.sqrt(to_shooter_x * to_shooter_x + to_shooter_y * to_shooter_y)

    best = None  # None, "light", or "heavy"
    if shooter_norm > 1e-9:
        cover_object = m.cover_object
        width, height = m.width, m.height
        for dx, dy, neighbor_norm in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            obj = int(cover_object[ny, nx])
            if obj not in (_LIGHT, _HEAVY):
                continue
            cos_angle = (dx * to_shooter_x + dy * to_shooter_y) / (neighbor_norm * shooter_norm)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.degrees(math.acos(cos_angle))
            if angle <= _ANGLE_THRESHOLD_DEG:
                if obj == _HEAVY:
                    return "heavy"  # nothing later can beat heavy
                if best is None:
                    best = "light"

    if best is not None:
        return best

    if area == _NEGATIVE:
        return "negative"
    return "open"
