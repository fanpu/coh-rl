"""Structural simulation constants (not gameplay numbers).

Gameplay numbers (damage, costs, ranges, ...) must come from `coh.data.GameData`,
never from literals in `coh/sim`. This module holds only tick-rate / grid
constants that define the simulation's structure.
"""

TICKS_PER_SECOND = 8
DT = 1.0 / TICKS_PER_SECOND
CELL_M = 2.0

# A vehicle only advances while its heading is within this many degrees of its
# desired travel direction; otherwise it spends the tick rotating in place.
VEHICLE_MOVE_ARC_DEG = 45.0
