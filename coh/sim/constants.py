"""Structural simulation constants (not gameplay numbers).

Gameplay numbers (damage, costs, ranges, ...) must come from `coh.data.GameData`,
never from literals in `coh/sim`. This module holds only tick-rate / grid
constants that define the simulation's structure.
"""

TICKS_PER_SECOND = 8
DT = 1.0 / TICKS_PER_SECOND
CELL_M = 2.0

# Performance tuning, not gameplay data: caps the vision system's per-origin
# mask cache (coh/sim/systems/vision.py) so a long game with many distinct
# unit positions can't grow it unboundedly. Eviction is FIFO and purely a
# cache (never affects results), so this has no effect on determinism.
VISION_MASK_CACHE_MAX = 4096

# A vehicle only advances while its heading is within this many degrees of its
# desired travel direction; otherwise it spends the tick rotating in place.
VEHICLE_MOVE_ARC_DEG = 45.0

# A building starts construction at this fraction of its full HP and gains the
# rest in proportion to construction progress (structural, not a balance knob).
CONSTRUCTION_START_HP_FRAC = 0.1

# How close (in cells, to the nearest footprint edge) a builder must be before
# it starts contributing construction progress.
BUILD_RANGE_CELLS = 1.5

# Maximum number of queued train/research items per building.
MAX_QUEUE_LEN = 5
