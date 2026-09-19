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
