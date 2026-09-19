"""Structural simulation constants (not gameplay numbers).

Gameplay numbers (damage, costs, ranges, ...) must come from `coh.data.GameData`,
never from literals in `coh/sim`. This module holds only tick-rate / grid
constants that define the simulation's structure.
"""

import math

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

# --- combat (coh/sim/systems/combat.py) ------------------------------------

# Squad formation: member `i` stands at `squad.pos + rotate(FORMATION[i],
# heading)` (metres, relative to the squad position) for the purpose of
# looking up the cover it is standing in. Member 0 stands on the squad
# position itself; the rest ring it. Squads with more members than slots wrap
# around the ring.
FORMATION_RADIUS_M = 1.5
FORMATION_SLOTS = 8
_RING = FORMATION_SLOTS - 1
FORMATION_OFFSETS: tuple[tuple[float, float], ...] = ((0.0, 0.0),) + tuple(
    (
        FORMATION_RADIUS_M * math.cos(2.0 * math.pi * i / _RING),
        FORMATION_RADIUS_M * math.sin(2.0 * math.pi * i / _RING),
    )
    for i in range(_RING)
)

# Behaviour thresholds for the `Attack` order, structural rather than
# gameplay-tuned: how long a squad keeps chasing a target it can no longer
# see, and how often it may re-plan a path toward a moving target.
ATTACK_ORDER_LOST_TARGET_S = 5.0
ATTACK_ORDER_REPATH_S = 1.0

# A squad only *auto*-acquires a building when its primary weapon's damage
# multiplier against that building's target type reaches this fraction (an
# acquisition policy, not a tuned gameplay number): rifles shouldn't wander
# off to plink at a bunker they can barely scratch. An explicit `Attack`
# order ignores this.
BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT = 0.25
