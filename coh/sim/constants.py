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

# --- team weapons / indirect fire (task 10) --------------------------------

# How close an infantry squad must get to an abandoned team weapon to re-crew
# it, and how close a `Move` target cell must be to the weapon for the order
# to count as a re-crew order at all. A reach tolerance, not a balance knob.
RECREW_RANGE_M = 2.0

# After combat automatically swings a team weapon onto a new bearing, it may
# not do so again for this many `setup_time`s: two (the teardown and setup the
# turn itself costs) plus one more of actually pointing that way. Without the
# extra one, two enemies on opposite sides would keep a crew turning forever
# without ever firing a shot.
AUTO_REFACE_HOLD_SETUPS = 3

# An explosion's damage falls off linearly from full at the impact point to
# this fraction at `aoe_radius`. Structural shape of the falloff curve; the
# radius and the damage itself are gameplay data on the weapon.
AOE_EDGE_FALLOFF = 0.5

# A building starts construction at this fraction of its full HP and gains the
# rest in proportion to construction progress (structural, not a balance knob).
CONSTRUCTION_START_HP_FRAC = 0.1

# How close (in cells, to the nearest footprint edge) a builder must be before
# it starts contributing construction progress.
BUILD_RANGE_CELLS = 1.5

# Maximum number of queued train/research items per building.
MAX_QUEUE_LEN = 5

# --- vehicles (coh/sim/systems/vehicle_combat.py, task 11) -----------------

# A hit counts as a rear hit when the angle between the target vehicle's
# heading and the direction from the target to the attacker exceeds this.
# It is the shape of the armour model (front/side vs rear), not a balance
# number: how much the rear is worth is `TargetMods.rear_penetration`.
REAR_ARC_DEG = 120.0

# A turret traverses this many times faster than its vehicle's hull rotates
# (`SquadDef.rotation_deg_s`), and may fire once it is pointing this close to
# the target.
TURRET_TRAVERSE_MULT = 2.0
TURRET_AIM_TOLERANCE_DEG = 5.0

# A destroyed vehicle turns the cell it died in into a crater/wreck (heavy
# area cover), but only when that cell is still open ground or road: a wreck
# never replaces a wall, hedge, water or building footprint.
WRECK_TERRAIN_CHAR = "c"
WRECK_REPLACES_TERRAIN = (".", "r")
