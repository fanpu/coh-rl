"""vehicle_combat — armour penetration, rear arcs, turrets and wrecks (task 11).

`systems/combat.py` owns the shot itself; this module owns the four places a
vehicle target or a vehicle shooter behaves differently, and `combat` calls
into it from four small seams:

1. `penetration_mult` — after a bullet (or an explosion) lands on a vehicle,
   one roll decides whether it goes through the armour. `p_pen = clamp(
   weapon.penetration[band] * tt.penetration * (tt.rear_penetration if rear
   else 1), 0, 1)`; a penetration does `damage * tt.damage`, a deflection
   `damage * tt.damage * weapon.deflection_damage_mult`. `rear` is decided by
   `is_rear_hit`: the shot comes from outside `REAR_ARC_DEG` of the *target's*
   heading. Infantry cover never applies to a vehicle -- the penetration roll
   replaces it.
2. `aim` — each tick, a vehicle points its main gun (`members[0]`, loadout
   index 0). A turreted gun (`arc_deg >= 360`) swings `turret_heading` toward
   the target at `TURRET_TRAVERSE_MULT * rotation_deg_s` and returns toward
   the hull heading when there is no target. A hull-mounted gun (`arc_deg <
   360`) instead turns the whole vehicle, but only while it is standing still:
   a vehicle following a path steers with `movement`, not with its gun.
3. `may_fire` — the main gun fires only once it is actually pointing at the
   target: within `TURRET_AIM_TOLERANCE_DEG` of the turret's heading, or
   within `arc_deg / 2` of the hull's. Secondary weapons (loadout index >= 1,
   e.g. hull MGs) are not arc-limited in M1.
4. `leave_wreck` — a destroyed vehicle leaves a heavy-cover wreck on the cell
   it died in (`combat` emits `vehicle_destroyed` rather than
   `squad_destroyed` for it), recorded on `GameState.terrain_changes` exactly
   like a crushed fence.

RNG discipline: exactly one `state.rng.random()` draw per hit that lands on a
vehicle, taken in `penetration_mult` immediately after the hit roll that
produced it (nothing between them draws). So a bullet at a vehicle draws
victim -> hit -> penetration, and an explosion draws one penetration per
vehicle it catches, in ascending squad id.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from coh.data.schema import TargetMods, WeaponDef
from coh.maps.format import cell_of
from coh.sim.constants import (
    CELL_M,
    DT,
    REAR_ARC_DEG,
    TURRET_AIM_TOLERANCE_DEG,
    TURRET_TRAVERSE_MULT,
    WRECK_REPLACES_TERRAIN,
    WRECK_TERRAIN_CHAR,
)
from coh.sim.state import Squad
from coh.sim.systems.movement import angle_diff, rotate_toward

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.data.schema import SquadDef
    from coh.sim.sim import Sim

__all__ = [
    "VEHICLE_DESTROYED",
    "is_vehicle",
    "is_rear_hit",
    "penetration_mult",
    "main_gun",
    "aim",
    "may_fire",
    "leave_wreck",
]

# The event kind a destroyed vehicle emits *instead of* `squad_destroyed`, so
# that a listener never sees both for the same squad.
VEHICLE_DESTROYED = "vehicle_destroyed"

_REAR_ARC_RAD = math.radians(REAR_ARC_DEG)
_AIM_TOLERANCE_RAD = math.radians(TURRET_AIM_TOLERANCE_DEG)
_FULL_CIRCLE_DEG = 360.0


def _bearing(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    delta = np.asarray(to_pos, dtype=float) - np.asarray(from_pos, dtype=float)
    return math.atan2(float(delta[1]), float(delta[0]))


def is_vehicle(sim: "Sim", squad: Squad) -> bool:
    sdef = sim.data.squads.get(squad.def_id)
    return sdef is not None and sdef.kind == "vehicle"


# ---------------------------------------------------------------------------
# Penetration
# ---------------------------------------------------------------------------


def is_rear_hit(victim: Squad, from_pos: np.ndarray) -> bool:
    """Does a shot from `from_pos` strike `victim` outside its frontal arc?"""
    to_attacker = np.asarray(from_pos, dtype=float) - victim.pos
    if float(np.linalg.norm(to_attacker)) < 1e-9:
        return False  # point blank on top of it: no meaningful facing
    return angle_diff(victim.heading, _bearing(victim.pos, from_pos)) > _REAR_ARC_RAD


def penetration_mult(
    sim: "Sim", weapon: WeaponDef, band: int, victim: Squad, from_pos: np.ndarray, mods: TargetMods
) -> float:
    """Roll this hit's penetration and return the damage multiplier it earns.

    `1.0` through the armour, `weapon.deflection_damage_mult` off it. Costs
    exactly one `state.rng` draw, always (even at p_pen 0 or 1), so the draw
    order does not depend on the numbers.
    """
    rear = is_rear_hit(victim, from_pos)
    p_pen = weapon.penetration[band] * mods.penetration * (mods.rear_penetration if rear else 1.0)
    p_pen = min(1.0, max(0.0, p_pen))
    penetrated = float(sim.state.rng.random()) < p_pen
    return 1.0 if penetrated else weapon.deflection_damage_mult


# ---------------------------------------------------------------------------
# Aiming: turret traverse and hull rotation
# ---------------------------------------------------------------------------


def main_gun(sim: "Sim", squad: Squad) -> WeaponDef | None:
    """The vehicle's main gun: the weapon its first living model carries."""
    if not squad.members:
        return None
    member = squad.members[0]
    if member.hp <= 0 or not member.weapon:
        return None
    return sim.data.weapons.get(member.weapon)


def aim(sim: "Sim", squad: Squad, aim_pos: np.ndarray | None) -> None:
    """Swing `squad`'s main gun toward `aim_pos` (None: no target) by one tick.

    A turreted gun traverses independently of the hull and re-centres on the
    hull when it has nothing to look at. A hull-mounted gun turns the whole
    vehicle, and only while the vehicle is standing still -- one that is
    following a path steers along it instead, and one whose target is already
    inside the hull arc has nothing to turn for.
    """
    sdef = sim.data.squads.get(squad.def_id)
    if sdef is None or sdef.kind != "vehicle":
        return
    gun = main_gun(sim, squad)
    if gun is None:
        return

    target_bearing = None if aim_pos is None else _bearing(squad.pos, aim_pos)

    if gun.arc_deg >= _FULL_CIRCLE_DEG:
        goal = squad.heading if target_bearing is None else target_bearing
        traverse = math.radians(sdef.rotation_deg_s * TURRET_TRAVERSE_MULT) * DT
        squad.turret_heading = rotate_toward(squad.turret_heading, goal, traverse)
        return

    if (
        target_bearing is not None
        and not squad.path
        and not squad.moving
        and angle_diff(squad.heading, target_bearing) > math.radians(gun.arc_deg) / 2.0
    ):
        squad.heading = rotate_toward(squad.heading, target_bearing, math.radians(sdef.rotation_deg_s) * DT)
    # A hull-mounted gun *is* the hull, as far as anything reading
    # `turret_heading` (the viewer) is concerned.
    squad.turret_heading = squad.heading


def may_fire(
    squad: Squad, sdef: "SquadDef", slot: int, weapon: WeaponDef, aim_pos: np.ndarray
) -> bool:
    """Is this weapon pointing close enough at `aim_pos` to fire this tick?"""
    if sdef.kind != "vehicle" or slot != 0:
        return True  # infantry/team weapons: combat's own arc rules apply
    target_bearing = _bearing(squad.pos, aim_pos)
    if weapon.arc_deg >= _FULL_CIRCLE_DEG:
        return angle_diff(squad.turret_heading, target_bearing) <= _AIM_TOLERANCE_RAD
    return angle_diff(squad.heading, target_bearing) <= math.radians(weapon.arc_deg) / 2.0


# ---------------------------------------------------------------------------
# Death
# ---------------------------------------------------------------------------


def leave_wreck(sim: "Sim", squad: Squad) -> None:
    """Turn the cell the vehicle died in into a wreck (heavy area cover).

    Only open ground and road become wrecks: a vehicle that dies on top of a
    hedge, a wall, water or a building footprint leaves the terrain alone
    rather than quietly making it passable or destroying the cover already
    there.
    """
    cx, cy = cell_of(squad.pos, CELL_M)
    if not (0 <= cx < sim.map.width and 0 <= cy < sim.map.height):
        return
    if str(sim.map.terrain[cy, cx]) not in WRECK_REPLACES_TERRAIN:
        return
    sim.map.set_terrain_cell((cx, cy), WRECK_TERRAIN_CHAR)
    sim.state.terrain_changes.append((cx, cy, WRECK_TERRAIN_CHAR))
