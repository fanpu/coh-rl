"""explosions — indirect fire: shell scatter and area-of-effect resolution.

A mortar's shot (task 10) never rolls to hit: `fire_shell` scatters the
round around its aim point instead, and `_explode` then damages and
suppresses everything hostile the blast catches -- squads (with a
per-model penetration roll for any vehicle among them, task 11) and
buildings alike.

Depends on `ballistics.py` (target descriptions, cover, suppression) and
`destruction.py` (death paths); imported by `combat.py`, which dispatches to
`fire_shell` from `ballistics.fire_member` -- see that module's docstring
for why that one call is the other way around.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from coh.data.schema import WeaponDef
from coh.sim.constants import AOE_EDGE_FALLOFF
from coh.sim.state import Building, Event, Squad
from coh.sim.systems import footprints, vehicle_combat
from coh.sim.systems.ballistics import (
    _NO_TEAM,
    _building_target_type,
    _building_team,
    _cover_for,
    _is_targetable,
    _member_pos,
    _team_of,
    add_suppression,
)
from coh.sim.systems.destruction import _destroy_building, _remove_member

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.systems.ballistics import _Target

__all__ = ["fire_shell"]


def fire_shell(sim: "Sim", squad: Squad, weapon: WeaponDef, band: int, target: "_Target") -> bool:
    """Lob one shell at `target` and resolve where it lands. Returns whether
    it hurt anything.

    There is no hit roll: the shell scatters instead. Each axis is offset by
    `N(0, scatter_m * d / max_range)`, drawn x then y so that the RNG stream
    stays fixed. Burst fields are ignored -- a mortar fires one bomb.
    """
    sigma = weapon.scatter_m * target.distance / weapon.ranges[2] if weapon.ranges[2] > 0 else 0.0
    rng = sim.state.rng
    offset_x = float(rng.normal(0.0, sigma))
    offset_y = float(rng.normal(0.0, sigma))
    impact = target.aim_pos + np.array([offset_x, offset_y])

    hurt = _explode(sim, squad, weapon, band, impact)
    sim.state.events.append(
        Event(
            kind="explosion",
            tick=sim.state.tick,
            data={
                "src": squad.id,
                "pos": [float(impact[0]), float(impact[1])],
                "radius": weapon.aoe_radius,
            },
        )
    )
    return hurt


def _falloff(distance: float, radius: float) -> float:
    """Linear from 1.0 at the impact point to `AOE_EDGE_FALLOFF` at `radius`."""
    return 1.0 - (1.0 - AOE_EDGE_FALLOFF) * (distance / radius)


def _explode(sim: "Sim", attacker: Squad, weapon: WeaponDef, band: int, impact: np.ndarray) -> bool:
    """Damage and suppress everything hostile inside `weapon.aoe_radius`.

    Only enemies of the shooter are caught: CoH mortars do hurt their own
    side, but M1's bots are not clever enough to be trusted with that.
    """
    radius = weapon.aoe_radius
    if radius <= 0.0:
        return False
    team = _team_of(sim, attacker.owner)
    hurt = False
    for sid in sorted(sim.state.squads):
        victim = sim.state.squads.get(sid)
        if victim is None or not _is_targetable(victim) or _team_of(sim, victim.owner) == team:
            continue
        hurt = _explode_on_squad(sim, attacker, weapon, band, impact, victim) or hurt
    for bid in sorted(sim.state.buildings):
        building = sim.state.buildings.get(bid)
        if building is None or _building_team(sim, building) in (team, _NO_TEAM):
            continue
        hurt = _explode_on_building(sim, weapon, impact, building) or hurt
    return hurt


def _explode_on_squad(
    sim: "Sim", attacker: Squad, weapon: WeaponDef, band: int, impact: np.ndarray, victim: Squad
) -> bool:
    sdef = sim.data.squads.get(victim.def_id)
    if sdef is None:
        return False
    mods = weapon.vs(sdef.target_type)
    is_vehicle = sdef.kind == "vehicle"

    # Snapshot the crew first: members are removed as they die, and every
    # model in the blast is meant to be caught where it stood when it went
    # off, not where the survivors shuffle to afterwards.
    caught = []
    for slot, member in enumerate(victim.members):
        if member.hp <= 0:
            continue
        distance = float(np.linalg.norm(_member_pos(sim, victim, slot) - impact))
        if distance > weapon.aoe_radius:
            continue
        cover = _cover_for(sim, victim, slot, impact)
        caught.append((member, distance, cover))
    if not caught:
        return False

    hurt = False
    for member, distance, cover in caught:
        # A vehicle caught in a blast rolls penetration (against the armour
        # facing the impact point) instead of taking infantry cover damage.
        if is_vehicle:
            cover_damage = vehicle_combat.penetration_mult(sim, weapon, band, victim, impact, mods)
        else:
            cover_damage = weapon.cover(cover).damage
        damage = weapon.damage * mods.damage * cover_damage * _falloff(distance, weapon.aoe_radius)
        if damage <= 0.0:
            continue
        hurt = True
        member.hp -= damage
        if member.hp <= 0.0:
            _remove_member(sim, victim, member)

    # One dose of suppression per shell, not per model, taken against the
    # cover of the model nearest the blast.
    _, _, best_cover = min(caught, key=lambda entry: entry[1])
    cover_mods = weapon.cover(best_cover)
    if _is_targetable(victim):  # it may have been wiped out by the blast
        add_suppression(sim, attacker, victim, weapon, band, cover_mods.suppression * mods.suppression)
    return hurt


def _explode_on_building(sim: "Sim", weapon: WeaponDef, impact: np.ndarray, building: Building) -> bool:
    cells = footprints.centres(sim, building)
    distance = math.sqrt(float(((cells - impact) ** 2).sum(axis=1).min()))
    if distance > weapon.aoe_radius:
        return False
    target_type = _building_target_type(sim, building)
    if target_type is None:
        return False
    damage = weapon.damage * weapon.vs(target_type).damage * _falloff(distance, weapon.aoe_radius)
    if damage <= 0.0:
        return False
    building.hp -= damage
    if building.hp <= 0.0:
        _destroy_building(sim, building)
    return True
