"""destruction — squad, vehicle and building death paths (tasks 8, 10, 11, 12).

The base of the combat split: this module has no dependency on
`ballistics.py`, `explosions.py` or `team_weapons.py` (they all depend on
it, directly or transitively), so that the split stays acyclic. The one
exception is `_forget_target`, which needs `team_weapons._clear_attack_order`
to drop a squad's `Attack` order on its dead target; since `team_weapons.py`
already imports this module at load time, that one call is deferred
(imported inside the function) to avoid a circular import at module load.

`_remove_member` is the single place a model actually dies: it decides
between `_promote_gunner` (another crewman steps up to a team weapon),
`_abandon_weapon` (the last crewman died -- the gun becomes a crewless
shell, task 10) and `_destroy_squad` (an ordinary squad, or a vehicle,
whose last member just died). `_destroy_building` and its
`on_building_destroyed` hook hand a collapsing (or captured-through-death)
building's garrison over to `systems/garrison.py` (task 12). `damage_member`
is `apply_hit`'s damage step, exposed for damage that isn't a bullet
(garrison collapse).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coh.sim import orders as orders_mod
from coh.sim.state import Event, Squad, SquadState, entity_ids
from coh.sim.systems import footprints, garrison, vehicle_combat

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Building, Member

__all__ = ["damage_member", "on_building_destroyed"]


# ---------------------------------------------------------------------------
# Damage -> death
# ---------------------------------------------------------------------------


def damage_member(sim: "Sim", squad: Squad, member: "Member", amount: float) -> None:
    """Hurt one model directly, retiring it (and its squad) if it dies.

    The raw damage path behind `apply_hit`, exposed for damage that isn't a
    bullet: task 12's garrison collapse.
    """
    member.hp -= amount
    if member.hp <= 0.0:
        _remove_member(sim, squad, member)


def _remove_member(sim: "Sim", squad: Squad, member: "Member") -> None:
    """A dead model leaves the squad, taking its weapon with it.

    A team weapon is the exception: it belongs to the gun, not to the man
    behind it. When the gunner falls the next crewman takes over
    (`_promote_gunner`), and when the last of them dies the weapon itself
    stays on the field as an abandoned shell (`_abandon_weapon`).
    """
    was_gunner = bool(squad.members) and squad.members[0] is member
    squad.members = [m for m in squad.members if m is not member]
    sdef = sim.data.squads.get(squad.def_id)
    is_team_weapon = sdef is not None and sdef.kind == "team_weapon"

    if not squad.has_alive_members:
        if is_team_weapon:
            _abandon_weapon(sim, squad)
        else:
            _destroy_squad(sim, squad)
        return

    if was_gunner and is_team_weapon and sdef.loadout:
        _promote_gunner(squad, sdef.loadout[0])


def _promote_gunner(squad: Squad, weapon_id: str) -> None:
    """The next crewman steps up to the gun, dropping his own weapon.

    His cooldown/reload bookkeeping belongs to the weapon he just let go of,
    so it resets with the hand-over.
    """
    gunner = squad.members[0]
    gunner.weapon = weapon_id
    gunner.next_ready_tick = 0
    gunner.shots_since_reload = 0
    gunner.burst_until_tick = 0


def _abandon_weapon(sim: "Sim", squad: Squad) -> None:
    """The last crewman died: leave the gun on the field, crewless.

    The `Squad` stays in `state.squads` as a shell so that it can be
    re-crewed. `owner` is left alone -- `abandoned` is what every other
    system reads -- but the shell is no longer visible, targetable, or
    counted towards its old owner's population and upkeep. A gun whose crew
    died inside a garrison is carried out of the building first (task 12),
    so that somebody can still walk up to it.

    Upgrades belonged to the crew that just died, not to the gun: they are
    dropped with them, so an enemy that re-crews the shell inherits a bare
    weapon rather than the dead squad's kit (and no half-paid purchase).
    """
    squad.members = []
    squad.abandoned = True
    squad.state = SquadState.IDLE
    squad.order = None
    squad.path = []
    squad.target_id = None
    squad.moving = False
    squad.suppression = 0.0
    squad.suppressed = False
    squad.pinned = False
    squad.reinforcing = False
    squad.recrew_target = None
    squad.reface_hold_tick = 0
    squad.upgrades = []
    squad.pending_upgrade = None
    squad.upgrade_done_tick = 0
    # A gun crewed inside a building is carried out to an exit cell rather
    # than left on the (impassable) footprint, where no squad could ever get
    # within `RECREW_RANGE_M` of it and the gun would be lost for good.
    # `settle=False`: the shell's own resting state is set above.
    garrison.leave(sim, squad, settle=False)
    _forget_target(sim, squad.id)
    sim.state.events.append(
        Event(
            kind="weapon_abandoned",
            tick=sim.state.tick,
            data={"id": squad.id, "owner": squad.owner, "def_id": squad.def_id,
                  "pos": [float(squad.pos[0]), float(squad.pos[1])]},
        )
    )


def _destroy_squad(sim: "Sim", squad: Squad) -> None:
    """Take a squad off the field.

    A vehicle leaves a wreck where it died and announces itself as a
    `vehicle_destroyed` rather than a `squad_destroyed` -- one event per
    death, never both (task 11).
    """
    kind = "squad_destroyed"
    if vehicle_combat.is_vehicle(sim, squad):
        vehicle_combat.leave_wreck(sim, squad)
        kind = vehicle_combat.VEHICLE_DESTROYED
    garrison.detach(sim, squad)
    sim.state.squads.pop(squad.id, None)
    _forget_target(sim, squad.id)
    sim.state.events.append(
        Event(
            kind=kind,
            tick=sim.state.tick,
            data={"id": squad.id, "owner": squad.owner, "def_id": squad.def_id,
                  "pos": [float(squad.pos[0]), float(squad.pos[1])]},
        )
    )


def _destroy_building(sim: "Sim", building: "Building") -> None:
    sim.state.buildings.pop(building.id, None)
    sim.map.stamp_footprint(building.cell, footprints.size(sim, building.def_id), blocked=False)
    _forget_target(sim, building.id)
    on_building_destroyed(sim, building)
    # After the hook: ejecting the garrison still needs the building's
    # geometry, and the geometry of a dead building can no longer change.
    footprints.forget(sim, building.id)
    sim.state.events.append(
        Event(
            kind="building_destroyed",
            tick=sim.state.tick,
            data={"id": building.id, "owner": building.owner, "def_id": building.def_id,
                  "cell": list(building.cell)},
        )
    )


def on_building_destroyed(sim: "Sim", building: "Building") -> None:
    """Eject the garrison (with collapse damage) and clear up after the wreck.

    Called after the building leaves `state.buildings` and its footprint is
    freed, before the `building_destroyed` event is emitted. The rules live in
    `systems/garrison.py`.
    """
    garrison.on_building_destroyed(sim, building)


def _forget_target(sim: "Sim", entity_id: int) -> None:
    """Clear every dangling reference to a destroyed entity."""
    # Deferred: `team_weapons.py` imports this module at load time (for
    # `_promote_gunner`), so importing it back at *this* module's top level
    # would be circular. By the time any squad dies both modules are already
    # fully loaded, so this costs nothing but a dict lookup.
    from coh.sim.systems import team_weapons

    for sid in entity_ids(sim.state.squads):
        squad = sim.state.squads[sid]
        if squad.target_id == entity_id:
            squad.target_id = None
        if isinstance(squad.order, orders_mod.Attack) and squad.order.target_id == entity_id:
            team_weapons._clear_attack_order(sim, squad)
