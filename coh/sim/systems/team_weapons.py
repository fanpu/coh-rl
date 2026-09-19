"""team_weapons — firing arcs, re-facing, gunner succession and re-crewing.

Task 10 owns team weapons as *objects* rather than squads: the crewed
weapon lives in `members[0]` and stays there as the crew dies
(`destruction._promote_gunner`), outlives the crew entirely as an abandoned
shell (`destruction._abandon_weapon`), and can be picked up again by any
infantry squad walking onto it (`try_recrew`, driven by `orders.apply_move`
and `movement`'s arrival). `set_facing` swings a gun onto a new bearing,
whether an agent asked for it (`SetFacing`) or `combat.py`'s acquisition
decided the fight had moved (`_auto_reface`, rate-limited by
`Squad.reface_hold_tick`).

`halt` (a squad stopping, the way `movement`'s arrival does) and
`_clear_attack_order` (dropping an `Attack` order) live here rather than in
`combat.py` because a team weapon's halt redeploys the gun; `combat.py`'s
Attack-order pursuit and `destruction.py`'s dangling-target cleanup both call
into this module for them (the latter via a deferred import -- see
`destruction._forget_target`).

Depends on `ballistics.py` (target descriptions) and `destruction.py`
(gunner succession); imported by `combat.py`, and lazily by
`ballistics.fire_member` for the firing-arc gate (see that module's
docstring).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from coh.data.schema import WeaponDef
from coh.sim.constants import AUTO_REFACE_HOLD_SETUPS, RECREW_RANGE_M
from coh.sim.state import Event, Squad, SquadState
from coh.sim.systems.ballistics import _describe, _lookup, _weapon_of
from coh.sim.systems.destruction import _forget_target, _promote_gunner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim

__all__ = ["halt", "set_facing", "can_recrew", "abandoned_weapon_near", "try_recrew"]


# ---------------------------------------------------------------------------
# Team weapons: the crewed weapon in slot 0, its firing arc, and its bearings
# ---------------------------------------------------------------------------


def _team_weapon(sim: "Sim", squad: Squad) -> WeaponDef | None:
    """The crewed weapon `members[0]` carries, if this is a team weapon squad.

    Member 0 is always the gunner (`_promote_gunner` keeps it that way when
    the gunner dies), so the weapon is read off the model rather than off the
    squad def: a re-crewed shell has the same weapon in the same slot.
    """
    sdef = sim.data.squads.get(squad.def_id)
    if sdef is None or sdef.kind != "team_weapon" or not squad.members:
        return None
    gunner = squad.members[0]
    if gunner.hp <= 0:
        return None
    return _weapon_of(sim, gunner)


def _arc_weapon(sim: "Sim", squad: Squad) -> WeaponDef | None:
    """The team weapon, if it has a firing arc narrow enough to matter."""
    weapon = _team_weapon(sim, squad)
    return weapon if weapon is not None and weapon.arc_deg < 360.0 else None


def _indirect_weapon(sim: "Sim", squad: Squad) -> WeaponDef | None:
    """The team weapon, if it lobs its shells over the terrain (mortars).

    None for a garrisoned crew: a mortar cannot be dropped through a roof, so
    the tube never fires from inside (task 12) and the squad neither gains
    indirect fire's LOS exemption nor its minimum range.
    """
    if squad.garrison_in is not None:
        return None
    weapon = _team_weapon(sim, squad)
    return weapon if weapon is not None and weapon.indirect else None


def _bearing(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    delta = to_pos - from_pos
    return math.atan2(float(delta[1]), float(delta[0]))


def _in_arc(squad: Squad, weapon: WeaponDef, aim_pos: np.ndarray) -> bool:
    """Is `aim_pos` within `arc_deg / 2` of where the gun points?"""
    if weapon.arc_deg >= 360.0:
        return True
    from coh.sim.systems import movement

    facing = squad.heading if squad.facing is None else squad.facing
    offset = movement.angle_diff(_bearing(squad.pos, aim_pos), facing)
    return offset <= math.radians(weapon.arc_deg) / 2.0


def set_facing(sim: "Sim", squad: Squad, direction: float, *, auto: bool) -> None:
    """Swing the gun to `direction` (radians), paying `2 x setup_time`.

    The squad tears down (`setup_time`), then `movement` walks it through its
    ordinary arrival path and sets it up again on the new bearing (another
    `setup_time`). `heading` moves with `facing`: the crew turns bodily, and
    `_on_arrival` derives the new `facing` from the heading.

    `auto` marks an acquisition-driven re-face, which is rate-limited by
    `squad.reface_hold_tick` so that two enemies on opposite sides cannot
    make the crew spin in place forever without ever firing. An explicit
    `SetFacing` (`auto=False`) is always honoured and clears the hold.
    """
    from coh.sim.systems import movement

    sdef = sim.data.squads.get(squad.def_id)
    if sdef is None:
        return
    ticks = movement.setup_ticks(sim, sdef)
    squad.heading = direction
    squad.facing = direction
    squad.path = []
    if not auto:
        squad.reface_hold_tick = 0
    if squad.garrison_in is not None:
        # A gun in a building covers every window: the bearing is recorded
        # but there is no arc to swing and nothing to tear down (task 12).
        return
    if ticks <= 0:  # a "team weapon" with no setup time simply pivots
        return
    squad.state = SquadState.TEARING_DOWN
    squad.setup_done_tick = sim.state.tick + ticks
    if auto:
        squad.reface_hold_tick = sim.state.tick + AUTO_REFACE_HOLD_SETUPS * ticks


def _auto_reface(sim: "Sim", squad: Squad, aim_pos: np.ndarray) -> None:
    if sim.state.tick < squad.reface_hold_tick:
        return
    set_facing(sim, squad, _bearing(squad.pos, aim_pos), auto=True)


# ---------------------------------------------------------------------------
# Halting and the `Attack` order
# ---------------------------------------------------------------------------


def halt(sim: "Sim", squad: Squad) -> None:
    """Stop walking, the way movement's arrival does.

    A team weapon that stops deploys again, so that it can actually fire the
    target it just walked into range of -- facing that target if it has one,
    otherwise its last travel direction.
    """
    squad.path = []
    if squad.state is not SquadState.MOVING:
        return
    from coh.sim.systems import movement

    sdef = sim.data.squads[squad.def_id]
    if sdef.kind == "team_weapon":
        squad.state = SquadState.SETTING_UP
        squad.facing = _stopping_facing(sim, squad)
        squad.setup_done_tick = sim.state.tick + movement.setup_ticks(sim, sdef)
    else:
        squad.state = SquadState.IDLE


def _stopping_facing(sim: "Sim", squad: Squad) -> float:
    """Where a team weapon that just stopped points its gun: at the target it
    stopped for, or failing that along its last travel direction."""
    if squad.target_id is not None:
        entity = _lookup(sim, squad.target_id)
        target = None if entity is None else _describe(sim, squad.pos, entity)
        if target is not None:
            return _bearing(squad.pos, target.aim_pos)
    return squad.heading


def _clear_attack_order(sim: "Sim", squad: Squad) -> None:
    squad.order = None
    squad.target_id = None
    squad.attack_last_seen_tick = -1
    squad.attack_last_repath_tick = -1
    halt(sim, squad)


# ---------------------------------------------------------------------------
# Re-crewing an abandoned team weapon
# ---------------------------------------------------------------------------


def can_recrew(sim: "Sim", squad: Squad) -> bool:
    """Could `squad` man an abandoned weapon? Infantry, two models or more."""
    sdef = sim.data.squads.get(squad.def_id)
    if sdef is None or sdef.kind in ("vehicle", "team_weapon") or squad.abandoned:
        return False
    return len(squad.alive_members) >= 2


def abandoned_weapon_near(sim: "Sim", pos: np.ndarray) -> Squad | None:
    """The nearest abandoned team weapon within `RECREW_RANGE_M` of `pos`."""
    best: Squad | None = None
    best_distance = RECREW_RANGE_M
    for sid in sorted(sim.state.squads):
        shell = sim.state.squads[sid]
        if not shell.abandoned:
            continue
        distance = float(np.linalg.norm(shell.pos - np.asarray(pos, dtype=float)))
        if distance <= best_distance and (best is None or distance < best_distance):
            best, best_distance = shell, distance
    return best


def try_recrew(sim: "Sim", squad: Squad) -> bool:
    """`squad` has arrived: if it was sent to man an abandoned weapon and is
    standing next to it, hand the gun over and retire the squad.

    The shell takes the arriving squad's owner and models (capped at the
    weapon's own crew size -- surplus models are lost), the new member 0
    takes the team weapon, and the gun starts setting up on the arriving
    squad's heading. The arriving squad leaves the field like a destroyed
    one, but as a `weapon_recrewed` rather than a `squad_destroyed`.
    """
    shell_id = squad.recrew_target
    squad.recrew_target = None
    if shell_id is None:
        return False
    shell = sim.state.squads.get(shell_id)
    if shell is None or not shell.abandoned or not can_recrew(sim, squad):
        return False
    if float(np.linalg.norm(shell.pos - squad.pos)) > RECREW_RANGE_M:
        return False
    shell_def = sim.data.squads.get(shell.def_id)
    if shell_def is None or not shell_def.loadout:
        return False

    from coh.sim.systems import movement

    crew = squad.alive_members[: shell_def.members]
    shell.owner = squad.owner
    shell.members = crew
    _promote_gunner(shell, shell_def.loadout[0])
    shell.abandoned = False
    shell.heading = squad.heading
    shell.facing = squad.heading
    shell.state = SquadState.SETTING_UP
    shell.setup_done_tick = sim.state.tick + movement.setup_ticks(sim, shell_def)

    sim.state.squads.pop(squad.id, None)
    _forget_target(sim, squad.id)
    sim.state.events.append(
        Event(
            kind="weapon_recrewed",
            tick=sim.state.tick,
            data={"id": shell.id, "owner": shell.owner, "def_id": shell.def_id,
                  "by": squad.id, "pos": [float(shell.pos[0]), float(shell.pos[1])]},
        )
    )
    return True
