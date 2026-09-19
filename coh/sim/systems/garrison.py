"""garrison — entering and leaving buildings, and building-death cleanup (task 12).

Only **neutral** buildings (`data.neutral`, `Building.neutral`) are enterable
in M1: they are the map's houses, and a `NeutralBuildingDef` is the only def
that carries a `capacity`. A house holds squads of one team at a time, up to
`capacity` squads.

A garrisoned squad:

- is `SquadState.GARRISONED` with `garrison_in` set and its id in the
  building's (sorted) `garrison` list;
- stands at the building's centre cell, so range, vision and cover are all
  measured from there without any special case (`combat` measures range from
  `squad.pos`, `vision` casts its mask from the same centre cell);
- sees and shoots *through its own building's footprint* — `los_ignore` is
  what `combat` hands `vision.has_los` to exempt those cells, mirroring what
  `vision.run` does with `ignore_block`;
- takes `"garrison"` cover (`combat._cover_for`, `suppression`'s recovery);
- accepts only `Ungarrison`, `Retreat`, `Attack`, `Stop`, `BuyUpgrade` and
  `SetFacing`. Everything that implies walking is rejected at validation time
  (`coh/sim/orders.py`), so nothing else in the sim has to worry about a
  garrisoned squad suddenly acquiring a path. `Retreat` is the exception: it
  exits first (`leave(settle=False)`) and then runs home.

Entry happens in `run`, one tick after `movement` has walked the squad to the
building: legality is re-checked then (the house may have filled up or been
taken meanwhile), and a squad that cannot get in is simply left outside with
its order cleared.

Exit cell rule (`exit_cell`, used by `Ungarrison`, `Retreat` and ejection):
among the infantry-passable cells directly bordering the footprint, pick the
one **furthest from the nearest enemy squad visible to the leaver's team**;
with no visible enemy, the one **nearest the owner's HQ**. Ties in either
case go to the lowest `(cy, cx)`. If the whole ring is blocked, fall back to
`pathfinding.nearest_adjacent_passable`.

`on_building_destroyed` is combat's hook and generalizes to *any* building:
eject the garrison (each model taking `economy.garrison_collapse_damage_frac`
of its max HP), release its builders, drop its production queue with no
refund, clear the capture point it was an observation post for, and cancel
any `Garrison` order aimed at it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from coh.maps.format import cell_of, center_of
from coh.sim import orders as orders_mod
from coh.sim import pathfinding
from coh.sim.constants import CELL_M, GARRISON_ENTER_RANGE_CELLS
from coh.sim.state import Building, Event, Squad, SquadState
from coh.sim.systems import footprints

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim

__all__ = [
    "run",
    "enter",
    "leave",
    "detach",
    "exit_cell",
    "occupants",
    "occupying_team",
    "capacity",
    "garrison_problem",
    "los_ignore",
    "on_building_destroyed",
]

_NO_CELLS: frozenset[tuple[int, int]] = frozenset()


# ---------------------------------------------------------------------------
# Geometry / occupancy queries
# ---------------------------------------------------------------------------


def centre_cell(sim: "Sim", building: Building) -> tuple[int, int]:
    """The building's centre cell — the same one `vision` casts sight from.

    Shared with `vision` rather than recomputed, so a garrisoned squad's
    position and the sight origin it is given can never drift apart.
    """
    from coh.sim.systems import vision

    return vision.building_center_cell(sim, building.cell, building.def_id)


def occupants(sim: "Sim", building: Building) -> list[Squad]:
    """The living squads actually inside `building`, in ascending id order."""
    found = []
    for sid in building.garrison:
        squad = sim.state.squads.get(sid)
        if squad is not None and squad.garrison_in == building.id and squad.alive_members:
            found.append(squad)
    return found


def occupying_team(sim: "Sim", building: Building) -> int | None:
    """The team holding `building`, or None if it is empty."""
    for squad in occupants(sim, building):
        player = sim.state.players.get(squad.owner)
        if player is not None:
            return player.team
    return None


def capacity(sim: "Sim", building: Building) -> int:
    """How many squads fit inside; 0 for a building that is not enterable."""
    ndef = sim.data.neutral.get(building.def_id)
    return 0 if ndef is None or not building.neutral else ndef.capacity


def garrison_problem(sim: "Sim", squad: Squad, building: Building) -> str:
    """Why `squad` may not enter `building`; `""` when it may.

    The squad's own `can_garrison` is checked by `orders.validate_garrison`;
    this is about the *building*, and is re-run on arrival because a house can
    fill up or change hands while the squad is walking to it.
    """
    if not building.neutral or sim.data.neutral.get(building.def_id) is None:
        return f"building {building.id} ({building.def_id}) is not enterable"
    inside = occupants(sim, building)
    team = sim.state.players[squad.owner].team
    holder = occupying_team(sim, building)
    if holder is not None and holder != team:
        return f"building {building.id} is occupied by team {holder}"
    if len(inside) >= capacity(sim, building):
        return f"building {building.id} is full ({capacity(sim, building)} squads)"
    return ""


def los_ignore(sim: "Sim", shooter: Squad, target: Squad | Building) -> frozenset[tuple[int, int]]:
    """Footprint cells that must not block the shooter -> target line.

    A garrisoned shooter fires out through its own walls, and a garrisoned
    target is shot at through the walls it is standing behind — exactly the
    own-footprint exemption `vision.run` applies to a garrisoned squad's sight
    mask, applied to both ends of the line.
    """
    cells = _NO_CELLS
    if shooter.garrison_in is not None:
        building = sim.state.buildings.get(shooter.garrison_in)
        if building is not None:
            cells = footprints.cells(sim, building)
    if isinstance(target, Squad) and target.garrison_in is not None:
        building = sim.state.buildings.get(target.garrison_in)
        if building is not None:
            cells = cells | footprints.cells(sim, building)
    return cells


def _range_to_footprint_cells(sim: "Sim", squad: Squad, building: Building) -> float:
    """Distance in cells from `squad` to the nearest footprint edge."""
    cx0, cy0 = building.cell
    w, h = footprints.size(sim, building.def_id)
    x, y = squad.pos[0] / CELL_M, squad.pos[1] / CELL_M
    dx = max(cx0 - x, 0.0, x - (cx0 + w))
    dy = max(cy0 - y, 0.0, y - (cy0 + h))
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# Entering
# ---------------------------------------------------------------------------


def enter(sim: "Sim", squad: Squad, building: Building) -> None:
    """Put `squad` inside `building`. The caller must have checked legality.

    A team weapon has to set its gun up again once inside: it goes
    `GARRISONED` like everyone else (so cover, vision and the arc bypass all
    key off `garrison_in`), but cannot fire until `setup_done_tick`.
    """
    from coh.sim.systems import movement

    squad.garrison_in = building.id
    if squad.id not in building.garrison:
        building.garrison.append(squad.id)
        building.garrison.sort()

    squad.pos = np.asarray(center_of(centre_cell(sim, building), CELL_M), dtype=float)
    squad.path = []
    squad.moving = False
    squad.order = None
    squad.state = SquadState.GARRISONED

    sdef = sim.data.squads.get(squad.def_id)
    if sdef is not None and sdef.kind == "team_weapon":
        squad.setup_done_tick = sim.state.tick + movement.setup_ticks(sim, sdef)

    sim.state.events.append(
        Event(
            kind="garrison_entered",
            tick=sim.state.tick,
            data={"squad": squad.id, "building": building.id, "owner": squad.owner},
        )
    )


# ---------------------------------------------------------------------------
# Leaving
# ---------------------------------------------------------------------------


def detach(sim: "Sim", squad: Squad, building: Building | None = None) -> Building | None:
    """Drop `squad` out of its building's roster without moving or settling it.

    Used by `combat` when a garrisoned squad dies or is abandoned, so that
    `Building.garrison` never keeps a dangling id (it is part of the state
    hash). `building` is passed explicitly when it has already been destroyed
    and so can no longer be looked up by id.
    """
    if building is None and squad.garrison_in is not None:
        building = sim.state.buildings.get(squad.garrison_in)
    squad.garrison_in = None
    if building is not None and squad.id in building.garrison:
        building.garrison.remove(squad.id)
    return building


def leave(sim: "Sim", squad: Squad, *, settle: bool, building: Building | None = None) -> None:
    """Take `squad` out of its building and stand it on an exit cell.

    `settle=True` also puts the squad back into a resting state (`IDLE`, or
    `SETTING_UP` for a team weapon that has to redeploy its gun).
    `Retreat` passes `settle=False`: `movement.start_path` sets `RETREATING`
    itself right afterwards, and so does `combat._abandon_weapon`, which uses
    this to carry a gun whose crew died indoors out to a cell somebody can
    actually reach. `building` is passed by the collapse path, whose building
    is already gone from `state.buildings`.
    """
    from coh.sim.systems import movement

    building = detach(sim, squad, building)
    if building is not None:
        cell = exit_cell(sim, squad, building)
        if cell is not None:
            squad.pos = np.asarray(center_of(cell, CELL_M), dtype=float)
    squad.path = []
    squad.moving = False
    if not settle:
        return

    sdef = sim.data.squads.get(squad.def_id)
    squad.state = SquadState.IDLE
    if sdef is not None and sdef.kind == "team_weapon":
        squad.state = SquadState.SETTING_UP
        squad.facing = squad.heading
        squad.setup_done_tick = sim.state.tick + movement.setup_ticks(sim, sdef)


def exit_cell(sim: "Sim", squad: Squad, building: Building) -> tuple[int, int] | None:
    """Where `squad` steps out of `building` (see the module docstring's rule)."""
    size = footprints.size(sim, building.def_id)
    ring = [
        cell
        for cell in pathfinding.adjacent_cells(building.cell, size, sim.map.width, sim.map.height)
        if sim.map.pass_inf[cell[1], cell[0]]
    ]
    if not ring:
        return pathfinding.nearest_adjacent_passable(
            sim.map.pass_inf, building.cell, size, cell_of(squad.pos, CELL_M)
        )

    threat = _nearest_visible_enemy_pos(sim, squad, building)
    if threat is not None:
        # Away from the enemy: maximise the distance, lowest (cy, cx) on ties.
        return min(ring, key=lambda c: (-_distance_m(c, threat), c[1], c[0]))

    home = _home_pos(sim, squad)
    if home is None:
        return min(ring, key=lambda c: (c[1], c[0]))
    return min(ring, key=lambda c: (_distance_m(c, home), c[1], c[0]))


def _distance_m(cell: tuple[int, int], pos: np.ndarray) -> float:
    return float(np.linalg.norm(center_of(cell, CELL_M) - pos))


def _nearest_visible_enemy_pos(sim: "Sim", squad: Squad, building: Building) -> np.ndarray | None:
    """Position of the enemy squad nearest `building` that `squad`'s team sees."""
    from coh.sim.systems import vision

    team = sim.state.players[squad.owner].team
    centre = center_of(centre_cell(sim, building), CELL_M)
    best: np.ndarray | None = None
    best_distance = math.inf
    for sid in sorted(sim.state.squads):
        other = sim.state.squads[sid]
        if other.abandoned or not other.alive_members:
            continue
        owner = sim.state.players.get(other.owner)
        if owner is None or owner.team == team:
            continue
        if not vision.is_visible(sim, team, other):
            continue
        distance = float(np.linalg.norm(other.pos - centre))
        if distance < best_distance:
            best, best_distance = other.pos, distance
    return best


def _home_pos(sim: "Sim", squad: Squad) -> np.ndarray | None:
    player = sim.state.players.get(squad.owner)
    hq = sim.state.buildings.get(player.hq_id) if player is not None else None
    if hq is None:
        return None
    return center_of(centre_cell(sim, hq), CELL_M)


# ---------------------------------------------------------------------------
# Building death: eject, damage, and clear everything that pointed at it
# ---------------------------------------------------------------------------


def on_building_destroyed(sim: "Sim", building: Building) -> None:
    """`combat`'s hook, for *any* building (see the module docstring).

    Called after the building has left `state.buildings` and its footprint has
    been freed, so ejected squads may stand on cells the walls used to block.
    """
    from coh.sim.systems import production, territory

    _eject_all(sim, building)
    production.release_builders_of(sim, building.id)
    building.queue.clear()  # lost with the building, no refund
    territory.clear_op_building(sim, building.id)
    _cancel_garrison_orders(sim, building.id)


def _eject_all(sim: "Sim", building: Building) -> None:
    """Throw everyone out of a collapsing building, hurting every model."""
    from coh.sim.systems import combat

    frac = sim.data.economy.garrison_collapse_damage_frac
    for squad in occupants(sim, building):
        leave(sim, squad, settle=True, building=building)
        sim.state.events.append(
            Event(
                kind="garrison_ejected",
                tick=sim.state.tick,
                data={
                    "squad": squad.id,
                    "building": building.id,
                    "owner": squad.owner,
                    "pos": [float(squad.pos[0]), float(squad.pos[1])],
                },
            )
        )
        sdef = sim.data.squads.get(squad.def_id)
        if sdef is None or frac <= 0.0:
            continue
        damage = frac * sdef.member_hp
        for member in list(squad.members):
            if member.hp > 0:
                combat.damage_member(sim, squad, member, damage)
    building.garrison.clear()


def _cancel_garrison_orders(sim: "Sim", building_id: int) -> None:
    from coh.sim.systems import combat

    for sid in sorted(sim.state.squads):
        squad = sim.state.squads[sid]
        order = squad.order
        if isinstance(order, orders_mod.Garrison) and order.building_id == building_id:
            squad.order = None
            combat.halt(sim, squad)


# ---------------------------------------------------------------------------
# Tick entry point: a walking `Garrison` order completes here
# ---------------------------------------------------------------------------


def run(sim: "Sim") -> None:
    """Advance the garrison system by one tick."""
    for sid in sorted(sim.state.squads):
        squad = sim.state.squads.get(sid)
        if squad is None:
            continue  # left the field earlier this tick
        if squad.garrison_in is not None:
            if squad.garrison_in not in sim.state.buildings:
                # Defensive: nothing should remove a building without going
                # through `on_building_destroyed`, but a squad must never be
                # left pointing at one that is gone.
                leave(sim, squad, settle=True)
            continue
        if isinstance(squad.order, orders_mod.Garrison):
            _try_enter(sim, squad, squad.order)


def _try_enter(sim: "Sim", squad: Squad, order: orders_mod.Garrison) -> None:
    if squad.path or squad.state is SquadState.TEARING_DOWN:
        return  # still walking there (or packing the gun up)

    building = sim.state.buildings.get(order.building_id)
    if (
        building is None
        or garrison_problem(sim, squad, building)
        or _range_to_footprint_cells(sim, squad, building) > GARRISON_ENTER_RANGE_CELLS
    ):
        # Arrived but cannot get in (full, taken, or nowhere near it): the
        # order is spent and the squad stays outside.
        squad.order = None
        squad.path = []
        return

    enter(sim, squad, building)
