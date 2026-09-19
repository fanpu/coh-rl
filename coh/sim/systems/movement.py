"""movement — pathfinding, steering, setup/teardown and retreat movement.

`start_path` is called by `coh/sim/orders.py`'s move-type appliers (`Move`,
`AttackMove`, `Retreat`, `Capture`, `Garrison`, `Build`) to resolve an order
into a `squad.path` and kick off any team-weapon teardown. `run` is the tick
system that walks squads along their paths.

Squad state during a move:
- team weapons that are `SET_UP`/`SETTING_UP` first spend the tick(s) in
  `TEARING_DOWN` before they start travelling;
- infantry/team-weapon heading snaps to the travel direction each tick;
- vehicles rotate toward the travel direction at `rotation_deg_s` and only
  advance while within `VEHICLE_MOVE_ARC_DEG` of it;
- a vehicle with `crushes_light_cover` clears a fence (`f`) cell's cover the
  moment it enters it, recording the mutation on `GameState.terrain_changes`;
- `AttackMove` halts (no advance) while `squad.target_id is not None`
  (combat sets that in a later task; tests set it directly);
- on arrival, `Move`/`AttackMove`/`Retreat` clear `squad.order` and go
  `IDLE`; `Capture`/`Garrison`/`Build` keep their order for later systems to
  act on; a team weapon that stops moving (arrival, or an immediate no-op
  order) enters `SETTING_UP` facing its last travel direction, then `SET_UP`
  after its weapon's `setup_time`;
- if the map changes underneath an in-flight path (a footprint gets
  stamped across the route), the squad re-plans to the same final goal the
  next time it would step into the now-blocked cell; a re-plan that finds
  nothing reachable drops the path and applies the same order-keep/clear
  rule as arrival, without actually treating it as one, while a re-plan
  that resolves to the squad's own current cell (its one remaining
  waypoint was the final goal and just got blocked) *is* treated as a real
  arrival, via `_on_arrival`.

`Sim.spawn_squad` also uses `setup_ticks` below: a freshly spawned team
weapon starts `SETTING_UP` (it's always deployed, never mid-march) rather
than `IDLE`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from coh.maps.format import cell_of, center_of
from coh.sim import orders as orders_mod
from coh.sim import pathfinding
from coh.sim.constants import CELL_M, DT, TICKS_PER_SECOND, VEHICLE_MOVE_ARC_DEG
from coh.sim.state import SquadState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import Squad

_VEHICLE_ARC_RAD = math.radians(VEHICLE_MOVE_ARC_DEG)

# Order types that clear on arrival; Capture/Garrison/Build instead keep
# their order for later systems to act on (see module docstring).
_ARRIVAL_CLEARS_ORDER = (orders_mod.Move, orders_mod.AttackMove, orders_mod.Retreat)


def run(sim: "Sim") -> None:
    """Advance the movement system by one tick."""
    for sid in sorted(sim.state.squads):
        _step_squad(sim, sim.state.squads[sid])


# ---------------------------------------------------------------------------
# Order resolution (called from coh/sim/orders.py at order-apply time)
# ---------------------------------------------------------------------------


def start_path(sim: "Sim", squad: "Squad", order: orders_mod.Order, goal_cell: tuple[int, int], moving_state: SquadState) -> None:
    """Resolve `order` into `squad.path`, storing it as the squad's order.

    `moving_state` is the state the squad should be in while travelling
    (`MOVING` for everything except `Retreat`, which uses `RETREATING`).
    """
    sdef = sim.data.squads[squad.def_id]
    is_vehicle = sdef.kind == "vehicle"
    start_cell = cell_of(squad.pos, CELL_M)

    path = pathfinding.find_path_cached(sim, is_vehicle, start_cell, goal_cell)
    squad.order = order
    if path is None:
        # Nothing reachable right now: the order stands, but there's nothing
        # to walk toward yet (e.g. a later map change might open one up).
        squad.path = []
        return

    squad.path = list(path[1:])  # drop the start cell; keep remaining waypoints

    if sdef.kind == "team_weapon" and squad.state in (SquadState.SET_UP, SquadState.SETTING_UP):
        squad.state = SquadState.TEARING_DOWN
        squad.setup_done_tick = sim.state.tick + setup_ticks(sim, sdef)
    elif not squad.path:
        _on_arrival(sim, squad, sdef)
    else:
        squad.state = moving_state


# ---------------------------------------------------------------------------
# Per-tick stepping
# ---------------------------------------------------------------------------


def _step_squad(sim: "Sim", squad: "Squad") -> None:
    sdef = sim.data.squads[squad.def_id]
    squad.moving = False

    if squad.state is SquadState.TEARING_DOWN:
        if sim.state.tick < squad.setup_done_tick:
            return
        squad.state = SquadState.RETREATING if isinstance(squad.order, orders_mod.Retreat) else SquadState.MOVING
        if not squad.path:
            _on_arrival(sim, squad, sdef)
        return

    if squad.state is SquadState.SETTING_UP:
        if sim.state.tick >= squad.setup_done_tick:
            squad.state = SquadState.SET_UP
        return

    if not squad.path:
        return

    if isinstance(squad.order, orders_mod.AttackMove) and squad.target_id is not None:
        return  # halted: engaging a target (combat handles the fight)

    is_vehicle = sdef.kind == "vehicle"

    # The map can change after a path was computed (a building finishes, a
    # footprint gets stamped across the route, ...). Cheaply revalidate just
    # the immediate next cell each tick; only re-plan when it's actually
    # gone bad, and only once (a failed re-plan drops the path rather than
    # retrying every tick).
    passable = sim.map.pass_veh if is_vehicle else sim.map.pass_inf
    next_cell = squad.path[0]
    if not passable[next_cell[1], next_cell[0]]:
        _replan_or_stop(sim, squad, sdef, is_vehicle)
        if not squad.path:
            return

    mult = _speed_mult(sim, squad)
    if mult <= 0.0:
        return

    if _advance(sim, squad, sdef, mult, is_vehicle):
        squad.moving = True

    if not squad.path:
        _on_arrival(sim, squad, sdef)


def _speed_mult(sim: "Sim", squad: "Squad") -> float:
    econ = sim.data.economy
    if squad.pinned:
        return 0.0
    if squad.suppressed:
        return econ.suppressed_speed_mult
    if squad.state is SquadState.RETREATING:
        return econ.retreat_speed_mult
    return 1.0


def _advance(sim: "Sim", squad: "Squad", sdef, mult: float, is_vehicle: bool) -> bool:
    """Move `squad` up to `speed * DT * mult` metres along its path.

    Returns whether the squad actually displaced this tick (a vehicle stuck
    rotating toward its next waypoint does not count).
    """
    remaining = sdef.speed * DT * mult
    moved = False

    while remaining > 1e-9 and squad.path:
        target_cell = squad.path[0]
        target_pos = center_of(target_cell, CELL_M)
        diff = target_pos - squad.pos
        dist = float(np.linalg.norm(diff))

        if dist < 1e-9:
            squad.path.pop(0)
            _enter_cell(sim, squad, sdef, target_cell)
            continue

        direction = diff / dist
        travel_heading = math.atan2(direction[1], direction[0])

        if is_vehicle:
            squad.heading = _rotate_toward(squad.heading, travel_heading, math.radians(sdef.rotation_deg_s) * DT)
            if _angle_diff(squad.heading, travel_heading) > _VEHICLE_ARC_RAD:
                break  # still turning onto the arc; no displacement this tick
        else:
            squad.heading = travel_heading

        step = min(remaining, dist)
        squad.pos = squad.pos + direction * step
        remaining -= step
        moved = moved or step > 1e-9

        if step >= dist - 1e-9:
            squad.path.pop(0)
            _enter_cell(sim, squad, sdef, target_cell)

    return moved


def _replan_or_stop(sim: "Sim", squad: "Squad", sdef, is_vehicle: bool) -> None:
    """The next cell on `squad.path` just became impassable: re-plan to the
    same final goal. If nothing is reachable any more, drop the path and
    apply the same order-keep/clear rule arrival would (but stay `IDLE`,
    since this isn't really an arrival). If the re-plan *succeeds* but
    resolves to the squad's own current cell (e.g. the one waypoint left was
    also the final goal, and it's now blocked, so `find_path`'s
    nearest-passable fallback lands back on `start_cell`), that's a genuine
    arrival -- go through `_on_arrival` rather than leaving `squad.path`
    empty with nothing having set the squad's state, which would strand it
    in `MOVING`/`RETREATING` forever (the `if not squad.path: return` guard
    at the top of `_step_squad` would then skip it every tick)."""
    goal_cell = squad.path[-1]
    start_cell = cell_of(squad.pos, CELL_M)
    path = pathfinding.find_path_cached(sim, is_vehicle, start_cell, goal_cell)
    if path is None:
        squad.path = []
        if isinstance(squad.order, _ARRIVAL_CLEARS_ORDER):
            squad.order = None
        squad.state = SquadState.IDLE
        return
    squad.path = list(path[1:])
    if not squad.path:
        _on_arrival(sim, squad, sdef)


def _enter_cell(sim: "Sim", squad: "Squad", sdef, cell: tuple[int, int]) -> None:
    if sdef.kind == "vehicle" and sdef.crushes_light_cover:
        cx, cy = cell
        if str(sim.map.terrain[cy, cx]) == "f":
            sim.map.set_terrain_cell(cell, ".")
            sim.state.terrain_changes.append((cx, cy, "."))


def _on_arrival(sim: "Sim", squad: "Squad", sdef) -> None:
    order = squad.order
    if isinstance(order, _ARRIVAL_CLEARS_ORDER):
        squad.order = None
    squad.state = SquadState.IDLE

    if sdef.kind == "team_weapon":
        squad.state = SquadState.SETTING_UP
        squad.facing = squad.heading
        squad.setup_done_tick = sim.state.tick + setup_ticks(sim, sdef)


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two headings (radians)."""
    return abs(_wrap(a - b))


def _rotate_toward(current: float, target: float, max_delta: float) -> float:
    diff = _wrap(target - current)
    diff = max(-max_delta, min(max_delta, diff))
    return _wrap(current + diff)


# ---------------------------------------------------------------------------
# Team-weapon setup/teardown timing (also used by Sim.spawn_squad: a freshly
# spawned team weapon starts SETTING_UP)
# ---------------------------------------------------------------------------


def weapon_setup_time(sim: "Sim", sdef) -> float:
    if not sdef.loadout:
        return 0.0
    weapon = sim.data.weapons.get(sdef.loadout[0])
    return weapon.setup_time if weapon is not None else 0.0


def setup_ticks(sim: "Sim", sdef) -> int:
    return max(0, math.ceil(weapon_setup_time(sim, sdef) * TICKS_PER_SECOND))
