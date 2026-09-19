"""territory — capture progress, ownership and supply connectivity (task 13).

Each tick:

1. Clear `PointState.op_building` on any point whose observation post has
   died (no longer in `state.buildings`).
2. For every non-HQ point, gather this tick's eligible capturers (see
   `_capturers_by_team`), then either stall (capturers of >=2 teams present),
   advance/neutralize progress toward the sole present team, or decay
   progress toward the point's resting value (owner's, or neutral) when no
   one is capturing.
3. If any point's `owner_team` changed this tick, recompute
   `state.connected` for every team.

HQ-sector points (see `hq_point_ids`) are permanently owned by their team:
`Capture` orders on them are rejected at validation time
(`coh.sim.orders.validate_capture`) and this system never touches them.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING


from coh.maps.format import center_of, distance
from coh.sim import orders as orders_mod
from coh.sim.constants import CELL_M, DT
from coh.sim.state import Event, SquadState, entity_ids

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.maps.format import PointDef
    from coh.sim.sim import Sim
    from coh.sim.state import PointState


def run(sim: "Sim") -> None:
    """Advance the territory system by one tick."""
    _clear_dead_op_buildings(sim)

    hq_points = hq_point_ids(sim)
    econ = sim.data.economy
    changed = False
    for pid in sorted(sim.state.points):
        if pid in hq_points:
            continue
        point_def = sim.map.points[pid]
        if _process_point(sim, pid, point_def, econ):
            changed = True

    if changed:
        recompute_connectivity(sim)


# ---------------------------------------------------------------------------
# HQ sectors
# ---------------------------------------------------------------------------


def hq_point_ids(sim: "Sim") -> set[str]:
    """Point ids of every team's HQ sectors (`Sim.hq_sectors`): permanently owned."""
    ids: set[str] = set()
    for sector_ids in sim.hq_sectors.values():
        for sector_id in sector_ids:
            sector = sim.map.sectors.get(sector_id)
            if sector is not None and sector.point_id is not None:
                ids.add(sector.point_id)
    return ids


def is_hq_point(sim: "Sim", point_id: str) -> bool:
    return point_id in hq_point_ids(sim)


# ---------------------------------------------------------------------------
# Observation posts
# ---------------------------------------------------------------------------


def _clear_dead_op_buildings(sim: "Sim") -> None:
    for pid in sorted(sim.state.points):
        point = sim.state.points[pid]
        if point.op_building is not None and point.op_building not in sim.state.buildings:
            point.op_building = None


def clear_op_building(sim: "Sim", building_id: int) -> None:
    """Forget `building_id` as an observation post, now, rather than on the
    next tick's sweep — a building's death is handled in one place (task 12),
    and nothing should see a point pointing at a building that is gone."""
    for pid in sorted(sim.state.points):
        point = sim.state.points[pid]
        if point.op_building == building_id:
            point.op_building = None


def _has_living_op(sim: "Sim", point: "PointState") -> bool:
    return point.op_building is not None and point.op_building in sim.state.buildings


# ---------------------------------------------------------------------------
# Capture / neutralize / decay
# ---------------------------------------------------------------------------


def _capturers_by_team(sim: "Sim", pid: str, point_def: "PointDef", econ) -> dict[int, float]:
    """Sum of `capture_rate` per team among squads eligible to capture `pid`."""
    center = center_of(point_def.cell, CELL_M)
    totals: dict[int, float] = {}
    for sid in entity_ids(sim.state.squads):
        squad = sim.state.squads[sid]
        order = squad.order
        if not isinstance(order, orders_mod.Capture) or order.point_id != pid:
            continue
        if squad.pinned or squad.state is SquadState.RETREATING:
            continue
        if squad.garrison_in is not None or squad.abandoned:
            continue
        if not squad.has_alive_members:
            continue
        sdef = sim.data.squads[squad.def_id]
        if sdef.capture_rate <= 0:
            continue
        dist = distance(squad.pos, center)
        if dist > econ.capture_radius:
            continue
        team = sim.state.players[squad.owner].team
        totals[team] = totals.get(team, 0.0) + sdef.capture_rate
    return totals


def _process_point(sim: "Sim", pid: str, point_def: "PointDef", econ) -> bool:
    """Advance one point's capture state by a tick; return whether its owner changed."""
    point = sim.state.points[pid]
    prev_owner = point.owner_team

    capturers = _capturers_by_team(sim, pid, point_def, econ)
    teams = sorted(capturers)

    if len(teams) > 1:
        # Contested by multiple teams: stalled, no progress change.
        point.capturing_team = None
    elif len(teams) == 1:
        team = teams[0]
        rate = capturers[team]
        if point.owner_team is not None and point.owner_team != team:
            _neutralize(sim, pid, point, team, rate, econ)
        else:
            _advance(sim, pid, point, team, rate, econ)
    else:
        _decay(point, econ)

    return point.owner_team != prev_owner


def _neutralize(sim: "Sim", pid: str, point: "PointState", team: int, rate: float, econ) -> None:
    """`team`'s capturers push an enemy-owned point's progress down toward 0."""
    point.capturing_team = team
    new_progress = point.progress - rate * DT / econ.neutralize_time_s
    if new_progress > 0.0:
        point.progress = new_progress
        return

    point.progress = 0.0
    if _has_living_op(sim, point):
        # A living observation post keeps the point from flipping neutral.
        return

    point.owner_team = None
    sim.state.events.append(
        Event(kind="point_neutralized", tick=sim.state.tick, data={"point_id": pid, "team": team})
    )


def _advance(sim: "Sim", pid: str, point: "PointState", team: int, rate: float, econ) -> None:
    """`team`'s capturers push a neutral/own point's progress up toward 1 (owned by `team`)."""
    was_owner = point.owner_team
    new_progress = point.progress + rate * DT / econ.capture_time_s
    if new_progress < 1.0:
        point.progress = new_progress
        point.capturing_team = team
        return

    point.progress = 1.0
    point.owner_team = team
    point.capturing_team = None
    _clear_capture_orders(sim, pid, team)
    if was_owner != team:
        sim.state.events.append(
            Event(kind="point_captured", tick=sim.state.tick, data={"point_id": pid, "team": team})
        )


def _decay(point: "PointState", econ) -> None:
    """No eligible capturers: progress drifts to the resting value at rate 1.0."""
    target = 1.0 if point.owner_team is not None else 0.0
    delta = DT / econ.capture_time_s
    if point.progress < target:
        point.progress = min(target, point.progress + delta)
    elif point.progress > target:
        point.progress = max(target, point.progress - delta)
    point.capturing_team = None


def _clear_capture_orders(sim: "Sim", pid: str, team: int) -> None:
    for sid in entity_ids(sim.state.squads):
        squad = sim.state.squads[sid]
        order = squad.order
        if isinstance(order, orders_mod.Capture) and order.point_id == pid:
            if sim.state.players[squad.owner].team == team:
                squad.order = None


# ---------------------------------------------------------------------------
# Supply connectivity
# ---------------------------------------------------------------------------


def recompute_connectivity(sim: "Sim") -> None:
    """`state.connected[team]`: sector ids reachable from `team`'s HQ sector(s)
    through a chain of sectors whose point is owned by `team`."""
    teams = sorted({player.team for player in sim.state.players.values()})
    connected: dict[int, list[int]] = {}
    for team in teams:
        hq_sectors = sim.hq_sectors.get(team, [])
        seen: set[int] = set(hq_sectors)
        queue: deque[int] = deque(hq_sectors)
        while queue:
            sid = queue.popleft()
            for nb in sim.map.sectors[sid].neighbors:
                if nb in seen:
                    continue
                nb_sector = sim.map.sectors[nb]
                owned = (
                    nb_sector.point_id is not None
                    and sim.state.points[nb_sector.point_id].owner_team == team
                )
                if owned:
                    seen.add(nb)
                    queue.append(nb)
        connected[team] = sorted(seen)
    sim.state.connected = connected
