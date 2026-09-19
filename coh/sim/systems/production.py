"""production — construction, training/research queues and squad upgrades (task 14).

Three independent jobs run each tick:

1. **Squad upgrades.** A `BuyUpgrade` order is paid for at issue time and
   recorded on the squad (`pending_upgrade` / `upgrade_done_tick`); when the
   tick arrives the upgrade's weapon replaces the first `count` member slots
   that don't already carry it. *Simplification for M1:* members restored by
   `Reinforce` come back with the squad def's base loadout, so they do not
   inherit a previously bought upgrade -- it has to be bought again.
2. **Construction.** `Build` places the (impassable) footprint immediately at
   `progress = 0` and `CONSTRUCTION_START_HP_FRAC` of its HP, and sends the
   builder to an adjacent cell. Every builder that is within
   `BUILD_RANGE_CELLS` of the footprint with the `Build` order still standing
   adds `DT / build_time` progress per tick -- so two builders finish in half
   the time -- and a proportional share of the remaining HP. Ordering a
   builder elsewhere simply stops its contribution; the site keeps whatever
   progress it has, and re-issuing the same `Build` on it resumes at no cost.
3. **Building queues.** A completed building ticks down the *head* of its
   FIFO queue only. A finished `train` item spawns the squad on the nearest
   passable cell south of the footprint; a finished `research` item adds the
   upgrade to the owning player.

Public helpers (`requirement_met`, `can_afford`, `pay`, `build_site_problem`,
...) are what `coh/sim/orders.py` validates Build/Train/Research/BuyUpgrade
against, so the rules live here rather than being split across both modules.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from coh.maps.format import center_of
from coh.sim import orders as orders_mod
from coh.sim.constants import (
    BUILD_RANGE_CELLS,
    CELL_M,
    CONSTRUCTION_START_HP_FRAC,
    DT,
    TICKS_PER_SECOND,
)
from coh.sim.state import Event, QueueItem, SquadState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.data.schema import BuildingDef, Cost, SquadUpgradeDef
    from coh.sim.sim import Sim
    from coh.sim.state import Building, Player, Squad

# Construction progress is accumulated by repeated float addition; anything
# this close to 1.0 counts as finished.
_PROGRESS_EPS = 1e-9

# Fixed 8-neighbour expansion order, so the "nearest passable cell" search
# below resolves ties the same way on every run and platform.
_RING_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
)


def run(sim: "Sim") -> None:
    """Advance the production system by one tick."""
    _finish_squad_upgrades(sim)
    _advance_construction(sim)
    _advance_queues(sim)


# ---------------------------------------------------------------------------
# Shared rules (also used by coh/sim/orders.py at validation time)
# ---------------------------------------------------------------------------


def requirement_met(sim: "Sim", player_id: int, req_id: str) -> bool:
    """A `requires` entry is satisfied by a researched upgrade or a completed
    building of that def owned by this player."""
    if req_id in sim.state.players[player_id].upgrades:
        return True
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        if building.owner == player_id and building.def_id == req_id and building.progress >= 1.0:
            return True
    return False


def missing_requirements(sim: "Sim", player_id: int, requires) -> list[str]:
    return [req for req in requires if not requirement_met(sim, player_id, req)]


def can_afford(player: "Player", cost: "Cost") -> bool:
    return (
        player.manpower >= cost.manpower
        and player.munitions >= cost.munitions
        and player.fuel >= cost.fuel
    )


def pay(player: "Player", cost: "Cost") -> None:
    player.manpower -= cost.manpower
    player.munitions -= cost.munitions
    player.fuel -= cost.fuel


def observation_post_id(sim: "Sim", faction: str) -> str | None:
    return sim.data.economy.op_building.get(faction)


def unfinished_building_at(sim: "Sim", player_id: int, def_id: str, cell: tuple[int, int]) -> "Building | None":
    """This player's still-under-construction building of `def_id` at `cell`."""
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        if (
            building.owner == player_id
            and building.def_id == def_id
            and building.cell == tuple(cell)
            and building.progress < 1.0
        ):
            return building
    return None


def footprint_cells(cell: tuple[int, int], footprint: tuple[int, int]) -> list[tuple[int, int]]:
    cx0, cy0 = cell
    w, h = footprint
    return [(cx0 + dx, cy0 + dy) for dy in range(h) for dx in range(w)]


def build_site_problem(sim: "Sim", player: "Player", bdef: "BuildingDef", cell: tuple[int, int]) -> str:
    """Why `bdef` cannot be placed at `cell` by `player`; `""` when it can.

    Observation posts follow their own rule (they sit *on* an owned, supplied
    capture point, whose cell need not be vehicle-passable open ground), so
    they skip the generic passability / in-supply footprint test.
    """
    if bdef.id == observation_post_id(sim, player.faction):
        return _op_site_problem(sim, player, bdef, cell)

    connected = set(sim.state.connected.get(player.team, ()))
    for cx, cy in footprint_cells(cell, bdef.footprint):
        if not (0 <= cx < sim.map.width and 0 <= cy < sim.map.height):
            return f"footprint cell {(cx, cy)} is out of bounds"
        if not sim.map.pass_veh[cy, cx]:
            return f"footprint cell {(cx, cy)} is not buildable terrain"
        if int(sim.map.sector_id[cy, cx]) not in connected:
            return f"footprint cell {(cx, cy)} is not in supplied territory"
    return _overlap_problem(sim, bdef, cell)


def _op_site_problem(sim: "Sim", player: "Player", bdef: "BuildingDef", cell: tuple[int, int]) -> str:
    connected = set(sim.state.connected.get(player.team, ()))
    for point_id in sorted(sim.state.points):
        point_def = sim.map.points[point_id]
        if point_def.cell != tuple(cell):
            continue
        point = sim.state.points[point_id]
        if point.owner_team != player.team or point_def.sector not in connected:
            return f"point {point_id!r} is not owned and supplied by team {player.team}"
        if point.op_building is not None and point.op_building in sim.state.buildings:
            return f"point {point_id!r} already has an observation post"
        cx0, cy0 = cell
        w, h = bdef.footprint
        if not (cx0 + w <= sim.map.width and cy0 + h <= sim.map.height):
            return f"observation post at {tuple(cell)} does not fit the map"
        return _overlap_problem(sim, bdef, cell)
    return f"cell {tuple(cell)} is not a capture point"


def _overlap_problem(sim: "Sim", bdef: "BuildingDef", cell: tuple[int, int]) -> str:
    wanted = set(footprint_cells(cell, bdef.footprint))
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        other = sim.data.buildings.get(building.def_id) or sim.data.neutral.get(building.def_id)
        size = other.footprint if other is not None else (1, 1)
        if wanted & set(footprint_cells(building.cell, size)):
            return f"footprint overlaps building {building.id}"
    return ""


def start_construction(sim: "Sim", owner: int, bdef: "BuildingDef", cell: tuple[int, int]) -> "Building":
    """Pay for and place a fresh construction site. Caller must have validated."""
    player = sim.state.players[owner]
    pay(player, bdef.cost)
    building = sim.spawn_building(owner, bdef.id, cell, complete=False)
    building.hp = bdef.hp * CONSTRUCTION_START_HP_FRAC

    if bdef.id == observation_post_id(sim, player.faction):
        for point_id in sorted(sim.state.points):
            if sim.map.points[point_id].cell == tuple(cell):
                sim.state.points[point_id].op_building = building.id

    sim.state.events.append(
        Event(
            kind="construction_started",
            tick=sim.state.tick,
            data={"building": building.id, "def_id": bdef.id, "owner": owner, "cell": list(building.cell)},
        )
    )
    return building


def start_squad_upgrade(sim: "Sim", squad: "Squad", udef: "SquadUpgradeDef") -> None:
    """Pay for and start a squad weapon upgrade. Caller must have validated."""
    pay(sim.state.players[squad.owner], udef.cost)
    squad.pending_upgrade = udef.id
    squad.upgrade_done_tick = sim.state.tick + max(0, math.ceil(udef.time * TICKS_PER_SECOND))


def enqueue(sim: "Sim", building: "Building", kind: str, item_id: str, cost: "Cost", seconds: float) -> None:
    """Pay for and append a queue item. Caller must have validated."""
    pay(sim.state.players[building.owner], cost)
    building.queue.append(QueueItem(kind=kind, item_id=item_id, remaining_s=seconds))


def research_queued(sim: "Sim", player_id: int, upgrade_id: str) -> bool:
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        if building.owner != player_id:
            continue
        if any(item.kind == "research" and item.item_id == upgrade_id for item in building.queue):
            return True
    return False


# ---------------------------------------------------------------------------
# Squad upgrades
# ---------------------------------------------------------------------------


def _finish_squad_upgrades(sim: "Sim") -> None:
    for squad_id in sorted(sim.state.squads):
        squad = sim.state.squads[squad_id]
        if squad.pending_upgrade is None:
            continue
        if not squad.alive_members:
            squad.pending_upgrade = None  # the squad died mid-purchase
            continue
        if sim.state.tick < squad.upgrade_done_tick:
            continue
        udef = sim.data.squad_upgrades[squad.pending_upgrade]
        _swap_weapons(squad, udef)
        squad.upgrades.append(udef.id)
        squad.pending_upgrade = None
        sim.state.events.append(
            Event(
                kind="upgrade_bought",
                tick=sim.state.tick,
                data={"squad": squad.id, "upgrade": udef.id, "owner": squad.owner},
            )
        )


def _swap_weapons(squad: "Squad", udef: "SquadUpgradeDef") -> None:
    remaining = udef.count
    for member in squad.members:
        if remaining <= 0:
            break
        if member.weapon == udef.weapon:
            continue
        member.weapon = udef.weapon
        remaining -= 1


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _advance_construction(sim: "Sim") -> None:
    builders_by_site: dict[int, list["Squad"]] = {}

    for squad_id in sorted(sim.state.squads):
        squad = sim.state.squads[squad_id]
        if squad.build_target is None:
            continue
        site = sim.state.buildings.get(squad.build_target)
        if site is None or site.progress >= 1.0 or not squad.alive_members:
            _release_builder(squad)
            continue
        if not isinstance(squad.order, orders_mod.Build):
            _release_builder(squad)  # re-tasked: stops contributing, site pauses
            continue
        if squad.state is not SquadState.CONSTRUCTING:
            if squad.path or not _in_build_range(sim, squad, site):
                continue  # still walking there (or stranded), no contribution
            squad.state = SquadState.CONSTRUCTING
        builders_by_site.setdefault(site.id, []).append(squad)

    for site_id in sorted(builders_by_site):
        site = sim.state.buildings[site_id]
        bdef = sim.data.buildings[site.def_id]
        builders = builders_by_site[site_id]
        delta = len(builders) * DT / bdef.build_time if bdef.build_time > 0 else 1.0
        progress = site.progress + delta
        # `build_time / DT` ticks of float addition land just shy of 1.0, so
        # snap: without this a site could never actually finish.
        progress = 1.0 if progress >= 1.0 - _PROGRESS_EPS else progress
        site.hp = min(bdef.hp, site.hp + (progress - site.progress) * (1.0 - CONSTRUCTION_START_HP_FRAC) * bdef.hp)
        site.progress = progress
        if progress >= 1.0:
            _complete_building(sim, site, builders)


def _complete_building(sim: "Sim", site: "Building", builders: list["Squad"]) -> None:
    for squad in builders:
        _release_builder(squad)
    sim.state.events.append(
        Event(
            kind="building_completed",
            tick=sim.state.tick,
            data={"building": site.id, "def_id": site.def_id, "owner": site.owner},
        )
    )


def _release_builder(squad: "Squad") -> None:
    """Detach a squad from its construction site, clearing a spent `Build` order."""
    squad.build_target = None
    if isinstance(squad.order, orders_mod.Build):
        squad.order = None
    if squad.state is SquadState.CONSTRUCTING:
        squad.state = SquadState.IDLE


def _in_build_range(sim: "Sim", squad: "Squad", site: "Building") -> bool:
    """Is the squad within `BUILD_RANGE_CELLS` of the site's footprint edge?"""
    bdef = sim.data.buildings[site.def_id]
    cx0, cy0 = site.cell
    w, h = bdef.footprint
    x, y = squad.pos[0] / CELL_M, squad.pos[1] / CELL_M
    dx = max(cx0 - x, 0.0, x - (cx0 + w))
    dy = max(cy0 - y, 0.0, y - (cy0 + h))
    return math.hypot(dx, dy) <= BUILD_RANGE_CELLS


# ---------------------------------------------------------------------------
# Train / research queues
# ---------------------------------------------------------------------------


def _advance_queues(sim: "Sim") -> None:
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        if building.owner is None or building.progress < 1.0 or not building.queue:
            continue
        head = building.queue[0]
        head.remaining_s -= DT
        if head.remaining_s > 0.0:
            continue
        building.queue.pop(0)
        if head.kind == "train":
            _spawn_trained_squad(sim, building, head.item_id)
        else:
            _finish_research(sim, building, head.item_id)


def _spawn_trained_squad(sim: "Sim", building: "Building", def_id: str) -> None:
    sdef = sim.data.squads[def_id]
    cell = _rally_cell(sim, building, sdef.kind == "vehicle")
    if cell is None:  # nowhere on the map this unit can stand
        return
    squad = sim.spawn_squad(building.owner, def_id, center_of(cell, CELL_M))
    sim.state.events.append(
        Event(
            kind="unit_trained",
            tick=sim.state.tick,
            data={"building": building.id, "unit": def_id, "squad": squad.id, "owner": building.owner},
        )
    )


def _finish_research(sim: "Sim", building: "Building", upgrade_id: str) -> None:
    player = sim.state.players[building.owner]
    if upgrade_id not in player.upgrades:
        player.upgrades.append(upgrade_id)
    sim.state.events.append(
        Event(
            kind="research_completed",
            tick=sim.state.tick,
            data={"building": building.id, "upgrade": upgrade_id, "owner": building.owner},
        )
    )


def _rally_cell(sim: "Sim", building: "Building", is_vehicle: bool) -> tuple[int, int] | None:
    """Nearest passable cell to the one just south of the footprint's bottom centre."""
    bdef = sim.data.buildings[building.def_id]
    cx0, cy0 = building.cell
    w, h = bdef.footprint
    passable = sim.map.pass_veh if is_vehicle else sim.map.pass_inf
    return _nearest_passable(passable, (cx0 + (w - 1) // 2, cy0 + h))


def _nearest_passable(passable: np.ndarray, cell: tuple[int, int]) -> tuple[int, int] | None:
    """8-connected BFS outward from `cell` over a fixed neighbour order."""
    height, width = passable.shape
    cx = min(max(cell[0], 0), width - 1)
    cy = min(max(cell[1], 0), height - 1)
    if passable[cy, cx]:
        return (cx, cy)

    seen = {(cx, cy)}
    queue: deque[tuple[int, int]] = deque([(cx, cy)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _RING_OFFSETS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            if passable[ny, nx]:
                return (nx, ny)
            queue.append((nx, ny))
    return None
