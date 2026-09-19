"""Scripted baseline bots: the bottom two rungs of the benchmark ladder.

Both talk to the environment through nothing but `Observation` + `Order`, and
both are **data-driven**: nothing here names a unit, building or upgrade id.
What counts as "capture-capable infantry" or "an infantry-producing building"
is derived from the `GameData` handed to `reset` and the legal-order set in
`obs.available`, so the same bots keep working when the fixture tables are
replaced by the real CoH1 ones.

`T1Capper`'s policy, in priority order per decision step:

1. keep every production building that can make capture-capable infantry
   busy, and put up such a building if the player has none yet;
2. retreat any squad that is down to a third of its models with an enemy
   within `RETREAT_ENEMY_RANGE_M`, and reinforce squads sitting in friendly
   reinforce range under strength;
3. send every otherwise-idle capture-capable squad to the nearest point its
   team does not hold that no friendly squad is already heading for; when
   there is nothing left to capture, attack-move onto enemy ground.

Each squad receives at most one order per step, and an order is never
re-issued to a squad that is already carrying it out (`SquadView.order` is
compared against what the bot is about to send) — so the order log stays
small and the invalid-order rate stays near zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from coh.data.schema import Cost, GameData
from coh.env import (
    AttackMove,
    Build,
    BuildingView,
    Capture,
    Observation,
    Order,
    PointView,
    Reinforce,
    Retreat,
    SquadView,
    Train,
)
from coh.maps.format import GameMap

# --- T1 policy constants (behavioural knobs, not game data) ------------------

# A squad this far below strength with an enemy this close runs home.
RETREAT_MEMBER_FRAC = 1.0 / 3.0
RETREAT_ENEMY_RANGE_M = 40.0
# How many unarmed capture-capable squads (construction squads, typically) the
# bot is willing to own before it insists on infantry that can shoot back.
MAX_UNARMED_SQUADS = 2
# How far from the HQ a new building may be placed, in cells.
BUILD_SEARCH_RADIUS_CELLS = 24
# Cells of clearance kept around every other building, so a builder can
# always reach the new site and the old ones stay reachable.
BUILD_CLEARANCE_CELLS = 1
# How far from a capture point to look for a cell infantry can actually stand
# on, when attack-moving onto it.
APPROACH_SEARCH_RADIUS_CELLS = 6


_SETTLED_STATES = frozenset({"idle", "set_up", "setting_up"})


class T0Idle:
    """The floor of the ladder: builds nothing, moves nothing, orders nothing."""

    name = "t0"

    def reset(self, player_id: int, map: GameMap, data: GameData) -> None:
        self.player_id = player_id

    def act(self, obs: Observation) -> list[Order]:
        return []


@dataclass(frozen=True)
class _Site:
    """A candidate building placement: top-left cell plus its footprint."""

    cell: tuple[int, int]
    footprint: tuple[int, int]


class T1Capper:
    """Trains cheap infantry, grabs territory, retreats what is about to die."""

    name = "t1"

    def __init__(self) -> None:
        self.player_id = -1

    # -- lifecycle --------------------------------------------------------

    def reset(self, player_id: int, map: GameMap, data: GameData) -> None:
        self.player_id = player_id
        self.map = map
        self.data = data
        # Permanently-owned HQ-sector points: `Capture` on them is always
        # rejected, so they must never be picked as a target.
        self._hq_point_ids = {
            map.sectors[start.sector].point_id
            for start in map.starts
            if map.sectors[start.sector].point_id is not None
        }
        self._capture_units = frozenset(
            def_id
            for def_id, sdef in data.squads.items()
            if sdef.capture_rate > 0 and sdef.kind == "infantry"
        )
        # Construction squads are typically unarmed: useful for grabbing the
        # first few points and putting up buildings, useless in a fight. The
        # bot keeps a couple and then only trains infantry that can shoot.
        self._armed_capture_units = frozenset(
            def_id
            for def_id in self._capture_units
            if any(weapon for weapon in data.squads[def_id].loadout)
        )
        self._infantry_producers = frozenset(
            def_id
            for def_id, bdef in data.buildings.items()
            if not bdef.is_hq and any(unit in self._capture_units for unit in bdef.produces)
        )

    # -- policy -----------------------------------------------------------

    def act(self, obs: Observation) -> list[Order]:
        # `obs.available` prices everything against the resources the player
        # has *now*, so a step that spent twice over would have its second
        # order rejected. One running purse covers the whole step.
        purse = [obs.manpower, obs.munitions, obs.fuel]
        spoken_for: set[int] = set()

        orders: list[Order] = []
        orders += self._production_orders(obs, spoken_for, purse)
        orders += self._survival_orders(obs, spoken_for, purse)
        orders += self._territory_orders(obs, spoken_for)
        return orders

    # -- 1. production ----------------------------------------------------

    def _production_orders(self, obs: Observation, spoken_for: set[int], purse: list[float]) -> list[Order]:
        # The new building goes first: it is the bigger win, and it should get
        # the manpower ahead of one more squad.
        orders: list[Order] = []

        build_order = self._build_order(obs, spoken_for)
        if build_order is not None:
            if self._afford(purse, self.data.buildings[build_order.structure].cost):
                orders.append(build_order)
            else:  # can't pay this step after all: free the builder up again
                spoken_for.discard(build_order.squad)

        # Population is spent the same way and needs the same running total:
        # two buildings can each look trainable on their own and not both be.
        pop = obs.pop_used
        idle_queues = {b.id for b in obs.own_buildings if not b.queue and b.progress >= 1.0}
        unarmed = self._unarmed_count(obs)
        for building_id, units in sorted(obs.available["train"].items()):
            if building_id not in idle_queues:
                continue
            unit = self._cheapest(u for u in units if u in self._armed_capture_units)
            fallback = unit is None and unarmed < MAX_UNARMED_SQUADS
            if fallback:
                unit = self._cheapest(u for u in units if u in self._capture_units)
            if unit is None:
                continue
            sdef = self.data.squads[unit]
            if pop + sdef.population > obs.pop_cap or not self._afford(purse, sdef.cost):
                continue
            pop += sdef.population
            if fallback:
                unarmed += 1
            orders.append(Train(building=building_id, unit=unit))
        return orders

    def _unarmed_count(self, obs: Observation) -> int:
        """Unarmed capture squads the player owns or has on order."""
        def is_unarmed(def_id: str) -> bool:
            return def_id in self._capture_units and def_id not in self._armed_capture_units

        owned = sum(1 for squad in obs.own_squads if is_unarmed(squad.def_id))
        queued = sum(
            1 for building in obs.own_buildings for item in building.queue if is_unarmed(item)
        )
        return owned + queued

    @staticmethod
    def _afford(purse: list[float], cost) -> bool:
        """Spend `cost` out of `purse` if it fits; report whether it did."""
        if purse[0] < cost.manpower or purse[1] < cost.munitions or purse[2] < cost.fuel:
            return False
        purse[0] -= cost.manpower
        purse[1] -= cost.munitions
        purse[2] -= cost.fuel
        return True

    def _cheapest(self, unit_ids: Iterable[str]) -> str | None:
        best, best_cost = None, None
        for unit_id in sorted(unit_ids):
            cost = self.data.squads[unit_id].cost
            total = cost.manpower + cost.munitions + cost.fuel
            if best_cost is None or total < best_cost:
                best, best_cost = unit_id, total
        return best

    def _build_order(self, obs: Observation, spoken_for: set[int]) -> Build | None:
        """Put up an infantry-producing building if the player has none yet."""
        if any(b.def_id in self._infantry_producers for b in obs.own_buildings):
            return None
        # A builder already on the job: leave it alone rather than re-ordering
        # it every step (which would restart its walk and never finish).
        for squad in obs.own_squads:
            if squad.order is not None and squad.order["type"] == "Build":
                spoken_for.add(squad.id)
                return None

        candidates = [s for s in obs.available["build"] if s in self._infantry_producers]
        structure = self._cheapest_structure(candidates)
        if structure is None:
            return None
        builder = self._free_builder(obs, structure, spoken_for)
        if builder is None:
            return None
        site = self._find_site(obs, structure)
        if site is None:
            return None
        spoken_for.add(builder.id)
        return Build(squad=builder.id, structure=structure, cell=site.cell)

    def _cheapest_structure(self, structure_ids: Iterable[str]) -> str | None:
        best, best_cost = None, None
        for structure_id in sorted(structure_ids):
            cost = self.data.buildings[structure_id].cost
            total = cost.manpower + cost.munitions + cost.fuel
            if best_cost is None or total < best_cost:
                best, best_cost = structure_id, total
        return best

    def _free_builder(self, obs: Observation, structure: str, spoken_for: set[int]) -> SquadView | None:
        for squad in obs.own_squads:
            if squad.id in spoken_for or squad.state == "retreating":
                continue
            if structure in self.data.squads[squad.def_id].builds:
                return squad
        return None

    def _find_site(self, obs: Observation, structure: str) -> _Site | None:
        """First clear, in-supply spot for `structure`, spiralling out from the HQ.

        Staying inside the HQ's own sector is what guarantees the site is in
        supplied territory: an HQ sector is always connected for its owner.
        """
        hq = self._hq(obs)
        if hq is None:
            return None
        footprint = self.data.buildings[structure].footprint
        hq_footprint = self.data.buildings[hq.def_id].footprint
        origin = (hq.cell[0] + hq_footprint[0] // 2, hq.cell[1] + hq_footprint[1] // 2)
        sector = int(self.map.sector_id[origin[1], origin[0]])
        blocked = self._blocked_cells(obs)

        for cell in _spiral(origin, BUILD_SEARCH_RADIUS_CELLS):
            if self._site_clear(cell, footprint, sector, blocked):
                return _Site(cell=cell, footprint=footprint)
        return None

    def _hq(self, obs: Observation) -> BuildingView | None:
        for building in obs.own_buildings:
            if self.data.buildings[building.def_id].is_hq:
                return building
        return None

    def _occupied_cells(self, obs: Observation) -> set[tuple[int, int]]:
        """Cells the player knows a building stands on: nothing can walk there."""
        return self._building_cells(obs, margin=0)

    def _blocked_cells(self, obs: Observation) -> set[tuple[int, int]]:
        """Footprints of every building the player knows about, plus clearance."""
        return self._building_cells(obs, margin=BUILD_CLEARANCE_CELLS)

    def _building_cells(self, obs: Observation, *, margin: int) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        known: list[tuple[tuple[int, int], str]] = [
            (b.cell, b.def_id)
            for b in obs.own_buildings + obs.ally_buildings + obs.enemy_buildings + obs.neutral_buildings
        ]
        known += [(g.cell, g.def_id) for g in obs.ghosts]
        for (cx0, cy0), def_id in known:
            w, h = self._footprint(def_id)
            for dy in range(-margin, h + margin):
                for dx in range(-margin, w + margin):
                    cells.add((cx0 + dx, cy0 + dy))
        return cells

    def _footprint(self, def_id: str) -> tuple[int, int]:
        bdef = self.data.buildings.get(def_id) or self.data.neutral.get(def_id)
        return bdef.footprint if bdef is not None else (1, 1)

    def _site_clear(
        self,
        cell: tuple[int, int],
        footprint: tuple[int, int],
        sector: int,
        blocked: set[tuple[int, int]],
    ) -> bool:
        cx0, cy0 = cell
        w, h = footprint
        if cx0 < 0 or cy0 < 0 or cx0 + w > self.map.width or cy0 + h > self.map.height:
            return False
        for dy in range(h):
            for dx in range(w):
                cx, cy = cx0 + dx, cy0 + dy
                if (cx, cy) in blocked:
                    return False
                if not self.map.pass_veh[cy, cx]:
                    return False
                if int(self.map.sector_id[cy, cx]) != sector:
                    return False
        return True

    # -- 2. survival ------------------------------------------------------

    def _survival_orders(self, obs: Observation, spoken_for: set[int], purse: list[float]) -> list[Order]:
        orders: list[Order] = []
        for squad in obs.own_squads:
            # A retreating squad takes no orders at all until it gets home.
            if squad.id in spoken_for or squad.state == "retreating":
                continue
            if self.data.squads[squad.def_id].kind == "vehicle":
                continue  # vehicles can neither retreat nor reinforce

            shattered = squad.members <= max(1, math.floor(squad.max_members * RETREAT_MEMBER_FRAC))
            # Retreating from *inside* the reinforce radius is pointless —
            # the squad is already home, and reinforcing is the better move.
            if shattered and not squad.in_reinforce_range:
                if self._enemy_near(squad, obs.enemy_squads, RETREAT_ENEMY_RANGE_M):
                    spoken_for.add(squad.id)
                    orders.append(Retreat(squad=squad.id))
                    continue

            if not (squad.members < squad.max_members and squad.in_reinforce_range):
                continue
            if squad.state == "garrisoned":
                continue  # a garrisoned squad cannot be reinforced
            if squad.order is not None and squad.order["type"] == "Reinforce":
                # Already reinforcing: it restores one model after another on
                # its own, and *any* other order would cancel it. Hands off.
                spoken_for.add(squad.id)
                continue
            if not self._afford(purse, self._reinforce_model_cost(squad.def_id)):
                continue
            spoken_for.add(squad.id)
            orders.append(Reinforce(squad=squad.id))
        return orders

    def _reinforce_model_cost(self, def_id: str) -> Cost:
        """What one replacement model costs — the same formula the sim uses."""
        sdef = self.data.squads[def_id]
        frac = self.data.economy.reinforce_base_cost_frac * sdef.reinforce_cost_mult / sdef.members
        return Cost(
            manpower=sdef.cost.manpower * frac,
            munitions=sdef.cost.munitions * frac,
            fuel=sdef.cost.fuel * frac,
        )

    @staticmethod
    def _enemy_near(squad: SquadView, enemies: list[SquadView], radius_m: float) -> bool:
        return any(math.dist(squad.pos, enemy.pos) <= radius_m for enemy in enemies)

    # -- 3. territory -----------------------------------------------------

    def _territory_orders(self, obs: Observation, spoken_for: set[int]) -> list[Order]:
        claimed = {
            squad.order["point_id"]
            for squad in obs.own_squads + obs.ally_squads
            if squad.order is not None and squad.order["type"] == "Capture"
        }
        capturable = [
            point
            for point in obs.points
            if point.id not in self._hq_point_ids and point.owner_team != obs.team
        ]
        hostile = [
            point
            for point in obs.points
            if point.owner_team != obs.team and (point.owner_team is not None or point.type == "victory")
        ]

        occupied = self._occupied_cells(obs)
        approaches = {point.id: self._approach(point, occupied) for point in hostile}

        orders: list[Order] = []
        attacked: set[str] = set()
        for squad in obs.own_squads:
            if squad.id in spoken_for or squad.state == "retreating":
                continue
            if squad.def_id not in self._capture_units:
                continue
            if not self._idle(squad):
                continue
            target = self._nearest(squad, (p for p in capturable if p.id not in claimed))
            if target is not None:
                claimed.add(target.id)
                orders.append(Capture(squad=squad.id, point_id=target.id))
                continue
            # Nothing left to take: push onto enemy ground. Attack-move at the
            # nearest *walkable* cell to the point (an HQ point sits under its
            # own footprint, so the point cell itself is often unreachable —
            # ordering it would fail to path and be re-issued every step), and
            # never at the cell the squad is already standing on.
            enemy_ground = self._nearest(
                squad,
                (
                    p
                    for p in hostile
                    if p.id not in attacked and approaches[p.id] not in (None, squad.cell)
                ),
            )
            if enemy_ground is not None:
                attacked.add(enemy_ground.id)
                orders.append(AttackMove(squad=squad.id, cell=approaches[enemy_ground.id]))
        return orders

    def _approach(self, point: PointView, occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
        """The nearest cell next to `point` that infantry can actually stand on.

        The terrain layer alone is not enough: buildings block their footprint
        too (an HQ point sits under its own HQ), and a cell the squad can never
        reach would make the order fail to path and be re-issued every step.
        """
        for cell in _spiral(point.cell, APPROACH_SEARCH_RADIUS_CELLS):
            cx, cy = cell
            if not (0 <= cx < self.map.width and 0 <= cy < self.map.height):
                continue
            if self.map.pass_inf[cy, cx] and cell not in occupied:
                return cell
        return None

    @staticmethod
    def _idle(squad: SquadView) -> bool:
        """Free to be re-tasked: no standing order, and not mid-manoeuvre."""
        return squad.order is None and squad.state in _SETTLED_STATES

    @staticmethod
    def _nearest(squad: SquadView, points: Iterable[PointView]) -> PointView | None:
        best, best_d2 = None, None
        for point in points:
            d2 = (point.cell[0] - squad.cell[0]) ** 2 + (point.cell[1] - squad.cell[1]) ** 2
            if best_d2 is None or d2 < best_d2:
                best, best_d2 = point, d2
        return best


def _spiral(origin: tuple[int, int], radius: int) -> Iterable[tuple[int, int]]:
    """Cells around `origin`, nearest first, in a fixed deterministic order."""
    ox, oy = origin
    cells = [
        (ox + dx, oy + dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    cells.sort(key=lambda c: ((c[0] - ox) ** 2 + (c[1] - oy) ** 2, c[1], c[0]))
    return cells
