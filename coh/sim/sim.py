"""The `Sim`: owns the state, the map copy, order intake and the tick loop.

`Sim` is the only entry point into the simulation. It is deterministic given
`(map, players, data, seed)` and the sequence of `issue()` / `tick()` calls:
the only randomness is `state.rng`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coh.data.schema import BuildingDef, GameData
from coh.maps.format import GameMap, StartDef, center_of
from coh.sim import orders as orders_mod
from coh.sim import systems
from coh.sim.orders import Order, OrderResult
from coh.sim.state import (
    Building,
    GameState,
    Member,
    Player,
    PointState,
    Squad,
    SquadState,
    state_hash,
)


class SimError(ValueError):
    """Raised for a setup that cannot produce a playable game."""


@dataclass(frozen=True)
class PlayerSetup:
    faction: str
    team: int
    start_slot: int


@dataclass(frozen=True)
class SimConfig:
    time_limit_s: float = 2700.0


def neutral_footprints(data: GameData) -> dict[str, tuple[int, int]]:
    """Neutral-building footprints for `coh.maps.format.load_map(footprints=...)`.

    `coh/maps` deliberately does not import `coh/data`, so a caller that has
    `GameData` loaded passes the authoritative footprints in.
    """
    return {def_id: nb.footprint for def_id, nb in data.neutral.items()}


class Sim:
    def __init__(
        self,
        game_map: GameMap,
        players: list[PlayerSetup],
        data: GameData,
        seed: int,
        config: SimConfig = SimConfig(),
    ) -> None:
        self.data = data
        self.config = config
        self.seed = seed
        self.player_setups = list(players)
        # The sim mutates the map (footprints, crushed cover, craters), so it
        # works on a private copy and never touches the caller's.
        self.map = game_map.copy()

        self.state = GameState(
            tick=0,
            rng=np.random.default_rng(seed),
            players={},
            squads={},
            buildings={},
            points={pid: PointState() for pid in self.map.points},
            connected={},
            tickets={},
            visible={},
            ghosts={},
            next_id=1,
        )
        # Pathfinding results, keyed (is_vehicle, start, goal, map.version) --
        # see coh/sim/pathfinding.py. Stale entries (from before a version
        # bump) are simply never looked up again.
        self._path_cache: dict[tuple, list[tuple[int, int]] | None] = {}

        self._setup_players()
        self._place_neutral_buildings()

    # -- setup -----------------------------------------------------------

    def _setup_players(self) -> None:
        starts = {s.slot: s for s in self.map.starts}
        econ = self.data.economy

        for player_id, setup in enumerate(self.player_setups):
            start = starts.get(setup.start_slot)
            if start is None:
                raise SimError(f"player {player_id}: map {self.map.name!r} has no start slot {setup.start_slot}")

            hq_def = self._hq_def(setup.faction)
            hq = self.spawn_building(player_id, hq_def.id, start.hq_cell)
            self.state.players[player_id] = Player(
                id=player_id,
                team=setup.team,
                faction=setup.faction,
                manpower=econ.start_resources.manpower,
                munitions=econ.start_resources.munitions,
                fuel=econ.start_resources.fuel,
                hq_id=hq.id,
            )
            self.spawn_squad(
                player_id,
                self._builder_def_id(setup.faction),
                center_of(self._builder_cell(start.hq_cell, hq_def.footprint)),
            )
            self._claim_hq_sector(start, setup.team)

        teams = sorted({setup.team for setup in self.player_setups})
        self.state.tickets = {team: float(econ.tickets) for team in teams}
        self.state.visible = {team: np.zeros((self.map.height, self.map.width), dtype=bool) for team in teams}
        self.state.ghosts = {team: {} for team in teams}
        self.state.connected = {
            team: sorted(
                {
                    starts[setup.start_slot].sector
                    for setup in self.player_setups
                    if setup.team == team
                }
            )
            for team in teams
        }

    def _hq_def(self, faction: str) -> BuildingDef:
        for def_id in sorted(self.data.buildings):
            bdef = self.data.buildings[def_id]
            if bdef.is_hq and bdef.faction == faction:
                return bdef
        raise SimError(f"no is_hq building for faction {faction!r}")

    def _builder_def_id(self, faction: str) -> str:
        """The squad that starts on the field: the faction's construction squad.

        If the data has no construction squad for this faction (fixture
        tables), fall back to any faction's — faction rules are enforced at
        Build/Train time, not at spawn.
        """
        builders = [def_id for def_id in sorted(self.data.squads) if self.data.squads[def_id].builds]
        for def_id in builders:
            if self.data.squads[def_id].faction == faction:
                return def_id
        if builders:
            return builders[0]
        raise SimError("no squad def can construct buildings; cannot place a starting builder")

    def _builder_cell(self, hq_cell: tuple[int, int], footprint: tuple[int, int]) -> tuple[int, int]:
        """Nearest infantry-passable cell south of the HQ footprint."""
        cx0, cy0 = hq_cell
        w, h = footprint
        centre_x = cx0 + (w - 1) / 2
        columns = sorted(range(self.map.width), key=lambda cx: (abs(cx - centre_x), cx))
        for cy in range(cy0 + h, self.map.height):
            for cx in columns:
                if self.map.pass_inf[cy, cx]:
                    return (cx, cy)
        raise SimError(f"no infantry-passable cell south of the HQ footprint at {hq_cell}")

    def _claim_hq_sector(self, start: StartDef, team: int) -> None:
        """A team owns its HQ sector's point from the start."""
        sector = self.map.sectors.get(start.sector)
        if sector is None or sector.point_id is None:
            return
        self.state.points[sector.point_id] = PointState(owner_team=team, progress=1.0)

    def _place_neutral_buildings(self) -> None:
        # Footprints are already stamped impassable by `load_map`, so this
        # only creates the entities.
        for placement in self.map.neutral_buildings:
            self.spawn_building(None, placement.def_id, placement.cell)

    # -- entity creation --------------------------------------------------

    def _take_id(self) -> int:
        entity_id = self.state.next_id
        self.state.next_id += 1
        return entity_id

    def spawn_squad(self, owner: int, def_id: str, pos) -> Squad:
        """Create a squad at a world position (meters). Faction is not checked."""
        sdef = self.data.squads.get(def_id)
        if sdef is None:
            raise SimError(f"unknown squad def {def_id!r}")
        squad = Squad(
            id=self._take_id(),
            owner=owner,
            def_id=def_id,
            pos=np.asarray(pos, dtype=float).copy(),
            heading=0.0,
            members=[Member(hp=sdef.member_hp, weapon=weapon) for weapon in sdef.loadout],
            state=SquadState.IDLE,
        )
        self.state.squads[squad.id] = squad
        return squad

    def spawn_building(
        self, owner: int | None, def_id: str, cell: tuple[int, int], complete: bool = True
    ) -> Building:
        """Create a building and stamp its footprint impassable.

        `def_id` may name a player building (`data.buildings`) or a neutral,
        enterable one (`data.neutral`); neutral defs always spawn unowned.
        """
        if def_id in self.data.buildings:
            bdef = self.data.buildings[def_id]
            footprint, hp, neutral = bdef.footprint, bdef.hp, False
        elif def_id in self.data.neutral:
            ndef = self.data.neutral[def_id]
            footprint, hp, neutral = ndef.footprint, ndef.hp, True
            owner = None
        else:
            raise SimError(f"unknown building def {def_id!r}")

        cx, cy = cell
        if not (0 <= cx and cx + footprint[0] <= self.map.width and 0 <= cy and cy + footprint[1] <= self.map.height):
            raise SimError(f"building {def_id!r} at {cell} with footprint {footprint} does not fit the map")

        building = Building(
            id=self._take_id(),
            owner=owner,
            def_id=def_id,
            cell=(cx, cy),
            hp=hp,
            progress=1.0 if complete else 0.0,
            neutral=neutral,
        )
        self.state.buildings[building.id] = building
        self.map.stamp_footprint(building.cell, footprint, blocked=True)
        return building

    # -- orders / tick ----------------------------------------------------

    def issue(self, player_id: int, orders: list[Order]) -> list[OrderResult]:
        """Validate and apply orders now; invalid ones are counted, never raised."""
        results: list[OrderResult] = []
        for order in orders:
            result = orders_mod.validate_order(self, player_id, order)
            if result.ok:
                orders_mod.apply_order(self, order)
            else:
                player = self.state.players.get(player_id)
                if player is not None:
                    player.invalid_orders += 1
            results.append(result)
        return results

    def tick(self) -> None:
        """Run one simulation tick. A no-op once the game has been won."""
        if self.state.winner is not None:
            return
        self.state.events.clear()
        for system in systems.SYSTEM_ORDER:
            system.run(self)
        self.state.tick += 1

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.tick()

    def state_hash(self) -> str:
        """sha256 over a canonical serialization (floats rounded to 1e-6)."""
        return state_hash(self.state)
