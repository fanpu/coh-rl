"""The fog-filtered, per-player `Observation` and the legal-order set.

`build_observation(sim, player_id)` is the single place where the simulation's
omniscient `GameState` is narrowed to what one player may legally know:

- **own / ally entities** are reported in full (order, cover, reinforce range);
- **enemy squads** appear only while `vision.is_visible` holds for the
  player's team, and never carry `order` / `cover` / `in_reinforce_range`;
- **enemy buildings** appear only while visible; once they leave vision the
  team's last-known state of them is reported as a `GhostView`;
- **points** are public knowledge (as in CoH: the minimap shows ownership and
  capture progress for everyone), except `connected`, which is supply
  information and is `None` for enemy-owned points.

Every view is a plain frozen dataclass of JSON types, and
`Observation.to_dict()` returns a plain, `json.dumps`-able structure that M2's
tensor / text encoders build on.

`available` lists only the options that are legal *right now* — each candidate
is run through the sim's own `validate_order`, so the set can never drift away
from what `Sim.issue` would accept (`"build"` is the one exception: it ignores
placement, since "where" is the agent's problem).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from coh.maps.cover import cover_at
from coh.maps.format import cell_of, center_of
from coh.sim.constants import CELL_M, DT
from coh.sim.orders import BuyUpgrade, Research, Train, order_to_dict, validate_order
from coh.sim.state import Building, Squad
from coh.sim.systems import economy, production, vision

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim


@dataclass(frozen=True)
class SquadView:
    id: int
    def_id: str
    owner: int
    pos: tuple[float, float]
    cell: tuple[int, int]
    members: int
    max_members: int
    hp_frac: float
    suppressed: bool
    pinned: bool
    state: str
    order: dict | None  # own/ally only; None for enemies
    cover: str | None  # own/ally only
    in_reinforce_range: bool | None  # own/ally only


@dataclass(frozen=True)
class BuildingView:
    id: int
    def_id: str
    owner: int | None
    cell: tuple[int, int]
    hp_frac: float
    progress: float
    queue: list[str]  # own/ally only; empty for enemy and neutral buildings
    garrison_count: int


@dataclass(frozen=True)
class GhostView:
    id: int
    def_id: str
    owner: int | None
    cell: tuple[int, int]
    last_seen_s: float


@dataclass(frozen=True)
class PointView:
    id: str
    name: str
    type: str
    cell: tuple[int, int]
    owner_team: int | None
    progress: float
    connected: bool | None  # None for enemy-owned points
    has_op: bool


@dataclass(frozen=True)
class Observation:
    player_id: int
    team: int
    faction: str
    time_s: float
    manpower: float
    munitions: float
    fuel: float
    income: dict[str, float]
    pop_used: int
    pop_cap: int
    upgrades: list[str]
    tickets: dict[int, float]
    own_squads: list[SquadView]
    ally_squads: list[SquadView]
    enemy_squads: list[SquadView]
    own_buildings: list[BuildingView]
    ally_buildings: list[BuildingView]
    enemy_buildings: list[BuildingView]
    ghosts: list[GhostView]
    neutral_buildings: list[BuildingView]
    points: list[PointView]
    available: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """A plain, `json.dumps`-able view of this observation.

        Tuples become lists so a round trip through JSON compares equal;
        integer dict keys (`tickets`, `available["train"]`, ...) are left as
        integers, which `json.dumps` renders as strings.
        """
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Entity views
# ---------------------------------------------------------------------------


def _squad_hp_frac(sim: "Sim", squad: Squad) -> float:
    sdef = sim.data.squads[squad.def_id]
    max_hp = sdef.members * sdef.member_hp
    if max_hp <= 0:
        return 0.0
    return max(0.0, min(1.0, sum(m.hp for m in squad.members if m.hp > 0) / max_hp))


def _building_hp_frac(sim: "Sim", building: Building) -> float:
    bdef = sim.data.buildings.get(building.def_id) or sim.data.neutral.get(building.def_id)
    max_hp = bdef.hp if bdef is not None else building.hp
    if max_hp <= 0:
        return 0.0
    return max(0.0, min(1.0, building.hp / max_hp))


def _rect_distance_m(pos: np.ndarray, cell: tuple[int, int], footprint: tuple[int, int]) -> float:
    """Distance in meters from `pos` to a building footprint's rectangle."""
    cx0, cy0 = cell
    w, h = footprint
    x0, y0 = cx0 * CELL_M, cy0 * CELL_M
    x1, y1 = (cx0 + w) * CELL_M, (cy0 + h) * CELL_M
    dx = max(x0 - pos[0], 0.0, pos[0] - x1)
    dy = max(y0 - pos[1], 0.0, pos[1] - y1)
    return float(np.hypot(dx, dy))


def in_reinforce_range(sim: "Sim", squad: Squad) -> bool:
    """Is this squad inside a friendly completed building's reinforce radius?"""
    team = sim.state.players[squad.owner].team
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        if building.owner is None or building.progress < 1.0:
            continue
        owner = sim.state.players.get(building.owner)
        if owner is None or owner.team != team:
            continue
        bdef = sim.data.buildings.get(building.def_id)
        if bdef is None or bdef.reinforce_radius <= 0:
            continue
        if _rect_distance_m(squad.pos, building.cell, bdef.footprint) <= bdef.reinforce_radius:
            return True
    return False


def _squad_cover(sim: "Sim", squad: Squad, threat_pos: np.ndarray | None) -> str:
    """Cover the squad currently benefits from, against `threat_pos`.

    With no known enemy to take cover *from*, the shooter position is the
    squad's own cell centre, which reduces `cover_at` to the cell's
    non-directional area/road cover.
    """
    cell = cell_of(squad.pos, CELL_M)
    from_pos = center_of(cell, CELL_M) if threat_pos is None else threat_pos
    return cover_at(sim.map, cell, from_pos)


def _nearest_threat_pos(squad: Squad, enemies: list[Squad]) -> np.ndarray | None:
    best, best_d2 = None, None
    for enemy in enemies:
        delta = enemy.pos - squad.pos
        d2 = float(delta[0] * delta[0] + delta[1] * delta[1])
        if best_d2 is None or d2 < best_d2:
            best, best_d2 = enemy.pos, d2
    return best


def _squad_view(sim: "Sim", squad: Squad, *, friendly: bool, threats: list[Squad]) -> SquadView:
    sdef = sim.data.squads[squad.def_id]
    return SquadView(
        id=squad.id,
        def_id=squad.def_id,
        owner=squad.owner,
        pos=(float(squad.pos[0]), float(squad.pos[1])),
        cell=cell_of(squad.pos, CELL_M),
        members=len(squad.alive_members),
        max_members=sdef.members,
        hp_frac=_squad_hp_frac(sim, squad),
        suppressed=squad.suppressed,
        pinned=squad.pinned,
        state=squad.state.value,
        order=(order_to_dict(squad.order) if friendly and squad.order is not None else None),
        cover=_squad_cover(sim, squad, _nearest_threat_pos(squad, threats)) if friendly else None,
        in_reinforce_range=in_reinforce_range(sim, squad) if friendly else None,
    )


def _building_view(sim: "Sim", building: Building, *, friendly: bool, garrison_count: int) -> BuildingView:
    return BuildingView(
        id=building.id,
        def_id=building.def_id,
        owner=building.owner,
        cell=building.cell,
        hp_frac=_building_hp_frac(sim, building),
        progress=building.progress,
        queue=[item.item_id for item in building.queue] if friendly else [],
        garrison_count=garrison_count,
    )


# ---------------------------------------------------------------------------
# Legal-order set
# ---------------------------------------------------------------------------


def _buildable(sim: "Sim", player_id: int) -> list[str]:
    """Structure ids this player's builders could place right now, anywhere.

    Location is deliberately ignored (the agent picks the cell and the sim
    validates it), so this checks only what does not depend on placement:
    the builder can build it, the faction matches, tech requirements are met
    and the player can pay for it.
    """
    player = sim.state.players[player_id]
    out: list[str] = []
    for squad_id in sorted(sim.state.squads):
        squad = sim.state.squads[squad_id]
        if squad.owner != player_id or not squad.alive_members:
            continue
        for structure in sim.data.squads[squad.def_id].builds:
            if structure in out:
                continue
            bdef = sim.data.buildings.get(structure)
            if bdef is None or bdef.faction != player.faction:
                continue
            if production.missing_requirements(sim, player_id, bdef.requires):
                continue
            if not production.can_afford(player, bdef.cost):
                continue
            out.append(structure)
    return sorted(out)


def available_orders(sim: "Sim", player_id: int) -> dict[str, Any]:
    """The currently legal train / build / research / squad-upgrade options."""
    train: dict[int, list[str]] = {}
    research: dict[int, list[str]] = {}
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        if building.owner != player_id:
            continue
        bdef = sim.data.buildings.get(building.def_id)
        if bdef is None:
            continue
        units = [u for u in bdef.produces if validate_order(sim, player_id, Train(building_id, u)).ok]
        if units:
            train[building_id] = units
        upgrades = [u for u in bdef.researches if validate_order(sim, player_id, Research(building_id, u)).ok]
        if upgrades:
            research[building_id] = upgrades

    squad_upgrades: dict[int, list[str]] = {}
    for squad_id in sorted(sim.state.squads):
        squad = sim.state.squads[squad_id]
        if squad.owner != player_id or not squad.alive_members:
            continue
        options = [
            upgrade_id
            for upgrade_id in sorted(sim.data.squad_upgrades)
            if squad.def_id in sim.data.squad_upgrades[upgrade_id].applies_to
            and validate_order(sim, player_id, BuyUpgrade(squad_id, upgrade_id)).ok
        ]
        if options:
            squad_upgrades[squad_id] = options

    return {
        "train": train,
        "build": _buildable(sim, player_id),
        "research": research,
        "squad_upgrades": squad_upgrades,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_observation(sim: "Sim", player_id: int) -> Observation:
    """The fog-filtered observation of `sim` for `player_id`."""
    player = sim.state.players[player_id]
    team = player.team

    own_squads: list[SquadView] = []
    ally_squads: list[SquadView] = []
    enemy_squads: list[SquadView] = []

    visible_enemy_squads = [
        sim.state.squads[sid]
        for sid in sorted(sim.state.squads)
        if sim.state.players[sim.state.squads[sid].owner].team != team
        and sim.state.squads[sid].alive_members
        and vision.is_visible(sim, team, sim.state.squads[sid])
    ]

    for squad_id in sorted(sim.state.squads):
        squad = sim.state.squads[squad_id]
        if not squad.alive_members:
            continue
        owner_team = sim.state.players[squad.owner].team
        if owner_team == team:
            view = _squad_view(sim, squad, friendly=True, threats=visible_enemy_squads)
            (own_squads if squad.owner == player_id else ally_squads).append(view)
        elif vision.is_visible(sim, team, squad):
            enemy_squads.append(_squad_view(sim, squad, friendly=False, threats=[]))

    own_buildings: list[BuildingView] = []
    ally_buildings: list[BuildingView] = []
    enemy_buildings: list[BuildingView] = []
    neutral_buildings: list[BuildingView] = []

    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        visible = vision.is_visible(sim, team, building)
        if building.neutral or building.owner is None:
            # Neutral buildings are map furniture: their positions are public,
            # but who is hiding inside is not.
            count = (
                len(building.garrison)
                if visible
                else sum(
                    1
                    for sid in building.garrison
                    if sid in sim.state.squads and sim.state.players[sim.state.squads[sid].owner].team == team
                )
            )
            neutral_buildings.append(_building_view(sim, building, friendly=False, garrison_count=count))
            continue
        owner_team = sim.state.players[building.owner].team
        if owner_team == team:
            view = _building_view(sim, building, friendly=True, garrison_count=len(building.garrison))
            (own_buildings if building.owner == player_id else ally_buildings).append(view)
        elif visible:
            enemy_buildings.append(
                _building_view(sim, building, friendly=False, garrison_count=len(building.garrison))
            )

    visible_enemy_building_ids = {view.id for view in enemy_buildings}
    ghosts = [
        GhostView(
            id=ghost.building_id,
            def_id=ghost.def_id,
            owner=ghost.owner,
            cell=ghost.cell,
            last_seen_s=ghost.last_seen_tick * DT,
        )
        for ghost in (
            sim.state.ghosts.get(team, {})[bid] for bid in sorted(sim.state.ghosts.get(team, {}))
        )
        if ghost.building_id not in visible_enemy_building_ids
    ]

    connected = set(sim.state.connected.get(team, ()))
    points = []
    for point_id in sorted(sim.map.points):
        point_def = sim.map.points[point_id]
        point = sim.state.points[point_id]
        enemy_owned = point.owner_team is not None and point.owner_team != team
        points.append(
            PointView(
                id=point_id,
                name=point_def.name,
                type=point_def.type,
                cell=point_def.cell,
                owner_team=point.owner_team,
                progress=point.progress,
                connected=None if enemy_owned else (point_def.sector in connected),
                has_op=economy.has_observation_post(sim, point),
            )
        )

    income = economy.income_per_min(sim, player_id)
    return Observation(
        player_id=player_id,
        team=team,
        faction=player.faction,
        time_s=sim.state.tick * DT,
        manpower=player.manpower,
        munitions=player.munitions,
        fuel=player.fuel,
        income={"manpower": income.manpower, "munitions": income.munitions, "fuel": income.fuel},
        pop_used=economy.pop_used(sim, player_id),
        pop_cap=economy.pop_cap(sim, player_id),
        upgrades=list(player.upgrades),
        tickets={team_id: sim.state.tickets[team_id] for team_id in sorted(sim.state.tickets)},
        own_squads=own_squads,
        ally_squads=ally_squads,
        enemy_squads=enemy_squads,
        own_buildings=own_buildings,
        ally_buildings=ally_buildings,
        enemy_buildings=enemy_buildings,
        ghosts=ghosts,
        neutral_buildings=neutral_buildings,
        points=points,
        available=available_orders(sim, player_id),
    )
