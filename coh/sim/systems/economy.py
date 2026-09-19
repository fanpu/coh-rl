"""economy — income, upkeep and population cap (task 14).

Income is computed **per player, per tick**. Sector connectivity is a *team*
property (`state.connected[team]`, maintained by `systems/territory.py`), but
income is not split: every player on the team receives the full income of
every connected point the team owns, matching CoH's team economy.

    income/min = base_income_per_min
               + Σ point_income[point.type] × (op_income_mult if the point
                 carries a completed observation post else 1)
                 over the team's *connected* owned points
    manpower  -= Σ upkeep_per_min of the player's own living squads
                 × Π upkeep_mult of the player's researched upgrades
    manpower   = max(manpower, base_income_per_min.manpower
                               × min_manpower_income_frac)

Each tick every player gains `income/min × DT / 60`.

Population works the same way: `pop_cap` counts the team's connected points,
`pop_used` counts only the player's own living squads plus the units they
have queued but not yet received. `income_per_min` / `pop_used` / `pop_cap`
are public: the env's observation builder reads them directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coh.data.schema import Cost, PointIncome
from coh.sim.constants import DT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from coh.sim.sim import Sim
    from coh.sim.state import PointState, Squad

_SECONDS_PER_MINUTE = 60.0


def run(sim: "Sim") -> None:
    """Advance the economy system by one tick."""
    for player_id in sorted(sim.state.players):
        player = sim.state.players[player_id]
        income = income_per_min(sim, player_id)
        share = DT / _SECONDS_PER_MINUTE
        player.manpower += income.manpower * share
        player.munitions += income.munitions * share
        player.fuel += income.fuel * share


# ---------------------------------------------------------------------------
# Income
# ---------------------------------------------------------------------------


def income_per_min(sim: "Sim", player_id: int) -> Cost:
    """This player's current resource income per minute, upkeep included."""
    econ = sim.data.economy
    player = sim.state.players[player_id]
    base = econ.base_income_per_min

    manpower, munitions, fuel = base.manpower, base.munitions, base.fuel
    for point_def, point in connected_points(sim, player.team):
        share = econ.point_income.get(point_def.type, PointIncome())
        mult = econ.op_income_mult if has_observation_post(sim, point) else 1.0
        manpower += share.manpower * mult
        munitions += share.munitions * mult
        fuel += share.fuel * mult

    floor = base.manpower * econ.min_manpower_income_frac
    manpower = max(manpower - upkeep_per_min(sim, player_id), floor)
    return Cost(manpower=manpower, munitions=munitions, fuel=fuel)


def upkeep_per_min(sim: "Sim", player_id: int) -> float:
    """Manpower upkeep of this player's living squads, after upgrade discounts."""
    player = sim.state.players[player_id]
    total = sum(sim.data.squads[squad.def_id].upkeep_per_min for squad in living_squads(sim, player_id))
    mult = 1.0
    for upgrade_id in player.upgrades:
        upgrade = sim.data.upgrades.get(upgrade_id)
        if upgrade is not None:
            mult *= upgrade.upkeep_mult
    return total * mult


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def pop_used(sim: "Sim", player_id: int) -> int:
    """Population of the player's living squads plus everything they have queued.

    Queued units already count so that a player cannot fill several queues
    past their cap and then be handed the units for free.
    """
    used = sum(sim.data.squads[squad.def_id].population for squad in living_squads(sim, player_id))
    for building_id in sorted(sim.state.buildings):
        building = sim.state.buildings[building_id]
        if building.owner != player_id:
            continue
        for item in building.queue:
            if item.kind == "train":
                used += sim.data.squads[item.item_id].population
    return used


def pop_cap(sim: "Sim", player_id: int) -> int:
    """Base population plus the team's connected points' bonuses, capped."""
    econ = sim.data.economy
    team = sim.state.players[player_id].team
    cap = econ.base_population
    for point_def, _point in connected_points(sim, team):
        cap += econ.point_income.get(point_def.type, PointIncome()).population
    return min(econ.max_population, cap)


# ---------------------------------------------------------------------------
# Shared lookups
# ---------------------------------------------------------------------------


def connected_points(sim: "Sim", team: int) -> list[tuple]:
    """`(PointDef, PointState)` for every point this team owns *and* supplies."""
    connected = set(sim.state.connected.get(team, ()))
    out = []
    for point_id in sorted(sim.state.points):
        point = sim.state.points[point_id]
        if point.owner_team != team:
            continue
        point_def = sim.map.points[point_id]
        if point_def.sector not in connected:
            continue
        out.append((point_def, point))
    return out


def has_observation_post(sim: "Sim", point: "PointState") -> bool:
    """Whether `point` carries a *completed*, still-living observation post."""
    if point.op_building is None:
        return False
    building = sim.state.buildings.get(point.op_building)
    return building is not None and building.progress >= 1.0


def living_squads(sim: "Sim", player_id: int) -> list["Squad"]:
    """The player's squads that still have a crew (abandoned shells excluded)."""
    out = []
    for squad_id in sorted(sim.state.squads):
        squad = sim.state.squads[squad_id]
        if squad.owner == player_id and not squad.abandoned and squad.alive_members:
            out.append(squad)
    return out
