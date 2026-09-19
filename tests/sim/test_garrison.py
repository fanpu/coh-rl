"""Tests for garrisons and building-death cleanup (task 12)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coh.data.schema import GameData
from coh.env.observation import build_observation
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import (
    Attack,
    Build,
    Capture,
    Garrison,
    Move,
    Reinforce,
    Retreat,
    SetFacing,
    Stop,
    Ungarrison,
)
from coh.sim.state import QueueItem, SquadState
from coh.sim.systems import combat, garrison, vision
from tests.helpers import fixture_data, make_sim, spawn

WIDTH, HEIGHT = 40, 30

STARTS = [
    {"slot": 0, "team": 0, "hq_cell": [2, 2], "sector": "a"},
    {"slot": 1, "team": 1, "hq_cell": [34, 22], "sector": "c"},
]

# A 5x4 barn: big enough that its own footprint sits between an occupant at
# the centre and anything shooting at it from the east, so the LOS exemption
# for a shooter's / target's own building is actually exercised.
BARN_CELL = (18, 8)
BARN_FOOTPRINT = (5, 4)
BARN_CENTRE_CELL = (20, 10)
EAST_CELL = (26, 10)
WEST_CELL = (10, 10)


def barn_sim(seed: int = 0, *, def_id: str = "barn", cell=BARN_CELL, data: GameData | None = None):
    sim = make_sim(
        seed=seed,
        starts=STARTS,
        data=data,
        neutral_buildings=[{"def": def_id, "cell": list(cell)}],
    )
    return sim, next(b for b in sim.state.buildings.values() if b.neutral)


def with_cover(weapon_id: str, **mods) -> GameData:
    """Fixture data with a `garrison` cover row added to one weapon."""
    from coh.data.schema import CoverMods

    data = fixture_data()
    weapons = dict(data.weapons)
    table = dict(weapons[weapon_id].cover_table)
    table["garrison"] = CoverMods(**mods)
    weapons[weapon_id] = replace(weapons[weapon_id], cover_table=table)
    return replace(data, weapons=weapons)


def garrison_now(sim, squad, building):
    """Put `squad` straight inside `building` (bypassing the walk)."""
    garrison.enter(sim, squad, building)
    return squad


def run_until(sim, predicate, max_ticks: int = 400) -> bool:
    for _ in range(max_ticks):
        if predicate():
            return True
        sim.tick()
    return predicate()


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def test_garrison_order_walks_to_the_building_and_enters():
    sim, barn = barn_sim()
    squad = spawn(sim, 0, "rifles", EAST_CELL)

    assert sim.issue(0, [Garrison(squad.id, barn.id)])[0].ok
    assert run_until(sim, lambda: squad.garrison_in is not None, 200)

    assert squad.state is SquadState.GARRISONED
    assert squad.order is None
    assert squad.path == []
    assert not squad.moving
    assert barn.garrison == [squad.id]
    assert tuple(np.floor(squad.pos / 2.0).astype(int)) == BARN_CENTRE_CELL


def test_garrison_on_a_player_building_is_invalid():
    sim, _ = barn_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    barracks = sim.spawn_building(0, "barracks", (12, 10))

    result = sim.issue(0, [Garrison(squad.id, barracks.id)])[0]
    assert not result.ok
    assert "enterable" in result.reason


def test_garrison_respects_capacity():
    sim, house = barn_sim(def_id="house_small", cell=(20, 9))  # capacity 1
    first = spawn(sim, 0, "rifles", (24, 10))
    second = spawn(sim, 0, "rifles", (25, 10))

    garrison_now(sim, first, house)
    result = sim.issue(0, [Garrison(second.id, house.id)])[0]
    assert not result.ok
    assert "full" in result.reason


def test_enemy_cannot_enter_an_occupied_building():
    sim, barn = barn_sim()
    holder = spawn(sim, 0, "rifles", EAST_CELL)
    garrison_now(sim, holder, barn)
    intruder = spawn(sim, 1, "rifles", WEST_CELL)

    result = sim.issue(1, [Garrison(intruder.id, barn.id)])[0]
    assert not result.ok
    assert "occupied" in result.reason
    assert sim.state.players[1].invalid_orders == 1


def test_a_building_taken_while_walking_leaves_the_squad_outside():
    sim, barn = barn_sim()
    walker = spawn(sim, 1, "rifles", EAST_CELL)
    assert sim.issue(1, [Garrison(walker.id, barn.id)])[0].ok

    holder = spawn(sim, 0, "rifles", WEST_CELL)
    garrison_now(sim, holder, barn)

    assert run_until(sim, lambda: walker.order is None, 200)
    assert walker.garrison_in is None
    assert walker.state is SquadState.IDLE
    assert barn.garrison == [holder.id]


# ---------------------------------------------------------------------------
# exit
# ---------------------------------------------------------------------------


def _ring_cells(sim, building):
    from coh.sim import pathfinding

    return set(pathfinding.adjacent_cells(building.cell, BARN_FOOTPRINT, sim.map.width, sim.map.height))


def test_ungarrison_puts_the_squad_on_an_adjacent_passable_cell():
    sim, barn = barn_sim()
    squad = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)

    assert sim.issue(0, [Ungarrison(squad.id)])[0].ok
    assert squad.garrison_in is None
    assert squad.state is SquadState.IDLE
    assert squad.order is None
    assert barn.garrison == []

    cell = tuple(np.floor(squad.pos / 2.0).astype(int))
    assert cell in _ring_cells(sim, barn)
    assert sim.map.pass_inf[cell[1], cell[0]]
    # No visible enemy: the cell nearest the owner's HQ (north-west of here).
    assert cell[0] == BARN_CELL[0] - 1


def test_ungarrison_exits_away_from_the_nearest_visible_enemy():
    sim, barn = barn_sim()
    squad = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    spawn(sim, 1, "rifles", WEST_CELL)  # west of the barn, and on the HQ side
    sim.tick()  # let vision see it

    assert sim.issue(0, [Ungarrison(squad.id)])[0].ok
    cell = tuple(np.floor(squad.pos / 2.0).astype(int))
    # Away from the west enemy: the east column of the ring.
    assert cell[0] == BARN_CELL[0] + BARN_FOOTPRINT[0]


def test_ungarrison_is_invalid_when_not_garrisoned():
    sim, _ = barn_sim()
    squad = spawn(sim, 0, "rifles", EAST_CELL)
    result = sim.issue(0, [Ungarrison(squad.id)])[0]
    assert not result.ok
    assert "not garrisoned" in result.reason


def test_a_team_weapon_leaving_a_garrison_sets_up_again():
    sim, barn = barn_sim()
    gun = garrison_now(sim, spawn(sim, 0, "hmg_team", EAST_CELL), barn)

    assert sim.issue(0, [Ungarrison(gun.id)])[0].ok
    assert gun.state is SquadState.SETTING_UP
    assert gun.setup_done_tick > sim.state.tick


def test_retreat_from_a_garrison_exits_and_runs_home():
    sim, barn = barn_sim()
    squad = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)

    assert sim.issue(0, [Retreat(squad.id)])[0].ok
    assert squad.garrison_in is None
    assert barn.garrison == []
    assert squad.state is SquadState.RETREATING
    cell = tuple(np.floor(squad.pos / 2.0).astype(int))
    assert cell in _ring_cells(sim, barn)


# ---------------------------------------------------------------------------
# which orders a garrisoned squad accepts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order_for", [
    lambda s, b: Move(s.id, (10, 10)),
    lambda s, b: Capture(s.id, "mid"),
    lambda s, b: Build(s.id, "barracks", (10, 10)),
    lambda s, b: Reinforce(s.id),
    lambda s, b: Garrison(s.id, b.id),
])
def test_orders_that_require_leaving_are_invalid_while_garrisoned(order_for):
    sim, barn = barn_sim()
    squad = garrison_now(sim, spawn(sim, 0, "engineers", EAST_CELL), barn)

    result = sim.issue(0, [order_for(squad, barn)])[0]
    assert not result.ok
    assert "garrisoned" in result.reason
    assert squad.garrison_in == barn.id


def test_stop_and_set_facing_leave_a_garrisoned_squad_inside():
    sim, barn = barn_sim()
    gun = garrison_now(sim, spawn(sim, 0, "hmg_team", EAST_CELL), barn)

    assert sim.issue(0, [Stop(gun.id)])[0].ok
    assert sim.issue(0, [SetFacing(gun.id, 90.0)])[0].ok
    assert gun.garrison_in == barn.id
    assert gun.state is SquadState.GARRISONED


def test_a_garrisoned_squad_never_walks_toward_an_attack_target():
    sim, barn = barn_sim()
    squad = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    far = spawn(sim, 1, "rifles", (38, 2))
    squad.order = Attack(squad.id, far.id)
    squad.attack_last_seen_tick = sim.state.tick

    for _ in range(10):
        sim.tick()
    assert squad.garrison_in == barn.id
    assert squad.path == []
    assert tuple(np.floor(squad.pos / 2.0).astype(int)) == BARN_CENTRE_CELL


# ---------------------------------------------------------------------------
# firing from inside
# ---------------------------------------------------------------------------


def test_a_garrisoned_squad_shoots_out_through_its_own_walls():
    sim, barn = barn_sim()
    shooter = garrison_now(sim, spawn(sim, 0, "garrison_rifles", EAST_CELL), barn)
    victim = spawn(sim, 1, "garrison_rifles", EAST_CELL)
    # Without the own-footprint exemption the barn blocks this line entirely.
    assert not vision.has_los(sim.map, shooter.pos, victim.pos)

    assert run_until(sim, lambda: shooter.target_id == victim.id, 20)
    assert run_until(sim, lambda: sum(m.hp for m in victim.members) < 4 * 80, 200)


def test_a_garrisoned_squad_is_shot_at_through_its_own_walls():
    sim, barn = barn_sim()
    inside = garrison_now(sim, spawn(sim, 0, "garrison_rifles", EAST_CELL), barn)
    outside = spawn(sim, 1, "garrison_rifles", EAST_CELL)

    assert run_until(sim, lambda: outside.target_id == inside.id, 20)


def test_a_garrisoned_squad_gains_no_suppression_from_small_arms():
    sim, barn = barn_sim()
    inside = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    gun = spawn(sim, 1, "hmg_team", EAST_CELL)
    gun.facing = np.pi  # pointing west, at the barn
    gun.heading = np.pi
    sim.tick()  # let vision find the barn

    assert sim.issue(1, [Attack(gun.id, inside.id)])[0].ok
    hp = sum(m.hp for m in inside.members)
    assert run_until(sim, lambda: sum(m.hp for m in inside.members) < hp, 400)
    assert inside.suppression == 0.0


def test_a_garrisoned_team_weapon_sets_up_then_ignores_its_arc():
    sim, barn = barn_sim()
    gun = garrison_now(sim, spawn(sim, 0, "hmg_team", EAST_CELL), barn)
    assert gun.state is SquadState.GARRISONED
    setup_tick = gun.setup_done_tick
    assert setup_tick > sim.state.tick

    gun.facing = 0.0  # arc points due east
    gun.heading = 0.0
    victim = spawn(sim, 1, "rifles", WEST_CELL)  # due west: outside a 90-degree arc

    # Nothing fires before the gun has set up inside the building.
    while sim.state.tick < setup_tick:
        sim.tick()
        assert sum(m.hp for m in victim.members) == 4 * 80
    assert run_until(sim, lambda: sum(m.hp for m in victim.members) < 4 * 80, 400)


def test_a_garrisoned_mortar_does_not_fire():
    sim, barn = barn_sim()
    mortar = garrison_now(sim, spawn(sim, 0, "mortar_team", EAST_CELL), barn)
    # Unarmed, so nothing knocks the barn down and lets the crew out.
    victim = spawn(sim, 1, "engineers", (32, 10))  # 24 m: inside the mortar's band

    explosions = 0
    for _ in range(30 * TICKS_PER_SECOND):
        sim.tick()
        explosions += len([e for e in sim.state.events if e.kind == "explosion"])
    assert explosions == 0
    assert mortar.target_id == victim.id  # it sees them; it just cannot shoot
    assert sum(m.hp for m in victim.members) == 4 * 60
    assert mortar.garrison_in == barn.id


# ---------------------------------------------------------------------------
# targeting a garrison
# ---------------------------------------------------------------------------


def test_an_unoccupied_neutral_building_is_not_a_valid_target():
    sim, barn = barn_sim()
    tank = spawn(sim, 1, "tank", EAST_CELL)
    sim.tick()

    assert not sim.issue(1, [Attack(tank.id, barn.id)])[0].ok
    assert tank.target_id is None


def test_an_occupied_neutral_building_becomes_a_valid_target():
    sim, barn = barn_sim()
    garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    tank = spawn(sim, 1, "tank", EAST_CELL)
    sim.tick()

    assert sim.issue(1, [Attack(tank.id, barn.id)])[0].ok
    assert run_until(sim, lambda: barn.hp < 900, 200)


def test_a_tank_auto_targets_an_occupied_building_over_its_occupants():
    sim, barn = barn_sim()
    inside = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    tank = spawn(sim, 1, "tank", EAST_CELL)

    assert run_until(sim, lambda: tank.target_id is not None, 20)
    assert tank.target_id == barn.id
    assert inside.garrison_in == barn.id


def test_small_arms_prefer_the_occupants_over_the_building():
    sim, barn = barn_sim()
    inside = garrison_now(sim, spawn(sim, 0, "garrison_rifles", EAST_CELL), barn)
    outside = spawn(sim, 1, "garrison_rifles", EAST_CELL)

    assert run_until(sim, lambda: outside.target_id is not None, 20)
    assert outside.target_id == inside.id


# ---------------------------------------------------------------------------
# the duel: cover is worth something
# ---------------------------------------------------------------------------


def _duel(seed: int) -> str:
    """One garrisoned squad vs an identical one in the open. Returns the winner."""
    sim, barn = barn_sim(seed=seed)
    inside = garrison_now(sim, spawn(sim, 0, "garrison_rifles", EAST_CELL), barn)
    outside = spawn(sim, 1, "garrison_rifles", EAST_CELL)

    for _ in range(300 * TICKS_PER_SECOND):
        if inside.id not in sim.state.squads:
            return "open"
        if outside.id not in sim.state.squads:
            return "garrison"
        sim.tick()
    return "draw"


@pytest.mark.slow
def test_garrisoned_rifles_beat_identical_rifles_in_the_open():
    results = [_duel(seed) for seed in range(15)]
    wins = results.count("garrison")
    assert wins >= 12, results


# ---------------------------------------------------------------------------
# building death
# ---------------------------------------------------------------------------


def test_a_destroyed_building_ejects_its_garrison_with_collapse_damage():
    sim, barn = barn_sim()
    inside = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    tank = spawn(sim, 1, "tank", EAST_CELL)
    frac = sim.data.economy.garrison_collapse_damage_frac
    max_hp = sim.data.squads["rifles"].member_hp

    assert run_until(sim, lambda: barn.id not in sim.state.buildings, 400)
    assert inside.garrison_in is None
    assert inside.state is SquadState.IDLE
    assert [m.hp for m in inside.members] == [max_hp - frac * max_hp] * 4
    cell = tuple(np.floor(inside.pos / 2.0).astype(int))
    assert cell in _ring_cells(sim, barn)
    assert sim.map.pass_inf[cell[1], cell[0]]
    assert [e for e in sim.state.events if e.kind == "garrison_ejected"][0].data["squad"] == inside.id
    # A destroyed neutral building is gone for good; its footprint is freed.
    assert sim.map.pass_inf[BARN_CENTRE_CELL[1], BARN_CENTRE_CELL[0]]


def test_collapse_damage_can_wipe_a_weakened_garrison():
    sim, barn = barn_sim()
    inside = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    for member in inside.members:
        member.hp = 1.0
    barn.hp = 1.0
    combat._destroy_building(sim, barn)

    assert inside.id not in sim.state.squads
    assert [e for e in sim.state.events if e.kind == "squad_destroyed"]


def test_destroying_a_building_clears_builders_queue_and_op():
    sim, _ = barn_sim()
    sim.state.players[0].manpower = 10_000

    builder = spawn(sim, 0, "engineers", (8, 10))
    assert sim.issue(0, [Build(builder.id, "barracks", (9, 10))])[0].ok
    site = sim.state.buildings[builder.build_target]
    assert run_until(sim, lambda: builder.state is SquadState.CONSTRUCTING, 200)

    site.queue.append(QueueItem(kind="train", item_id="rifles", remaining_s=5.0))
    sim.state.points["west"].op_building = site.id

    combat._destroy_building(sim, site)

    assert builder.build_target is None
    assert builder.order is None
    assert builder.state is SquadState.IDLE
    assert site.queue == []
    assert sim.state.points["west"].op_building is None


def test_destroying_a_building_clears_pending_garrison_orders():
    sim, barn = barn_sim()
    walker = spawn(sim, 0, "rifles", (30, 10))
    assert sim.issue(0, [Garrison(walker.id, barn.id)])[0].ok
    assert walker.state is SquadState.MOVING

    combat._destroy_building(sim, barn)

    assert walker.order is None
    assert walker.path == []
    assert walker.state is SquadState.IDLE


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------


def test_a_shell_on_the_building_applies_garrison_cover_to_the_occupants():
    """Task 10's AOE path, end to end now that garrisons are real: a blast
    centred on the building hurts the men inside through `garrison` cover."""
    data = with_cover("mortar", damage=0.5)
    sim, barn = barn_sim(data=data)
    inside = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    gunner = spawn(sim, 1, "mortar_team", (32, 10))
    mortar = data.weapons["mortar"]

    assert combat._cover_for(sim, inside, 0, inside.pos) == "garrison"
    combat._explode(sim, gunner, mortar, 0, inside.pos.copy())

    # Member 0 stands on the impact point: no falloff, just the cover mod.
    assert inside.members[0].hp == 80 - mortar.damage * 0.5
    assert all(m.hp < 80 for m in inside.members)


def test_a_garrison_is_deterministic():
    def play(seed: int) -> str:
        sim, barn = barn_sim(seed=seed)
        squad = spawn(sim, 0, "garrison_rifles", (30, 10))
        sim.issue(0, [Garrison(squad.id, barn.id)])
        spawn(sim, 1, "garrison_rifles", WEST_CELL)
        sim.run(60 * TICKS_PER_SECOND)
        return sim.state_hash()

    assert play(3) == play(3)
    assert play(3) != play(4)


def test_observation_reports_neutral_garrison_counts():
    sim, barn = barn_sim()
    inside = garrison_now(sim, spawn(sim, 0, "rifles", EAST_CELL), barn)
    sim.tick()

    mine = build_observation(sim, 0)
    assert next(b for b in mine.neutral_buildings if b.id == barn.id).garrison_count == 1
    assert next(s for s in mine.own_squads if s.id == inside.id).cover == "garrison"

    # Team 1 has nothing nearby: the barn is not visible, so neither the
    # occupants nor their number are leaked.
    theirs = build_observation(sim, 1)
    assert next(b for b in theirs.neutral_buildings if b.id == barn.id).garrison_count == 0
    assert not any(s.id == inside.id for s in theirs.enemy_squads)

    # With eyes on the barn they see the count, and the squad itself.
    spawn(sim, 1, "rifles", EAST_CELL)
    sim.tick()
    theirs = build_observation(sim, 1)
    assert next(b for b in theirs.neutral_buildings if b.id == barn.id).garrison_count == 1
    assert any(s.id == inside.id for s in theirs.enemy_squads)
