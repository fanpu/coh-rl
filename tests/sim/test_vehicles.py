"""Tests for vehicles: penetration, rear arcs, turrets and wrecks (task 11)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from coh.data.schema import GameData
from coh.maps.cover import cover_at
from coh.maps.format import center_of
from coh.sim.constants import (
    DT,
    REAR_ARC_DEG,
    TURRET_AIM_TOLERANCE_DEG,
    TURRET_TRAVERSE_MULT,
    WRECK_TERRAIN_CHAR,
)
from coh.sim.orders import Capture, Garrison, Move, Reinforce, Retreat
from coh.sim.state import SquadState
from coh.sim.systems import combat, movement, suppression, vehicle_combat, vision
from tests.helpers import fixture_data, make_sim, spawn

WIDTH, HEIGHT = 60, 20

STARTS = [
    {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
    {"slot": 1, "team": 1, "hq_cell": [WIDTH - 5, HEIGHT - 5], "sector": "a"},
]

TARGET_CELL = (25, 10)
SHOOTER_CELL = (30, 10)  # 10 m due east of TARGET_CELL
BIG_HP = 1.0e6

EAST, WEST = 0.0, math.pi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rows(*cells: tuple[tuple[int, int], str]) -> list[str]:
    grid = [["."] * WIDTH for _ in range(HEIGHT)]
    for (cx, cy), ch in cells:
        grid[cy][cx] = ch
    return ["".join(row) for row in grid]


def veh_sim(terrain: list[str] | None = None, seed: int = 0, data: GameData | None = None):
    return make_sim(terrain if terrain is not None else rows(), seed=seed, data=data, starts=STARTS)


def with_weapon(weapon_id: str, **changes) -> GameData:
    data = fixture_data()
    weapons = dict(data.weapons)
    weapons[weapon_id] = replace(weapons[weapon_id], **changes)
    return replace(data, weapons=weapons)


def bearing(from_pos, to_pos) -> float:
    delta = np.asarray(to_pos, dtype=float) - np.asarray(from_pos, dtype=float)
    return math.atan2(delta[1], delta[0])


def disarm(squad):
    for member in squad.members:
        member.weapon = ""
    return squad


def gunner(sim, owner: int, def_id: str, cell, target_pos) -> object:
    """A vehicle sitting at `cell` with hull and turret already on `target_pos`."""
    squad = spawn(sim, owner, def_id, cell)
    squad.heading = bearing(squad.pos, target_pos)
    squad.turret_heading = squad.heading
    return squad


def deployed(sim, owner: int, def_id: str, cell, facing: float = 0.0):
    """A team weapon already SET_UP and facing `facing` (radians, 0 = east)."""
    squad = spawn(sim, owner, def_id, cell)
    squad.state = SquadState.SET_UP
    squad.heading = facing
    squad.facing = facing
    return squad


def shot_damages(seed: int, victim_heading: float, shots: int = 200) -> list[float]:
    """Damage dealt by `shots` guaranteed tank-gun hits on a stationary tank."""
    sim = veh_sim(seed=seed, data=with_weapon("tank_gun", accuracy=(1.0, 1.0, 1.0)))
    victim = disarm(spawn(sim, 1, "tank", TARGET_CELL))
    victim.members[0].hp = BIG_HP
    victim.heading = victim_heading
    shooter = gunner(sim, 0, "tank", SHOOTER_CELL, victim.pos)
    vision.run(sim)

    damages: list[float] = []
    for _ in range(shots):
        before = victim.members[0].hp
        shooter.members[0].next_ready_tick = sim.state.tick
        combat.run(sim)
        damages.append(before - victim.members[0].hp)
        sim.state.tick += 1
    return damages


def tick_systems(sim, n: int) -> None:
    """`n` movement -> vision -> combat -> suppression ticks."""
    for _ in range(n):
        movement.run(sim)
        vision.run(sim)
        combat.run(sim)
        suppression.run(sim)
        sim.state.tick += 1


def events(sim, kind: str) -> list:
    return [e for e in sim.state.events if e.kind == kind]


# ---------------------------------------------------------------------------
# penetration
# ---------------------------------------------------------------------------

# tank_gun vs the `tank` target table: 160 damage through the armour,
# 160 * deflection_damage_mult (0.1) off it.
PENETRATED, DEFLECTED = 160.0, 16.0


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_frontal_hits_penetrate_about_half_the_time(seed: int) -> None:
    # p_pen = penetration[band] 1.0 * tt.penetration 0.5 = 0.5.
    damages = shot_damages(seed, victim_heading=EAST)  # facing the shooter, which is due east
    assert set(damages) == {PENETRATED, DEFLECTED}
    pens = damages.count(PENETRATED)
    # 200 trials, p = 0.5 -> sd ~7.07; +-4 sd is +-28.3.
    assert 71 <= pens <= 129, pens


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_rear_hits_always_penetrate(seed: int) -> None:
    # The victim faces west, the shooter is due east: p_pen = 0.5 * 2.0 = 1.0.
    damages = shot_damages(seed, victim_heading=WEST, shots=60)
    assert damages == [PENETRATED] * 60


def test_deflections_use_the_weapons_deflection_damage_mult() -> None:
    data = with_weapon("tank_gun", accuracy=(1.0, 1.0, 1.0), deflection_damage_mult=0.25)
    sim = veh_sim(data=data)
    victim = disarm(spawn(sim, 1, "tank", TARGET_CELL))
    victim.members[0].hp = BIG_HP
    victim.heading = EAST  # frontal: deflections are possible
    shooter = gunner(sim, 0, "tank", SHOOTER_CELL, victim.pos)
    vision.run(sim)

    damages = []
    for _ in range(40):
        before = victim.members[0].hp
        shooter.members[0].next_ready_tick = sim.state.tick
        combat.run(sim)
        damages.append(before - victim.members[0].hp)
        sim.state.tick += 1

    assert set(damages) == {160.0, 40.0}


def test_rear_arc_is_measured_from_the_targets_heading() -> None:
    sim = veh_sim()
    tank = spawn(sim, 1, "tank", TARGET_CELL)
    tank.heading = EAST  # pointing +x

    def hit_from(angle_deg: float) -> bool:
        angle = math.radians(angle_deg)
        from_pos = tank.pos + np.array([math.cos(angle), math.sin(angle)]) * 10.0
        return vehicle_combat.is_rear_hit(tank, from_pos)

    assert not hit_from(0.0)  # dead ahead
    assert not hit_from(90.0)  # flank
    assert not hit_from(REAR_ARC_DEG - 1.0)
    assert hit_from(REAR_ARC_DEG + 1.0)
    assert hit_from(180.0)  # dead astern


def test_rifles_barely_scratch_a_tank() -> None:
    sim = veh_sim()
    tank = disarm(spawn(sim, 1, "tank", TARGET_CELL))
    tank.heading = EAST
    rifles = spawn(sim, 0, "rifles", SHOOTER_CELL)
    vision.run(sim)

    start = tank.members[0].hp
    for _ in range(40):
        for member in rifles.members:
            member.next_ready_tick = sim.state.tick
        combat.run(sim)
        sim.state.tick += 1

    dealt = start - tank.members[0].hp
    # They connect constantly (the tank is 10 m away and cannot miss being
    # hit) but the target table turns 20 damage a bullet into 0.2: a minute
    # of massed rifle fire does not amount to a tenth of the tank.
    assert dealt > 0.0
    assert dealt < sim.data.squads["tank"].member_hp * 0.1


def test_a_mortar_shell_rolls_penetration_against_a_vehicle_it_lands_on() -> None:
    """An AOE hit on a vehicle uses the same penetration model as a bullet."""

    def blast_damage(penetration: float, impact_from_behind: bool) -> float:
        data = with_weapon(
            "mortar",
            deflection_damage_mult=0.0,
            target_table={
                "armour_tank": replace(
                    fixture_data().weapons["mortar"].vs("armour_tank"),
                    penetration=penetration,
                    rear_penetration=2.0,
                )
            },
        )
        sim = veh_sim(data=data)
        tank = disarm(spawn(sim, 1, "tank", TARGET_CELL))
        tank.members[0].hp = BIG_HP
        # The mortar is due west, so a tank facing east is shelled in the rear.
        tank.heading = EAST if impact_from_behind else WEST
        # 20 m due west of the tank: past the tube's minimum range and inside
        # its crew's sight.
        mortar = deployed(sim, 0, "mortar_team", (15, 10), facing=EAST)
        vision.run(sim)

        before = tank.members[0].hp
        for _ in range(30):
            mortar.members[0].next_ready_tick = sim.state.tick
            combat.run(sim)
            sim.state.tick += 1
        return before - tank.members[0].hp

    assert blast_damage(penetration=0.0, impact_from_behind=False) == 0.0
    assert blast_damage(penetration=1.0, impact_from_behind=False) > 0.0
    # 0.5 frontal, doubled to 1.0 from behind: the shooter is due west, so a
    # tank facing west is hit in the rear.
    assert blast_damage(penetration=0.5, impact_from_behind=True) > 0.0


# ---------------------------------------------------------------------------
# turrets and hull arcs
# ---------------------------------------------------------------------------


def test_turret_must_traverse_before_the_first_shot() -> None:
    sim = veh_sim()
    target = disarm(spawn(sim, 1, "rifles", TARGET_CELL))
    for member in target.members:
        member.hp = BIG_HP
    tank = spawn(sim, 0, "tank", SHOOTER_CELL)
    tank.heading = EAST  # the target is due west: a 180-degree traverse
    tank.turret_heading = EAST
    vision.run(sim)

    # 45 deg/s hull rotation * 2.0 traverse mult = 11.25 degrees a tick, so
    # 180 degrees takes 16 ticks before the gun is within tolerance.
    traverse_ticks = math.ceil(
        (180.0 - TURRET_AIM_TOLERANCE_DEG)
        / (sim.data.squads["tank"].rotation_deg_s * TURRET_TRAVERSE_MULT * DT)
    )
    fired_at = None
    for tick in range(traverse_ticks + 8):
        combat.run(sim)
        if events(sim, "shot") and fired_at is None:
            fired_at = tick
        sim.state.tick += 1

    assert fired_at == traverse_ticks - 1  # aiming happens before firing, same tick
    assert movement.angle_diff(tank.turret_heading, WEST) < 1e-9


def test_turret_returns_to_the_hull_heading_with_no_target() -> None:
    sim = veh_sim()
    tank = spawn(sim, 0, "tank", SHOOTER_CELL)
    tank.heading = EAST
    tank.turret_heading = WEST

    for _ in range(64):
        combat.run(sim)
        sim.state.tick += 1

    assert math.isclose(tank.turret_heading, EAST, abs_tol=1e-9)


def test_hull_mounted_gun_waits_for_the_hull_to_turn() -> None:
    sim = veh_sim()
    target = disarm(spawn(sim, 1, "rifles", TARGET_CELL))
    for member in target.members:
        member.hp = BIG_HP
    assault = spawn(sim, 0, "assault_gun", SHOOTER_CELL)
    assault.heading = EAST  # target is due west, hull arc is 60 degrees
    vision.run(sim)

    # It rotates at the hull rate (45 deg/s = 5.625 deg/tick) and may fire
    # once the bearing is inside arc_deg / 2 = 30 degrees.
    rotate_ticks = math.ceil(
        (180.0 - 30.0) / (sim.data.squads["assault_gun"].rotation_deg_s * DT)
    )
    fired_at = None
    for tick in range(rotate_ticks + 8):
        combat.run(sim)
        if events(sim, "shot") and fired_at is None:
            fired_at = tick
        sim.state.tick += 1

    assert fired_at == rotate_ticks - 1
    off_bearing = movement.angle_diff(assault.heading, bearing(assault.pos, target.pos))
    assert off_bearing <= math.radians(sim.data.weapons["hull_gun"].arc_deg / 2.0)


def test_a_hull_mounted_vehicle_does_not_turn_while_following_a_path() -> None:
    sim = veh_sim()
    assault = spawn(sim, 0, "assault_gun", SHOOTER_CELL)
    assault.heading = EAST
    assault.path = [(35, 10)]
    behind = np.array(assault.pos) - np.array([20.0, 0.0])

    vehicle_combat.aim(sim, assault, behind)

    assert assault.heading == EAST


def test_secondary_weapons_are_not_arc_limited() -> None:
    sim = veh_sim()
    sdef = sim.data.squads["assault_gun"]
    hull_gun = sim.data.weapons["hull_gun"]
    assault = spawn(sim, 0, "assault_gun", SHOOTER_CELL)
    assault.heading = EAST
    behind = assault.pos - np.array([20.0, 0.0])

    assert not vehicle_combat.may_fire(assault, sdef, 0, hull_gun, behind)
    assert vehicle_combat.may_fire(assault, sdef, 1, hull_gun, behind)


def test_a_moving_vehicle_still_fires() -> None:
    sim = veh_sim()
    target = disarm(spawn(sim, 1, "rifles", TARGET_CELL))
    for member in target.members:
        member.hp = BIG_HP
    tank = gunner(sim, 0, "tank", (34, 10), target.pos)
    sim.issue(0, [Move(squad=tank.id, cell=(31, 10))])
    vision.run(sim)

    fired_while_moving = False
    for _ in range(24):
        sim.state.events.clear()
        movement.run(sim)
        combat.run(sim)
        if tank.moving and events(sim, "shot"):
            fired_while_moving = True
        sim.state.tick += 1

    assert not tank.path  # it really did drive the whole way
    assert fired_while_moving


# ---------------------------------------------------------------------------
# death and wrecks
# ---------------------------------------------------------------------------


def kill_tank(sim, tank) -> None:
    """Drop `tank` to 0 HP through the ordinary damage path."""
    shooter = gunner(sim, 0, "tank", SHOOTER_CELL, tank.pos)
    vision.run(sim)
    for _ in range(200):
        if tank.id not in sim.state.squads:
            return
        shooter.members[0].next_ready_tick = sim.state.tick
        combat.run(sim)
        sim.state.tick += 1
    raise AssertionError("the tank survived 200 shots")


def test_a_destroyed_vehicle_leaves_a_heavy_cover_wreck() -> None:
    sim = veh_sim()
    tank = disarm(spawn(sim, 1, "tank", TARGET_CELL))
    version_before = sim.map.version

    kill_tank(sim, tank)

    assert tank.id not in sim.state.squads
    destroyed = events(sim, "vehicle_destroyed")
    assert [e.data["id"] for e in destroyed] == [tank.id]
    assert not events(sim, "squad_destroyed")
    assert str(sim.map.terrain[TARGET_CELL[1], TARGET_CELL[0]]) == WRECK_TERRAIN_CHAR
    assert cover_at(sim.map, TARGET_CELL, np.array([0.0, 0.0])) == "heavy"
    assert sim.state.terrain_changes[-1] == (*TARGET_CELL, WRECK_TERRAIN_CHAR)
    assert sim.map.version > version_before


def test_a_wreck_never_overwrites_terrain_that_is_not_open_ground() -> None:
    sim = veh_sim(rows((TARGET_CELL, "f")))
    tank = disarm(spawn(sim, 1, "tank", TARGET_CELL))

    kill_tank(sim, tank)

    assert str(sim.map.terrain[TARGET_CELL[1], TARGET_CELL[0]]) == "f"
    assert not sim.state.terrain_changes


def test_a_wreck_replaces_a_road_cell() -> None:
    sim = veh_sim(rows((TARGET_CELL, "r")))
    tank = disarm(spawn(sim, 1, "tank", TARGET_CELL))

    kill_tank(sim, tank)

    assert str(sim.map.terrain[TARGET_CELL[1], TARGET_CELL[0]]) == WRECK_TERRAIN_CHAR


def test_destroyed_infantry_still_emits_squad_destroyed() -> None:
    sim = veh_sim()
    victim = disarm(spawn(sim, 1, "rifles", TARGET_CELL))
    victim.members = victim.members[:1]
    shooter = gunner(sim, 0, "tank", SHOOTER_CELL, victim.pos)
    vision.run(sim)

    for _ in range(60):
        if victim.id not in sim.state.squads:
            break
        shooter.members[0].next_ready_tick = sim.state.tick
        combat.run(sim)
        sim.state.tick += 1

    assert events(sim, "squad_destroyed")
    assert not events(sim, "vehicle_destroyed")
    assert not sim.state.terrain_changes


# ---------------------------------------------------------------------------
# orders a vehicle may not give
# ---------------------------------------------------------------------------


def test_vehicles_cannot_capture_garrison_retreat_or_reinforce() -> None:
    sim = veh_sim()
    tank = spawn(sim, 0, "tank", TARGET_CELL)
    hq = sim.state.buildings[sim.state.players[0].hq_id]

    orders = [
        Capture(squad=tank.id, point_id="mid"),
        Garrison(squad=tank.id, building_id=hq.id),
        Retreat(squad=tank.id),
        Reinforce(squad=tank.id),
    ]
    results = sim.issue(0, orders)

    assert [r.ok for r in results] == [False, False, False, False]
    assert sim.state.players[0].invalid_orders == len(orders)


def test_a_tank_fight_is_reproducible_from_the_seed() -> None:
    """The penetration draw must not disturb the fixed RNG order."""

    def run_fight() -> str:
        sim = veh_sim(seed=3)
        west = gunner(sim, 1, "tank", TARGET_CELL, center_of(SHOOTER_CELL))
        gunner(sim, 0, "tank", SHOOTER_CELL, west.pos)
        sim.run(160)
        return sim.state_hash()

    assert run_fight() == run_fight()


def test_infantry_can_still_be_ordered_to_garrison() -> None:
    """`can_garrison` is false for vehicles and true for infantry; which
    buildings are enterable at all is task 12's rule (neutral ones)."""
    sim = veh_sim()
    house = sim.spawn_building(None, "house", (20, 2))
    rifles = spawn(sim, 0, "rifles", TARGET_CELL)

    assert sim.issue(0, [Garrison(squad=rifles.id, building_id=house.id)])[0].ok


# ---------------------------------------------------------------------------
# duels
# ---------------------------------------------------------------------------

DUEL_GUN_CELL = (15, 10)
DUEL_TANK_CELL = (37, 10)  # 44 m east of the gun: long range for both
DUEL_TICKS = 8 * 100  # 100 seconds


def duel(seed: int, gun_facing: float) -> str:
    """Run an AT-gun-vs-tank duel to a conclusion; returns the winner."""
    sim = veh_sim(seed=seed)
    gun = deployed(sim, 0, "at_team_duel", DUEL_GUN_CELL, facing=gun_facing)
    tank = gunner(sim, 1, "duel_tank", DUEL_TANK_CELL, gun.pos)

    for _ in range(DUEL_TICKS):
        tick_systems(sim, 1)
        if tank.id not in sim.state.squads:
            return "gun"
        if gun.abandoned:
            return "tank"
    return "neither"


@pytest.mark.parametrize("seeds", [range(15)])
def test_the_at_gun_beats_a_tank_it_is_already_pointing_at(seeds) -> None:
    winners = [duel(seed, gun_facing=EAST) for seed in seeds]
    assert winners.count("gun") > len(winners) / 2, winners


@pytest.mark.parametrize("seeds", [range(15)])
def test_the_at_gun_loses_to_a_tank_that_starts_behind_it(seeds) -> None:
    winners = [duel(seed, gun_facing=WEST) for seed in seeds]
    assert winners.count("tank") > len(winners) / 2, winners
