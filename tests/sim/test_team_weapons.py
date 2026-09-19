"""Tests for team weapons (setup, arcs, re-crewing) and indirect fire (task 10)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from coh.data.schema import GameData
from coh.sim.constants import AOE_EDGE_FALLOFF, TICKS_PER_SECOND
from coh.sim.orders import Attack, Move, SetFacing, Stop
from coh.sim.state import SquadState
from coh.sim.systems import combat, movement, vision
from tests.helpers import fixture_data, make_sim, spawn

WIDTH, HEIGHT = 60, 20

STARTS = [
    {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
    {"slot": 1, "team": 1, "hq_cell": [WIDTH - 5, HEIGHT - 5], "sector": "a"},
]

GUN_CELL = (25, 10)
EAST_CELL = (30, 10)  # 10 m due east of GUN_CELL: inside a 90-degree east arc
BIG_HP = 1.0e6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rows(*cells: tuple[tuple[int, int], str]) -> list[str]:
    grid = [["."] * WIDTH for _ in range(HEIGHT)]
    for (cx, cy), ch in cells:
        grid[cy][cx] = ch
    return ["".join(row) for row in grid]


def tw_sim(terrain: list[str] | None = None, seed: int = 0, data: GameData | None = None):
    return make_sim(terrain if terrain is not None else rows(), seed=seed, data=data, starts=STARTS)


def with_weapon(weapon_id: str, **changes) -> GameData:
    data = fixture_data()
    weapons = dict(data.weapons)
    weapons[weapon_id] = replace(weapons[weapon_id], **changes)
    return replace(data, weapons=weapons)


def trim(squad, count: int = 1, hp: float | None = None, disarm: bool = False):
    squad.members = squad.members[:count]
    for member in squad.members:
        if hp is not None:
            member.hp = hp
        if disarm:
            member.weapon = ""
    return squad


def dummy(sim, cell, owner: int = 1, def_id: str = "rifles", count: int = 1, hp: float = BIG_HP):
    """An unarmed, practically immortal squad: a pure damage sponge."""
    return trim(spawn(sim, owner, def_id, cell), count=count, hp=hp, disarm=True)


def damage_taken(squad, hp: float = BIG_HP) -> float:
    return sum(hp - m.hp for m in squad.members)


def deployed(sim, owner: int, def_id: str, cell, facing: float = 0.0):
    """A team weapon already SET_UP and facing `facing` (radians, 0 = east)."""
    squad = spawn(sim, owner, def_id, cell)
    squad.state = SquadState.SET_UP
    squad.heading = facing
    squad.facing = facing
    return squad


def fire_rounds(sim, shooter, n: int) -> None:
    """`n` combat ticks with `shooter`'s cooldowns cleared before each."""
    vision.run(sim)
    for _ in range(n):
        for member in shooter.members:
            member.next_ready_tick = sim.state.tick
        combat.run(sim)
        sim.state.tick += 1


def tick_systems(sim, n: int, ready=None) -> None:
    """`n` full movement -> vision -> combat ticks (the real SYSTEM_ORDER subset)."""
    for _ in range(n):
        movement.run(sim)
        vision.run(sim)
        if ready is not None and ready.id in sim.state.squads:
            for member in ready.members:
                member.next_ready_tick = sim.state.tick
        combat.run(sim)
        sim.state.tick += 1


def events(sim, kind: str) -> list:
    return [e for e in sim.state.events if e.kind == kind]


def setup_ticks(sim, def_id: str) -> int:
    return movement.setup_ticks(sim, sim.data.squads[def_id])


def bearing(from_pos, to_pos) -> float:
    delta = np.asarray(to_pos, dtype=float) - np.asarray(from_pos, dtype=float)
    return math.atan2(delta[1], delta[0])


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def test_hmg_does_not_fire_before_setup_completes():
    sim = tw_sim()
    hmg = spawn(sim, 0, "hmg_team", GUN_CELL)
    target = dummy(sim, EAST_CELL)
    assert hmg.state is SquadState.SETTING_UP

    ticks = setup_ticks(sim, "hmg_team")
    assert ticks == 3 * TICKS_PER_SECOND

    tick_systems(sim, ticks, ready=hmg)
    assert hmg.state is SquadState.SETTING_UP
    assert damage_taken(target) == 0.0

    tick_systems(sim, 1, ready=hmg)
    assert hmg.state is SquadState.SET_UP
    assert damage_taken(target) > 0.0


# ---------------------------------------------------------------------------
# arcs
# ---------------------------------------------------------------------------

# 126 degrees off due east, 10 m away: well outside the hmg's 90-degree arc.
OFF_ARC_CELL = (22, 14)


def test_off_arc_target_is_not_engaged_until_the_weapon_has_re_faced():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    target = dummy(sim, OFF_ARC_CELL)

    offset = abs(bearing(hmg.pos, target.pos))
    assert math.degrees(offset) > sim.data.weapons["hmg"].arc_deg / 2

    vision.run(sim)
    combat.run(sim)
    # It cannot shoot sideways; it starts turning the gun instead.
    assert damage_taken(target) == 0.0
    assert hmg.state is SquadState.TEARING_DOWN
    assert hmg.facing == pytest.approx(bearing(hmg.pos, target.pos))

    ticks = setup_ticks(sim, "hmg_team")
    sim.state.tick += 1
    tick_systems(sim, 2 * ticks - 1, ready=hmg)
    assert hmg.state is not SquadState.SET_UP  # re-facing costs 2 x setup_time
    assert damage_taken(target) == 0.0

    tick_systems(sim, 1, ready=hmg)
    assert hmg.state is SquadState.SET_UP
    assert damage_taken(target) > 0.0


def test_a_target_already_inside_the_arc_never_triggers_a_re_face():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    target = dummy(sim, EAST_CELL)

    fire_rounds(sim, hmg, 10)

    assert hmg.state is SquadState.SET_UP
    assert hmg.facing == 0.0
    assert damage_taken(target) > 0.0


def test_an_in_arc_target_is_preferred_over_a_closer_out_of_arc_one():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    near_off_arc = dummy(sim, (23, 12))
    far_in_arc = dummy(sim, (33, 10))

    fire_rounds(sim, hmg, 5)

    assert hmg.target_id == far_in_arc.id
    assert hmg.state is SquadState.SET_UP
    assert damage_taken(near_off_arc) == 0.0
    assert damage_taken(far_in_arc) > 0.0


def test_crew_small_arms_are_not_arc_limited():
    """Only the team weapon in slot 0 is gated by the arc; the crew is not."""
    data = fixture_data()
    squads = dict(data.squads)
    squads["hmg_team"] = replace(squads["hmg_team"], loadout=("hmg", "rifle", "rifle"))
    sim = tw_sim(data=replace(data, squads=squads))

    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    target = dummy(sim, OFF_ARC_CELL)

    vision.run(sim)
    combat.run(sim)

    rifle = sim.data.weapons["rifle"]
    # The hmg itself held fire (it is tearing down to re-face), but the two
    # riflemen shot: at most 2 rifle bullets' worth of damage was dealt.
    assert hmg.state is SquadState.TEARING_DOWN
    assert 0.0 < damage_taken(target) <= 2 * rifle.damage


def test_auto_re_face_hysteresis_stops_the_gun_spinning_forever():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    west = dummy(sim, (20, 10))
    east = dummy(sim, (30, 10))
    # Both sides of a gun facing north: neither is in the 90-degree arc.
    hmg.heading = hmg.facing = -math.pi / 2

    ticks = setup_ticks(sim, "hmg_team")
    vision.run(sim)
    combat.run(sim)
    assert hmg.state is SquadState.TEARING_DOWN
    faced = hmg.target_id
    hold = hmg.reface_hold_tick
    assert hold == sim.state.tick + 3 * ticks

    sim.state.tick += 1
    tick_systems(sim, 2 * ticks, ready=hmg)
    assert hmg.state is SquadState.SET_UP

    # The target it turned to vanishes; the other one is still behind it.
    sim.state.squads.pop(faced)
    other = east if faced == west.id else west
    tick_systems(sim, hold - sim.state.tick, ready=hmg)
    assert hmg.state is SquadState.SET_UP  # no second re-face during the hold
    assert damage_taken(other) == 0.0

    # Once the hold expires it may turn again.
    tick_systems(sim, 1, ready=hmg)
    assert hmg.state is SquadState.TEARING_DOWN


def test_a_mortar_never_needs_to_re_face():
    sim = tw_sim()
    mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
    dummy(sim, (25, 17))  # due south, 14 m: outside any 90-degree arc

    vision.run(sim)
    combat.run(sim)

    assert sim.data.weapons["mortar"].arc_deg == 360.0
    assert mortar.state is SquadState.SET_UP


# ---------------------------------------------------------------------------
# SetFacing
# ---------------------------------------------------------------------------


def test_set_facing_costs_two_setup_times_and_leaves_the_gun_on_the_new_bearing():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    ticks = setup_ticks(sim, "hmg_team")

    assert sim.issue(0, [SetFacing(hmg.id, 90.0)])[0].ok
    assert hmg.state is SquadState.TEARING_DOWN
    assert hmg.facing == pytest.approx(math.pi / 2)  # 90 deg clockwise = +y = south

    sim.state.tick += 1
    tick_systems(sim, 2 * ticks - 1)
    assert hmg.state is not SquadState.SET_UP

    tick_systems(sim, 1)
    assert hmg.state is SquadState.SET_UP
    assert hmg.facing == pytest.approx(math.pi / 2)
    assert hmg.order is None


def test_set_facing_lets_the_gun_engage_what_it_could_not_before():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    target = dummy(sim, (25, 15))  # due south

    assert sim.issue(0, [SetFacing(hmg.id, 90.0)])[0].ok
    sim.state.tick += 1
    tick_systems(sim, 2 * setup_ticks(sim, "hmg_team"), ready=hmg)

    assert hmg.state is SquadState.SET_UP
    assert damage_taken(target) > 0.0


def test_set_facing_is_invalid_for_anything_but_a_team_weapon():
    sim = tw_sim()
    rifles = spawn(sim, 0, "rifles", GUN_CELL)

    result = sim.issue(0, [SetFacing(rifles.id, 90.0)])[0]

    assert not result.ok
    assert "team weapon" in result.reason
    assert sim.state.players[0].invalid_orders == 1
    assert rifles.state is SquadState.IDLE


def test_explicit_set_facing_ignores_the_auto_re_face_hold():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    hmg.reface_hold_tick = sim.state.tick + 1000

    assert sim.issue(0, [SetFacing(hmg.id, 180.0)])[0].ok

    assert hmg.state is SquadState.TEARING_DOWN
    assert hmg.facing == pytest.approx(math.pi)
    assert hmg.reface_hold_tick == 0


# ---------------------------------------------------------------------------
# crew loss / abandonment
# ---------------------------------------------------------------------------


def killer(sim, cell, owner: int = 1):
    """An immortal rifle squad that will shoot whatever it can see."""
    return trim(spawn(sim, owner, "rifles", cell), count=4, hp=BIG_HP)


def test_killing_the_gunner_hands_the_weapon_to_the_next_crewman():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    hmg.members[0].hp = 10.0  # the only crewman who can die
    for member in hmg.members[1:]:
        member.hp = BIG_HP
    enemy = killer(sim, EAST_CELL)

    for _ in range(200):
        tick_systems(sim, 1, ready=enemy)
        if len(hmg.members) == 2:
            break

    assert len(hmg.members) == 2
    assert hmg.members[0].weapon == "hmg"
    assert hmg.members[0].next_ready_tick == 0  # cooldown state reset with the weapon
    assert hmg.members[0].shots_since_reload == 0
    assert not hmg.abandoned

    before = damage_taken(enemy)
    fire_rounds(sim, hmg, 3)
    assert damage_taken(enemy) > before  # the weapon keeps firing


def test_wiping_the_crew_leaves_an_abandoned_shell_rather_than_destroying_it():
    sim = tw_sim()
    hmg = trim(deployed(sim, 0, "hmg_team", GUN_CELL), count=1, hp=10.0)
    hmg.suppression = 0.4
    enemy = killer(sim, EAST_CELL)

    tick_systems(sim, 40, ready=enemy)

    assert hmg.id in sim.state.squads  # not destroyed
    assert hmg.abandoned
    assert hmg.members == []
    assert hmg.state is SquadState.IDLE
    assert hmg.order is None and hmg.path == [] and hmg.target_id is None
    assert hmg.suppression == 0.0
    assert enemy.target_id is None  # an abandoned shell is not a target
    assert [e for e in events(sim, "weapon_abandoned") if e.data["id"] == hmg.id]
    assert not [e for e in events(sim, "squad_destroyed") if e.data["id"] == hmg.id]


def test_an_abandoned_shell_neither_scouts_nor_draws_fire():
    sim = tw_sim()
    shell = abandoned_shell(sim, GUN_CELL)
    enemy = killer(sim, EAST_CELL)

    vision.run(sim)
    combat.run(sim)

    cx, cy = GUN_CELL
    assert not sim.state.visible[0][cy, cx]  # a crewless gun sees nothing
    assert enemy.target_id is None  # and is not worth shooting at
    assert shell.abandoned


def test_a_rifle_squad_wiping_a_crew_can_then_take_the_gun_and_fire_it():
    sim = tw_sim()
    hmg = trim(deployed(sim, 0, "hmg_team", GUN_CELL), count=1, hp=10.0)
    enemy = killer(sim, EAST_CELL)
    victim = dummy(sim, (20, 10), owner=0)  # something for the captured gun to shoot

    tick_systems(sim, 40, ready=enemy)
    assert hmg.abandoned

    assert sim.issue(1, [Move(enemy.id, GUN_CELL)])[0].ok
    assert enemy.recrew_target == hmg.id
    tick_systems(sim, 200)

    assert enemy.id not in sim.state.squads
    assert hmg.owner == 1
    assert not hmg.abandoned
    assert len(hmg.members) == sim.data.squads["hmg_team"].members  # 4 rifles -> 3 crew
    assert hmg.members[0].weapon == "hmg"
    assert [m.weapon for m in hmg.members[1:]] == ["rifle", "rifle"]
    assert [e for e in events(sim, "weapon_recrewed") if e.data["id"] == hmg.id]

    tick_systems(sim, 3 * TICKS_PER_SECOND, ready=hmg)
    assert hmg.state is SquadState.SET_UP
    assert damage_taken(victim) > 0.0


# ---------------------------------------------------------------------------
# re-crewing
# ---------------------------------------------------------------------------


def abandoned_shell(sim, cell, owner: int = 0, def_id: str = "hmg_team"):
    shell = spawn(sim, owner, def_id, cell)
    shell.abandoned = True
    shell.members = []
    shell.state = SquadState.IDLE
    return shell


def test_move_onto_an_abandoned_weapon_re_crews_it():
    sim = tw_sim()
    shell = abandoned_shell(sim, GUN_CELL)
    rescuer = spawn(sim, 1, "rifles", (25, 15))

    assert sim.issue(1, [Move(rescuer.id, GUN_CELL)])[0].ok
    assert rescuer.recrew_target == shell.id
    for _ in range(200):
        tick_systems(sim, 1)
        if rescuer.id not in sim.state.squads:
            break

    assert rescuer.id not in sim.state.squads
    assert shell.owner == 1
    assert shell.members[0].weapon == "hmg"
    assert shell.state is SquadState.SETTING_UP  # the new crew has to deploy it
    assert shell.facing == shell.heading == pytest.approx(-math.pi / 2)  # came from the south
    recrewed = [e for e in events(sim, "weapon_recrewed") if e.data["id"] == shell.id]
    assert recrewed and recrewed[0].data["owner"] == 1


def test_re_crew_works_for_a_squad_of_either_team():
    for owner, taker in ((0, 1), (1, 0)):
        sim = tw_sim()
        shell = abandoned_shell(sim, GUN_CELL, owner=owner)
        rescuer = spawn(sim, taker, "rifles", (25, 14))
        assert sim.issue(taker, [Move(rescuer.id, GUN_CELL)])[0].ok
        tick_systems(sim, 200)
        assert shell.owner == taker
        assert not shell.abandoned


def test_a_single_survivor_cannot_re_crew():
    sim = tw_sim()
    shell = abandoned_shell(sim, GUN_CELL)
    rescuer = trim(spawn(sim, 1, "rifles", (25, 15)), count=1, hp=80.0)

    assert sim.issue(1, [Move(rescuer.id, GUN_CELL)])[0].ok
    assert rescuer.recrew_target is None
    tick_systems(sim, 200)

    assert rescuer.id in sim.state.squads
    assert shell.abandoned
    assert shell.members == []


def test_an_abandoned_shell_takes_no_orders_from_its_old_owner():
    sim = tw_sim()
    shell = abandoned_shell(sim, GUN_CELL, owner=0)

    result = sim.issue(0, [Move(shell.id, (30, 10))])[0]

    assert not result.ok
    assert "abandoned" in result.reason
    assert shell.path == [] and shell.order is None
    assert not sim.issue(0, [SetFacing(shell.id, 90.0)])[0].ok


def test_a_team_weapon_cannot_re_crew_another_team_weapon():
    sim = tw_sim()
    shell = abandoned_shell(sim, GUN_CELL)
    other = spawn(sim, 1, "at_team", (25, 15))

    assert sim.issue(1, [Move(other.id, GUN_CELL)])[0].ok

    assert other.recrew_target is None


def test_any_other_order_forgets_the_re_crew_target():
    sim = tw_sim()
    shell = abandoned_shell(sim, GUN_CELL)
    rescuer = spawn(sim, 1, "rifles", (25, 15))

    assert sim.issue(1, [Move(rescuer.id, GUN_CELL)])[0].ok
    assert rescuer.recrew_target == shell.id

    assert sim.issue(1, [Stop(rescuer.id)])[0].ok
    assert rescuer.recrew_target is None


def test_a_move_that_stops_short_of_the_weapon_does_not_re_crew():
    sim = tw_sim()
    shell = abandoned_shell(sim, GUN_CELL)
    rescuer = spawn(sim, 1, "rifles", (25, 15))

    assert sim.issue(1, [Move(rescuer.id, (25, 13))])[0].ok  # 6 m short
    assert rescuer.recrew_target is None
    tick_systems(sim, 200)

    assert rescuer.id in sim.state.squads
    assert shell.abandoned


# ---------------------------------------------------------------------------
# indirect fire
# ---------------------------------------------------------------------------

# A short hedgerow between the mortar and its target: no LOS, but a friendly
# scout standing next to the target still sees it.
HEDGE = rows(*(((28, cy), "H") for cy in range(8, 13)))
MORTAR_TARGET_CELL = (32, 10)  # 14 m from GUN_CELL: past the mortar's 10 m min range


def test_a_mortar_shells_a_target_its_own_crew_cannot_see():
    sim = tw_sim(HEDGE, data=with_weapon("mortar", scatter_m=0.0))
    mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
    target = dummy(sim, MORTAR_TARGET_CELL)

    vision.run(sim)
    assert not vision.has_los(sim.map, mortar.pos, target.pos)
    assert not vision.is_visible(sim, 0, target)

    fire_rounds(sim, mortar, 5)
    assert damage_taken(target) == 0.0  # nobody on the team can see it

    dummy(sim, (31, 10), owner=0)  # an (unarmed) spotter walks up
    fire_rounds(sim, mortar, 3)

    assert vision.is_visible(sim, 0, target)
    assert damage_taken(target) > 0.0


def test_a_mortar_never_fires_inside_its_minimum_range():
    sim = tw_sim(data=with_weapon("mortar", scatter_m=0.0))
    mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
    close = dummy(sim, (28, 10))  # 6 m: inside min_range 10

    fire_rounds(sim, mortar, 20)

    assert mortar.target_id is None
    assert damage_taken(close) == 0.0
    assert events(sim, "explosion") == []


def test_a_mortar_shell_damages_two_adjacent_squads_with_linear_falloff():
    sim = tw_sim(data=with_weapon("mortar", scatter_m=0.0))
    mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
    near = dummy(sim, (35, 10))
    far = dummy(sim, (36, 10))  # 2 m from `near`, inside the 4 m blast

    fire_rounds(sim, mortar, 1)

    weapon = sim.data.weapons["mortar"]
    assert mortar.target_id == near.id
    assert damage_taken(near) == pytest.approx(weapon.damage)
    edge_falloff = 1.0 - (1.0 - AOE_EDGE_FALLOFF) * (2.0 / weapon.aoe_radius)
    assert damage_taken(far) == pytest.approx(weapon.damage * edge_falloff)

    blast = events(sim, "explosion")
    assert len(blast) == 1
    assert blast[0].data["radius"] == weapon.aoe_radius
    assert blast[0].data["pos"] == pytest.approx(list(near.pos))
    assert len(events(sim, "shot")) == 1


def test_a_squad_outside_the_blast_radius_is_untouched():
    sim = tw_sim(data=with_weapon("mortar", scatter_m=0.0))
    mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
    near = dummy(sim, (35, 10))
    outside = dummy(sim, (38, 10))  # 6 m away: beyond aoe_radius 4

    fire_rounds(sim, mortar, 1)

    assert damage_taken(near) > 0.0
    assert damage_taken(outside) == 0.0


def test_a_mortar_shell_suppresses_every_enemy_squad_in_the_blast():
    sim = tw_sim(data=with_weapon("mortar", scatter_m=0.0, nearby_suppression_radius=0.0))
    mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
    near = dummy(sim, (35, 10))
    far = dummy(sim, (36, 10))

    fire_rounds(sim, mortar, 1)

    weapon = sim.data.weapons["mortar"]
    assert near.suppression == pytest.approx(weapon.suppression[1])
    assert far.suppression == pytest.approx(weapon.suppression[1])


def test_a_mortar_does_not_shell_its_own_side():
    sim = tw_sim(data=with_weapon("mortar", scatter_m=0.0))
    mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
    enemy = dummy(sim, (35, 10))
    friend = dummy(sim, (36, 10), owner=0)

    fire_rounds(sim, mortar, 1)

    assert damage_taken(enemy) > 0.0
    assert damage_taken(friend) == 0.0
    assert friend.suppression == 0.0


def test_a_mortar_shell_damages_a_building_inside_the_blast():
    sim = tw_sim(data=with_weapon("mortar", scatter_m=0.0))
    mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
    building = sim.spawn_building(1, "barracks", (36, 10))
    target = dummy(sim, (35, 10))
    hp_before = building.hp

    fire_rounds(sim, mortar, 1)

    assert damage_taken(target) > 0.0
    assert building.hp < hp_before


def test_mortar_scatter_is_gaussian_around_the_target_and_grows_with_range():
    weapon = fixture_data().weapons["mortar"]
    shells = 250
    for seed in (0, 1, 2):
        sim = tw_sim(seed=seed)
        mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
        target = dummy(sim, (35, 10), hp=BIG_HP)

        fire_rounds(sim, mortar, shells)

        blast = np.array([e.data["pos"] for e in events(sim, "explosion")], dtype=float)
        assert len(blast) == shells
        distance = float(np.linalg.norm(target.pos - mortar.pos))
        sigma = weapon.scatter_m * distance / weapon.ranges[2]

        offsets = blast - target.pos
        # Mean is 0 +- 4 standard errors; spread matches sigma within 25%.
        assert np.abs(offsets.mean(axis=0)).max() <= 4 * sigma / math.sqrt(shells)
        assert offsets.std(axis=0) == pytest.approx(sigma, rel=0.25)


def test_indirect_fire_does_not_need_line_of_sight_but_a_direct_gun_does():
    sim = tw_sim(HEDGE)
    dummy(sim, (31, 10), owner=0)  # unarmed spotter, keeps the target visible
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    target = dummy(sim, MORTAR_TARGET_CELL)

    fire_rounds(sim, hmg, 10)

    assert vision.is_visible(sim, 0, target)
    assert damage_taken(target) == 0.0  # direct fire still needs LOS


def test_mortar_fire_is_deterministic_for_a_given_seed():
    def run() -> list[list[float]]:
        sim = tw_sim(seed=5)
        mortar = deployed(sim, 0, "mortar_team", GUN_CELL)
        dummy(sim, (35, 10))
        fire_rounds(sim, mortar, 20)
        return [e.data["pos"] for e in events(sim, "explosion")]

    assert run() == run()


# ---------------------------------------------------------------------------
# interaction with existing orders
# ---------------------------------------------------------------------------


def test_an_attack_order_still_needs_the_target_inside_the_arc():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)
    target = dummy(sim, OFF_ARC_CELL)
    vision.run(sim)

    assert sim.issue(0, [Attack(hmg.id, target.id)])[0].ok
    fire_rounds(sim, hmg, 1)

    assert damage_taken(target) == 0.0


def test_a_team_weapon_that_stops_moving_faces_its_travel_direction():
    sim = tw_sim()
    hmg = deployed(sim, 0, "hmg_team", GUN_CELL)

    assert sim.issue(0, [Move(hmg.id, (25, 15))])[0].ok  # due south
    tick_systems(sim, 400)

    assert hmg.state is SquadState.SET_UP
    assert hmg.facing == pytest.approx(math.pi / 2, abs=1e-6)
