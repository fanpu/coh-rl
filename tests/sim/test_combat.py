"""Tests for the combat system: acquisition, shots, cover, damage, deaths (task 8)."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import pytest

from coh.data.schema import GameData, TargetMods
from coh.maps.format import center_of
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import Attack, Move
from coh.sim.state import SquadState
from coh.sim.systems import combat, vision
from tests.helpers import fixture_data, make_sim, spawn

WIDTH, HEIGHT = 60, 20

# Test cells sit in the middle of the map, out of sight of either HQ (sight
# 30 m = 15 cells) and of the builder squads that spawn next to them.
SHOOTER_CELL = (25, 10)
TARGET_CELL = (30, 10)  # exactly 10 m away: rifle short band, accuracy 1.0
NEAR_SIDE_CELL = (29, 10)  # between target and shooter
FAR_SIDE_CELL = (31, 10)  # behind the target, away from the shooter

STARTS = [
    {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
    {"slot": 1, "team": 1, "hq_cell": [WIDTH - 5, HEIGHT - 5], "sector": "a"},
]

BIG_HP = 1.0e6  # a target that never dies, so damage dealt == hits x damage


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rows(*cells: tuple[tuple[int, int], str]) -> list[str]:
    """Open terrain with the given `(cell, char)` overrides."""
    grid = [["."] * WIDTH for _ in range(HEIGHT)]
    for (cx, cy), ch in cells:
        grid[cy][cx] = ch
    return ["".join(row) for row in grid]


def combat_sim(terrain: list[str] | None = None, seed: int = 0, data: GameData | None = None):
    return make_sim(terrain if terrain is not None else rows(), seed=seed, data=data, starts=STARTS)


def trim(squad, count: int = 1, hp: float | None = None, disarm: bool = False):
    """Keep only `count` members (so member 0 stands on the squad's own cell)."""
    squad.members = squad.members[:count]
    for member in squad.members:
        if hp is not None:
            member.hp = hp
        if disarm:
            member.weapon = ""
    return squad


def dummy(sim, cell, owner: int = 1, def_id: str = "rifles", count: int = 1, hp: float = BIG_HP):
    """An unarmed, practically immortal enemy squad: a pure damage sponge."""
    return trim(spawn(sim, owner, def_id, cell), count=count, hp=hp, disarm=True)


def fire_rounds(sim, shooter, n: int) -> None:
    """Run `n` combat ticks, clearing `shooter`'s cooldowns before each.

    Vision is computed once up front: nothing moves in these tests, so the
    per-team visible grids never change.
    """
    vision.run(sim)
    for _ in range(n):
        for member in shooter.members:
            member.next_ready_tick = sim.state.tick
        combat.run(sim)
        sim.state.tick += 1


def damage_taken(squad, hp: float = BIG_HP) -> float:
    return hp * len(squad.members) - sum(m.hp for m in squad.members)


def with_weapon(weapon_id: str, **changes) -> GameData:
    """Fixture data with one weapon def field-replaced."""
    data = fixture_data()
    weapons = dict(data.weapons)
    weapons[weapon_id] = replace(weapons[weapon_id], **changes)
    return replace(data, weapons=weapons)


def shot_events(sim):
    return [e for e in sim.state.events if e.kind == "shot"]


# ---------------------------------------------------------------------------
# basic firing, range, damage
# ---------------------------------------------------------------------------


def test_short_range_rifle_always_hits_and_removes_expected_hp():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)

    fire_rounds(sim, shooter, 30)

    rifle = sim.data.weapons["rifle"]
    assert damage_taken(target) == 30 * rifle.damage
    assert shooter.target_id == target.id
    assert all(e.data["hit"] for e in shot_events(sim))


def test_shot_event_carries_ids_and_positions():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)

    fire_rounds(sim, shooter, 1)

    events = shot_events(sim)
    assert len(events) == 1
    data = events[0].data
    assert data["src"] == shooter.id
    assert data["dst"] == target.id
    assert data["hit"] is True
    assert data["src_pos"] == [float(shooter.pos[0]), float(shooter.pos[1])]
    assert data["dst_pos"] == [float(target.pos[0]), float(target.pos[1])]


def test_target_out_of_range_is_never_acquired_or_damaged():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    # rifle max range is 30 m = 15 cells; 20 cells away is well beyond it.
    target = dummy(sim, (SHOOTER_CELL[0] + 20, SHOOTER_CELL[1]))

    fire_rounds(sim, shooter, 20)

    assert shooter.target_id is None
    assert damage_taken(target) == 0.0
    assert shot_events(sim) == []


def test_band_and_cooldown_bookkeeping():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    dummy(sim, TARGET_CELL)

    vision.run(sim)
    combat.run(sim)

    rifle = sim.data.weapons["rifle"]
    lo, hi = rifle.cooldown
    member = shooter.members[0]
    assert member.next_ready_tick >= sim.state.tick + math.ceil(lo * TICKS_PER_SECOND)
    assert member.next_ready_tick <= sim.state.tick + math.ceil(hi * TICKS_PER_SECOND)

    # on cooldown: no second shot this tick or the next
    combat.run(sim)
    assert len(shot_events(sim)) == 1


def test_no_line_of_sight_through_a_hedgerow_means_no_fire():
    sim = combat_sim(rows((NEAR_SIDE_CELL, "H")))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)

    fire_rounds(sim, shooter, 20)

    assert damage_taken(target) == 0.0
    assert target.id not in (shooter.target_id,)
    assert shot_events(sim) == []


# ---------------------------------------------------------------------------
# cover
# ---------------------------------------------------------------------------


def _hits_with_terrain(terrain: list[str], seed: int, trials: int) -> int:
    sim = combat_sim(terrain, seed=seed)
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    dummy(sim, TARGET_CELL)
    fire_rounds(sim, shooter, trials)
    events = shot_events(sim)
    assert len(events) == trials  # one firing per round, cooldowns cleared
    return sum(1 for e in events if e.data["hit"])


def test_heavy_cover_between_shooter_and_target_halves_the_hits():
    trials = 600
    # A wall cell on the shooter's side of the target gives heavy cover
    # (rifle heavy cover accuracy mult = 0.5); the same wall behind the
    # target does nothing.
    covered = _hits_with_terrain(rows((NEAR_SIDE_CELL, "w")), seed=11, trials=trials)
    flanked = _hits_with_terrain(rows((FAR_SIDE_CELL, "w")), seed=11, trials=trials)

    expected = trials * fixture_data().weapons["rifle"].cover("heavy").accuracy
    assert abs(covered - expected) <= 0.15 * expected
    assert flanked == trials  # accuracy 1.0, no cover applies


def test_heavy_cover_halves_hits_across_many_seeds():
    trials = 200
    expected = trials * fixture_data().weapons["rifle"].cover("heavy").accuracy
    sigma = math.sqrt(trials * 0.5 * 0.5)
    for seed in range(8):
        covered = _hits_with_terrain(rows((NEAR_SIDE_CELL, "w")), seed=seed, trials=trials)
        assert abs(covered - expected) <= 4 * sigma, (seed, covered)


def test_heavy_cover_reduces_damage_per_hit():
    sim = combat_sim(rows((NEAR_SIDE_CELL, "w")), data=with_weapon("rifle", accuracy=(1.0, 1.0, 1.0)))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)

    fire_rounds(sim, shooter, 10)

    rifle = sim.data.weapons["rifle"]
    hits = sum(1 for e in shot_events(sim) if e.data["hit"])
    assert 0 < hits < 10  # heavy cover also halves accuracy
    assert damage_taken(target) == hits * rifle.damage * rifle.cover("heavy").damage


def test_negative_cover_increases_damage_per_hit():
    sim = combat_sim(rows((TARGET_CELL, "r")))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)

    fire_rounds(sim, shooter, 10)

    rifle = sim.data.weapons["rifle"]
    # accuracy 1.0 x 1.25 clamps back to 1.0, so all ten shots land.
    assert damage_taken(target) == 10 * rifle.damage * rifle.cover("negative").damage


def test_garrisoned_target_uses_the_garrison_cover_row():
    sim = combat_sim(data=with_weapon("rifle", accuracy=(1.0, 1.0, 1.0)))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)
    building = sim.spawn_building(1, "barracks", (35, 5))
    target.garrison_in = building.id  # task 12 owns real garrison entry

    fire_rounds(sim, shooter, 10)

    rifle = sim.data.weapons["rifle"]
    hits = sum(1 for e in shot_events(sim) if e.data["hit"])
    assert hits > 0
    assert damage_taken(target) == hits * rifle.damage * rifle.cover("garrison").damage
    assert target.suppression == 0.0  # garrison suppression mult is 0 in the fixture


def test_members_stand_on_their_formation_offsets_for_cover_lookup():
    from coh.sim.constants import CELL_M, FORMATION_RADIUS_M

    sim = combat_sim()
    squad = spawn(sim, 0, "rifles", TARGET_CELL)
    squad_cell = TARGET_CELL

    assert combat._member_cell(sim, squad, 0) == squad_cell  # member 0 on the squad cell
    for slot in range(1, len(squad.members)):
        cx, cy = combat._member_cell(sim, squad, slot)
        # every ring slot is within one cell of the squad's own cell
        assert abs(cx - squad_cell[0]) <= math.ceil(FORMATION_RADIUS_M / CELL_M)
        assert abs(cy - squad_cell[1]) <= math.ceil(FORMATION_RADIUS_M / CELL_M)

    # an offset that lands somewhere infantry cannot stand falls back
    blocked = combat_sim(rows(((TARGET_CELL[0] + 1, TARGET_CELL[1]), "w")))
    walled = spawn(blocked, 0, "rifles", TARGET_CELL)
    cells = {combat._member_cell(blocked, walled, slot) for slot in range(len(walled.members))}
    assert (TARGET_CELL[0] + 1, TARGET_CELL[1]) not in cells


# ---------------------------------------------------------------------------
# deaths
# ---------------------------------------------------------------------------


def test_dead_members_are_removed_and_the_empty_squad_is_deleted():
    sim = combat_sim()
    shooter = spawn(sim, 0, "rifles", SHOOTER_CELL)  # four rifles
    target = spawn(sim, 1, "rifles", TARGET_CELL)
    sdef = sim.data.squads["rifles"]
    rifle = sim.data.weapons["rifle"]
    shots_needed = math.ceil(sdef.members * sdef.member_hp / rifle.damage / len(shooter.members))

    vision.run(sim)
    seen_partial = False
    for _ in range(shots_needed + 2):
        for member in shooter.members:
            member.next_ready_tick = sim.state.tick
        combat.run(sim)
        sim.state.tick += 1
        if target.id in sim.state.squads and 0 < len(target.members) < sdef.members:
            seen_partial = True
        if target.id not in sim.state.squads:
            break

    assert seen_partial, "members should die one at a time before the squad is gone"
    assert target.id not in sim.state.squads
    assert all(m.hp > 0 for m in target.members)  # dead members removed, not kept at <= 0
    assert shooter.target_id is None  # dangling target cleared
    assert [e for e in sim.state.events if e.kind == "squad_destroyed" and e.data["id"] == target.id]


# ---------------------------------------------------------------------------
# bursts, moving, indirect, team weapons
# ---------------------------------------------------------------------------


def test_burst_weapon_resolves_rate_of_fire_times_burst_duration_bullets():
    data = with_weapon("hmg", burst=(2.5, 2.5), accuracy=(1.0, 1.0, 1.0))
    sim = combat_sim(data=data)
    shooter = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
    shooter.state = SquadState.SET_UP
    target = dummy(sim, TARGET_CELL, count=4)

    fire_rounds(sim, shooter, 1)

    hmg = sim.data.weapons["hmg"]
    bullets = round(hmg.rate_of_fire * 2.5)
    assert bullets == 25
    assert damage_taken(target) == bullets * hmg.damage
    assert len(shot_events(sim)) == 1  # one event per firing, not per bullet


def test_team_weapon_does_not_fire_before_it_is_set_up():
    sim = combat_sim()
    shooter = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
    assert shooter.state is SquadState.SETTING_UP
    target = dummy(sim, TARGET_CELL)

    fire_rounds(sim, shooter, 5)
    assert damage_taken(target) == 0.0

    shooter.state = SquadState.SET_UP
    fire_rounds(sim, shooter, 1)
    assert damage_taken(target) > 0.0


def test_moving_shooter_with_zero_moving_accuracy_never_fires():
    sim = combat_sim(data=with_weapon("rifle", moving_accuracy=0.0))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)
    shooter.moving = True

    fire_rounds(sim, shooter, 20)
    assert damage_taken(target) == 0.0

    shooter.moving = False
    fire_rounds(sim, shooter, 1)
    assert damage_taken(target) > 0.0


def test_moving_shooter_applies_the_moving_accuracy_multiplier():
    trials = 400
    rifle = fixture_data().weapons["rifle"]
    expected = trials * rifle.moving_accuracy
    sigma = math.sqrt(trials * rifle.moving_accuracy * (1 - rifle.moving_accuracy))
    for seed in (0, 1, 2, 3):
        sim = combat_sim(seed=seed)
        shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
        target = dummy(sim, TARGET_CELL)
        shooter.moving = True
        fire_rounds(sim, shooter, trials)
        hits = damage_taken(target) / rifle.damage
        assert abs(hits - expected) <= 4 * sigma, (seed, hits)


def test_indirect_weapons_do_not_fire_in_this_task():
    sim = combat_sim()
    shooter = spawn(sim, 0, "mortar_team", (SHOOTER_CELL[0], SHOOTER_CELL[1]))
    shooter.state = SquadState.SET_UP
    target = dummy(sim, TARGET_CELL)

    fire_rounds(sim, shooter, 10)

    assert damage_taken(target) == 0.0
    assert shot_events(sim) == []


def test_below_min_range_no_shot():
    data = with_weapon("rifle", min_range=12.0)
    sim = combat_sim(data=data)
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)  # 10 m away

    fire_rounds(sim, shooter, 10)
    assert damage_taken(target) == 0.0


def _next_ready_after_one_firing(sim, shooter) -> int:
    fired_at = sim.state.tick
    fire_rounds(sim, shooter, 1)
    return shooter.members[0].next_ready_tick - fired_at


def test_suppressed_attacker_loses_accuracy_and_gains_cooldown():
    econ = fixture_data().economy
    trials = 400
    expected = trials * econ.suppressed_accuracy_mult
    sigma = math.sqrt(trials * econ.suppressed_accuracy_mult * (1 - econ.suppressed_accuracy_mult))
    for seed in (0, 1, 2):
        sim = combat_sim(seed=seed)
        shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
        target = dummy(sim, TARGET_CELL)
        shooter.suppressed = True  # task 9 sets this; we set it by hand
        fire_rounds(sim, shooter, trials)
        hits = damage_taken(target) / sim.data.weapons["rifle"].damage
        assert abs(hits - expected) <= 4 * sigma, (seed, hits)

    # cooldown: a fixed 2 s cooldown becomes 2 s x suppressed_cooldown_mult
    sim = combat_sim(data=with_weapon("rifle", cooldown=(2.0, 2.0)))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    dummy(sim, TARGET_CELL)
    assert _next_ready_after_one_firing(sim, shooter) == math.ceil(2.0 * TICKS_PER_SECOND)

    shooter.suppressed = True
    delay = _next_ready_after_one_firing(sim, shooter)
    assert delay == math.ceil(2.0 * econ.suppressed_cooldown_mult * TICKS_PER_SECOND)


def test_reload_delay_is_added_on_every_nth_firing():
    sim = combat_sim(data=with_weapon("rifle", cooldown=(1.0, 1.0), reload=(4.0, 4.0), reload_every=3))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    dummy(sim, TARGET_CELL)

    plain = math.ceil(1.0 * TICKS_PER_SECOND)
    reloading = math.ceil((1.0 + 4.0) * TICKS_PER_SECOND)
    delays = [_next_ready_after_one_firing(sim, shooter) for _ in range(6)]

    assert delays == [plain, plain, reloading, plain, plain, reloading]
    assert shooter.members[0].shots_since_reload == 0


def test_cooldown_range_mult_is_applied_per_band():
    data = with_weapon("rifle", cooldown=(1.0, 1.0), cooldown_range_mult=(1.0, 2.0, 3.0))
    plain = 1.0 * TICKS_PER_SECOND

    # band 0: 10 m
    sim = combat_sim(data=data)
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    dummy(sim, TARGET_CELL)
    assert _next_ready_after_one_firing(sim, shooter) == math.ceil(plain * 1.0)

    # band 1: 16 m (short 10 < d <= medium 20)
    sim = combat_sim(data=data)
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    dummy(sim, (SHOOTER_CELL[0] + 8, SHOOTER_CELL[1]))
    assert _next_ready_after_one_firing(sim, shooter) == math.ceil(plain * 2.0)

    # band 2: 26 m (medium 20 < d <= long 30); a spotter supplies the vision
    sim = combat_sim(data=data)
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    dummy(sim, (SHOOTER_CELL[0] + 13, SHOOTER_CELL[1]))
    spawn(sim, 0, "engineers", (SHOOTER_CELL[0] + 11, SHOOTER_CELL[1]))
    assert _next_ready_after_one_firing(sim, shooter) == math.ceil(plain * 3.0)


# ---------------------------------------------------------------------------
# suppression accumulation (task 9 consumes it)
# ---------------------------------------------------------------------------


def test_suppression_accumulates_per_bullet_and_clamps():
    sim = combat_sim(data=with_weapon("rifle", damage=0.0))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)
    rifle = sim.data.weapons["rifle"]

    fire_rounds(sim, shooter, 3)
    assert target.suppression == 3 * rifle.suppression[0]
    assert target.last_hit_tick == sim.state.tick - 1
    assert target.last_attacker_pos == (float(shooter.pos[0]), float(shooter.pos[1]))

    fire_rounds(sim, shooter, 100)
    assert target.suppression == 1.0


def test_suppression_immune_squads_accumulate_nothing():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    tank = dummy(sim, TARGET_CELL, def_id="tank")

    fire_rounds(sim, shooter, 10)

    assert tank.suppression == 0.0
    rifle = sim.data.weapons["rifle"]
    # vs armour_tank the rifle's damage mod is 0.01; no cover mod on vehicles
    assert damage_taken(tank) == pytest.approx(10 * rifle.damage * rifle.vs("armour_tank").damage)


# ---------------------------------------------------------------------------
# acquisition
# ---------------------------------------------------------------------------


def _priority_data(infantry_priority: float, tank_priority: float) -> GameData:
    rifle = fixture_data().weapons["rifle"]
    table = dict(rifle.target_table)
    table["infantry"] = TargetMods(priority=infantry_priority)
    table["armour_tank"] = replace(rifle.vs("armour_tank"), priority=tank_priority)
    return with_weapon("rifle", target_table=table)


def test_acquisition_prefers_higher_priority_then_nearer():
    sim = combat_sim(data=_priority_data(infantry_priority=5.0, tank_priority=0.0))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    tank = dummy(sim, (SHOOTER_CELL[0] + 2, SHOOTER_CELL[1]), def_id="tank")  # nearer
    infantry = dummy(sim, (SHOOTER_CELL[0] + 10, SHOOTER_CELL[1]))

    vision.run(sim)
    combat.run(sim)
    assert shooter.target_id == infantry.id

    # equal priority: the nearer one wins
    sim2 = combat_sim(data=_priority_data(infantry_priority=0.0, tank_priority=0.0))
    shooter2 = trim(spawn(sim2, 0, "rifles", SHOOTER_CELL))
    near = dummy(sim2, (SHOOTER_CELL[0] + 2, SHOOTER_CELL[1]), def_id="tank")
    dummy(sim2, (SHOOTER_CELL[0] + 10, SHOOTER_CELL[1]))
    vision.run(sim2)
    combat.run(sim2)
    assert shooter2.target_id == near.id
    assert tank.id != infantry.id


def test_acquisition_ties_break_on_lower_id():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    first = dummy(sim, (SHOOTER_CELL[0], SHOOTER_CELL[1] + 4))
    second = dummy(sim, (SHOOTER_CELL[0], SHOOTER_CELL[1] - 4))
    assert first.id < second.id

    vision.run(sim)
    combat.run(sim)
    assert shooter.target_id == first.id


def test_target_is_kept_while_valid_and_dropped_when_it_leaves_range():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)

    vision.run(sim)
    combat.run(sim)
    assert shooter.target_id == target.id

    target.pos = center_of((SHOOTER_CELL[0] + 25, SHOOTER_CELL[1]))
    vision.run(sim)
    combat.run(sim)
    assert shooter.target_id is None


def test_neutral_buildings_are_never_auto_targeted():
    sim = make_sim(
        rows(),
        starts=STARTS,
        neutral_buildings=[{"def": "house", "cell": [33, 11]}],
    )
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))

    vision.run(sim)
    combat.run(sim)

    assert shooter.target_id is None
    house = next(b for b in sim.state.buildings.values() if b.neutral)
    assert house.hp == sim.data.neutral["house"].hp


def test_retreating_pinned_and_constructing_squads_do_not_fire():
    for state in (SquadState.RETREATING, SquadState.CONSTRUCTING):
        sim = combat_sim()
        shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
        target = dummy(sim, TARGET_CELL)
        shooter.state = state
        fire_rounds(sim, shooter, 5)
        assert damage_taken(target) == 0.0, state

    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)
    shooter.pinned = True
    fire_rounds(sim, shooter, 5)
    assert damage_taken(target) == 0.0


def test_retreating_target_takes_the_retreat_accuracy_penalty():
    trials = 400
    sdef = fixture_data().squads["rifles"]
    expected = trials * sdef.retreat_received_accuracy
    sigma = math.sqrt(trials * 0.25)
    for seed in (0, 1, 2):
        sim = combat_sim(seed=seed)
        shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
        target = dummy(sim, TARGET_CELL)
        target.state = SquadState.RETREATING
        fire_rounds(sim, shooter, trials)
        hits = damage_taken(target) / sim.data.weapons["rifle"].damage
        assert abs(hits - expected) <= 4 * sigma, (seed, hits)


# ---------------------------------------------------------------------------
# buildings as targets
# ---------------------------------------------------------------------------


def test_enemy_building_is_damaged_and_destroyed():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    building = sim.spawn_building(1, "barracks", (29, 9))
    building.hp = 3 * sim.data.weapons["rifle"].damage

    footprint = sim.data.buildings["barracks"].footprint
    assert not sim.map.pass_inf[9, 29]

    fire_rounds(sim, shooter, 2)
    assert shooter.target_id == building.id
    assert building.hp == sim.data.weapons["rifle"].damage

    fire_rounds(sim, shooter, 1)
    assert building.id not in sim.state.buildings
    assert shooter.target_id is None
    assert [e for e in sim.state.events if e.kind == "building_destroyed" and e.data["id"] == building.id]
    for dx in range(footprint[0]):
        for dy in range(footprint[1]):
            assert sim.map.pass_inf[9 + dy, 29 + dx]


def test_every_shot_at_a_building_lands():
    """Buildings are large static targets: no accuracy roll at all."""
    # the rifle's long-band accuracy is 0.25, yet all 40 shots must land
    sim = combat_sim(data=with_weapon("rifle", accuracy=(0.25, 0.25, 0.25)))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    building = sim.spawn_building(1, "barracks", (29, 9))
    building.hp = 1.0e6
    start_hp = building.hp

    fire_rounds(sim, shooter, 40)

    events = shot_events(sim)
    assert len(events) == 40
    assert all(e.data["hit"] for e in events)
    rifle = sim.data.weapons["rifle"]
    assert start_hp - building.hp == 40 * rifle.damage * rifle.vs("building_light").damage


def test_buildings_are_not_auto_targeted_below_the_damage_mod_threshold():
    table = dict(fixture_data().weapons["rifle"].target_table)
    table["building_light"] = TargetMods(damage=0.1)
    sim = combat_sim(data=with_weapon("rifle", target_table=table))
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    sim.spawn_building(1, "barracks", (29, 9))

    vision.run(sim)
    combat.run(sim)

    assert shooter.target_id is None


# ---------------------------------------------------------------------------
# Attack order
# ---------------------------------------------------------------------------


def test_attack_order_validation():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    friend = trim(spawn(sim, 0, "rifles", (26, 12)))
    target = trim(spawn(sim, 1, "rifles", TARGET_CELL), disarm=True)
    vision.run(sim)

    assert not sim.issue(0, [Attack(shooter.id, 999999)])[0].ok
    assert not sim.issue(0, [Attack(shooter.id, friend.id)])[0].ok
    assert sim.issue(0, [Attack(shooter.id, target.id)])[0].ok
    assert shooter.order == Attack(shooter.id, target.id)

    # an unseen enemy cannot be attack-ordered
    hidden = trim(spawn(sim, 1, "rifles", (55, 2)), disarm=True)
    assert not sim.issue(0, [Attack(shooter.id, hidden.id)])[0].ok

    neutral_sim = make_sim(rows(), starts=STARTS, neutral_buildings=[{"def": "house", "cell": [27, 11]}])
    walker = trim(spawn(neutral_sim, 0, "rifles", SHOOTER_CELL))
    vision.run(neutral_sim)
    house = next(b for b in neutral_sim.state.buildings.values() if b.neutral)
    assert not neutral_sim.issue(0, [Attack(walker.id, house.id)])[0].ok


def test_attack_order_paths_toward_an_out_of_range_target_then_stops():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    far_cell = (SHOOTER_CELL[0] + 20, SHOOTER_CELL[1])
    target = dummy(sim, far_cell)
    # make the target visible to team 0 so the order validates
    sim.state.visible[0][:, :] = True

    assert sim.issue(0, [Attack(shooter.id, target.id)])[0].ok

    from coh.sim.systems import movement

    start_x = float(shooter.pos[0])
    for _ in range(200):
        movement.run(sim)
        vision.run(sim)
        combat.run(sim)
        sim.state.tick += 1
        if shooter.target_id == target.id:
            break

    assert shooter.target_id == target.id
    assert float(shooter.pos[0]) > start_x  # it walked into range
    assert shooter.path == []
    assert shooter.order == Attack(shooter.id, target.id)

    shooter.moving = False
    fire_rounds(sim, shooter, 60)  # it may stop at long range: accuracy 0.25
    assert damage_taken(target) > 0.0


def test_team_weapon_redeploys_when_its_attack_order_walks_it_into_range():
    from coh.sim.systems import movement

    sim = combat_sim()
    hmg = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
    hmg.state = SquadState.SET_UP
    target = dummy(sim, (SHOOTER_CELL[0] + 20, SHOOTER_CELL[1]))  # 40 m: out of hmg range
    # a friendly spotter keeps the target visible while the hmg walks in
    spawn(sim, 0, "engineers", (SHOOTER_CELL[0] + 18, SHOOTER_CELL[1]))
    vision.run(sim)
    assert sim.issue(0, [Attack(hmg.id, target.id)])[0].ok

    for _ in range(400):
        movement.run(sim)
        vision.run(sim)
        combat.run(sim)
        sim.state.tick += 1
        if hmg.state is SquadState.SET_UP and hmg.target_id == target.id:
            break

    assert hmg.target_id == target.id
    assert hmg.state is SquadState.SET_UP  # it tore down, walked, and set up again
    assert damage_taken(target) > 0.0


def test_attack_on_an_in_range_target_settles_out_of_moving_and_fires():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)
    vision.run(sim)

    # walking somewhere else when the Attack order lands
    assert sim.issue(0, [Move(shooter.id, (SHOOTER_CELL[0], SHOOTER_CELL[1] + 5))])[0].ok
    assert shooter.state is SquadState.MOVING
    assert sim.issue(0, [Attack(shooter.id, target.id)])[0].ok
    assert shooter.path == []  # the move is cancelled immediately

    combat.run(sim)

    assert shooter.state is SquadState.IDLE  # not stuck MOVING forever
    assert shooter.target_id == target.id
    assert len(shot_events(sim)) == 1
    assert damage_taken(target) > 0.0


def test_set_up_team_weapon_attacking_an_in_range_target_does_not_tear_down():
    sim = combat_sim()
    hmg = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
    hmg.state = SquadState.SET_UP
    target = dummy(sim, TARGET_CELL, count=4)  # hmg range 35 m; target at 10 m
    vision.run(sim)

    assert sim.issue(0, [Attack(hmg.id, target.id)])[0].ok
    assert hmg.state is SquadState.SET_UP

    fire_rounds(sim, hmg, 1)

    assert hmg.state is SquadState.SET_UP  # never went TEARING_DOWN / SETTING_UP
    assert hmg.target_id == target.id
    assert damage_taken(target) > 0.0  # the main weapon actually fired


def test_attack_pursuit_state_lives_on_the_squad_and_is_hashed():
    from coh.sim.state import canonical_state

    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)
    vision.run(sim)
    sim.state.tick = 12

    assert shooter.attack_last_seen_tick == -1
    assert shooter.attack_last_repath_tick == -1
    assert sim.issue(0, [Attack(shooter.id, target.id)])[0].ok
    assert shooter.attack_last_seen_tick == 12  # validation just saw it
    assert shooter.attack_last_repath_tick == -1  # may re-path at once

    squad_row = next(r for r in canonical_state(sim.state)["squads"] if r["id"] == shooter.id)
    assert squad_row["attack_last_seen_tick"] == 12
    assert squad_row["attack_last_repath_tick"] == -1

    before = sim.state_hash()
    shooter.attack_last_seen_tick = 13
    assert sim.state_hash() != before  # the fields are covered by the hash

    shooter.attack_last_seen_tick = 12
    combat.run(sim)
    sim.issue(0, [Attack(shooter.id, target.id)])
    # cleared with the order
    combat._clear_attack_order(sim, shooter)
    assert shooter.attack_last_seen_tick == -1
    assert shooter.attack_last_repath_tick == -1


def test_attack_order_is_cleared_when_the_target_dies():
    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = trim(spawn(sim, 1, "rifles", TARGET_CELL), disarm=True)
    vision.run(sim)
    assert sim.issue(0, [Attack(shooter.id, target.id)])[0].ok

    fire_rounds(sim, shooter, 10)

    assert target.id not in sim.state.squads
    assert shooter.order is None
    assert shooter.target_id is None


def test_attack_order_is_cleared_after_the_target_stays_unseen():
    from coh.sim.constants import ATTACK_ORDER_LOST_TARGET_S

    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)
    sim.state.visible[0][:, :] = True
    assert sim.issue(0, [Attack(shooter.id, target.id)])[0].ok

    combat.run(sim)
    sim.state.tick += 1
    sim.state.visible[0][:, :] = False

    for _ in range(int(ATTACK_ORDER_LOST_TARGET_S * TICKS_PER_SECOND) + 2):
        combat.run(sim)
        sim.state.tick += 1

    assert shooter.order is None
    assert shooter.target_id is None


def test_stop_order_clears_an_attack_order():
    from coh.sim.orders import Stop

    sim = combat_sim()
    shooter = trim(spawn(sim, 0, "rifles", SHOOTER_CELL))
    target = dummy(sim, TARGET_CELL)
    vision.run(sim)
    sim.issue(0, [Attack(shooter.id, target.id)])
    sim.issue(0, [Stop(shooter.id)])
    assert shooter.order is None


# ---------------------------------------------------------------------------
# determinism and performance
# ---------------------------------------------------------------------------


def test_combat_is_deterministic_for_a_given_seed():
    hashes = []
    for _ in range(2):
        sim = combat_sim(seed=5)
        spawn(sim, 0, "rifles", SHOOTER_CELL)
        spawn(sim, 1, "rifles", TARGET_CELL)
        sim.run(60)
        hashes.append(sim.state_hash())
    assert hashes[0] == hashes[1]


@pytest.mark.perf
def test_idle_squads_out_of_range_tick_combat_cheaply():
    sim = combat_sim()
    # 17 cells (34 m) apart: beyond the 30 m rifle range, so the squared-
    # distance filter must reject every pair before any per-pair work.
    for i in range(20):
        spawn(sim, 0, "rifles", (20 + i, 1))
        spawn(sim, 1, "rifles", (20 + i, HEIGHT - 2))
    vision.run(sim)
    combat.run(sim)  # warm any caches

    ticks = 200
    start = time.perf_counter()
    for _ in range(ticks):
        combat.run(sim)
        sim.state.tick += 1
    per_tick_ms = (time.perf_counter() - start) / ticks * 1000.0

    assert all(s.target_id is None for s in sim.state.squads.values())
    assert per_tick_ms < 3.0, f"{per_tick_ms:.3f} ms/tick"
