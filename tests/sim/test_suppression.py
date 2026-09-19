"""Tests for suppression thresholds/recovery, spill, retreat and reinforce (task 9)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coh.data.schema import GameData
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import Move, Reinforce, Retreat
from coh.sim.state import SquadState
from coh.sim.systems import combat, movement, suppression, vision
from tests.helpers import fixture_data, make_sim, spawn

WIDTH, HEIGHT = 60, 20

STARTS = [
    {"slot": 0, "team": 0, "hq_cell": [1, 1], "sector": "a"},
    {"slot": 1, "team": 1, "hq_cell": [WIDTH - 5, HEIGHT - 5], "sector": "a"},
]

SHOOTER_CELL = (25, 10)
TARGET_CELL = (30, 10)  # 10 m from the shooter: hmg short range band


def rows(*cells: tuple[tuple[int, int], str]) -> list[str]:
    grid = [["."] * WIDTH for _ in range(HEIGHT)]
    for (cx, cy), ch in cells:
        grid[cy][cx] = ch
    return ["".join(row) for row in grid]


def suppression_sim(terrain: list[str] | None = None, seed: int = 0, data: GameData | None = None):
    return make_sim(terrain if terrain is not None else rows(), seed=seed, data=data, starts=STARTS)


def with_weapon(weapon_id: str, **changes) -> GameData:
    data = fixture_data()
    weapons = dict(data.weapons)
    weapons[weapon_id] = replace(weapons[weapon_id], **changes)
    return replace(data, weapons=weapons)


def with_squad(def_id: str, **changes) -> GameData:
    data = fixture_data()
    squads = dict(data.squads)
    squads[def_id] = replace(squads[def_id], **changes)
    return replace(data, squads=squads)


def disarm(squad) -> None:
    for member in squad.members:
        member.weapon = ""


def run_ticks(sim, n: int) -> None:
    """`combat.run` then `suppression.run`, `n` times: bullet accumulation
    followed by decay/thresholds, exactly as `Sim.tick`'s `SYSTEM_ORDER` does."""
    vision.run(sim)
    for _ in range(n):
        combat.run(sim)
        suppression.run(sim)
        sim.state.tick += 1


# ---------------------------------------------------------------------------
# suppressed / pinned thresholds, hysteresis, recovery
# ---------------------------------------------------------------------------


def test_hmg_pins_a_rifle_squad_in_the_open_within_a_few_seconds():
    # The hmg crew is made immortal here: the point of this test is what
    # happens to the *target*, not a race over who dies of return fire first.
    sim = suppression_sim(data=with_squad("hmg_team", member_hp=1.0e6))
    hmg = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
    hmg.state = SquadState.SET_UP
    target = spawn(sim, 1, "rifles", TARGET_CELL)

    run_ticks(sim, 3 * TICKS_PER_SECOND)

    assert target.pinned
    assert target.suppressed

    # A pinned squad neither moves...
    pos_before = target.pos.copy()
    sim.issue(1, [Move(squad=target.id, cell=(5, 5))])
    movement.run(sim)
    assert np.array_equal(target.pos, pos_before)

    # ...nor fires (it has an enemy -- the hmg -- well within its own range).
    before_events = len(sim.state.events)
    combat.run(sim)
    fired = [
        e
        for e in sim.state.events[before_events:]
        if e.kind == "shot" and e.data["src"] == target.id
    ]
    assert fired == []


def test_same_squad_behind_heavy_cover_is_not_pinned_in_the_same_time():
    sim = suppression_sim(
        rows((TARGET_CELL, "c")),  # area heavy cover on the target's own cell
        data=with_squad("hmg_team", member_hp=1.0e6),
    )
    hmg = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
    hmg.state = SquadState.SET_UP
    target = spawn(sim, 1, "rifles", TARGET_CELL)
    # Trim to member 0, the one that actually stands on the squad's own cell
    # (the others ring it at formation offsets, which may land off the
    # single covered cell); this isolates the cover lookup this test cares
    # about from the "which member gets hit" randomness.
    target.members = target.members[:1]

    run_ticks(sim, 3 * TICKS_PER_SECOND)

    assert not target.pinned


def test_suppression_decays_and_flags_clear_with_hysteresis(no_combat):
    sim = suppression_sim()
    squad = spawn(sim, 0, "rifles", TARGET_CELL)
    sup = sim.data.squads["rifles"].suppression

    squad.suppression = sup.pin_at
    suppression.run(sim)
    assert squad.pinned
    assert squad.suppressed

    # Hovering just below `pin_at` but above `pin_recover` keeps it pinned
    # (hysteresis): only dropping under `pin_recover` clears the flag.
    squad.suppression = (sup.pin_at + sup.pin_recover) / 2.0
    suppression.run(sim)
    assert squad.pinned

    squad.last_hit_tick = -1  # long out of combat: recovery gets the noncombat multiplier
    for _ in range(400):
        suppression.run(sim)
        sim.state.tick += 1
        if squad.suppression <= 0.0:
            break

    assert squad.suppression == 0.0
    assert not squad.pinned
    assert not squad.suppressed


def test_heavier_cover_recovers_suppression_faster(no_combat):
    econ = fixture_data().economy
    assert econ.cover_recovery_mult["heavy"] > econ.cover_recovery_mult["open"]  # sanity on the fixture

    def decay_ticks(cover_cell: str | None) -> int:
        sim = suppression_sim(rows((TARGET_CELL, cover_cell)) if cover_cell else None)
        squad = spawn(sim, 0, "rifles", TARGET_CELL)
        squad.suppression = 0.9
        squad.last_hit_tick = -1000  # well past the noncombat delay from tick 0
        # Recovery only looks up cover once the squad has actually been hit
        # (it aims `cover_at` at the last attacker's direction); an arbitrary
        # past attacker position is enough to make that lookup live.
        squad.last_attacker_pos = (float(SHOOTER_CELL[0]), float(SHOOTER_CELL[1]))
        ticks = 0
        while squad.suppression > 0.0 and ticks < 2000:
            suppression.run(sim)
            sim.state.tick += 1
            ticks += 1
        return ticks

    assert decay_ticks("c") < decay_ticks(None)


# ---------------------------------------------------------------------------
# nearby-suppression spill (combat.add_suppression)
# ---------------------------------------------------------------------------


def test_spill_suppresses_a_neighbour_within_radius_but_not_beyond():
    data = with_weapon("hmg", nearby_suppression_radius=6.0, nearby_suppression_mult=0.5)
    sim = suppression_sim(data=data)
    weapon = sim.data.weapons["hmg"]

    attacker = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
    victim = spawn(sim, 1, "rifles", TARGET_CELL)
    near = spawn(sim, 1, "rifles", (TARGET_CELL[0] + 2, TARGET_CELL[1]))  # 4 m away: inside radius
    far = spawn(sim, 1, "rifles", (TARGET_CELL[0] + 10, TARGET_CELL[1]))  # 20 m away: outside radius
    same_side_as_attacker = spawn(sim, 0, "rifles", (TARGET_CELL[0] + 1, TARGET_CELL[1]))

    combat.add_suppression(sim, attacker, victim, weapon, band=0, mult=1.0)

    s = weapon.suppression[0]
    assert victim.suppression == pytest.approx(s)
    assert near.suppression == pytest.approx(s * weapon.nearby_suppression_mult)
    assert far.suppression == 0.0
    assert same_side_as_attacker.suppression == 0.0  # spill only hits the victim's own side

    assert near.last_hit_tick == sim.state.tick
    assert near.last_attacker_pos is None  # spill never sets the aim point, only the victim itself does


def test_spill_skips_a_retreating_neighbour():
    data = with_weapon("hmg", nearby_suppression_radius=6.0, nearby_suppression_mult=0.5)
    sim = suppression_sim(data=data)
    weapon = sim.data.weapons["hmg"]

    attacker = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
    victim = spawn(sim, 1, "rifles", TARGET_CELL)
    near = spawn(sim, 1, "rifles", (TARGET_CELL[0] + 2, TARGET_CELL[1]))
    near.state = SquadState.RETREATING

    combat.add_suppression(sim, attacker, victim, weapon, band=0, mult=1.0)

    assert near.suppression == 0.0


# ---------------------------------------------------------------------------
# retreat
# ---------------------------------------------------------------------------


def test_retreat_clears_suppression_and_ignores_further_orders(no_combat):
    sim = suppression_sim()
    squad = spawn(sim, 0, "rifles", TARGET_CELL)
    squad.suppression = 0.9
    squad.suppressed = True
    squad.pinned = True

    (result,) = sim.issue(0, [Retreat(squad=squad.id)])
    assert result.ok
    assert squad.suppression == 0.0
    assert not squad.suppressed
    assert not squad.pinned
    assert squad.state is SquadState.RETREATING

    (blocked,) = sim.issue(0, [Move(squad=squad.id, cell=(5, 5))])
    assert not blocked.ok


def test_retreating_squad_reaches_its_hq_and_settles_idle(no_combat):
    sim = suppression_sim()
    squad = spawn(sim, 0, "rifles", TARGET_CELL)
    sim.issue(0, [Retreat(squad=squad.id)])

    sim.run(300)

    assert squad.state is SquadState.IDLE
    assert squad.order is None


def test_retreating_squad_takes_reduced_hits():
    """Task 8's `_retreat_accuracy_mult` scales incoming accuracy by
    `retreat_received_accuracy` while `state is RETREATING`. Compare total
    damage dealt over the same number of firings against an (immortal, so
    member deaths never make the two runs' RNG streams diverge) disarmed
    target, standing vs. retreating."""
    retreat_mult = fixture_data().squads["rifles"].retreat_received_accuracy
    assert 0.0 < retreat_mult < 1.0
    immortal_rifles = with_squad("rifles", member_hp=1.0e6)

    def damage_taken(retreating: bool, seed: int) -> float:
        sim = suppression_sim(seed=seed, data=immortal_rifles)
        shooter = spawn(sim, 0, "hmg_team", SHOOTER_CELL)
        shooter.state = SquadState.SET_UP
        target = spawn(sim, 1, "rifles", TARGET_CELL)
        disarm(target)  # only the shooter's hits matter here
        if retreating:
            target.state = SquadState.RETREATING
        vision.run(sim)
        for _ in range(40):
            for m in shooter.members:
                m.next_ready_tick = sim.state.tick
            combat.run(sim)
            sim.state.tick += 1
        return sum(1.0e6 - m.hp for m in target.members)

    standing = damage_taken(False, seed=3)
    retreating = damage_taken(True, seed=3)
    assert retreating < standing


def test_vehicle_cannot_retreat():
    sim = suppression_sim()
    squad = spawn(sim, 0, "tank", TARGET_CELL)
    (result,) = sim.issue(0, [Retreat(squad=squad.id)])
    assert not result.ok


def test_team_weapon_retreats_instantly_without_teardown_delay(no_combat):
    sim = suppression_sim()
    squad = spawn(sim, 0, "hmg_team", TARGET_CELL)
    squad.state = SquadState.SET_UP

    (result,) = sim.issue(0, [Retreat(squad=squad.id)])
    assert result.ok
    assert squad.state is SquadState.RETREATING  # not TEARING_DOWN


# ---------------------------------------------------------------------------
# reinforce
# ---------------------------------------------------------------------------


def _near_hq_cell() -> tuple[int, int]:
    # Just south-east of the [1, 1]-footprint-[4, 4] HQ, well within its
    # `reinforce_radius` of 20 m.
    return (4, 6)


def test_reinforce_restores_members_and_charges_manpower(no_combat, monkeypatch):
    from coh.sim.systems import economy

    monkeypatch.setattr(economy, "run", lambda sim: None)  # isolate the reinforce cost from income

    sim = suppression_sim()
    squad = spawn(sim, 0, "rifles", _near_hq_cell())
    squad.members = squad.members[:2]  # missing 2 of 4
    player = sim.state.players[0]
    manpower_before = player.manpower

    sdef = sim.data.squads["rifles"]
    econ = sim.data.economy
    per_model = sdef.cost.manpower / sdef.members * econ.reinforce_base_cost_frac * sdef.reinforce_cost_mult

    (result,) = sim.issue(0, [Reinforce(squad=squad.id)])
    assert result.ok
    assert squad.reinforcing
    assert player.manpower == pytest.approx(manpower_before - per_model)

    sim.run(200)  # well past 2 models' worth of reinforce time

    assert len(squad.members) == sdef.members
    assert all(m.hp == sdef.member_hp for m in squad.members)
    assert not squad.reinforcing
    assert squad.order is None
    assert player.manpower == pytest.approx(manpower_before - 2 * per_model)


def test_reinforce_gives_the_new_member_the_slot_default_weapon(no_combat):
    sim = suppression_sim()
    squad = spawn(sim, 0, "rifles", _near_hq_cell())
    squad.members = squad.members[:3]  # missing the 4th (slot 3)
    sdef = sim.data.squads["rifles"]

    sim.issue(0, [Reinforce(squad=squad.id)])
    sim.run(int(sdef.build_time / sdef.members * TICKS_PER_SECOND) + 4)

    assert len(squad.members) == 4
    assert squad.members[3].weapon == sdef.loadout[3]


def test_reinforce_fails_away_from_any_reinforce_point():
    sim = suppression_sim()
    squad = spawn(sim, 0, "rifles", SHOOTER_CELL)  # far from the HQ
    squad.members = squad.members[:2]

    (result,) = sim.issue(0, [Reinforce(squad=squad.id)])
    assert not result.ok


def test_reinforce_fails_when_the_player_cannot_afford_it():
    sim = suppression_sim()
    squad = spawn(sim, 0, "rifles", _near_hq_cell())
    squad.members = squad.members[:2]
    sim.state.players[0].manpower = 0.0

    (result,) = sim.issue(0, [Reinforce(squad=squad.id)])
    assert not result.ok


def test_reinforce_is_invalid_when_squad_is_already_full():
    sim = suppression_sim()
    squad = spawn(sim, 0, "rifles", _near_hq_cell())
    (result,) = sim.issue(0, [Reinforce(squad=squad.id)])
    assert not result.ok
