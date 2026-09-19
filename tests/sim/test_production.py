"""Tests for coh.sim.systems.production (task 14): construction, train/research
queues, tech gating and squad weapon upgrades.

Default map: three vertical sectors, 'a' (cx 0..13) is team 0's HQ sector and
the only one it supplies at the start, 'b' (14..25) is no-man's land. Team 0's
HQ is a 4x4 footprint at (2, 2); its 'west' strategic point sits at (6, 14).

Fixture tech tree (see tests/data/fixtures/): engineers build `barracks`,
`tank_depot` (requires the barracks) and `op_us`; pioneers build
`panzer_command` (requires the researched `phase_2` upgrade) and `op_wehr`.
"""

from __future__ import annotations

import pytest

from coh.sim.constants import CONSTRUCTION_START_HP_FRAC, MAX_QUEUE_LEN
from coh.sim.orders import Build, BuyUpgrade, Move, Research, Train
from coh.sim.sim import PlayerSetup
from coh.sim.state import SquadState
from tests.helpers import make_sim, spawn

BARRACKS_TICKS = 30 * 8  # barracks build_time 30 s
TANK_DEPOT_TICKS = 40 * 8


def _rich(sim, player_id=0):
    player = sim.state.players[player_id]
    player.manpower = player.munitions = player.fuel = 10_000.0
    return player


def _site(sim, def_id, owner=0):
    """The one building of `def_id` owned by `owner`."""
    return next(
        sim.state.buildings[bid]
        for bid in sorted(sim.state.buildings)
        if sim.state.buildings[bid].def_id == def_id and sim.state.buildings[bid].owner == owner
    )


def _build_now(sim, player_id, structure, cell, builder_cell=None, *, ticks, builder=None):
    """Order `structure` from a builder standing next to `cell` and run it to done."""
    if builder is None:
        faction = sim.state.players[player_id].faction
        builder = spawn(sim, player_id, "engineers" if faction == "us" else "pioneers", builder_cell)
    assert sim.issue(player_id, [Build(squad=builder.id, structure=structure, cell=cell)])[0].ok
    sim.run(ticks)
    site = _site(sim, structure, player_id)
    assert site.progress == 1.0
    return builder, site


# ---------------------------------------------------------------------------
# Build validation
# ---------------------------------------------------------------------------


def test_build_requires_a_prerequisite_building():
    sim = make_sim()
    _rich(sim)
    builder = spawn(sim, 0, "engineers", (7, 11))

    (result,) = sim.issue(0, [Build(squad=builder.id, structure="tank_depot", cell=(8, 10))])
    assert result.ok is False
    assert "requires" in result.reason and "barracks" in result.reason
    assert sim.state.players[0].invalid_orders == 1


def test_build_requires_funds():
    sim = make_sim()
    sim.state.players[0].manpower = 10.0  # barracks costs 200
    builder = spawn(sim, 0, "engineers", (7, 11))

    (result,) = sim.issue(0, [Build(squad=builder.id, structure="barracks", cell=(8, 10))])
    assert result.ok is False
    assert "afford" in result.reason
    assert not any(b.def_id == "barracks" for b in sim.state.buildings.values())


def test_build_outside_supplied_territory_is_rejected():
    sim = make_sim()
    _rich(sim)
    builder = spawn(sim, 0, "engineers", (18, 11))

    (result,) = sim.issue(0, [Build(squad=builder.id, structure="barracks", cell=(18, 10))])
    assert result.ok is False
    assert "supplied territory" in result.reason


def test_build_on_blocked_cells_is_rejected():
    sim = make_sim()
    _rich(sim)
    builder = spawn(sim, 0, "engineers", (7, 11))

    # (2, 2) is the HQ footprint: stamped impassable.
    (result,) = sim.issue(0, [Build(squad=builder.id, structure="barracks", cell=(2, 2))])
    assert result.ok is False
    assert "buildable terrain" in result.reason


def test_build_of_another_factions_structure_is_rejected():
    sim = make_sim()
    _rich(sim)
    builder = spawn(sim, 0, "engineers", (7, 11))
    builder.def_id = "pioneers"  # a wehr builder in a us player's hands

    (result,) = sim.issue(0, [Build(squad=builder.id, structure="panzer_command", cell=(8, 10))])
    assert result.ok is False
    assert "wehr building" in result.reason


def test_build_of_a_structure_the_squad_cannot_make_is_rejected():
    sim = make_sim()
    _rich(sim)
    rifles = spawn(sim, 0, "rifles", (7, 11))

    (result,) = sim.issue(0, [Build(squad=rifles.id, structure="barracks", cell=(8, 10))])
    assert result.ok is False
    assert "cannot build" in result.reason


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_places_the_site_immediately_and_charges_once():
    sim = make_sim()
    player = sim.state.players[0]
    builder = spawn(sim, 0, "engineers", (7, 11))
    before = player.manpower

    assert sim.issue(0, [Build(squad=builder.id, structure="barracks", cell=(8, 10))])[0].ok

    site = _site(sim, "barracks")
    bdef = sim.data.buildings["barracks"]
    assert player.manpower == before - bdef.cost.manpower
    assert site.progress == 0.0
    assert site.hp == pytest.approx(bdef.hp * CONSTRUCTION_START_HP_FRAC)
    assert builder.build_target == site.id
    assert not sim.map.pass_veh[10, 8]  # footprint stamped impassable
    assert any(e.kind == "construction_started" for e in sim.state.events)


def test_one_builder_completes_in_build_time_and_two_in_half():
    sim = make_sim()
    builder = spawn(sim, 0, "engineers", (7, 11))
    assert sim.issue(0, [Build(squad=builder.id, structure="barracks", cell=(8, 10))])[0].ok
    site = _site(sim, "barracks")

    sim.run(BARRACKS_TICKS - 1)
    assert site.progress < 1.0
    assert builder.state is SquadState.CONSTRUCTING

    sim.tick()
    assert site.progress == 1.0
    assert site.hp == pytest.approx(sim.data.buildings["barracks"].hp)
    assert builder.order is None
    assert builder.build_target is None
    assert builder.state is SquadState.IDLE
    assert any(e.kind == "building_completed" and e.data["building"] == site.id for e in sim.state.events)

    two = make_sim()
    first = spawn(two, 0, "engineers", (7, 11))
    second = spawn(two, 0, "engineers", (7, 12))
    assert two.issue(0, [Build(squad=first.id, structure="barracks", cell=(8, 10))])[0].ok
    assert two.issue(0, [Build(squad=second.id, structure="barracks", cell=(8, 10))])[0].ok

    two.run(BARRACKS_TICKS // 2 - 1)
    assert _site(two, "barracks").progress < 1.0
    two.tick()
    assert _site(two, "barracks").progress == 1.0


def test_assisting_an_unfinished_site_costs_nothing():
    sim = make_sim()
    first = spawn(sim, 0, "engineers", (7, 11))
    assert sim.issue(0, [Build(squad=first.id, structure="barracks", cell=(8, 10))])[0].ok
    after_paying = sim.state.players[0].manpower

    second = spawn(sim, 0, "engineers", (7, 12))
    assert sim.issue(0, [Build(squad=second.id, structure="barracks", cell=(8, 10))])[0].ok

    assert sim.state.players[0].manpower == after_paying
    assert len([b for b in sim.state.buildings.values() if b.def_id == "barracks"]) == 1


def test_builder_ordered_away_pauses_construction_and_can_resume():
    sim = make_sim()
    builder = spawn(sim, 0, "engineers", (7, 11))
    assert sim.issue(0, [Build(squad=builder.id, structure="barracks", cell=(8, 10))])[0].ok
    site = _site(sim, "barracks")

    sim.run(80)
    paused_at = site.progress
    paused_hp = site.hp
    assert 0.0 < paused_at < 1.0

    assert sim.issue(0, [Move(squad=builder.id, cell=(3, 3))])[0].ok
    sim.run(80)
    assert site.progress == paused_at
    assert site.hp == paused_hp
    assert builder.build_target is None

    # Resuming is free and picks up where it left off.
    before = sim.state.players[0].manpower
    assert sim.issue(0, [Build(squad=builder.id, structure="barracks", cell=(8, 10))])[0].ok
    assert sim.state.players[0].manpower == before
    sim.run(400)
    assert site.progress == 1.0


def test_observation_post_is_built_on_an_owned_point_and_registered():
    sim = make_sim()
    _rich(sim)
    _builder, op = _build_now(sim, 0, "op_us", (6, 14), (5, 14), ticks=2)

    assert sim.state.points["west"].op_building == op.id


def test_observation_post_off_a_point_or_on_an_unowned_one_is_rejected():
    sim = make_sim()
    _rich(sim)
    builder = spawn(sim, 0, "engineers", (5, 14))

    (off_point,) = sim.issue(0, [Build(squad=builder.id, structure="op_us", cell=(5, 10))])
    assert off_point.ok is False
    assert "not a capture point" in off_point.reason

    (enemy_point,) = sim.issue(0, [Build(squad=builder.id, structure="op_us", cell=(33, 14))])
    assert enemy_point.ok is False
    assert "not owned and supplied" in enemy_point.reason


def test_second_observation_post_on_the_same_point_is_rejected():
    sim = make_sim()
    _rich(sim)
    _build_now(sim, 0, "op_us", (6, 14), (5, 14), ticks=2)

    another = spawn(sim, 0, "engineers", (5, 15))
    (result,) = sim.issue(0, [Build(squad=another.id, structure="op_us", cell=(6, 14))])
    assert result.ok is False
    assert "already has an observation post" in result.reason


# ---------------------------------------------------------------------------
# Train / research queues
# ---------------------------------------------------------------------------


def test_train_deducts_queues_and_spawns_south_of_the_building():
    sim = make_sim()
    player = _rich(sim)
    hq = sim.state.buildings[player.hq_id]
    before = player.manpower

    assert sim.issue(0, [Train(building=hq.id, unit="engineers")])[0].ok
    assert player.manpower == before - sim.data.squads["engineers"].cost.manpower
    assert [(q.kind, q.item_id) for q in hq.queue] == [("train", "engineers")]

    sim.run(15 * 8 - 1)  # engineers build_time 15 s
    assert hq.queue and hq.queue[0].remaining_s > 0

    sim.tick()
    assert hq.queue == []
    event = next(e for e in sim.state.events if e.kind == "unit_trained")
    trained = sim.state.squads[event.data["squad"]]
    assert trained.def_id == "engineers"
    assert trained.state is SquadState.IDLE
    # HQ footprint (2,2)+4x4 -> bottom-centre south cell is (3, 6).
    assert (int(trained.pos[0] // 2), int(trained.pos[1] // 2)) == (3, 6)


def test_queue_processes_the_head_only_and_in_order():
    sim = make_sim()
    player = _rich(sim)
    hq = sim.state.buildings[player.hq_id]

    assert sim.issue(0, [Train(building=hq.id, unit="engineers")])[0].ok
    assert sim.issue(0, [Research(building=hq.id, upgrade="research_1")])[0].ok

    sim.run(8)
    assert hq.queue[0].remaining_s == pytest.approx(15 - 1)
    assert hq.queue[1].remaining_s == pytest.approx(30)  # untouched behind the head


def test_queue_length_is_capped():
    from dataclasses import replace

    from tests.helpers import fixture_data

    base = fixture_data()  # plenty of population, so only the cap can bite
    sim = make_sim(data=replace(base, economy=replace(base.economy, base_population=100)))
    player = _rich(sim)
    hq = sim.state.buildings[player.hq_id]

    for _ in range(MAX_QUEUE_LEN):
        assert sim.issue(0, [Train(building=hq.id, unit="engineers")])[0].ok
    (overflow,) = sim.issue(0, [Train(building=hq.id, unit="engineers")])
    assert overflow.ok is False
    assert "queue is full" in overflow.reason


def test_train_respects_the_population_cap():
    sim = make_sim()
    player = _rich(sim)
    hq = sim.state.buildings[player.hq_id]

    # cap 22, starting engineers use 4; four more queued engineers reach 20.
    for _ in range(4):
        assert sim.issue(0, [Train(building=hq.id, unit="engineers")])[0].ok
    (over,) = sim.issue(0, [Train(building=hq.id, unit="engineers")])
    assert over.ok is False
    assert "population cap" in over.reason


def test_train_from_an_unfinished_building_is_rejected():
    sim = make_sim()
    _rich(sim)
    builder = spawn(sim, 0, "engineers", (7, 11))
    assert sim.issue(0, [Build(squad=builder.id, structure="barracks", cell=(8, 10))])[0].ok

    (result,) = sim.issue(0, [Train(building=_site(sim, "barracks").id, unit="rifles")])
    assert result.ok is False
    assert "under construction" in result.reason


def test_train_of_a_unit_the_building_does_not_produce_is_rejected():
    sim = make_sim()
    player = _rich(sim)
    (result,) = sim.issue(0, [Train(building=player.hq_id, unit="rifles")])
    assert result.ok is False
    assert "does not produce" in result.reason


def test_research_completes_once_and_cannot_be_re_queued():
    sim = make_sim()
    player = _rich(sim)
    hq = sim.state.buildings[player.hq_id]

    assert sim.issue(0, [Research(building=hq.id, upgrade="research_1")])[0].ok
    (duplicate,) = sim.issue(0, [Research(building=hq.id, upgrade="research_1")])
    assert duplicate.ok is False
    assert "already queued" in duplicate.reason

    sim.run(30 * 8)  # research_1 time 30 s
    assert "research_1" in player.upgrades
    assert any(e.kind == "research_completed" for e in sim.state.events)

    (again,) = sim.issue(0, [Research(building=hq.id, upgrade="research_1")])
    assert again.ok is False
    assert "already has" in again.reason


# ---------------------------------------------------------------------------
# Tech trees, end to end through orders
# ---------------------------------------------------------------------------


def test_us_tech_tree_barracks_unlocks_tank_depot_which_trains_a_tank():
    sim = make_sim()
    _rich(sim)
    builder = spawn(sim, 0, "engineers", (7, 11))
    assert sim.issue(0, [Build(squad=builder.id, structure="tank_depot", cell=(4, 10))])[0].ok is False

    _build_now(sim, 0, "barracks", (8, 10), ticks=BARRACKS_TICKS, builder=builder)
    _, depot = _build_now(sim, 0, "tank_depot", (4, 10), ticks=TANK_DEPOT_TICKS, builder=builder)

    assert sim.issue(0, [Train(building=depot.id, unit="tank")])[0].ok
    sim.run(60 * 8)  # tank build_time 60 s
    assert any(s.def_id == "tank" and s.owner == 0 for s in sim.state.squads.values())


def test_wehr_tech_tree_phase_research_unlocks_panzer_command():
    sim = make_sim(
        players=[
            PlayerSetup(faction="wehr", team=0, start_slot=0),
            PlayerSetup(faction="us", team=1, start_slot=1),
        ]
    )
    player = _rich(sim)
    hq = sim.state.buildings[player.hq_id]
    pioneer = spawn(sim, 0, "pioneers", (7, 11))

    (blocked,) = sim.issue(0, [Build(squad=pioneer.id, structure="panzer_command", cell=(8, 10))])
    assert blocked.ok is False
    assert "phase_2" in blocked.reason

    assert sim.issue(0, [Research(building=hq.id, upgrade="phase_2")])[0].ok
    sim.run(20 * 8)  # phase_2 time 20 s
    assert "phase_2" in player.upgrades

    assert sim.issue(0, [Build(squad=pioneer.id, structure="panzer_command", cell=(8, 10))])[0].ok
    sim.run(TANK_DEPOT_TICKS + 8)
    command = _site(sim, "panzer_command")
    assert command.progress == 1.0

    assert sim.issue(0, [Train(building=command.id, unit="tank")])[0].ok
    sim.run(60 * 8)
    assert any(s.def_id == "tank" and s.owner == 0 for s in sim.state.squads.values())


# ---------------------------------------------------------------------------
# Squad weapon upgrades
# ---------------------------------------------------------------------------


def test_buy_upgrade_charges_munitions_and_swaps_weapons_after_its_time():
    sim = make_sim()
    player = sim.state.players[0]
    player.munitions = 100.0
    sim.spawn_building(0, "barracks", (8, 10))  # `bar` requires a completed barracks
    rifles = spawn(sim, 0, "rifles", (5, 10))

    assert sim.issue(0, [BuyUpgrade(squad=rifles.id, upgrade="bar")])[0].ok
    assert player.munitions == pytest.approx(40.0)  # bar costs 60
    assert rifles.pending_upgrade == "bar"

    sim.run(20 * 8)  # bar time 20 s
    assert [m.weapon for m in rifles.members] == ["rifle"] * 4

    sim.tick()
    assert [m.weapon for m in rifles.members] == ["bar", "bar", "rifle", "rifle"]
    assert rifles.upgrades == ["bar"]
    assert rifles.pending_upgrade is None
    assert any(e.kind == "upgrade_bought" and e.data["squad"] == rifles.id for e in sim.state.events)


def test_buy_upgrade_outside_supplied_territory_is_rejected():
    sim = make_sim()
    sim.state.players[0].munitions = 100.0
    sim.spawn_building(0, "barracks", (8, 10))
    rifles = spawn(sim, 0, "rifles", (20, 10))  # sector 'b', not supplied

    (result,) = sim.issue(0, [BuyUpgrade(squad=rifles.id, upgrade="bar")])
    assert result.ok is False
    assert "supplied friendly territory" in result.reason


def test_buy_upgrade_needs_its_requirement_a_free_slot_and_munitions():
    sim = make_sim()
    player = sim.state.players[0]
    rifles = spawn(sim, 0, "rifles", (5, 10))

    player.munitions = 100.0
    (no_req,) = sim.issue(0, [BuyUpgrade(squad=rifles.id, upgrade="bar")])
    assert no_req.ok is False and "requires" in no_req.reason

    sim.spawn_building(0, "barracks", (8, 10))
    player.munitions = 10.0
    (broke,) = sim.issue(0, [BuyUpgrade(squad=rifles.id, upgrade="bar")])
    assert broke.ok is False and "afford" in broke.reason

    player.munitions = 500.0
    player.upgrades.append("research_1")  # unlocks the second fixture upgrade
    assert sim.issue(0, [BuyUpgrade(squad=rifles.id, upgrade="bar")])[0].ok
    (no_slot,) = sim.issue(0, [BuyUpgrade(squad=rifles.id, upgrade="bar_upgrade")])
    assert no_slot.ok is False and "upgrade slot" in no_slot.reason


def test_buy_upgrade_on_a_squad_it_does_not_apply_to_is_rejected():
    sim = make_sim()
    sim.state.players[0].munitions = 100.0
    sim.spawn_building(0, "barracks", (8, 10))
    engineers = spawn(sim, 0, "engineers", (5, 10))

    (result,) = sim.issue(0, [BuyUpgrade(squad=engineers.id, upgrade="bar")])
    assert result.ok is False
    assert "does not apply" in result.reason


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_a_full_production_run_is_deterministic():
    def play(seed):
        sim = make_sim(seed=seed)
        _rich(sim)
        _build_now(sim, 0, "barracks", (8, 10), (7, 11), ticks=BARRACKS_TICKS)
        barracks = _site(sim, "barracks")
        sim.issue(0, [Train(building=barracks.id, unit="rifles")])
        sim.run(200)
        return sim.state_hash()

    assert play(5) == play(5)
