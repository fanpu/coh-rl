"""Tests for the sim core: start state, orders, tick loop, state hashing."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from coh.maps.format import GameMap, cell_of, center_of
from coh.sim.pathfinding import adjacent_cells
from coh.sim import orders as orders_mod
from coh.sim import systems
from coh.sim.orders import (
    Attack,
    AttackMove,
    Build,
    BuyUpgrade,
    Capture,
    Garrison,
    Move,
    Reinforce,
    Research,
    Retreat,
    SetFacing,
    Stop,
    Train,
    Ungarrison,
    order_from_dict,
    order_to_dict,
)
from coh.sim.state import Event, SquadState
from tests.helpers import fixture_data, make_map, make_sim, spawn

# --------------------------------------------------------------------------
# start state
# --------------------------------------------------------------------------


def test_start_state_has_hq_and_builder_per_player():
    sim = make_sim()

    assert sorted(sim.state.players) == [0, 1]
    for pid, player in sim.state.players.items():
        hq = sim.state.buildings[player.hq_id]
        assert hq.owner == pid
        assert sim.data.buildings[hq.def_id].is_hq
        assert hq.progress == 1.0

        builders = [s for s in sim.state.squads.values() if s.owner == pid]
        assert len(builders) == 1
        assert sim.data.squads[builders[0].def_id].builds  # a construction-capable squad


def test_start_resources_and_tickets_come_from_data():
    sim = make_sim()
    econ = fixture_data().economy
    for player in sim.state.players.values():
        assert player.manpower == econ.start_resources.manpower
        assert player.munitions == econ.start_resources.munitions
        assert player.fuel == econ.start_resources.fuel
    assert sim.state.tickets == {0: float(econ.tickets), 1: float(econ.tickets)}


def test_hq_footprint_is_stamped_impassable():
    sim = make_sim()
    hq = sim.state.buildings[sim.state.players[0].hq_id]
    cx, cy = hq.cell
    w, h = sim.data.buildings[hq.def_id].footprint

    assert not sim.map.pass_inf[cy : cy + h, cx : cx + w].any()
    assert not sim.map.pass_veh[cy : cy + h, cx : cx + w].any()
    assert sim.map.los_block[cy : cy + h, cx : cx + w].all()


def test_sim_owns_a_private_copy_of_the_map():
    game_map = make_map()
    sim = make_sim(game_map=game_map)
    assert sim.map is not game_map
    hq_x, hq_y = game_map.starts[0].hq_cell
    assert game_map.pass_inf[hq_y, hq_x]  # the caller's map is not stamped


def test_hq_footprint_is_centred_on_the_starts_hq_cell():
    """`StartDef.hq_cell` names the middle of the HQ, not its top-left corner:
    anchoring it top-left buried one seat two cells deeper in its corner than
    the other on an otherwise symmetric map."""
    sim = make_sim()
    for player_id, player in sim.state.players.items():
        hq = sim.state.buildings[player.hq_id]
        w, h = sim.data.buildings[hq.def_id].footprint
        anchor = sim.map.starts[sim.player_setups[player_id].start_slot].hq_cell
        assert hq.cell == (anchor[0] - w // 2, anchor[1] - h // 2)


def test_builder_spawns_on_the_side_of_the_hq_facing_the_map_centre():
    sim = make_sim()
    hq = sim.state.buildings[sim.state.players[0].hq_id]
    cx, cy = hq.cell
    w, h = sim.data.buildings[hq.def_id].footprint
    builder = next(s for s in sim.state.squads.values() if s.owner == 0)

    bx, by = cell_of(builder.pos)
    assert sim.map.pass_inf[by, bx]
    assert (bx, by) in adjacent_cells((cx, cy), (w, h), sim.map.width, sim.map.height)
    # Player 0's HQ is in the north-west: the centre lies south-east of it.
    assert bx >= cx + w - 1 and by >= cy + h - 1


def _mirror_map(width=40, height=30, hq=(8, 6)):
    """A map that is its own 180-degree rotation, with the two `hq_cell`s
    placed as rotational images (see `Sim._hq_footprint_cell` for the
    convention an even footprint needs)."""
    return make_map(
        ["." * width] * height,
        sectors=["a" * 14 + "b" * 12 + "c" * 14] * height,
        points=[
            {"id": "west", "name": "West", "type": "strategic", "cell": [6, 7]},
            {"id": "mid", "name": "Mid", "type": "victory", "cell": [width // 2, height // 2]},
            {"id": "east", "name": "East", "type": "strategic", "cell": [width - 1 - 6, height - 1 - 7]},
        ],
        starts=[
            {"slot": 0, "team": 0, "hq_cell": list(hq), "sector": "a"},
            {"slot": 1, "team": 1, "hq_cell": [width - hq[0], height - hq[1]], "sector": "c"},
        ],
    )


def test_the_two_seats_are_exact_180_degree_images_on_a_symmetric_map():
    """Seat asymmetry is a silent bias in every mirror match: the HQ
    footprint and the starting builder must rotate onto each other."""
    sim = make_sim(game_map=_mirror_map())
    w, h = sim.map.width, sim.map.height

    def rotate(cells):
        return {(w - 1 - cx, h - 1 - cy) for cx, cy in cells}

    def hq_cells(player_id):
        hq = sim.state.buildings[sim.state.players[player_id].hq_id]
        fw, fh = sim.data.buildings[hq.def_id].footprint
        return {(hq.cell[0] + dx, hq.cell[1] + dy) for dx in range(fw) for dy in range(fh)}

    def builder_cell(player_id):
        squad = next(s for s in sim.state.squads.values() if s.owner == player_id)
        return cell_of(squad.pos)

    assert rotate(hq_cells(0)) == hq_cells(1)
    assert rotate({builder_cell(0)}) == {builder_cell(1)}


def test_points_start_neutral_except_hq_sectors():
    sim = make_sim()
    hq_sectors = {sim.map.starts[p.start_slot].sector: p.team for p in sim.player_setups}

    assert set(sim.state.points) == set(sim.map.points)
    for pid, point in sim.state.points.items():
        sector = sim.map.points[pid].sector
        if sector in hq_sectors:
            assert point.owner_team == hq_sectors[sector]
            assert point.progress == 1.0
        else:
            assert point.owner_team is None
            assert point.progress == 0.0
        assert point.capturing_team is None
        assert point.op_building is None


def test_connected_sectors_start_as_sorted_lists_of_hq_sectors():
    sim = make_sim()
    for setup in sim.player_setups:
        sector = sim.map.starts[setup.start_slot].sector
        connected = sim.state.connected[setup.team]
        assert isinstance(connected, list)
        assert connected == sorted(connected)
        assert sector in connected


def test_neutral_buildings_are_placed_as_entities():
    sim = make_sim(neutral_buildings=[{"def": "house", "cell": [18, 10]}])
    neutral = [b for b in sim.state.buildings.values() if b.neutral]
    assert len(neutral) == 1
    house = neutral[0]
    assert house.owner is None
    assert house.cell == (18, 10)
    assert house.hp == sim.data.neutral["house"].hp

    w, h = sim.data.neutral["house"].footprint
    assert not sim.map.pass_inf[10 : 10 + h, 18 : 18 + w].any()


def test_neutral_footprints_are_not_stamped_again_by_the_sim(monkeypatch):
    """`load_map` already stamped them; re-stamping would inflate `map.version`."""
    game_map = make_map(neutral_buildings=[{"def": "house", "cell": [18, 10]}, {"def": "house", "cell": [24, 20]}])
    assert game_map.version == 0  # load_map stamped the neutral footprints already

    stamped: list[tuple[int, int]] = []
    original = GameMap.stamp_footprint

    def spy(self, cell, size, blocked):
        stamped.append(cell)
        return original(self, cell, size, blocked)

    monkeypatch.setattr(GameMap, "stamp_footprint", spy)
    sim = make_sim(game_map=game_map)

    hq_cells = [sim.state.buildings[p.hq_id].cell for p in sim.state.players.values()]
    assert stamped == hq_cells  # the two HQs, and nothing else
    assert sim.map.version == len(hq_cells)


def test_entity_ids_share_one_counter():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    building = sim.spawn_building(0, "barracks", (10, 20))
    other = spawn(sim, 0, "rifles", (12, 10))
    assert squad.id < building.id < other.id
    assert building.id not in sim.state.squads
    assert squad.id not in sim.state.buildings


def test_spawned_squad_members_match_the_def():
    sim = make_sim()
    squad = spawn(sim, 1, "hmg_team", (20, 15))
    sdef = sim.data.squads["hmg_team"]
    assert squad.owner == 1
    # a team weapon spawns already deployed (SETTING_UP), not mid-march
    assert squad.state is SquadState.SETTING_UP
    assert [m.weapon for m in squad.members] == list(sdef.loadout)
    assert all(m.hp == sdef.member_hp for m in squad.members)
    assert np.allclose(squad.pos, center_of((20, 15)))


def test_spawned_non_team_weapon_squad_starts_idle():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    assert squad.state is SquadState.IDLE


# --------------------------------------------------------------------------
# orders
# --------------------------------------------------------------------------

ALL_ORDERS = [
    Move(squad=3, cell=(4, 5)),
    AttackMove(squad=3, cell=(4, 5)),
    Attack(squad=3, target_id=9),
    Capture(squad=3, point_id="mid"),
    Garrison(squad=3, building_id=7),
    Ungarrison(squad=3),
    Retreat(squad=3),
    Reinforce(squad=3),
    SetFacing(squad=3, direction_deg=90.0),
    Build(squad=3, structure="barracks", cell=(6, 7)),
    Train(building=7, unit="rifles"),
    Research(building=7, upgrade="research_1"),
    BuyUpgrade(squad=3, upgrade="bar"),
    Stop(squad=3),
]


def test_every_order_type_is_covered_by_the_round_trip_list():
    assert {type(o) for o in ALL_ORDERS} == set(orders_mod.ORDER_TYPES.values())


@pytest.mark.parametrize("order", ALL_ORDERS, ids=lambda o: type(o).__name__)
def test_order_dict_round_trip(order):
    as_dict = order_to_dict(order)
    assert as_dict["type"] == type(order).__name__
    assert order_from_dict(as_dict) == order
    # dicts survive a JSON round trip (plain types only)
    import json

    assert order_from_dict(json.loads(json.dumps(as_dict))) == order


def test_order_from_dict_rejects_unknown_type_and_bad_fields():
    with pytest.raises(ValueError, match="unknown order type"):
        order_from_dict({"type": "Nope", "squad": 1})
    with pytest.raises(ValueError, match="Move"):
        order_from_dict({"type": "Move", "squad": 1, "nonsense": 2})
    with pytest.raises(ValueError, match="missing 'type'"):
        order_from_dict({"squad": 1})


def test_valid_order_is_stored_on_the_squad():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    (result,) = sim.issue(0, [Move(squad=squad.id, cell=(12, 12))])
    assert result.ok and result.reason == ""
    assert squad.order == Move(squad=squad.id, cell=(12, 12))
    assert sim.state.players[0].invalid_orders == 0


def test_order_for_another_players_squad_is_rejected_and_counted():
    sim = make_sim()
    squad = spawn(sim, 1, "rifles", (20, 15))
    (result,) = sim.issue(0, [Move(squad=squad.id, cell=(21, 15))])
    assert result.ok is False
    assert "owner" in result.reason
    assert sim.state.players[0].invalid_orders == 1
    assert squad.order is None


def test_order_for_unknown_entity_is_rejected():
    sim = make_sim()
    results = sim.issue(0, [Move(squad=999, cell=(2, 2)), Train(building=999, unit="rifles")])
    assert [r.ok for r in results] == [False, False]
    assert all("no such" in r.reason for r in results)
    assert sim.state.players[0].invalid_orders == 2


def test_order_from_unknown_player_is_rejected_without_raising():
    sim = make_sim()
    (result,) = sim.issue(42, [Retreat(squad=1)])
    assert result.ok is False
    assert "player" in result.reason


def test_out_of_bounds_cell_is_rejected():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    results = sim.issue(
        0,
        [
            Move(squad=squad.id, cell=(-1, 0)),
            Move(squad=squad.id, cell=(sim.map.width, 0)),
            Build(squad=squad.id, structure="barracks", cell=(0, sim.map.height)),
        ],
    )
    assert [r.ok for r in results] == [False, False, False]
    assert all("out of bounds" in r.reason for r in results)


def test_retreating_squad_accepts_no_orders():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    squad.state = SquadState.RETREATING
    (result,) = sim.issue(0, [Move(squad=squad.id, cell=(11, 10))])
    assert result.ok is False
    assert "retreating" in result.reason
    assert sim.state.players[0].invalid_orders == 1


def test_stop_clears_order_path_and_target():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    squad.order = Move(squad=squad.id, cell=(12, 12))
    squad.path = [(11, 11), (12, 12)]
    squad.target_id = 5
    (result,) = sim.issue(0, [Stop(squad=squad.id)])
    assert result.ok
    assert squad.order is None
    assert squad.path == []
    assert squad.target_id is None


def test_building_orders_enqueue():
    sim = make_sim()
    hq = sim.state.buildings[sim.state.players[0].hq_id]
    sim.state.players[0].munitions = 500
    results = sim.issue(0, [Train(building=hq.id, unit="engineers"), Research(building=hq.id, upgrade="research_1")])
    assert all(r.ok for r in results)
    assert [(q.kind, q.item_id) for q in hq.queue] == [("train", "engineers"), ("research", "research_1")]


def test_order_on_another_players_building_is_rejected():
    sim = make_sim()
    enemy_hq = sim.state.buildings[sim.state.players[1].hq_id]
    (result,) = sim.issue(0, [Train(building=enemy_hq.id, unit="rifles")])
    assert result.ok is False
    assert "owner" in result.reason


def test_neutral_building_order_is_rejected():
    sim = make_sim(neutral_buildings=[{"def": "house", "cell": [18, 10]}])
    house = next(b for b in sim.state.buildings.values() if b.neutral)
    (result,) = sim.issue(0, [Train(building=house.id, unit="rifles")])
    assert result.ok is False


def test_every_order_type_has_a_registered_handler():
    for cls in orders_mod.ORDER_TYPES.values():
        assert cls in orders_mod.ORDER_HANDLERS


# --------------------------------------------------------------------------
# tick loop
# --------------------------------------------------------------------------


def test_tick_advances_the_clock_and_runs_systems_in_spec_order():
    sim = make_sim()
    assert [m.__name__.rsplit(".", 1)[1] for m in systems.SYSTEM_ORDER] == [
        "production",
        "movement",
        "vision",
        "combat",
        "suppression",
        "garrison",
        "territory",
        "economy",
        "victory",
    ]
    sim.run(5)
    assert sim.state.tick == 5


def test_tick_runs_each_system_once_per_tick(monkeypatch):
    sim = make_sim()
    calls: list[str] = []
    for module in systems.SYSTEM_ORDER:
        name = module.__name__.rsplit(".", 1)[1]
        monkeypatch.setattr(module, "run", lambda s, name=name: calls.append(name))
    sim.tick()
    assert calls == [m.__name__.rsplit(".", 1)[1] for m in systems.SYSTEM_ORDER]


def test_events_are_cleared_at_the_start_of_each_tick():
    sim = make_sim()
    sim.state.events.append(Event(kind="shot", tick=0, data={}))
    sim.tick()
    assert sim.state.events == []


def test_tick_is_a_no_op_once_a_winner_is_set():
    sim = make_sim()
    sim.run(3)
    sim.state.winner = 0
    before = sim.state_hash()
    sim.run(10)
    assert sim.state.tick == 3
    assert sim.state_hash() == before


def test_a_finished_game_drops_its_stale_events():
    sim = make_sim()
    sim.run(3)
    sim.state.winner = 0
    sim.state.events.append(Event(kind="game_over", tick=3, data={"winner": 0}))

    sim.tick()

    assert sim.state.events == []


def test_orders_after_the_game_is_over_are_refused_without_being_counted():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    sim.run(3)
    sim.state.winner = 1
    before = sim.state_hash()

    results = sim.issue(0, [Move(squad=squad.id, cell=(12, 12)), Retreat(squad=squad.id)])

    assert [r.ok for r in results] == [False, False]
    assert all("game over" in r.reason for r in results)
    assert sim.state.players[0].invalid_orders == 0  # not the agent's fault
    assert sim.state_hash() == before


# --------------------------------------------------------------------------
# hashing / determinism
# --------------------------------------------------------------------------


def _scripted(sim):
    """Issue a fixed order script; used by the determinism tests."""
    squads = sorted(sim.state.squads)
    sim.issue(0, [Move(squad=squads[0], cell=(12, 12)), Capture(squad=squads[0], point_id="west")])
    sim.issue(1, [AttackMove(squad=squads[1], cell=(20, 15))])


def test_same_seed_and_orders_give_the_same_hash_after_80_ticks():
    a, b = make_sim(seed=11), make_sim(seed=11)
    _scripted(a)
    _scripted(b)
    a.run(80)
    b.run(80)
    assert a.state.tick == b.state.tick == 80
    assert a.state_hash() == b.state_hash()


def test_different_seeds_diverge_once_the_rng_is_consumed(monkeypatch):
    from coh.sim.systems import movement

    def jitter(sim) -> None:
        for sid in sorted(sim.state.squads):
            sim.state.squads[sid].pos += sim.state.rng.normal(size=2)

    monkeypatch.setattr(movement, "run", jitter)

    a, b, a2 = make_sim(seed=1), make_sim(seed=2), make_sim(seed=1)
    for sim in (a, b, a2):
        sim.run(80)
    assert a.state_hash() == a2.state_hash()
    assert a.state_hash() != b.state_hash()


def test_hash_ignores_events_and_visibility_grids():
    sim = make_sim()
    before = sim.state_hash()
    sim.state.events.append(Event(kind="death", tick=0, data={"squad": 1}))
    sim.state.visible[0][:] = True
    assert sim.state_hash() == before


def test_hash_reflects_later_task_squad_fields():
    sim = make_sim()
    squad = spawn(sim, 0, "rifles", (10, 10))
    before = sim.state_hash()
    for field_name, value in (
        ("abandoned", True),
        ("turret_heading", 1.25),
        ("last_attacker_pos", (3.0, 4.0)),
        ("moving", True),
    ):
        setattr(squad, field_name, value)
        assert sim.state_hash() != before, field_name
        before = sim.state_hash()


def test_hash_changes_with_state_and_is_a_sha256_hex_digest():
    sim = make_sim()
    before = sim.state_hash()
    assert len(before) == 64 and all(c in "0123456789abcdef" for c in before)
    spawn(sim, 0, "rifles", (10, 10))
    assert sim.state_hash() != before


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    from tests.helpers import make_sim
    from coh.sim.orders import Move
    sim = make_sim(seed=5)
    sim.issue(0, [Move(squad=min(sim.state.squads), cell=(9, 9))])
    sim.run(20)
    print(sim.state_hash())
    """
)


def test_hash_is_stable_across_processes_and_hash_seeds():
    import os
    import pathlib

    root = str(pathlib.Path(__file__).resolve().parents[2])
    script = _SUBPROCESS_SCRIPT.format(root=root)
    digests = set()
    for hash_seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1


# ---------------------------------------------------------------------------
# entity_ids: the invariant the tick loops rely on instead of sorting
# ---------------------------------------------------------------------------


def test_entity_dicts_stay_in_ascending_id_order():
    """`state.squads` / `state.buildings` iterate in ascending id, always.

    `coh.sim.state.entity_ids` is plain insertion-order iteration and the tick
    systems use it in place of `sorted(...)`. That is only equivalent because
    ids are handed out by a monotonic counter and entities are only ever
    appended or popped -- if anything ever re-inserts an entity (or ids stop
    growing), this test fails and `entity_ids` must go back to sorting.
    """
    from coh.sim.state import entity_ids

    sim = make_sim(seed=3)
    spawn(sim, 0, "rifles", (6, 6))
    spawn(sim, 1, "rifles", (7, 6))
    victim = spawn(sim, 0, "hmg_team", (8, 6))
    sim.spawn_building(0, "barracks", (2, 10))

    # Kill one off in the middle, then spawn again on top of the hole.
    for member in victim.members:
        member.hp = 0.0
    combat_mod = systems.combat
    combat_mod._destroy_squad(sim, victim)
    spawn(sim, 1, "rifles", (9, 6))
    sim.spawn_building(1, "barracks", (20, 10))

    assert entity_ids(sim.state.squads) == sorted(sim.state.squads)
    assert entity_ids(sim.state.buildings) == sorted(sim.state.buildings)
