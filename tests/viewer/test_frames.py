"""`coh.viewer.frames` turns a replay into the JSON the canvas app consumes."""

from __future__ import annotations

import base64
import gzip
import json
from dataclasses import replace

import numpy as np
import pytest

from coh.data.schema import Cost
from coh.env import CohEnv
from coh.maps.format import map_from_ascii
from coh.sim.constants import TICKS_PER_SECOND
from coh.sim.orders import Move, Train
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints
from coh.viewer.frames import build_frames, frames_json_gz
from tests.helpers import DEFAULT_POINTS, DEFAULT_STARTS, default_sectors, fixture_data
from tests.viewer.conftest import play

EVERY = 2
FRAME_KEYS = {"t", "players", "tickets", "points", "squads", "buildings", "events", "vis", "terrain_delta"}


@pytest.fixture(scope="module")
def frames(short_replay):
    return build_frames(short_replay, EVERY, data=fixture_data())


# --- structure --------------------------------------------------------------


def test_frame_count_covers_the_whole_replay(short_replay, frames):
    # tick 0 plus one frame every EVERY ticks up to the final tick.
    assert short_replay.final_tick == int(30.0 * TICKS_PER_SECOND)
    assert len(frames["frames"]) == short_replay.final_tick // EVERY + 1
    assert frames["frames"][0]["t"] == 0
    assert frames["frames"][-1]["t"] == short_replay.final_tick
    ticks = [f["t"] for f in frames["frames"]]
    assert ticks == sorted(ticks)


def test_every_frame_has_the_required_keys(frames):
    for frame in frames["frames"]:
        assert FRAME_KEYS <= set(frame)


def test_static_sections_describe_the_map_and_players(frames):
    assert frames["map"]["width"] == 96 and frames["map"]["height"] == 96
    assert len(frames["map"]["terrain"]) == 96
    assert all(len(row) == 96 for row in frames["map"]["terrain"])
    assert len(frames["map"]["sectors"]) == 96
    assert frames["map"]["points"], "the client needs the point definitions"
    assert [p["team"] for p in frames["players"]] == [0, 1]


def test_meta_carries_what_the_client_must_not_hardcode(frames):
    meta = frames["meta"]
    assert meta["ticks_per_second"] == TICKS_PER_SECOND
    assert meta["every_ticks"] == EVERY
    # formation offsets: member 0 on the squad position, the rest ringing it
    assert meta["formation_offsets"][0] == [0.0, 0.0]
    assert len(meta["formation_offsets"]) == 8
    assert meta["squad_defs"]["hmg_team"]["kind"] == "team_weapon"
    assert meta["squad_defs"]["hmg_team"]["arc_deg"] < 360
    assert meta["building_defs"]["hq_us"]["footprint"] == [4, 4]


def test_squad_and_building_entries_are_renderable(frames):
    squads = [s for frame in frames["frames"] for s in frame["squads"]]
    assert squads
    for squad in squads:
        assert {"id", "o", "def", "kind", "x", "y", "h", "th", "n", "max", "hp", "sup", "st"} <= set(squad)
        assert squad["kind"] in ("infantry", "team_weapon", "vehicle")
        assert 0 <= squad["n"] <= squad["max"]
        assert 0.0 <= squad["hp"] <= 1.0
    buildings = [b for frame in frames["frames"] for b in frame["buildings"]]
    assert buildings
    for building in buildings:
        assert {"id", "o", "def", "cx", "cy", "w", "h", "hp", "prog", "n"} <= set(building)


def test_floats_are_rounded_to_two_places(frames):
    for frame in frames["frames"]:
        for squad in frame["squads"]:
            assert squad["x"] == round(squad["x"], 2)
            assert squad["h"] == round(squad["h"], 2)


# --- visibility -------------------------------------------------------------


def test_vis_decodes_to_the_map_shape(frames):
    width, height = frames["map"]["width"], frames["map"]["height"]
    first = frames["frames"][0]["vis"]
    assert set(first) == {"0", "1"}, "both teams' visibility is present in the first frame"
    for team, packed in first.items():
        grid = np.unpackbits(np.frombuffer(base64.b64decode(packed), dtype=np.uint8))
        assert grid.size == height * width or grid.size == 8 * ((height * width + 7) // 8)
        assert grid[: height * width].reshape(height, width).shape == (height, width)


def test_vis_is_only_resent_when_it_changes(frames):
    """The client keeps the last grid it saw, so unchanged frames omit it."""
    omitted = sum(1 for frame in frames["frames"][1:] if not frame["vis"])
    assert omitted > 0, "a static tick should not repeat the visibility grid"
    assert any(frame["vis"] for frame in frames["frames"][1:]), "vision does change during a match"


# --- events -----------------------------------------------------------------


def test_events_are_carried_and_merged_between_frames(frames):
    events = [e for frame in frames["frames"] for e in frame["events"]]
    assert events, "a 30 s match emits events"
    for frame, previous in zip(frames["frames"][1:], frames["frames"]):
        for event in frame["events"]:
            assert {"k", "t", "d"} <= set(event)
            # every event lands in the frame covering the ticks since the last one
            assert previous["t"] < event["t"] <= frame["t"]


@pytest.mark.slow
def test_a_longer_match_carries_the_combat_and_capture_events():
    """30 s is barely a skirmish; the kinds the viewer animates need longer."""
    frames = build_frames(play(180.0), EVERY, data=fixture_data())
    kinds = {e["k"] for frame in frames["frames"] for e in frame["events"]}
    assert {"shot", "point_captured", "unit_trained"} <= kinds
    shots = [e for frame in frames["frames"] for e in frame["events"] if e["k"] == "shot"]
    for shot in shots:
        assert len(shot["d"]["src_pos"]) == 2 and len(shot["d"]["dst_pos"]) == 2
        assert isinstance(shot["d"]["hit"], bool)


def test_no_event_tick_is_dropped(short_replay, frames):
    """Every tick of the match belongs to exactly one frame's event window."""
    covered = set()
    previous = 0
    for frame in frames["frames"][1:]:
        covered.update(range(previous + 1, frame["t"] + 1))
        previous = frame["t"]
    assert covered == set(range(1, short_replay.final_tick + 1))


# --- serialization ----------------------------------------------------------


def test_the_stream_is_json_serializable_and_compresses(frames):
    blob = json.dumps(frames, allow_nan=False)
    assert json.loads(blob)["map"]["name"] == "hedgerow_crossing"
    payload = frames_json_gz(frames)
    assert gzip.decompress(payload) == json.dumps(frames, separators=(",", ":"), allow_nan=False).encode()
    assert len(payload) < len(blob)


def test_every_ticks_must_be_positive(short_replay):
    with pytest.raises(ValueError, match="every_ticks"):
        build_frames(short_replay, 0, data=fixture_data())


# --- terrain deltas ---------------------------------------------------------


def test_terrain_delta_is_an_empty_list_when_nothing_is_crushed(frames):
    assert all(frame["terrain_delta"] == [] for frame in frames["frames"])


def _crushable_data():
    """Fixture tables tweaked so an HQ can hand out a tank immediately."""
    data = fixture_data()
    return replace(
        data,
        economy=replace(
            data.economy,
            start_resources=Cost(2000, 2000, 2000),
            base_population=200,
            max_population=200,
        ),
        buildings={**data.buildings, "hq_us": replace(data.buildings["hq_us"], produces=("engineers", "tank"))},
        squads={
            **data.squads,
            "tank": replace(data.squads["tank"], build_time=1.0, cost=Cost(0, 0, 0), population=1),
        },
    )


@pytest.mark.slow
def test_terrain_delta_appears_when_a_vehicle_crushes_a_fence():
    data = _crushable_data()
    width, height = 40, 30
    rows = ["." * 20 + "f" + "." * (width - 21)] * height
    game_map = map_from_ascii(
        terrain_rows=rows,
        sectors_rows=default_sectors(width, height),
        points=DEFAULT_POINTS,
        starts=DEFAULT_STARTS,
        name="fence",
        footprints=neutral_footprints(data),
    )
    env = CohEnv(
        map_name="fence",
        players=[PlayerSetup("us", 0, 0), PlayerSetup("us", 1, 1)],
        seed=0,
        decision_interval_s=0.5,
        config=SimConfig(time_limit_s=120.0),
        data=data,
        game_map=game_map,
    )
    env.reset()
    sim = env.sim
    hq = next(b for b in sim.state.buildings.values() if b.def_id == "hq_us" and b.owner == 0)

    env.step({0: [Train(building=hq.id, unit="tank")], 1: []})
    tank = None
    for _ in range(60):
        env.step({0: [], 1: []})
        vehicles = [s for s in sim.state.squads.values() if s.def_id == "tank" and s.owner == 0]
        if vehicles:
            tank = vehicles[0]
            break
    assert tank is not None, "the tank was never delivered"

    env.step({0: [Move(squad=tank.id, cell=(30, 14))], 1: []})
    for _ in range(200):
        env.step({0: [], 1: []})
        if sim.state.terrain_changes:
            break
    assert sim.state.terrain_changes, "the tank never reached the fence"
    env.step({0: [], 1: []})

    frames = build_frames(env.to_replay(), EVERY, data=data, game_map=game_map)
    deltas = [delta for frame in frames["frames"] for delta in frame["terrain_delta"]]
    assert deltas == [[20, 14, "."]]
    # and the delta is reported once, on the frame that covers the crush
    carrying = [f for f in frames["frames"] if f["terrain_delta"]]
    assert len(carrying) == 1
