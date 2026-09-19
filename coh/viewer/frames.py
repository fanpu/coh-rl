"""Re-simulate a replay into a compact frame stream for the canvas viewer.

`build_frames` walks `coh.replay.resimulate` and snapshots the sim every
`every_ticks` ticks. The result is plain JSON-serializable data: the browser
never sees a `Sim`, only this.

Shape (keys are short because there is one frame every quarter second of a
ten-minute match)::

    {"meta":    {...},                      # constants the client needs
     "map":     {"name", "width", "height", "cell_m",
                 "terrain": ["...."], "sectors": [[int]], "points": [...]},
     "players": [{"id", "team", "faction"}],
     "frames":  [{"t", "players", "tickets", "points", "squads", "buildings",
                  "events", "vis", "terrain_delta"}],
     "winner":  int | None}

Three things are sent differentially, because they dominate the payload:

- `vis` (per team, `numpy.packbits` of the visibility grid, base64) is only
  present for a team whose grid changed since the previous frame — the client
  keeps the last one it saw;
- `terrain_delta` carries only the terrain edits made since the previous frame
  (the client applies them to the base grid);
- `events` merges every event emitted since the previous frame, so none are
  lost between snapshots.

Floats are rounded to 2 decimal places: the viewer interpolates positions
between frames anyway, and centimetre precision is well below one pixel.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from coh.data.loader import load_game_data
from coh.data.schema import GameData
from coh.maps.format import GameMap, load_map
from coh.replay.replay import Replay, resimulate
from coh.sim.constants import CELL_M, FORMATION_OFFSETS, TICKS_PER_SECOND
from coh.sim.sim import Sim, neutral_footprints
from coh.sim.state import Building, Squad
from coh.sim.systems.economy import pop_cap, pop_used

FRAMES_VERSION = 1
_DP = 2


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _r(x: Any) -> float:
    """Round to `_DP` places, normalizing -0.0 and non-finite values to 0.0."""
    value = float(x)
    if not math.isfinite(value):
        return 0.0
    return round(value, _DP) + 0.0


def _round_any(value: Any) -> Any:
    """Recursively round floats inside arbitrary (JSON-ish) event payloads."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _r(value)
    if isinstance(value, (np.floating, np.integer)):
        return _round_any(value.item())
    if isinstance(value, np.ndarray):
        return [_round_any(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _round_any(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_any(v) for v in value]
    return value


def _pack_visible(grid: np.ndarray) -> str:
    """Bit-pack a bool visibility grid (row-major) into base64 ASCII."""
    return base64.b64encode(np.packbits(np.ascontiguousarray(grid, dtype=bool)).tobytes()).decode()


# ---------------------------------------------------------------------------
# static sections
# ---------------------------------------------------------------------------


def _squad_meta(data: GameData) -> dict[str, dict[str, Any]]:
    """Per-squad-def display facts: kind, size, arc, speed, sight."""
    out: dict[str, dict[str, Any]] = {}
    for def_id in sorted(data.squads):
        sdef = data.squads[def_id]
        weapons = [w for w in sdef.loadout if w]
        arc = 360.0
        setup_time = 0.0
        max_range = 0.0
        for weapon_id in weapons:
            weapon = data.weapons.get(weapon_id)
            if weapon is None:
                continue
            arc = min(arc, float(weapon.arc_deg))
            setup_time = max(setup_time, float(weapon.setup_time))
            max_range = max(max_range, float(weapon.ranges[2]))
        out[def_id] = {
            "kind": sdef.kind,
            "members": sdef.members,
            "member_hp": _r(sdef.member_hp),
            "arc_deg": _r(arc),
            "setup_time": _r(setup_time),
            "range": _r(max_range),
            "build_time": _r(sdef.build_time),
            "speed": _r(sdef.speed),
            "sight": _r(sdef.sight),
            "faction": sdef.faction,
        }
    return out


def _building_meta(data: GameData) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for def_id in sorted(data.buildings):
        bdef = data.buildings[def_id]
        out[def_id] = {
            "footprint": list(bdef.footprint),
            "hp": _r(bdef.hp),
            "build_time": _r(bdef.build_time),
            "is_hq": bdef.is_hq,
            "faction": bdef.faction,
            "neutral": False,
        }
    for def_id in sorted(data.neutral):
        ndef = data.neutral[def_id]
        out[def_id] = {
            "footprint": list(ndef.footprint),
            "hp": _r(ndef.hp),
            "is_hq": False,
            "faction": "",
            "neutral": True,
            "capacity": ndef.capacity,
        }
    return out


def _map_section(game_map: GameMap) -> dict[str, Any]:
    points = []
    for point_id in sorted(game_map.points):
        point = game_map.points[point_id]
        points.append(
            {
                "id": point.id,
                "name": point.name,
                "type": point.type,
                "cell": list(point.cell),
                "sector": point.sector,
            }
        )
    return {
        "name": game_map.name,
        "width": game_map.width,
        "height": game_map.height,
        "cell_m": CELL_M,
        "terrain": ["".join(row) for row in game_map.terrain.tolist()],
        "sectors": game_map.sector_id.astype(int).tolist(),
        "points": points,
    }


# ---------------------------------------------------------------------------
# per-frame sections
# ---------------------------------------------------------------------------


def _squad_frame(sim: Sim, squad: Squad) -> dict[str, Any]:
    sdef = sim.data.squads.get(squad.def_id)
    max_members = sdef.members if sdef is not None else len(squad.members)
    member_hp = [_r(m.hp) for m in squad.members]
    max_hp = (sdef.member_hp * max_members) if sdef is not None else max(sum(member_hp), 1.0)
    out: dict[str, Any] = {
        "id": squad.id,
        "o": squad.owner,
        "def": squad.def_id,
        "kind": sdef.kind if sdef is not None else "infantry",
        "x": _r(squad.pos[0]),
        "y": _r(squad.pos[1]),
        "h": _r(squad.heading),
        "th": _r(squad.turret_heading),
        "n": len(squad.alive_members),
        "max": max_members,
        "hp": _r(sum(member_hp) / max_hp if max_hp > 0 else 0.0),
        "mhp": member_hp,
        "w": [m.weapon for m in squad.members],
        "sup": 2 if squad.pinned else (1 if squad.suppressed else 0),
        "st": squad.state.value,
    }
    if squad.garrison_in is not None:
        out["g"] = squad.garrison_in
    if squad.facing is not None:
        out["fa"] = _r(squad.facing)
    if squad.abandoned:
        out["ab"] = 1
    if squad.reinforcing:
        out["re"] = 1
    if squad.target_id is not None:
        out["tg"] = squad.target_id
    if squad.order is not None:
        out["ord"] = type(squad.order).__name__
    if squad.state.name == "SETTING_UP" and squad.setup_done_tick > sim.state.tick:
        out["setup"] = squad.setup_done_tick - sim.state.tick
    return out


def _building_frame(sim: Sim, building: Building) -> dict[str, Any]:
    if building.neutral:
        bdef = sim.data.neutral.get(building.def_id)
    else:
        bdef = sim.data.buildings.get(building.def_id)
    footprint = tuple(bdef.footprint) if bdef is not None else (2, 2)
    max_hp = float(bdef.hp) if bdef is not None else max(building.hp, 1.0)
    out: dict[str, Any] = {
        "id": building.id,
        "o": building.owner,
        "def": building.def_id,
        "cx": building.cell[0],
        "cy": building.cell[1],
        "w": footprint[0],
        "h": footprint[1],
        "hp": _r(building.hp / max_hp if max_hp > 0 else 0.0),
        "prog": _r(building.progress),
        "n": len(building.garrison),
    }
    if building.neutral:
        out["nu"] = 1
    if building.queue:
        out["q"] = [[q.kind, q.item_id, _r(q.remaining_s)] for q in building.queue]
    return out


def _players_frame(sim: Sim) -> list[dict[str, Any]]:
    out = []
    for player_id in sorted(sim.state.players):
        player = sim.state.players[player_id]
        out.append(
            {
                "mp": _r(player.manpower),
                "mu": _r(player.munitions),
                "fu": _r(player.fuel),
                "pop": pop_used(sim, player_id),
                "cap": pop_cap(sim, player_id),
            }
        )
    return out


def _points_frame(sim: Sim) -> list[dict[str, Any]]:
    out = []
    for point_id in sorted(sim.state.points):
        point = sim.state.points[point_id]
        out.append(
            {
                "id": point_id,
                "owner": point.owner_team,
                "progress": _r(point.progress),
                "c": point.capturing_team,
                "op": point.op_building is not None,
            }
        )
    return out


def _snapshot(sim: Sim, events: list[dict], terrain_delta: list, last_vis: dict[int, str]) -> dict[str, Any]:
    vis: dict[str, str] = {}
    for team in sorted(sim.state.visible):
        packed = _pack_visible(sim.state.visible[team])
        if last_vis.get(team) != packed:
            last_vis[team] = packed
            vis[str(team)] = packed

    return {
        "t": sim.state.tick,
        "players": _players_frame(sim),
        "tickets": [_r(sim.state.tickets[team]) for team in sorted(sim.state.tickets)],
        "points": _points_frame(sim),
        "squads": [_squad_frame(sim, sim.state.squads[sid]) for sid in sorted(sim.state.squads)],
        "buildings": [_building_frame(sim, sim.state.buildings[bid]) for bid in sorted(sim.state.buildings)],
        "events": events,
        "vis": vis,
        "terrain_delta": terrain_delta,
    }


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def resolve_data(replay: Replay, data: GameData | None = None, data_dir: str | Path | None = None) -> GameData:
    """Pick the stat tables for a replay: explicit `data` > `data_dir` > replay hint."""
    if data is not None:
        return data
    if data_dir is not None:
        return load_game_data(Path(data_dir))
    if replay.data_dir:
        return load_game_data(Path(replay.data_dir))
    return load_game_data()


def build_frames(
    replay: Replay,
    every_ticks: int = 2,
    *,
    data: GameData | None = None,
    data_dir: str | Path | None = None,
    game_map: GameMap | None = None,
) -> dict[str, Any]:
    """Re-simulate `replay` and return its frame stream (see module docstring).

    A frame is emitted for the opening state (tick 0), for every tick that is
    a multiple of `every_ticks`, and for the final tick of the match.
    """
    if every_ticks < 1:
        raise ValueError(f"every_ticks must be >= 1, got {every_ticks}")

    data = resolve_data(replay, data, data_dir)
    if game_map is None:
        game_map = load_map(replay.map_name, footprints=neutral_footprints(data))

    # The opening state, before any tick has run. `Sim` copies the map, so
    # building it here does not disturb the sim `resimulate` builds next.
    opening = Sim(
        game_map=game_map,
        players=replay.players,
        data=data,
        seed=replay.seed,
        config=replay.config,
    )

    out: dict[str, Any] = {
        "version": FRAMES_VERSION,
        "meta": {
            "ticks_per_second": TICKS_PER_SECOND,
            "every_ticks": every_ticks,
            "cell_m": CELL_M,
            "formation_offsets": [[_r(ox), _r(oy)] for ox, oy in FORMATION_OFFSETS],
            "squad_defs": _squad_meta(data),
            "building_defs": _building_meta(data),
            "research_times": {
                **{k: _r(v.time) for k, v in sorted(data.upgrades.items())},
                **{k: _r(v.time) for k, v in sorted(data.squad_upgrades.items())},
            },
            "seed": replay.seed,
            "final_tick": replay.final_tick,
        },
        "map": _map_section(game_map),
        "players": [
            {"id": i, "team": p.team, "faction": p.faction} for i, p in enumerate(replay.players)
        ],
        "frames": [],
        "winner": None,
    }

    last_vis: dict[int, str] = {}
    out["frames"].append(_snapshot(opening, [], [], last_vis))

    pending_events: list[dict] = []
    terrain_seen = 0
    sim: Sim | None = None
    for sim in resimulate(replay, data=data, game_map=game_map):
        for event in sim.state.events:
            # Sim systems stamp `Event.tick` with `sim.state.tick` *before* `Sim.tick`
            # advances the counter, so an event's raw tick is the state time at which
            # it was *decided*, one tick before its effects are visible. Frames are
            # keyed by post-increment tick ("t" is the state time the frame snapshots),
            # so we shift the serialized event time by +1 to match: this guarantees
            # every event's "t" falls inside (previous_frame.t, frame.t] for the frame
            # that carries it, which is the contract the viewer client relies on.
            pending_events.append({"k": event.kind, "t": event.tick + 1, "d": _round_any(event.data)})
        if sim.state.tick % every_ticks == 0:
            delta = [[cx, cy, ch] for cx, cy, ch in sim.state.terrain_changes[terrain_seen:]]
            terrain_seen = len(sim.state.terrain_changes)
            out["frames"].append(_snapshot(sim, pending_events, delta, last_vis))
            pending_events = []

    if sim is not None:
        if pending_events or out["frames"][-1]["t"] != sim.state.tick:
            delta = [[cx, cy, ch] for cx, cy, ch in sim.state.terrain_changes[terrain_seen:]]
            out["frames"].append(_snapshot(sim, pending_events, delta, last_vis))
        out["winner"] = sim.state.winner

    return out


def frames_json_gz(frames: dict[str, Any], compresslevel: int = 6) -> bytes:
    """Serialize a frame stream to gzip-compressed UTF-8 JSON."""
    blob = json.dumps(frames, separators=(",", ":"), allow_nan=False).encode()
    return gzip.compress(blob, compresslevel=compresslevel)
