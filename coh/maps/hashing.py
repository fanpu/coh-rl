"""A stable content hash over a `GameMap`'s authored definition.

Covers exactly what decides a match's trajectory and is authored in the map
file: the name, the terrain grid, the sector grid, the points, the player
starts and the neutral-building placements. Derived layers (`pass_inf`,
`area_cover`, ...) follow from the terrain grid, and `version` counts in-match
mutation, so neither is hashed -- which is what makes the hash of a map the
sim has been chewing on the same as the hash of the map as loaded.
"""

from __future__ import annotations

import hashlib
import json

from coh.maps.format import GameMap


def canonical_map(game_map: GameMap) -> str:
    """The canonical JSON string `map_hash` digests."""
    body = {
        "name": game_map.name,
        "width": game_map.width,
        "height": game_map.height,
        "terrain": ["".join(str(ch) for ch in row) for row in game_map.terrain],
        "sectors": [[int(sid) for sid in row] for row in game_map.sector_id],
        "points": [
            {
                "id": point.id,
                "name": point.name,
                "type": point.type,
                "cell": list(point.cell),
                "sector": point.sector,
            }
            for _, point in sorted(game_map.points.items())
        ],
        "starts": [
            {"slot": start.slot, "team": start.team, "hq_cell": list(start.hq_cell), "sector": start.sector}
            for start in sorted(game_map.starts, key=lambda s: s.slot)
        ],
        "neutral_buildings": sorted(
            [placement.def_id, list(placement.cell)] for placement in game_map.neutral_buildings
        ),
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def map_hash(game_map: GameMap) -> str:
    """sha256 over `canonical_map(game_map)`."""
    return hashlib.sha256(canonical_map(game_map).encode()).hexdigest()
