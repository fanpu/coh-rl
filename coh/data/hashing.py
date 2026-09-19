"""A stable content hash over a loaded `GameData`.

A replay is only reproducible against the stat tables it was recorded with,
so it records this hash and refuses to re-simulate against anything else.
The hash is taken over a canonical JSON dump of the dataclass tree: keys
sorted, tuples rendered as lists, so it depends on the values and not on YAML
key order, file layout or dict insertion order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from coh.data.schema import GameData


def canonical_data(data: GameData) -> str:
    """The canonical JSON string `data_hash` digests (useful in diffs)."""
    return json.dumps(_plain(asdict(data)), sort_keys=True, separators=(",", ":"))


def data_hash(data: GameData) -> str:
    """sha256 over `canonical_data(data)`."""
    return hashlib.sha256(canonical_data(data).encode()).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
