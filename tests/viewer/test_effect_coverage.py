"""Every event kind the sim can emit must have a deliberate rendering decision.

The viewer is allowed to *ignore* a kind (`draw: null` — points pulse
themselves, `game_over` is a banner) and it degrades gracefully on kinds it has
never heard of, but silently falling through to the generic marker because a
new system landed is a bug. This test reads both sides and compares them, so
adding an `Event(kind=...)` to `coh/sim` fails here until the viewer decides
what to do with it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "coh" / "sim"
VIEWER_JS = REPO / "viewer" / "viewer.js"

# `kind="..."` covers both `Event(kind="shot")` and `kind = "squad_destroyed"`;
# `QueueItem(kind=kind, ...)` passes a variable, so build-queue kinds ("train",
# "research") are correctly not picked up here.
_KIND_LITERAL = re.compile(r'kind\s*=\s*"([a-z_]+)"')
_KIND_CONSTANT = re.compile(r'^[A-Z][A-Z_]*_(?:DESTROYED|EVENT)\s*=\s*"([a-z_]+)"', re.M)


def sim_event_kinds() -> set[str]:
    kinds: set[str] = set()
    for path in sorted(SIM.rglob("*.py")):
        text = path.read_text()
        kinds.update(_KIND_LITERAL.findall(text))
        kinds.update(_KIND_CONSTANT.findall(text))
    return kinds


def _js_object_keys(marker: str) -> set[str]:
    text = VIEWER_JS.read_text()
    start = text.index(marker)
    end = text.index("\n};", start)
    return set(re.findall(r"^\s{2}([a-z_]+):", text[start:end], re.M))


def viewer_effect_kinds() -> set[str]:
    return _js_object_keys("var EFFECTS = {")


def viewer_marker_kinds() -> set[str]:
    return _js_object_keys("var MARKER_COLOURS = {")


def test_the_extractors_find_the_kinds_we_know_about():
    """Guard against a regex that silently matches nothing."""
    kinds = sim_event_kinds()
    assert {"shot", "explosion", "vehicle_destroyed", "game_over"} <= kinds
    assert len(kinds) >= 15
    assert "train" not in kinds and "research" not in kinds
    assert len(viewer_effect_kinds()) >= 15


def test_every_sim_event_kind_is_handled_by_the_viewer():
    missing = sim_event_kinds() - viewer_effect_kinds()
    assert not missing, (
        "these event kinds fall through to the generic marker; give each one an "
        "entry in EFFECTS in viewer/viewer.js (use `draw: null` if it is shown "
        "elsewhere): %s" % sorted(missing)
    )


def test_the_viewer_does_not_handle_kinds_the_sim_never_emits():
    stale = viewer_effect_kinds() - sim_event_kinds()
    assert not stale, "EFFECTS handles kinds the sim no longer emits: %s" % sorted(stale)


def test_deaths_and_captures_are_marked_on_the_timeline():
    markers = viewer_marker_kinds()
    assert {"squad_destroyed", "vehicle_destroyed", "building_destroyed"} <= markers
    assert {"point_captured", "point_neutralized", "game_over"} <= markers
    assert markers <= sim_event_kinds()
