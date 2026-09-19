"""What a team fog view is allowed to show — asserted in a real browser.

The canvas app receives omniscient frames and filters them client-side, so
"team 0's view" is only as honest as this file makes it. Each test drives a
hand-built scenario with hand-built visibility bitmaps, so there is exactly one
right answer for every entity, and checks the outcome three ways: the
renderer's own draw counts, the pixels on the canvas, and what a click hits.
"""

from __future__ import annotations

import pytest

from tests.viewer.conftest import (  # noqa: F401  (fixtures are used by name)
    needs_chromium,
    playwright_or_skip,
    shoot,
)

pytestmark = pytest.mark.slow

playwright_or_skip()


# One frame stream, three frames, and visibility we choose ourselves:
#
#   frame t=0: team 0 sees a disc around cell (20, 20) only
#   frames t=2, t=4: it sees a disc around cell (70, 70) instead
#
#   #700 (player 0, team 0) at cell (72, 72) — own, inside the t=4 disc
#   #701 (player 1, team 1) at cell (18, 18) — seen at t=0, lost by t=4 -> ghost
#   #702 (player 1, team 1) at cell (60, 20) — in neither disc -> never seen
#   #703 (neutral)          at cell (62, 26) — out of vision, but map knowledge
#   #800 (player 0)  squad at cell (70, 70) — own, inside the t=4 disc
#   #801 (player 1)  squad at cell (52, 30) — in neither disc -> hidden
FOG_SCENARIO_JS = """(() => {
  var size = window.__viewer.mapSize();
  var W = size.W, H = size.H;
  function packVis(inside) {
    var bytes = new Uint8Array(Math.ceil(W * H / 8));
    for (var cy = 0; cy < H; cy++) {
      for (var cx = 0; cx < W; cx++) {
        if (!inside(cx, cy)) continue;
        var n = cy * W + cx;
        bytes[n >> 3] |= 128 >> (n & 7);
      }
    }
    var out = '';
    for (var i = 0; i < bytes.length; i++) out += String.fromCharCode(bytes[i]);
    return btoa(out);
  }
  function disc(ox, oy, r) {
    return function (cx, cy) { return (cx - ox) * (cx - ox) + (cy - oy) * (cy - oy) <= r * r; };
  }
  var visA = packVis(disc(20, 20, 8));
  var visB = packVis(disc(70, 70, 8));

  var base = window.__viewer.data().frames[0];
  var sq0 = base.squads[0] || {th: 0, mhp: [60], w: [''], sup: 0, st: 'idle'};
  function sq(o) { var x = JSON.parse(JSON.stringify(sq0)); for (var k in o) x[k] = o[k]; return x; }
  var bl0 = base.buildings[0];
  function bl(o) { var x = JSON.parse(JSON.stringify(bl0)); for (var k in o) x[k] = o[k]; return x; }

  var squads = [
    sq({id: 800, o: 0, def: 'rifles', kind: 'infantry', x: 141, y: 141, h: 0, n: 4, max: 4, hp: 1}),
    sq({id: 801, o: 1, def: 'pioneers', kind: 'infantry', x: 105, y: 61, h: 0, n: 4, max: 4, hp: 1})
  ];
  var buildings = [
    bl({id: 700, o: 0, def: 'barracks', cx: 72, cy: 72, w: 3, h: 3, hp: 1, prog: 1, n: 0, q: []}),
    bl({id: 701, o: 1, def: 'barracks', cx: 18, cy: 18, w: 3, h: 3, hp: 1, prog: 1, n: 0, q: []}),
    bl({id: 702, o: 1, def: 'tank_depot', cx: 60, cy: 20, w: 3, h: 3, hp: 1, prog: 1, n: 0, q: []}),
    bl({id: 703, o: null, def: 'barn', cx: 62, cy: 26, w: 5, h: 4, hp: 1, prog: 1, n: 2, nu: 1, q: []})
  ];
  if (window.__fogDropUnseen) {
    buildings = buildings.filter(function (b) { return b.id !== 702; });
  }

  function frame(t, vis, events) {
    return {
      t: t, players: base.players, tickets: base.tickets, points: base.points,
      squads: JSON.parse(JSON.stringify(squads)),
      buildings: JSON.parse(JSON.stringify(buildings)),
      events: events || [], vis: {"0": vis, "1": vis}, terrain_delta: []
    };
  }
  var events = [
    // out of vision: an enemy blast, and an exchange between two enemies
    {k: 'explosion', t: 4, d: {src: 801, pos: [105, 61], radius: 6}},
    {k: 'shot', t: 4, d: {src: 801, dst: 802, hit: true, src_pos: [105, 61], dst_pos: [110, 66]}},
    // in vision at one end: our own squad firing out into the dark
    {k: 'shot', t: 4, d: {src: 800, dst: 801, hit: true, src_pos: [141, 141], dst_pos: [105, 61]}}
  ];

  var data = JSON.parse(JSON.stringify(window.__viewer.data()));
  data.frames = [frame(0, visA), frame(2, visB), frame(4, visB, events)];
  data.winner = null;
  window.__viewer.inject(data);
  window.__viewer.camera(96, 96, 4.2);
  return true;
})()"""

# Somewhere inside the never-seen building's footprint, in world metres.
UNSEEN_BUILDING = (123.0, 43.0)
ENEMY_SQUAD = (105.0, 61.0)
OWN_BUILDING = (146.0, 146.0)
NEUTRAL_BUILDING = (128.0, 56.0)


def load_scenario(page, drop_unseen=False):
    page.evaluate("window.__fogDropUnseen = %s" % ("true" if drop_unseen else "false"))
    page.evaluate(FOG_SCENARIO_JS)
    page.evaluate("window.__viewer.seekTick(4)")
    page.evaluate("window.__viewer.redraw()")


def set_fog(page, mode):
    """Cycle `F` until the requested fog mode is reached (0/1/2), then redraw."""
    for _ in range(4):
        if page.evaluate("window.__viewer.state().fogMode") == mode:
            break
        page.keyboard.press("f")
    assert page.evaluate("window.__viewer.state().fogMode") == mode
    page.evaluate("window.__viewer.redraw()")


def pick_world(page, world):
    screen = page.evaluate("window.__viewer.screenOf(%f, %f)" % world)
    return page.evaluate("window.__viewer.pickAt(%f, %f)" % (screen[0], screen[1]))


@needs_chromium
def test_omniscient_mode_draws_everything(page):
    load_scenario(page)
    set_fog(page, 0)
    stats = page.evaluate("window.__viewer.stats()")
    assert stats["buildings"] == 4 and stats["ghosts"] == 0 and stats["hiddenBuildings"] == 0
    assert stats["squads"] == 2 and stats["hiddenSquads"] == 0
    assert stats["effects"] == 3 and stats["hiddenEffects"] == 0
    assert page.errors == []


@needs_chromium
def test_team_view_hides_enemy_buildings_squads_and_effects(page, tmp_path):
    load_scenario(page)
    set_fog(page, 1)
    shoot(page, tmp_path / "fog-team0.png")

    stats = page.evaluate("window.__viewer.stats()")
    # own barracks + the neutral barn are live; the enemy barracks is a
    # remembered ghost; the enemy depot was never seen, so it is not drawn
    assert stats["buildings"] == 2, stats
    assert stats["ghosts"] == 1, stats
    assert stats["hiddenBuildings"] == 1, stats
    # the enemy infantry is out of vision
    assert stats["squads"] == 1 and stats["hiddenSquads"] == 1, stats
    # the enemy blast and the enemy-to-enemy shot are dropped; our own shot,
    # which starts on a squad we can see, is kept
    assert stats["effects"] == 1 and stats["hiddenEffects"] == 2, stats

    assert page.evaluate("window.__viewer.buildingFog(700)") == "live"
    assert page.evaluate("window.__viewer.buildingFog(701)") == "ghost"
    assert page.evaluate("window.__viewer.buildingFog(702)") == "hidden"
    assert page.evaluate("window.__viewer.buildingFog(703)") == "live"
    assert page.errors == []


@needs_chromium
def test_a_building_seen_earlier_becomes_a_ghost_when_vision_moves_away(page):
    load_scenario(page)
    set_fog(page, 1)

    page.evaluate("window.__viewer.seekTick(0)")   # inside the first disc
    page.evaluate("window.__viewer.redraw()")
    assert page.evaluate("window.__viewer.buildingFog(701)") == "live"

    page.evaluate("window.__viewer.seekTick(4)")   # vision has moved away
    page.evaluate("window.__viewer.redraw()")
    assert page.evaluate("window.__viewer.buildingFog(701)") == "ghost"

    # seeking back must not leave stale memory behind
    page.evaluate("window.__viewer.seekTick(0)")
    page.evaluate("window.__viewer.redraw()")
    assert page.evaluate("window.__viewer.buildingFog(701)") == "live"
    assert page.errors == []


@needs_chromium
def test_a_ghost_inspector_shows_last_seen_and_no_live_state(page):
    load_scenario(page)
    set_fog(page, 1)
    page.evaluate("window.__viewer.seekTick(4)")
    page.evaluate("window.__viewer.select('building', 701)")
    page.evaluate("window.__viewer.redraw()")

    body = page.locator("#panel-body").inner_text().lower()
    assert "last seen" in body, body
    assert "health" not in body, body
    assert "construction" not in body, body
    assert "queue" not in body and "garrison" not in body, body
    assert "last known" in page.locator("#panel-title").inner_text().lower()
    assert page.errors == []


@needs_chromium
def test_a_never_seen_enemy_building_leaves_no_pixels(page):
    load_scenario(page)
    set_fog(page, 1)
    with_hidden = page.evaluate("window.__viewer.sampleWorld(%f, %f, 40)" % UNSEEN_BUILDING)

    # the identical scene with that building simply absent must look the same
    load_scenario(page, drop_unseen=True)
    set_fog(page, 1)
    without = page.evaluate("window.__viewer.sampleWorld(%f, %f, 40)" % UNSEEN_BUILDING)
    assert with_hidden == without, "a never-seen enemy building left pixels on the canvas"

    # and with the fog off it plainly does show up
    load_scenario(page)
    set_fog(page, 0)
    assert page.evaluate("window.__viewer.sampleWorld(%f, %f, 40)" % UNSEEN_BUILDING) != with_hidden
    assert page.errors == []


@needs_chromium
def test_hidden_entities_are_not_clickable(page):
    load_scenario(page)
    set_fog(page, 0)
    assert pick_world(page, ENEMY_SQUAD) == {"kind": "squad", "id": 801}
    assert pick_world(page, UNSEEN_BUILDING) == {"kind": "building", "id": 702}

    set_fog(page, 1)
    assert pick_world(page, ENEMY_SQUAD) is None, "a hidden enemy squad must not be selectable"
    assert pick_world(page, UNSEEN_BUILDING) is None, "a never-seen building must not be selectable"
    assert page.errors == []


@needs_chromium
def test_own_and_neutral_entities_stay_visible_and_clickable(page):
    load_scenario(page)
    set_fog(page, 1)

    assert pick_world(page, OWN_BUILDING) == {"kind": "building", "id": 700}
    # a neutral building is map knowledge: still drawn and still clickable
    assert pick_world(page, NEUTRAL_BUILDING) == {"kind": "building", "id": 703}
    assert page.errors == []


@needs_chromium
def test_a_neutral_buildings_garrison_is_only_shown_while_visible(page):
    """The barn holds two squads; out of vision you should not know that."""
    load_scenario(page)
    set_fog(page, 0)
    assert page.evaluate("window.__viewer.stats()")["badges"] == 1, "the barn's garrison badge"

    set_fog(page, 1)
    # the barn itself is still drawn — you know the building is there — but its
    # occupants are not something team 0 can see from across the map
    assert page.evaluate("window.__viewer.buildingFog(703)") == "live"
    assert page.evaluate("window.__viewer.stats()")["badges"] == 0
    assert page.errors == []


@needs_chromium
def test_a_real_match_hides_something_in_team_view(page):
    """The scenarios above are rigged; check the real frame stream too."""
    page.evaluate("window.__viewer.seek(0.5)")
    page.evaluate("window.__viewer.redraw()")
    omniscient = page.evaluate("window.__viewer.stats()")
    assert omniscient["hiddenBuildings"] == 0
    assert omniscient["hiddenSquads"] == 0
    assert omniscient["hiddenEffects"] == 0

    set_fog(page, 1)
    team0 = page.evaluate("window.__viewer.stats()")
    hidden = team0["hiddenBuildings"] + team0["hiddenSquads"] + team0["hiddenEffects"]
    assert hidden > 0, "mid-match team 0 should not see the whole map: %s" % team0
    assert team0["buildings"] < omniscient["buildings"], team0
    assert page.errors == []
