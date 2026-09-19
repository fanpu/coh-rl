"""Drive the canvas app in a real headless Chromium.

Skipped cleanly when Playwright or a Chromium/Chrome binary is missing (this
repo deliberately does not run `playwright install`; it points Playwright at
whatever browser the machine already has — see `tests/viewer/chromium.py`).

The harness lives in `conftest.py`, shared with `test_viewer_fog.py`:
everything is on a leash there — the page opens with `#paused` so it never
races an animation loop, every Playwright call has an explicit timeout, and an
autouse `hard_deadline` fixture fails a test that blows its budget rather than
letting it stall the suite (`evaluate()` has no timeout of its own).

Screenshots land in the test's `tmp_path` so a failure can be eyeballed; the
unit-showcase test also refreshes `docs/img/viewer-units.png`.
"""

from __future__ import annotations

import pytest

from tests.viewer.conftest import (  # noqa: F401  (fixtures are used by name)
    ACTION_MS,
    DOC_IMAGES,
    canvas_digest,
    canvas_variance,
    needs_chromium,
    open_page,
    playwright_or_skip,
    shoot,
)

pytestmark = pytest.mark.slow

playwright_or_skip()


# --- the basics -------------------------------------------------------------


@needs_chromium
def test_the_page_loads_the_frame_stream(page):
    state = page.evaluate("window.__viewer.state()")
    assert state["frames"] > 100
    assert page.locator("#error").is_hidden()
    assert page.locator("#hud").is_visible()
    assert page.locator("#bar").is_visible()
    assert page.errors == []


@needs_chromium
def test_the_hud_is_populated(page, tmp_path):
    page.evaluate("window.__viewer.seek(0.5)")
    shoot(page, tmp_path / "hud.png")
    assert page.locator("#clock").inner_text() != "0:00"
    assert "MP" in page.locator("#p0").inner_text()
    assert "POP" in page.locator("#p1").inner_text()
    assert page.locator("#tickets").inner_text().strip()


@needs_chromium
def test_scrubbing_renders_a_non_blank_canvas(page, tmp_path):
    seen = []
    for i, fraction in enumerate((0.1, 0.5, 0.9)):
        page.evaluate("window.__viewer.seek(%s)" % fraction)
        shoot(page, tmp_path / ("scrub%d.png" % i))
        variance = canvas_variance(page)
        assert variance > 50, "canvas looks blank at %s (variance %.1f)" % (fraction, variance)
        seen.append(page.evaluate("window.__viewer.state().tick"))
    assert seen == sorted(seen) and seen[0] < seen[-1]
    assert page.errors == []


@needs_chromium
def test_fog_and_cover_toggles(page, tmp_path):
    page.evaluate("window.__viewer.seek(0.5)")
    assert page.evaluate("window.__viewer.state().fogMode") == 0

    page.keyboard.press("f")
    shoot(page, tmp_path / "fog-team0.png")
    assert page.evaluate("window.__viewer.state().fogMode") == 1
    assert "team 0" in page.locator("#btn-fog").inner_text()
    assert canvas_variance(page) > 50
    # a team view must actually withhold something; what exactly is nailed
    # down entity by entity in test_viewer_fog.py
    stats = page.evaluate("window.__viewer.stats()")
    assert stats["hiddenSquads"] + stats["hiddenBuildings"] + stats["hiddenEffects"] > 0, stats

    page.keyboard.press("f")
    assert page.evaluate("window.__viewer.state().fogMode") == 2
    page.keyboard.press("f")
    assert page.evaluate("window.__viewer.state().fogMode") == 0

    page.keyboard.press("c")
    shoot(page, tmp_path / "cover.png")
    assert page.evaluate("window.__viewer.state().showCover") is True
    page.keyboard.press("c")
    assert page.evaluate("window.__viewer.state().showCover") is False
    assert page.errors == []


@needs_chromium
def test_playback_controls(page):
    page.evaluate("window.__viewer.seek(0.2)")
    before = page.evaluate("window.__viewer.state().tick")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    after = page.evaluate("window.__viewer.state().tick")
    assert after > before
    page.keyboard.press("ArrowLeft")
    assert page.evaluate("window.__viewer.state().tick") < after

    assert page.evaluate("window.__viewer.state().speed") == 1
    page.keyboard.press("+")
    assert page.evaluate("window.__viewer.state().speed") == 2
    page.keyboard.press("-")
    page.keyboard.press("-")
    assert page.evaluate("window.__viewer.state().speed") == 0.5

    page.keyboard.press(" ")
    assert page.evaluate("window.__viewer.state().playing") is True
    page.keyboard.press(" ")
    assert page.evaluate("window.__viewer.state().playing") is False
    assert page.errors == []


@needs_chromium
def test_clicking_a_squad_opens_the_inspector(page, tmp_path):
    page.evaluate("window.__viewer.seek(0.5)")
    opened = page.evaluate(
        """(() => {
          var f = frames[frameIndexFor(tick)];
          if (!f.squads.length) return null;
          window.__viewer.select('squad', f.squads[0].id);
          return f.squads[0].def;
        })()"""
    )
    assert opened, "the match should have squads at half time"
    shoot(page, tmp_path / "panel.png")
    assert page.locator("#panel").is_visible()
    body = page.locator("#panel-body").inner_text()
    assert "members" in body and "hp" in body, body
    assert page.errors == []


@needs_chromium
def test_the_winner_banner_appears_at_the_end(page):
    assert page.locator("#banner").is_hidden()
    page.evaluate("window.__viewer.seek(1)")
    page.evaluate("window.__viewer.redraw()")
    has_game_over = page.evaluate("allEvents.some(function (e) { return e.k === 'game_over'; })")
    if not has_game_over:
        pytest.skip("this match ran out of time without a game_over event")
    assert page.locator("#banner").is_visible()
    assert "win" in page.locator("#banner").inner_text().lower() \
        or "draw" in page.locator("#banner").inner_text().lower()
    assert page.errors == []


# --- robustness -------------------------------------------------------------


@needs_chromium
def test_unknown_kinds_and_terrain_never_throw(page, tmp_path):
    """The sim keeps growing; the viewer must degrade, not explode."""
    page.evaluate(
        """(() => {
          var f = JSON.parse(JSON.stringify(frames[frameIndexFor(tick)]));
          var s = JSON.parse(JSON.stringify(f.squads[0] || {id: 1, o: 0, x: 40, y: 40, h: 0, n: 1, max: 1,
                                                            hp: 1, sup: 0, st: 'idle', th: 0, mhp: [1], w: ['']}));
          s.id = 4242; s.kind = 'hovercraft'; s.def = 'flying_saucer'; s.x = 60; s.y = 60;
          f.squads = f.squads.concat([s]);
          f.events = [
            {k: 'quantum_strike', t: f.t, d: {pos: [60, 60]}},
            {k: 'no_position_at_all', t: f.t, d: {}},
            {k: 'shot', t: f.t, d: {src: 1, dst: 2, hit: true}}
          ];
          var data = JSON.parse(JSON.stringify(D));
          data.frames = [f];
          data.map.terrain[10] = 'Z'.repeat(data.map.width);
          window.__viewer.inject(data);
        })()"""
    )
    shoot(page, tmp_path / "unknown.png")
    assert canvas_variance(page) > 50
    assert page.errors == []


@needs_chromium
def test_a_missing_frame_stream_shows_an_error_not_a_blank_page(browser, broken_url):
    pg = open_page(browser, broken_url, width=900, height=600)
    try:
        pg.wait_for_selector("#error:not([hidden])", timeout=ACTION_MS)
        assert "404" in pg.locator("#error-msg").inner_text()
        assert pg.locator("#loading").is_hidden()
    finally:
        pg.close()


# --- the full unit / effect showcase ----------------------------------------

# One synthetic frame holding every unit presentation and every event effect
# the scripted fixture agents never actually produce. This is the regression
# test for the rendering paths *and* the source of `docs/img/viewer-units.png`.
SHOWCASE_JS = """(() => {
  var f = JSON.parse(JSON.stringify(frames[frameIndexFor(tick)]));
  var base = f.squads[0] || {id: 1, o: 0, def: 'rifles', kind: 'infantry', x: 0, y: 0, h: 0, th: 0,
                             n: 1, max: 1, hp: 1, mhp: [60], w: [''], sup: 0, st: 'idle'};
  function sq(o) { var s = JSON.parse(JSON.stringify(base)); for (var k in o) s[k] = o[k]; return s; }
  f.squads = [
    sq({id: 900, o: 0, def: 'hmg_team', kind: 'team_weapon', x: 70, y: 86, h: 0.6, fa: 0.6,
        n: 3, max: 3, hp: 1, st: 'set_up'}),
    sq({id: 901, o: 0, def: 'mortar_team', kind: 'team_weapon', x: 108, y: 86, h: 1.1,
        n: 3, max: 3, hp: 1, st: 'setting_up', setup: 18}),
    sq({id: 902, o: 1, def: 'at_team', kind: 'team_weapon', x: 146, y: 86, h: 2.2, fa: 3.3,
        n: 0, max: 3, hp: 0, ab: 1}),
    sq({id: 903, o: 1, def: 'tank', kind: 'vehicle', x: 70, y: 118, h: 0.7, th: 2.4,
        n: 1, max: 1, hp: 0.7}),
    sq({id: 904, o: 1, def: 'pioneers', kind: 'infantry', x: 108, y: 118, h: 0,
        n: 3, max: 4, hp: 0.3, sup: 2, st: 'retreating'}),
    sq({id: 905, o: 0, def: 'rifles', kind: 'infantry', x: 146, y: 118, h: 0,
        n: 4, max: 6, hp: 0.65, sup: 1, re: 1}),
    sq({id: 906, o: 0, def: 'engineers', kind: 'infantry', x: 70, y: 150, h: 0,
        n: 4, max: 4, hp: 1, g: 910})
  ];
  var b0 = f.buildings[0];
  function bl(o) { var b = JSON.parse(JSON.stringify(b0)); for (var k in o) b[k] = o[k]; return b; }
  f.buildings = [
    bl({id: 910, o: 0, def: 'barracks', cx: 33, cy: 74, w: 3, h: 3, hp: 1, prog: 1, n: 2, q: []}),
    bl({id: 911, o: 1, def: 'tank_depot', cx: 52, cy: 74, w: 3, h: 3, hp: 0.45, prog: 0.4, n: 0, q: []}),
    bl({id: 912, o: 0, def: 'hq_us', cx: 70, cy: 74, w: 4, h: 4, hp: 0.8, prog: 1, n: 0,
        q: [['train', 'rifles', 9], ['research', 'phase_2', 20]]})
  ];
  f.events = [
    {k: 'shot', t: f.t, d: {src: 900, dst: 904, hit: true, src_pos: [70, 86], dst_pos: [108, 118]}},
    {k: 'shot', t: f.t, d: {src: 904, dst: 900, hit: false, src_pos: [108, 118], dst_pos: [70, 86]}},
    {k: 'explosion', t: f.t, d: {src: 901, pos: [128, 136], radius: 6}},
    {k: 'vehicle_destroyed', t: f.t, d: {id: 990, owner: 1, def_id: 'tank', pos: [88, 136]}},
    {k: 'squad_destroyed', t: f.t, d: {id: 991, owner: 1, def_id: 'pioneers', pos: [160, 136]}},
    {k: 'weapon_abandoned', t: f.t, d: {id: 902, owner: 1, def_id: 'at_team', pos: [146, 86]}},
    {k: 'weapon_recrewed', t: f.t, d: {id: 900, owner: 0, def_id: 'hmg_team', by: 905, pos: [70, 86]}},
    {k: 'garrison_ejected', t: f.t, d: {squad: 906, building: 910, owner: 0, pos: [70, 152]}},
    {k: 'garrison_entered', t: f.t, d: {squad: 906, building: 910, owner: 0}},
    {k: 'building_completed', t: f.t, d: {building: 910, def_id: 'barracks', owner: 0}},
    {k: 'construction_started', t: f.t, d: {building: 911, def_id: 'tank_depot', owner: 1, cell: [52, 74]}},
    {k: 'unit_trained', t: f.t, d: {building: 912, unit: 'rifles', squad: 905, owner: 0}},
    {k: 'research_completed', t: f.t, d: {building: 912, upgrade: 'phase_2', owner: 0}},
    {k: 'upgrade_bought', t: f.t, d: {squad: 905, upgrade: 'bar', owner: 0}},
    {k: 'reinforced', t: f.t, d: {squad: 905, owner: 0, def_id: 'rifles'}}
  ];
  f.terrain_delta = [];

  // A second, event-free frame a few ticks later so the playhead can sit
  // mid-animation: parked exactly on the events every effect would be at age 0.
  var after = JSON.parse(JSON.stringify(f));
  after.t = f.t + 8;
  after.events = [];
  after.vis = {};

  var data = JSON.parse(JSON.stringify(D));
  data.frames = [f, after];
  data.winner = 0;
  window.__viewer.inject(data);
  window.__viewer.seekTick(f.t + 4);
  window.__viewer.select('building', 912);
  window.__viewer.camera(108, 118, 7.6);
  return f.events.length;
})()"""


@needs_chromium
def test_every_unit_presentation_and_effect_renders(page, tmp_path):
    page.evaluate("window.__viewer.seek(0.6)")   # a populated HUD behind the showcase
    page.evaluate("window.__viewer.redraw()")
    before = canvas_digest(page)
    events = page.evaluate(SHOWCASE_JS)
    assert events == 15
    shoot(page, tmp_path / "viewer-units.png")

    assert page.errors == []
    assert canvas_variance(page) > 50, "the showcase frame rendered blank"
    assert canvas_digest(page) != before, "the canvas did not change"
    # the inspector is showing the HQ's production queue
    body = page.locator("#panel-body").inner_text()
    assert "Rifles" in body and "s left" in body, body

    # keep the documented screenshot in step with the renderer
    DOC_IMAGES.mkdir(parents=True, exist_ok=True)
    shoot(page, DOC_IMAGES / "viewer-units.png")
