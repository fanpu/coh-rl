"""Drive the 3D battlefield in a real headless Chromium with a real WebGL2 context.

The harness in `conftest.py` launches Chrome with software GL (SwiftShader via
ANGLE, DISPLAY cleared — see the comment on `LAUNCH_ARGS`), so these are not
mocked: three.js compiles real shaders and really rasterises. Skipped cleanly
when Playwright or a Chromium/Chrome binary is missing, and skipped with a
clear reason if the browser turns out to have no WebGL at all.

The fog assertions are the point of the file. The 3D view and the 2D tactical
map share one data layer, so "team 0 cannot see that" has to mean the same
thing in both: here it is checked against the *scene graph* — a hidden enemy
building is not among the objects three.js draws, and it is not among the
proxy volumes a click is tested against.
"""

from __future__ import annotations

import math

import pytest

from tests.viewer.conftest import (  # noqa: F401  (fixtures are used by name)
    ACTION_MS,
    needs_chromium,
    open_page,
    playwright_or_skip,
    shoot,
    wait_loaded,
)
from tests.viewer.showcase import SHOWCASE_EVENTS, SHOWCASE_JS
from tests.viewer.test_viewer_fog import FOG_SCENARIO_JS

pytestmark = pytest.mark.slow

playwright_or_skip()


# --- fixtures ---------------------------------------------------------------


# Loading the page costs several seconds under software GL: Chromium compiles
# every shader and the ground texture is painted texel by texel. One page is
# shared by the whole module and reset between tests instead — which is both
# cheaper and a small exercise of the reset path.
@pytest.fixture(scope="module")
def shared_page3d(browser, viewer_url):
    pg = open_page(browser, viewer_url, view="3d", width=980, height=660)
    try:
        wait_loaded(pg)
        yield pg
    finally:
        pg.close()


@pytest.fixture
def page3d(shared_page3d):
    """The shared 3D page, restored to a known state before and after each test."""
    pg = shared_page3d
    if pg.evaluate("window.__viewer.state().view") != "3d":
        pytest.skip("this browser has no WebGL, so the 3D view cannot be exercised")
    reset_page(pg)
    pg.errors.clear()
    yield pg
    reset_page(pg)


def reset_page(pg):
    """Undo whatever the last test did: payload, fog, selection, camera, view."""
    if pg.evaluate("!!window.__viewerInjected"):
        pg.evaluate("window.__viewer.reload()")
        wait_loaded(pg)
    if pg.evaluate("window.__viewer.state().view") != "3d":
        pg.keyboard.press("v")
    for _ in range(4):
        if pg.evaluate("window.__viewer.state().fogMode") == 0:
            break
        pg.keyboard.press("f")
    pg.evaluate("document.getElementById('panel-close').click()")
    pg.keyboard.press("Home")
    pg.evaluate("window.__viewer.seek(0)")


def gl_canvas_variance(pg):
    """Spread of the pixels three.js actually rasterised into `#cv3`."""
    return pg.evaluate(
        """(() => {
          var cv = document.getElementById('cv3');
          var px = window.__viewer.readPixels3d(0, 0, cv.width, cv.height);
          var n = 0, sum = 0, sq = 0;
          for (var i = 0; i < px.length; i += 4 * 97) {
            var v = (px[i] + px[i + 1] + px[i + 2]) / 3;
            n++; sum += v; sq += v * v;
          }
          return sq / n - (sum / n) * (sum / n);
        })()"""
    )


def gl_canvas_digest(pg):
    return pg.evaluate(
        """(() => {
          var cv = document.getElementById('cv3');
          var px = window.__viewer.readPixels3d(0, 0, cv.width, cv.height);
          var h = 2166136261;
          for (var i = 0; i < px.length; i += 4 * 31) {
            h ^= px[i] + px[i + 1] * 3 + px[i + 2] * 7; h = (h * 16777619) | 0;
          }
          return h;
        })()"""
    )


# --- the basics -------------------------------------------------------------


@needs_chromium
def test_the_3d_view_is_the_default_and_boots_clean(page3d):
    state = page3d.evaluate("window.__viewer.state()")
    assert state["view"] == "3d"
    assert state["webgl"] is True
    assert state["frames"] > 100
    assert page3d.locator("#cv3").is_visible()
    assert page3d.locator("#cv").is_hidden()
    assert page3d.locator("#hud").is_visible()
    assert page3d.errors == []


@needs_chromium
def test_a_real_webgl_context_renders_a_non_blank_scene(page3d, tmp_path):
    renderer = page3d.evaluate(
        """(() => { const c = document.createElement('canvas');
           const g = c.getContext('webgl2') || c.getContext('webgl');
           if (!g) return null;
           const d = g.getExtension('WEBGL_debug_renderer_info');
           return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'webgl'; })()"""
    )
    assert renderer, "no WebGL context at all"
    page3d.evaluate("window.__viewer.seek(0.5)")
    shoot(page3d, tmp_path / "3d.png")
    variance = gl_canvas_variance(page3d)
    assert variance > 50, "the WebGL canvas looks blank (variance %.1f, renderer %s)" % (
        variance, renderer,
    )
    stats = page3d.evaluate("window.__viewer.scene3dStats()")
    assert stats["drawCalls"] > 5, stats
    assert stats["triangles"] > 1000, stats
    assert stats["features"] > 100, "the map's hedgerows/trees/walls should be instanced: %s" % stats
    assert page3d.errors == []


@needs_chromium
def test_v_toggles_the_views_and_the_hash_remembers_the_choice(page3d):
    assert "view=3d" in page3d.evaluate("window.location.hash")

    page3d.keyboard.press("v")
    assert page3d.evaluate("window.__viewer.state().view") == "2d"
    assert "view=2d" in page3d.evaluate("window.location.hash")
    assert page3d.locator("#cv").is_visible()
    assert page3d.locator("#cv3").is_hidden()
    assert page3d.evaluate("window.__viewer.scene3dStats()") is not None

    page3d.keyboard.press("v")
    assert page3d.evaluate("window.__viewer.state().view") == "3d"
    assert "view=3d" in page3d.evaluate("window.location.hash")
    assert page3d.locator("#cv3").is_visible()
    assert page3d.errors == []


@needs_chromium
def test_stepping_frames_changes_the_rendered_image(page3d):
    page3d.evaluate("window.__viewer.seek(0.35)")
    page3d.evaluate("window.__viewer.redraw()")
    before = gl_canvas_digest(page3d)
    page3d.evaluate("window.__viewer.seek(0.75)")
    page3d.evaluate("window.__viewer.redraw()")
    assert gl_canvas_digest(page3d) != before, "the scene did not change between frames"
    assert page3d.errors == []


@needs_chromium
def test_the_camera_rotates_zooms_and_resets(page3d):
    page3d.evaluate("window.__viewer.seek(0.5)")
    page3d.keyboard.press("Home")
    home = page3d.evaluate("window.__viewer.state3d()")

    # right-drag rotates, synchronously on the mousemove
    page3d.mouse.move(490, 330)
    page3d.mouse.down(button="right")
    page3d.mouse.move(620, 330)
    page3d.mouse.up(button="right")
    dragged = page3d.evaluate("window.__viewer.state3d()")
    assert dragged["yaw"] != home["yaw"], (home, dragged)

    # Q/E rotate too, but the held key is applied on animation frames, so poll
    page3d.keyboard.down("q")
    try:
        held = None
        for _ in range(20):
            held = page3d.evaluate("window.__viewer.state3d()")
            if held["yaw"] != dragged["yaw"]:
                break
            page3d.wait_for_timeout(100)
        assert held["yaw"] != dragged["yaw"], (dragged, held)
    finally:
        page3d.keyboard.up("q")

    # the wheel zooms in, and the pitch steepens as it comes down
    page3d.mouse.move(490, 330)
    page3d.mouse.wheel(0, -600)
    zoomed = page3d.evaluate("window.__viewer.state3d()")
    assert zoomed["dist"] < home["dist"]
    assert zoomed["pitch"] < home["pitch"], "pitch should steepen as you zoom in"

    page3d.keyboard.press("Home")
    back = page3d.evaluate("window.__viewer.state3d()")
    assert abs(back["dist"] - home["dist"]) < 0.5
    assert abs(back["yaw"] - home["yaw"]) < 1e-6
    assert page3d.errors == []


@needs_chromium
def test_clicking_selects_and_the_inspector_matches_the_2d_one(page3d):
    page3d.evaluate("window.__viewer.seek(0.5)")
    page3d.evaluate("window.__viewer.redraw()")
    hit = page3d.evaluate(
        """(() => {
          var f = window.__viewer.frame();
          for (var i = 0; i < f.squads.length; i++) {
            var s = f.squads[i];
            if (s.g !== undefined) continue;
            var p = window.__viewer.screenOf(s.x, s.y);
            var got = window.__viewer.pickAt(p[0], p[1]);
            if (got && got.kind === 'squad') return {want: s.id, got: got};
          }
          return null;
        })()"""
    )
    assert hit, "no squad was pickable in the 3D view at half time"
    page3d.evaluate("window.__viewer.select('squad', %d)" % hit["got"]["id"])
    page3d.evaluate("window.__viewer.redraw()")
    assert page3d.locator("#panel").is_visible()
    body = page3d.locator("#panel-body").inner_text()
    assert "members" in body and "hp" in body, body
    assert page3d.errors == []


# --- fog --------------------------------------------------------------------


def load_fog_scenario(pg):
    pg.evaluate("window.__fogDropUnseen = false")
    pg.evaluate(FOG_SCENARIO_JS)
    pg.evaluate("window.__viewer.seekTick(4)")
    pg.evaluate("window.__viewer.redraw()")


def set_fog(pg, mode):
    for _ in range(4):
        if pg.evaluate("window.__viewer.state().fogMode") == mode:
            break
        pg.keyboard.press("f")
    assert pg.evaluate("window.__viewer.state().fogMode") == mode
    pg.evaluate("window.__viewer.redraw()")


@needs_chromium
def test_team_fog_removes_hidden_enemies_from_the_3d_scene_graph(page3d, tmp_path):
    """The same rigged scenario `test_viewer_fog.py` uses, checked in the scene.

    #700 own barracks, #701 enemy barracks seen earlier, #702 enemy depot never
    seen, #703 neutral barn; #800 own squad, #801 enemy squad out of vision.
    """
    load_fog_scenario(page3d)

    set_fog(page3d, 0)
    omniscient = page3d.evaluate("window.__viewer.scene3dStats()")
    assert omniscient["buildings"] == 4, omniscient
    assert omniscient["ghosts"] == 0, omniscient
    assert omniscient["units"] == 2, omniscient

    set_fog(page3d, 1)
    shoot(page3d, tmp_path / "3d-fog.png")
    team0 = page3d.evaluate("window.__viewer.scene3dStats()")
    # the never-seen enemy depot is simply not in the scene; the one team 0 saw
    # earlier is there only as a translucent last-known volume
    assert team0["buildings"] == 2, team0
    assert team0["ghosts"] == 1, team0
    assert team0["units"] == 1, team0
    assert team0["buildings"] + team0["ghosts"] < omniscient["buildings"]

    # and none of the hidden things are clickable, because they are not among
    # the proxy volumes the raycast is run against
    assert page3d.evaluate("window.__viewer.buildingFog(702)") == "hidden"
    assert page3d.evaluate("window.__viewer.buildingFog(701)") == "ghost"
    assert page3d.errors == []


@needs_chromium
def test_hidden_entities_are_unpickable_in_3d(page3d):
    load_fog_scenario(page3d)

    set_fog(page3d, 0)
    visible = page3d.evaluate(
        """(() => {
          function at(x, y) {
            var p = window.__viewer.screenOf(x, y);
            return window.__viewer.pickAt(p[0], p[1]);
          }
          return {squad: at(105, 61), depot: at(123, 43), own: at(146, 146)};
        })()"""
    )
    assert visible["squad"] == {"kind": "squad", "id": 801}, visible
    assert visible["depot"] == {"kind": "building", "id": 702}, visible

    set_fog(page3d, 1)
    hidden = page3d.evaluate(
        """(() => {
          function at(x, y) {
            var p = window.__viewer.screenOf(x, y);
            return window.__viewer.pickAt(p[0], p[1]);
          }
          return {squad: at(105, 61), depot: at(123, 43), own: at(146, 146)};
        })()"""
    )
    assert hidden["squad"] is None, "a hidden enemy squad must not be selectable in 3D"
    assert hidden["depot"] is None, "a never-seen enemy building must not be selectable in 3D"
    assert hidden["own"] == {"kind": "building", "id": 700}, "own things stay clickable"
    assert page3d.errors == []


@needs_chromium
def test_a_real_match_hides_something_in_the_3d_team_view(page3d):
    page3d.evaluate("window.__viewer.seek(0.5)")
    page3d.evaluate("window.__viewer.redraw()")
    omniscient = page3d.evaluate("window.__viewer.scene3dStats()")
    set_fog(page3d, 1)
    team0 = page3d.evaluate("window.__viewer.scene3dStats()")
    assert team0["buildings"] < omniscient["buildings"], (omniscient, team0)
    stats = page3d.evaluate("window.__viewer.stats()")
    assert stats["hiddenBuildings"] + stats["hiddenSquads"] + stats["hiddenEffects"] > 0, stats
    assert page3d.errors == []


# --- the showcase and terrain deltas ----------------------------------------


@needs_chromium
def test_the_showcase_renders_every_unit_kind_and_effect_in_3d(page3d, tmp_path):
    page3d.evaluate("window.__viewer.redraw()")
    before = gl_canvas_digest(page3d)
    assert page3d.evaluate(SHOWCASE_JS) == SHOWCASE_EVENTS
    page3d.evaluate("window.__viewer.redraw()")
    shoot(page3d, tmp_path / "3d-showcase.png")

    stats = page3d.evaluate("window.__viewer.scene3dStats()")
    # tanks, a halftrack, three team weapons, four infantry squads and an
    # unknown kind; the two garrisoned squads are badges on the farmhouse
    assert stats["units"] == 10, stats
    assert stats["soldiers"] >= 15, stats
    assert stats["buildings"] == 6, stats
    assert stats["puffs"] > 10, "the blasts should be throwing fire and smoke: %s" % stats
    assert stats["wrecks"] == 1, "a destroyed vehicle leaves a hull behind: %s" % stats
    assert gl_canvas_variance(page3d) > 50
    assert gl_canvas_digest(page3d) != before
    assert page3d.errors == []


@needs_chromium
def test_a_terrain_delta_updates_the_scene(page3d):
    """A crushed fence disappears and a new crater grows a rim."""
    counts = page3d.evaluate(
        """(() => {
          var data = JSON.parse(JSON.stringify(window.__viewer.data()));
          var f = JSON.parse(JSON.stringify(window.__viewer.frame()));

          // frame 0: a fence line and clear ground next to it
          function paint(cy, from, to, ch) {
            var row = data.map.terrain[cy];
            data.map.terrain[cy] = row.slice(0, from) + ch.repeat(to - from) + row.slice(to);
          }
          paint(60, 20, 30, 'f');
          paint(62, 20, 30, '.');
          f.terrain_delta = [];
          f.events = [];

          // frame 1: the fence is crushed flat and shells have cratered the field
          var after = JSON.parse(JSON.stringify(f));
          after.t = f.t + 2;
          after.terrain_delta = [];
          for (var cx = 20; cx < 30; cx++) after.terrain_delta.push([cx, 60, '.']);
          for (var cx2 = 22; cx2 < 28; cx2++) after.terrain_delta.push([cx2, 62, 'c']);

          data.frames = [f, after];
          window.__viewer.inject(data);
          window.__viewer.camera3(50, 122, 60, 0);
          window.__viewer.seekTick(f.t);
          window.__viewer.redraw();
          var a = window.__viewer.scene3dStats().features;
          window.__viewer.seekTick(after.t);
          window.__viewer.redraw();
          var b = window.__viewer.scene3dStats().features;
          return {before: a, after: b};
        })()"""
    )
    # ten fence panels go, six crater rims arrive
    assert counts["after"] == counts["before"] - 10 + 6, counts
    assert page3d.errors == []


# --- the low-quality fallback -----------------------------------------------


@needs_chromium
def test_quality_low_drops_shadows_and_still_renders(browser, viewer_url, tmp_path):
    pg = open_page(browser, viewer_url, view="3d", query="?quality=low", width=980, height=660)
    try:
        wait_loaded(pg)
        if pg.evaluate("window.__viewer.state().view") != "3d":
            pytest.skip("this browser has no WebGL")
        assert pg.evaluate("window.__viewer.state().quality") == "low"
        pg.evaluate("window.__viewer.seek(0.5)")
        shoot(pg, tmp_path / "3d-low.png")
        stats = pg.evaluate("window.__viewer.scene3dStats()")
        assert stats["shadows"] is False, stats
        assert stats["drawCalls"] > 5, stats
        assert gl_canvas_variance(pg) > 50
        assert pg.errors == []
    finally:
        pg.close()


# --- the no-WebGL fallback --------------------------------------------------


@needs_chromium
def test_without_webgl_the_page_falls_back_to_the_2d_map_with_a_notice(browser, viewer_url):
    """A browser that cannot give us a context must still show the replay.

    `HTMLCanvasElement.getContext` is stubbed to refuse WebGL before any of the
    page's own scripts run, which is as close to a machine without a GPU as we
    can get from here.
    """
    pg = browser.new_page(viewport={"width": 900, "height": 620})
    pg.set_default_timeout(ACTION_MS)
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append("pageerror: %s" % e))
    try:
        pg.add_init_script(
            """(() => {
              const real = HTMLCanvasElement.prototype.getContext;
              HTMLCanvasElement.prototype.getContext = function (kind) {
                if (String(kind).indexOf('webgl') === 0) return null;
                return real.apply(this, arguments);
              };
            })()"""
        )
        pg.goto(viewer_url + "#paused&view=3d", wait_until="load")
        wait_loaded(pg)
        state = pg.evaluate("window.__viewer.state()")
        assert state["webgl"] is False, state
        assert state["view"] == "2d", "it must fall back, not leave a dead canvas"
        assert pg.locator("#cv").is_visible()
        assert pg.locator("#cv3").is_hidden()
        assert pg.locator("#notice").is_visible()
        assert "webgl" in pg.locator("#notice-msg").inner_text().lower()
        # and the tactical map really is drawing (redraw first: `stats()`
        # reports the last render, and the page may not have had one yet)
        pg.evaluate("window.__viewer.redraw()")
        assert pg.evaluate("window.__viewer.stats().buildings") > 0
        assert errors == []
    finally:
        pg.close()


@needs_chromium
def test_a_blocked_unit_is_announced_over_the_building(page3d):
    """`unit_blocked` has no effect on the map, so it has to be words.

    The showcase blocks a rifle squad coming out of barracks #913; the badge
    must land on the overlay, in its own colour, near that building.
    """
    page3d.evaluate(SHOWCASE_JS)
    page3d.evaluate("window.__viewer.redraw()")
    found = page3d.evaluate(
        """(() => {
          // barracks #913 sits at cell (33, 50) with a 3x3 footprint
          var p = window.__viewer.screenOf((33 + 1.5) * 2, (50 + 1.5) * 2);
          var ov = document.getElementById('ov');
          var g = ov.getContext('2d');
          var dpr = window.devicePixelRatio > 2 ? 2 : (window.devicePixelRatio || 1);
          var x = Math.max(0, Math.round((p[0] - 70) * dpr));
          var y = Math.max(0, Math.round((p[1] - 80) * dpr));
          var w = Math.min(Math.round(140 * dpr), ov.width - x);
          var h = Math.min(Math.round(110 * dpr), ov.height - y);
          if (w <= 0 || h <= 0) return {onScreen: false};
          var d = g.getImageData(x, y, w, h).data;
          var warm = 0, any = 0;
          for (var i = 0; i < d.length; i += 4) {
            if (d[i + 3] < 100) continue;
            any++;
            // the badge is #e2705f: clearly red-dominant, unlike the cream labels
            if (d[i] > 150 && d[i] - d[i + 1] > 45 && d[i] - d[i + 2] > 35) warm++;
          }
          return {onScreen: true, warm: warm, any: any};
        })()"""
    )
    assert found["onScreen"], "the barracks projected off screen"
    assert found["warm"] > 20, "no 'blocked' badge over the stuck barracks: %s" % found
    assert page3d.errors == []


@needs_chromium
def test_the_opening_shot_is_close_enough_to_see_individual_soldiers(browser, viewer_url):
    """On load the camera looks down the watched team's axis, not at the map.

    This needs its own page: the thing under test is the state the viewer opens
    in, which is exactly what the shared page's `Home` reset throws away.
    """
    pg = open_page(browser, viewer_url, view="3d", width=980, height=660)
    try:
        wait_loaded(pg)
        if pg.evaluate("window.__viewer.state().view") != "3d":
            pytest.skip("this browser has no WebGL")
        opening = pg.evaluate("window.__viewer.state3d()")
        pg.keyboard.press("Home")
        whole_map = pg.evaluate("window.__viewer.state3d()")

        assert opening["dist"] < whole_map["dist"] / 2, (opening, whole_map)
        # a 1.8 m figure at this height is tens of pixels tall, not one or two
        height = pg.evaluate("window.innerHeight")
        metres = 2 * opening["dist"] * math.tan(math.radians(42) / 2)
        assert metres < 70, "opening zoom is too far out: %.0f m of ground" % metres
        assert height / metres * 1.8 > 15, "a soldier would be under 15 px tall"
        # and it is looking somewhere inside the map, not off the edge
        size = pg.evaluate("window.__viewer.mapSize()")
        assert 0 <= opening["x"] <= size["W"] * size["cell"], opening
        assert 0 <= opening["y"] <= size["H"] * size["cell"], opening
        assert pg.errors == []
    finally:
        pg.close()
