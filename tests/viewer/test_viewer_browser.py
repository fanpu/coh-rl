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
from tests.viewer.showcase import SHOWCASE_EVENTS, SHOWCASE_JS

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
          var f = window.__viewer.frame();
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
    has_game_over = page.evaluate("window.__viewer.events().some(function (e) { return e.k === 'game_over'; })")
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
          var f = JSON.parse(JSON.stringify(window.__viewer.frame()));
          var s = JSON.parse(JSON.stringify(f.squads[0] || {id: 1, o: 0, x: 40, y: 40, h: 0, n: 1, max: 1,
                                                            hp: 1, sup: 0, st: 'idle', th: 0, mhp: [1], w: ['']}));
          s.id = 4242; s.kind = 'hovercraft'; s.def = 'flying_saucer'; s.x = 60; s.y = 60;
          f.squads = f.squads.concat([s]);
          f.events = [
            {k: 'quantum_strike', t: f.t, d: {pos: [60, 60]}},
            {k: 'no_position_at_all', t: f.t, d: {}},
            {k: 'shot', t: f.t, d: {src: 1, dst: 2, hit: true}}
          ];
          var data = JSON.parse(JSON.stringify(window.__viewer.data()));
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

# `tests/viewer/showcase.py` builds one synthetic frame holding every unit
# presentation and every event effect the scripted fixture agents never
# actually produce. It is shared with the 3D tests and with the screenshot
# tool, so all three look at exactly the same scene.


@needs_chromium
def test_every_unit_presentation_and_effect_renders(page, tmp_path):
    page.evaluate("window.__viewer.seek(0.6)")   # a populated HUD behind the showcase
    page.evaluate("window.__viewer.redraw()")
    before = canvas_digest(page)
    events = page.evaluate(SHOWCASE_JS)
    assert events == SHOWCASE_EVENTS
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
