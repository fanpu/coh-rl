"""Drive the canvas app in a real headless Chromium.

Skipped cleanly when Playwright or a Chromium/Chrome binary is missing (this
repo deliberately does not run `playwright install`; it points Playwright at
whatever browser the machine already has — see `tests/viewer/chromium.py`).

Screenshots land in the test's `tmp_path` so a failure can be eyeballed.
"""

from __future__ import annotations

import socketserver
import threading

import pytest

from coh.viewer.__main__ import STATIC_DIR, make_handler
from coh.viewer.frames import build_frames, frames_json_gz
from tests.helpers import fixture_data
from tests.viewer.chromium import find_chromium
from tests.viewer.conftest import play

pytestmark = pytest.mark.slow

sync_playwright = pytest.importorskip("playwright.sync_api", reason="playwright is not installed").sync_playwright

CHROMIUM = find_chromium()
needs_chromium = pytest.mark.skipif(CHROMIUM is None, reason="no Chromium/Chrome binary on this machine")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture(scope="module")
def viewer_url():
    """Serve a 3-minute match's frames on an ephemeral port for the test."""
    frames = build_frames(play(180.0), 2, data=fixture_data())
    server = _Server(("127.0.0.1", 0), make_handler(frames_json_gz(frames)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    if CHROMIUM is None:
        pytest.skip("no Chromium/Chrome binary on this machine")
    with sync_playwright() as pw:
        instance = pw.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox"])
        yield instance
        instance.close()


@pytest.fixture(scope="module")
def broken_url():
    """A server that serves the page but no frames, to exercise the error path."""
    import http.server

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def log_message(self, fmt, *args):
            pass

    server = _Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def page(browser, viewer_url):
    """A freshly loaded viewer page; `page.errors` collects console errors."""
    errors: list[str] = []
    pg = browser.new_page(viewport={"width": 1280, "height": 860})
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append("pageerror: %s" % e))
    pg.goto(viewer_url, wait_until="load")
    for _ in range(150):
        if pg.evaluate("!!(window.__viewer && window.__viewer.state().frames)"):
            break
        pg.wait_for_timeout(200)
    pg.evaluate("window.__viewer.setPlaying(false)")
    pg.errors = errors
    yield pg
    pg.close()


def shoot(pg, path):
    """Force a synchronous redraw, then capture (never wait on rAF timing)."""
    pg.evaluate("window.__viewer.redraw()")
    pg.screenshot(path=str(path), timeout=60_000)


def canvas_variance(pg):
    """Spread of the rendered pixels — a blank canvas scores ~0."""
    return pg.evaluate(
        """(() => {
          var cv = document.getElementById('cv');
          var d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
          var n = 0, sum = 0, sq = 0;
          for (var i = 0; i < d.length; i += 4 * 97) {
            var v = (d[i] + d[i + 1] + d[i + 2]) / 3;
            n++; sum += v; sq += v * v;
          }
          return sq / n - (sum / n) * (sum / n);
        })()"""
    )


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
    pg = browser.new_page(viewport={"width": 900, "height": 600})
    try:
        pg.goto(broken_url, wait_until="load")
        pg.wait_for_selector("#error:not([hidden])", timeout=15_000)
        assert "404" in pg.locator("#error-msg").inner_text()
        assert pg.locator("#loading").is_hidden()
    finally:
        pg.close()
