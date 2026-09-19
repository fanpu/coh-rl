"""Shared viewer-test scaffolding: replays, the browser harness, and a deadline.

The browser pieces live here rather than in one test module so that several
modules (`test_viewer_browser.py`, `test_viewer_fog.py`) share a single
Chromium instance and a single served frame stream.
"""

from __future__ import annotations

import contextlib
import http.server
import os
import signal
import socketserver
import threading
from pathlib import Path

import pytest

from coh.agents import AGENTS
from coh.env import CohEnv
from coh.maps.format import load_map
from coh.replay.replay import Replay
from coh.sim.sim import PlayerSetup, SimConfig, neutral_footprints
from coh.viewer.__main__ import STATIC_DIR, make_handler
from coh.viewer.frames import build_frames, frames_json_gz
from tests.helpers import fixture_data
from tests.viewer.chromium import find_chromium

MAP_NAME = "hedgerow_crossing"
SHORT_MATCH_S = 30.0


def play(seconds: float, agent: str = "t1", seed: int = 0) -> Replay:
    """Play `seconds` of `hedgerow_crossing` on fixture data and return the replay."""
    data = fixture_data()
    game_map = load_map(MAP_NAME, footprints=neutral_footprints(data))
    players = [PlayerSetup("us", 0, 0), PlayerSetup("wehr", 1, 1)]
    env = CohEnv(
        map_name=MAP_NAME,
        players=players,
        seed=seed,
        decision_interval_s=2.0,
        config=SimConfig(time_limit_s=seconds),
        data=data,
        game_map=game_map,
    )
    obs = env.reset()
    agents = [AGENTS[agent](), AGENTS[agent]()]
    for player_id, bot in enumerate(agents):
        bot.reset(player_id, game_map, data)
    done = False
    while not done:
        orders = {pid: agents[pid].act(obs[pid]) for pid in env.player_ids}
        obs, _rewards, done, _infos = env.step(orders)
    return env.to_replay(data_dir="tests/data/fixtures")


@pytest.fixture(scope="session")
def short_replay() -> Replay:
    """A 30 s scripted match — long enough for captures, shots and training."""
    return play(SHORT_MATCH_S)


# ---------------------------------------------------------------------------
# Hard deadline
# ---------------------------------------------------------------------------

# Playwright's sync `evaluate()` has no timeout of its own: if the page's main
# thread wedges, the call blocks forever and takes the whole suite with it.
# This is the backstop — a test that blows its budget fails loudly instead of
# stalling. SIGALRM only fires on the main thread of a POSIX process, so on
# anything else the guard is a no-op and the per-call Playwright timeouts are
# the only protection.
_CAN_ALARM = hasattr(signal, "SIGALRM")


class DeadlineExceeded(AssertionError):
    pass


@contextlib.contextmanager
def deadline(seconds: float, what: str):
    """Fail the test if the body has not finished within `seconds`.

    NOT nesting-safe: there is only one interval timer per process, so an inner
    `deadline` overwrites the outer one's alarm and, on exit, disarms it
    entirely — the outer budget is silently lost. Use one per call stack. The
    module-scoped fixtures and the autouse per-test guard never overlap
    (fixture setup finishes before the test body starts).
    """
    if not _CAN_ALARM or threading.current_thread() is not threading.main_thread():
        yield
        return

    def fire(signum, frame):
        raise DeadlineExceeded(f"{what} exceeded its {seconds:g}s deadline")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# Browser harness
# ---------------------------------------------------------------------------

CHROMIUM = find_chromium()
DOC_IMAGES = Path(__file__).resolve().parents[2] / "docs" / "img"

# A screenshot is never byte-identical across machines (fonts, GL backend,
# driver), so a test that writes into `docs/img/` leaves the tree dirty after
# every run. Tests write to their own `tmp_path` and only refresh the
# documented images when this is explicitly asked for:
#
#     COH_REFRESH_DOCS_IMG=1 uv run pytest tests/viewer -q
REFRESH_DOCS_IMG = os.environ.get("COH_REFRESH_DOCS_IMG") == "1"


def doc_image(name: str, fallback: Path) -> Path:
    """Where a documented screenshot should be written this run."""
    if not REFRESH_DOCS_IMG:
        return fallback / name
    DOC_IMAGES.mkdir(parents=True, exist_ok=True)
    return DOC_IMAGES / name

MATCH_S = 120.0          # long enough for captures, training and firefights
ACTION_MS = 15_000       # any single Playwright action
NAV_MS = 20_000          # page load
TEST_DEADLINE_S = 90.0   # hard ceiling per test
SETUP_DEADLINE_S = 90.0  # hard ceiling for module-scoped setup
# Software GL, so the 3D view is exercised against a *real* WebGL2 context
# rather than skipped. Chrome's SwiftShader is a Vulkan ICD, and ANGLE picks an
# X11/Vulkan display whenever DISPLAY is set - which then fails with
# "xcb_connect() failed" and leaves the page with no WebGL at all. Clearing
# DISPLAY for the browser process is what makes SwANGLE fall back to its
# headless surfaceless path. (Nine flag combinations were tried; none help
# while DISPLAY is set, and none are needed once it is not.)
LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
]
BROWSER_ENV = {k: v for k, v in os.environ.items() if k != "DISPLAY"}

needs_chromium = pytest.mark.skipif(CHROMIUM is None, reason="no Chromium/Chrome binary on this machine")


def playwright_or_skip():
    """`sync_playwright`, or skip the whole module if Playwright is absent."""
    return pytest.importorskip("playwright.sync_api", reason="playwright is not installed").sync_playwright


@pytest.fixture(autouse=True)
def hard_deadline(request):
    wants = {"page", "page3d", "browser"}
    if not wants & set(request.fixturenames):
        yield
        return
    with deadline(TEST_DEADLINE_S, request.node.name):
        yield


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(handler_cls):
    """Run `handler_cls` on an ephemeral port; yields the URL, always shuts down."""
    server = _Server(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the frames server thread did not stop"


@pytest.fixture(scope="session")
def viewer_url():
    """Serve one match's frames on an ephemeral port for the whole session."""
    with deadline(SETUP_DEADLINE_S, "building the frame stream"):
        frames = build_frames(play(MATCH_S), 2, data=fixture_data())
    yield from _serve(make_handler(frames_json_gz(frames)))


@pytest.fixture(scope="session")
def broken_url():
    """Serve the page but no frames, to exercise the client's error path."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def log_message(self, fmt, *args):
            pass

    yield from _serve(Handler)


@pytest.fixture(scope="session")
def browser():
    if CHROMIUM is None:
        pytest.skip("no Chromium/Chrome binary on this machine")
    with playwright_or_skip()() as pw:
        with deadline(SETUP_DEADLINE_S, "launching Chromium"):
            instance = pw.chromium.launch(
                executable_path=CHROMIUM, args=LAUNCH_ARGS, env=BROWSER_ENV
            )
        try:
            yield instance
        finally:
            instance.close()


def open_page(browser, url, width=1280, height=860, view="2d", query=""):
    """A page with explicit timeouts everywhere; never auto-plays.

    `view` picks the renderer through the URL hash the app itself uses
    (`#paused&view=2d`), so these tests drive the same deep link a human would.
    The default is the 2D tactical map: the pixel-level fog assertions below
    are written against that canvas.
    """
    pg = browser.new_page(viewport={"width": width, "height": height})
    pg.set_default_timeout(ACTION_MS)
    pg.set_default_navigation_timeout(NAV_MS)
    errors: list[str] = []
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append("pageerror: %s" % e))
    pg.errors = errors
    pg.goto(url + query + "#paused&view=" + view, wait_until="load", timeout=NAV_MS)
    return pg


def wait_loaded(pg, tries=60, gap_ms=250):
    """Poll (rather than `wait_for_function`, which proved flaky on this page)."""
    for _ in range(tries):
        if pg.evaluate("!!(window.__viewer && window.__viewer.state().frames)"):
            return
        pg.wait_for_timeout(gap_ms)
    raise AssertionError("the frame stream never finished loading")


@pytest.fixture
def page(browser, viewer_url):
    """A freshly loaded, paused viewer page; `page.errors` collects console errors."""
    pg = open_page(browser, viewer_url)
    try:
        wait_loaded(pg)
        assert pg.evaluate("window.__viewer.state().playing") is False, "#paused should not auto-play"
        yield pg
    finally:
        pg.close()


def shoot(pg, path):
    """Force a synchronous redraw, then capture (never wait on rAF timing)."""
    pg.evaluate("window.__viewer.redraw()")
    pg.screenshot(path=str(path), timeout=ACTION_MS)


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


def canvas_digest(pg):
    """A cheap fingerprint of the rendered pixels, to prove a redraw changed something."""
    return pg.evaluate(
        """(() => {
          var cv = document.getElementById('cv');
          var d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
          var h = 2166136261;
          for (var i = 0; i < d.length; i += 4 * 31) { h ^= d[i] + d[i + 1] * 3 + d[i + 2] * 7; h = (h * 16777619) | 0; }
          return h;
        })()"""
    )
