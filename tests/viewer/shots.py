"""Drive the viewer in a real browser and write screenshots.

Not a test: this is the tool used to *look* at the renderer while working on
it, and to regenerate the images the README shows. It reuses the browser
harness from `conftest.py` so the screenshots come out of the same Chromium,
with the same software-WebGL flags, as the tests.

    uv run python -m tests.viewer.shots --replay /tmp/m.replay.json --out /tmp/shots
    uv run python -m tests.viewer.shots --docs        # refresh docs/img/*
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from coh.replay.replay import load
from coh.viewer.__main__ import make_handler
from coh.viewer.frames import build_frames, frames_json_gz
from tests.helpers import fixture_data
from tests.viewer.conftest import (
    BROWSER_ENV,
    CHROMIUM,
    LAUNCH_ARGS,
    NAV_MS,
    DOC_IMAGES,
    _Server,
    play,
)
from tests.viewer.showcase import SHOWCASE_JS

WIDTH, HEIGHT = 1600, 950


def serve(payload: bytes):
    server = _Server(("127.0.0.1", 0), make_handler(payload))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, "http://127.0.0.1:%d/" % server.server_address[1]


def open_viewer(browser, url, view="3d", query=""):
    pg = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
    pg.set_default_timeout(30_000)
    errors: list[str] = []
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append("pageerror: %s" % e))
    pg.errors = errors
    pg.goto(url + query + "#paused&view=" + view, wait_until="load", timeout=NAV_MS)
    for _ in range(120):
        if pg.evaluate("!!(window.__viewer && window.__viewer.state().frames)"):
            break
        pg.wait_for_timeout(250)
    else:
        raise SystemExit("the frame stream never loaded")
    return pg


def shot(pg, path: Path, label: str = ""):
    pg.evaluate("window.__viewer.redraw()")
    path.parent.mkdir(parents=True, exist_ok=True)
    pg.screenshot(path=str(path), timeout=30_000)
    print(f"  {path}  {label}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", default=None, help="a replay JSON to render (default: a fresh fixture match)")
    ap.add_argument("--out", default="/tmp/shots", help="directory for the screenshots")
    ap.add_argument("--docs", action="store_true", help="also write docs/img/viewer-3d*.png")
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args(argv)

    if CHROMIUM is None:
        print("no Chromium/Chrome binary", file=sys.stderr)
        return 1

    if args.replay:
        frames = build_frames(load(args.replay), args.every, data=fixture_data())
    else:
        frames = build_frames(play(150.0), args.every, data=fixture_data())
    print(f"{len(frames['frames'])} frames")
    payload = frames_json_gz(frames)

    out = Path(args.out)
    server, thread, url = serve(payload)
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=CHROMIUM, args=LAUNCH_ARGS, env=BROWSER_ENV)
            try:
                pg = open_viewer(browser, url)
                gl = pg.evaluate(
                    "(() => { const c = document.createElement('canvas');"
                    " const g = c.getContext('webgl2'); if (!g) return 'none';"
                    " const d = g.getExtension('WEBGL_debug_renderer_info');"
                    " return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'webgl2'; })()"
                )
                print("GL:", gl)
                print("view:", pg.evaluate("window.__viewer.state().view"))

                tick, (bx, by), n = busiest(frames)
                print(f"busiest tick {tick}: {n} combat events around ({bx:.0f}, {by:.0f})")
                pg.evaluate("window.__viewer.seekTick(%d)" % tick)

                # a few camera angles over the busiest moment of the match
                cams = cameras(bx, by)
                for name, js in cams:
                    pg.evaluate(js)
                    shot(pg, out / f"{name}.png")

                print("scene:", json.dumps(pg.evaluate("window.__viewer.scene3dStats()")))
                print("fps:", measure_fps(pg))

                if args.docs:
                    pg.evaluate(cams[1][1])
                    shot(pg, DOC_IMAGES / "viewer-3d.png", "(real match)")

                # fog view
                pg.keyboard.press("f")
                shot(pg, out / "fog.png", "(team 0)")
                if args.docs:
                    shot(pg, DOC_IMAGES / "viewer-3d-fog.png", "(team fog)")
                pg.keyboard.press("f")
                pg.keyboard.press("f")

                # the 2D tactical map, at the same moment: fitted to the whole
                # map (which is what a tactical map is for) with the squad in
                # the thick of it inspected
                pg.keyboard.press("v")
                pg.keyboard.press("Home")
                pg.evaluate(
                    """(() => {
                      var f = window.__viewer.frame(), best = null, bd = 1e9;
                      f.squads.forEach(function (s) {
                        var d = Math.hypot(s.x - %f, s.y - %f);
                        if (s.g === undefined && d < bd) { bd = d; best = s; }
                      });
                      if (best) window.__viewer.select('squad', best.id);
                    })()""" % (bx, by)
                )
                shot(pg, out / "tactical.png", "(2D)")
                if args.docs:
                    shot(pg, DOC_IMAGES / "viewer.png", "(2D tactical map)")
                pg.keyboard.press("v")
                pg.evaluate(cams[2][1])

                # the synthetic showcase
                pg.evaluate(SHOWCASE_JS)
                pg.evaluate("window.__viewer.camera3(104, 100, 74, 0.35)")
                shot(pg, out / "showcase.png")
                if args.docs:
                    shot(pg, DOC_IMAGES / "viewer-3d-showcase.png", "(showcase)")
                print("showcase scene:", json.dumps(pg.evaluate("window.__viewer.scene3dStats()")))
                print("errors:", pg.errors)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    return 0


def cameras(x, y):
    """Four looks at the same moment: map, operational, tactical, low angle."""
    # Every preset states its pitch offset: it is sticky, so a shot taken
    # after the low-angle one would otherwise inherit its tilt.
    return [
        ("overview", "window.__viewer.camera3(96, 96, 190, 0.0, 0)"),
        ("battle", "window.__viewer.camera3(%f, %f, 58, 0.55, 0)" % (x, y)),
        ("close", "window.__viewer.camera3(%f, %f, 30, 2.3, 0.05)" % (x, y)),
        ("low", "window.__viewer.camera3(%f, %f, 40, 3.9, -0.28)" % (x, y)),
    ]


def busiest(frames):
    """The tick with the most combat, and where on the map it is happening."""
    best_t, best_n, best_pos = 0, -1, (96.0, 96.0)
    for f in frames["frames"]:
        pts = []
        for e in f["events"]:
            if e["k"] not in ("shot", "explosion", "squad_destroyed"):
                continue
            d = e.get("d") or {}
            p = d.get("src_pos") or d.get("pos") or d.get("dst_pos")
            if isinstance(p, list) and len(p) == 2:
                pts.append(p)
        if pts and len(pts) > best_n:
            best_t, best_n = f["t"], len(pts)
            best_pos = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
    return best_t, best_pos, best_n


def measure_fps(pg) -> float:
    return pg.evaluate(
        """(() => new Promise(res => {
          let n = 0; const t0 = performance.now();
          function tick() {
            window.__viewer.redraw(); n++;
            if (performance.now() - t0 < 2000) requestAnimationFrame(tick);
            else res(Math.round(n / ((performance.now() - t0) / 1000) * 10) / 10);
          }
          requestAnimationFrame(tick);
        }))()"""
    )


if __name__ == "__main__":
    raise SystemExit(main())
