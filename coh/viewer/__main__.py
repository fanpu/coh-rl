#!/usr/bin/env python
"""Build a replay's frame stream and serve the canvas viewer.

    uv run python -m coh.viewer match.replay.json --data-dir tests/data/fixtures

Serves the static page from `viewer/` and the frame stream at `/frames.json`
(gzip-encoded, built once into memory at startup). With `--no-serve` the frame
stream is written to `--out` instead, which is handy for tests and for
shipping a replay as a single file.
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import time
from pathlib import Path

from coh.replay.replay import load
from coh.viewer.frames import build_frames, frames_json_gz

STATIC_DIR = Path(__file__).resolve().parents[2] / "viewer"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m coh.viewer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("replay", help="path to a replay JSON file")
    parser.add_argument("--port", type=int, default=8000, help="port to serve on (0 = pick a free one)")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="stat tables directory (default: the replay's own `data_dir`, else the packaged tables)",
    )
    parser.add_argument("--every", type=int, default=2, help="emit a frame every N ticks (default 2)")
    parser.add_argument("--no-serve", action="store_true", help="write the frame stream instead of serving it")
    parser.add_argument("--out", default=None, help="with --no-serve: gzip JSON output path")
    return parser.parse_args(argv)


def make_handler(payload: bytes) -> type[http.server.SimpleHTTPRequestHandler]:
    """A static handler for `viewer/` that also answers `/frames.json`."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def do_GET(self):  # noqa: N802 - stdlib naming
            if self.path.split("?")[0] in ("/frames.json", "/frames.json.gz"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def end_headers(self):
            # The page and its assets are rebuilt constantly during
            # development; never let a browser serve a stale viewer.js.
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
            pass  # the startup banner is the only output we want

    return Handler


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        replay = load(args.replay)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: cannot read replay {args.replay}: {e}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    frames = build_frames(replay, args.every, data_dir=args.data_dir)
    payload = frames_json_gz(frames)
    build_s = time.perf_counter() - started
    print(
        f"built {len(frames['frames'])} frames from {args.replay} "
        f"in {build_s:.1f}s ({len(payload) / 1e6:.2f} MB gzipped)"
    )

    if args.no_serve:
        out = Path(args.out or "frames.json.gz")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        print(f"wrote {out}")
        return 0

    if not STATIC_DIR.is_dir():
        print(f"error: static viewer directory not found at {STATIC_DIR}", file=sys.stderr)
        return 1

    with _Server((args.host, args.port), make_handler(payload)) as httpd:
        print(f"viewer at http://{args.host}:{httpd.server_address[1]}/  (ctrl-c to stop)", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
