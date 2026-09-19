"""Replay viewer: turns a replay into frame JSON and serves the canvas app.

`coh.viewer.frames` re-simulates a replay and snapshots it into a compact,
JSON-serializable frame stream; `python -m coh.viewer` builds that stream and
serves it (gzipped) next to the static page in `viewer/`.

The browser app consumes only the frame JSON — it never imports or mirrors
simulation logic.
"""

from coh.viewer.frames import build_frames, frames_json_gz

__all__ = ["build_frames", "frames_json_gz"]
