"""Locate a Chromium/Chrome binary for the headless viewer check.

Playwright's own browsers are not downloaded in this repo (`playwright
install` is deliberately not run); instead we point Playwright at whatever
Chromium is already on the machine. Returns `None` when there is none, so
callers can skip cleanly.
"""

from __future__ import annotations

import glob
import os
import shutil

_BUNDLE_GLOBS = (
    "/opt/pw-browsers/chromium*/chrome-linux*/chrome",
    "/opt/pw-browsers/chromium*/chrome-linux*/headless_shell",
)
_SYSTEM_NAMES = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome")


def find_chromium() -> str | None:
    """Path to a usable Chromium/Chrome executable, or None."""
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH"), "/opt/pw-browsers"]
    for root in roots:
        if not root:
            continue
        for pattern in _BUNDLE_GLOBS:
            pattern = pattern.replace("/opt/pw-browsers", root, 1)
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]
    for name in _SYSTEM_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None
