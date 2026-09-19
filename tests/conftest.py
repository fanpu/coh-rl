"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: wall-clock performance assertions")
    config.addinivalue_line("markers", "slow: whole-match runs (seconds, not milliseconds)")


@pytest.fixture
def no_combat(monkeypatch):
    """Stub the combat system out for this test.

    Tests that park hostile squads on top of each other (territory capture,
    movement halting, ...) are isolating one system; without this they would
    also be testing whether the squads shoot each other to death first.
    """
    from coh.sim.systems import combat

    monkeypatch.setattr(combat, "run", lambda sim: None)
