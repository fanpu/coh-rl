"""Scripted agents. They see the world only through `coh.env`."""

from coh.agents.base import Agent
from coh.agents.scripted import T0Idle, T1Capper

#: Ladder rung id -> constructor, for `scripts/play_match.py` and the runner.
AGENTS = {"t0": T0Idle, "t1": T1Capper}

__all__ = ["AGENTS", "Agent", "T0Idle", "T1Capper"]
