# coh-rl

A Company of Heroes 1 mechanics clone used as a reinforcement-learning environment: a deterministic tick-based sim (`coh/sim`), a Gym-style wrapper (`coh/env`), and a browser-based replay viewer, driven entirely by data tables scraped/estimated into `coh/data`. See `docs/superpowers/specs/2026-09-19-coh-rl-env-design.md` for the design and `docs/superpowers/plans/2026-09-19-m1-playable-sim.md` for the implementation plan.

Setup and usage:

```
uv sync
uv run pytest
uv run python scripts/play_match.py
uv run python -m coh.viewer <replay>
```
