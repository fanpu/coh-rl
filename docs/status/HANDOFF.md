# CoH-RL — handoff / current state

Last updated: 2026-09-19 (end of the M1 execution session). Read this first when
resuming in a new conversation.

## What this project is

A headless, deterministic, mechanics-faithful 2D clone of Company of Heroes 1
(US vs Wehrmacht) used as an RL environment and as a benchmark for API-driven decision
models (TypeSafe AI's **Jev**, LLMs) against scripted bots and RL policies.

- Spec (binding): `docs/superpowers/specs/2026-09-19-coh-rl-env-design.md`
- M1 plan: `docs/superpowers/plans/2026-09-19-m1-playable-sim.md`
- Data sources survey: `docs/data-sources.md`
- Full run ledger (every task, review, fix round, ruling): `docs/status/m1-ledger.md`
- Final-review findings files: `docs/status/final-findings-{sim,env,viewer}.md`

## Working agreements (from the user)

- Commit often, push straight to `main`; no feature branches / PRs unless asked.
- Subagents: `opus` for hard work, `sonnet` for reviews and mechanical tasks.
  **Never use `fable` for subagents.** Always pass the model explicitly.
- The user wants CoH-like **3D-looking** visuals (the sim stays 2D).
- Process used: superpowers brainstorming → spec → plan → subagent-driven development
  (fresh implementer per task in an isolated git worktree, task review, fix rounds,
  final whole-branch review, one fix wave, scoped re-review).

## State of `main`

All 18 plan tasks except Task 3 are implemented, reviewed and merged, plus a final
whole-branch review (3 reviewers), its fix waves, and a scoped re-review.

| Area | Status |
|---|---|
| Scaffold, data schema + loader (now with strict cross-validation) | done |
| Map format, directional cover, `hedgerow_crossing` (exactly seat-symmetric; same-bot mirrors 7–6–3) | done |
| Sim core: state, typed orders (payload-hardened, fuzz-tested), tick loop, `state_hash` | done |
| Pathfinding/movement, vision/fog (per-cell LOS), combat (split into combat/ballistics/explosions/team_weapons/destruction), suppression/retreat/reinforce, team weapons + mortars + re-crewing, vehicles (penetration, rear arc, turrets, wrecks), garrisons, territory/supply (spec contesting), economy/production/tech/upgrades, victory | done |
| Env (PettingZoo-style parallel, fog-filtered observations, order-issue rotation, reward paid once), T0/T1 bots, replays (data/map identity + final-hash verification), `scripts/play_match.py`, `coh/env/match.py` | done |
| 2D tactical-map viewer with correct team fog (buildings ghosts, effects filtered) | done |
| Bench (`scripts/bench_sim.py`): ~1,085 ticks/s stress (60 squads), ~7,000 ticks/s bot matches, single core | done |
| Scenario tests (fixture data), ladder sanity (T1 > T0 10/10), seat-fairness test, golden state hashes | done |
| 3D viewer (three.js vendored, ES modules, static assets inside the package; `V` toggles 3D / 2D tactical map; same fog rules in both) | done |
| **Task 3 — real CoH1 stat tables** | **NOT on main** (see below) |

Everything runs on the fixture tables in `tests/data/fixtures` —
pass `--data-dir tests/data/fixtures` to scripts until Task 3 lands.

### Test status

`uv run pytest -q` → **602 passed** (~2 min, includes real-Chrome 2D + WebGL 3D browser
tests under SwiftShader); `git status` is clean after a run (doc screenshots are only
refreshed with `COH_REFRESH_DOCS_IMG=1`). Nothing is known-red.

## In flight when this was written

Nothing. The 3D viewer and its polish round (larger infantry with team rings, smooth
ground texture, HQ→centre opening camera, `unit_blocked` badge, brighter tracers) are
merged and pushed. Unverified: real-GPU frame rate (only software GL was available;
~36 draw calls for a real match, ~75 for the showcase).

## Open decision for the user: Task 3 (real CoH1 stats)

The Task 3 agent (scrape coh-stats.com mirror → `coh/data/tables/*.yaml`) was **stopped
by the user** mid-run and the harness forbids relaunching without an explicit ask. Its
uncommitted work was preserved, unreviewed and untested, on branch
**`wip/task3-scraper`** (commit `023529c`): `tools/scrape_coh_stats.py` (36 KB),
`tools/roster.yaml`, `tools/patch_2602.yaml`, generated `coh/data/tables/*.yaml`
(weapons.yaml 124 KB), and an edited `coh/data/loader.py` (based on an OLD loader —
do not merge that file; main's loader has since gained strict validation).
To finish Task 3: start from main, bring over `tools/` and `coh/data/tables/`, make the
tables load under main's stricter loader (expect fixes: `generic_target_types`, strictly
ascending ranges for odd weapons, enum kinds, team-weapon slot 0, op_building, reachable
tech tree), write `tests/data/test_real_tables.py` per the plan, then do **Task 18b**:
re-point `tests/scenarios/test_mechanic_outcomes.py` (`SCENARIO_UNITS` + `data` fixture)
at the real tables, re-tune values marked `estimated` only, re-check bot thresholds and
seat fairness on real data, re-record golden hashes if they are switched to real data.

## After M1

- **M2 (benchmark):** JSON/tensor observation encoders + action masks, Jev adapter
  (typed-schema decisions, 70–500 ms; see `docs/superpowers/specs/...` §4–5), generic
  LLM adapter, T2/T3 bots (use team weapons, vehicles, garrisons, flanking, supply cuts),
  match runner + Elo + results tables, 2v2 map and tests (teams-as-lists already
  supported; check the deferred "two HQ sectors" test), veterancy + Kampfkraft, viewer
  live websocket mode + agent-inspection panel, real-time-budget timing mode.
- **M3 (RL):** vectorized multi-process rollouts on the GB10 (20 ARM cores), PPO with
  invalid-action masking vs the ladder + self-play, checkpoints entered into the
  benchmark. Orders from policies may be numpy-typed — already normalized and tested.

## Deferred minors worth doing eventually (full list in the ledger)

- `coh/sim/orders.py` (~950 lines) and `coh/data/loader.py` (~800 lines) are large;
  split by domain. `combat.py` still re-exports ~40 private names for old test imports.
- Several `coh/sim/constants.py` values are really gameplay data (AOE_EDGE_FALLOFF,
  REAR_ARC_DEG, TURRET_TRAVERSE_MULT, BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT, …) → move
  to `economy.yaml`.
- No stress-state hash oracle in the bench; add one.
- Loader's strictly-ascending `ranges` rule is unverified against real CoH1 outliers.
- Viewer: no real-GPU frame-rate measurement yet (only SwiftShader); billboards are a
  screen-space overlay; `view2d.js` ~840 lines; `/frames.json` is omniscient (fog is a
  client-side filter — fine for replays, not for live spectating of hidden-info games).
- Tests: multi-HQ-sector connectivity (2v2), re-crew races, abandon→recrew→abandon
  cycle, garrison-key vs heavy recovery cover, "reinforced" event assertion.

## Rulings made on the user's behalf (what each costs if wrong)

See every `Ruling:` line in `docs/status/m1-ledger.md`. The consequential ones:

1. Vision uses per-cell Bresenham LOS instead of the plan's perimeter rays (plan defect: holes).
2. Shots at buildings always hit (brief was self-contradictory).
3. Mortars may fire with direct LOS; mortar AOE has no friendly fire (squads or buildings) in M1.
4. Only NEUTRAL buildings are enterable; walking orders are invalid while garrisoned (Retreat auto-exits).
5. Team weapons spawn SETTING_UP; movement re-plans around newly blocked cells; failed re-plan drops the path.
6. Capture CONTEST follows the spec (any eligible enemy infantry presence stalls), progress still needs a Capture order.
7. Maps require strict point-type symmetry (reversed an earlier "fuel/munitions swap pair" ruling); HQ footprints are centred on `hq_cell`; builder/rally cells face the map centre.
8. Terminal reward is paid exactly once; per-step order-issue sequence rotates across players.
9. `Replay` gained `final_tick`, `data_hash`, `map_hash`; `resimulate(verify=True)` by default.
10. T1 prefers armed capture-capable infantry (≤ 2 unarmed builders); bot thresholds are fixture-calibrated.
11. Tasks were run in parallel git worktrees and merged to main ahead of review several times to keep throughput; final review split across three reviewers by area; one extra mini fix (numpy order-log) after the scoped re-review because RL needs it.

## How to run things

```
uv sync
uv run pytest -q                                   # full suite (~100 s; needs Chrome for viewer tests, else they skip)
uv run python scripts/play_match.py --map hedgerow_crossing --p0 t1 --p1 t1 \
    --data-dir tests/data/fixtures --out m.replay.json
uv run python -m coh.viewer m.replay.json --data-dir tests/data/fixtures   # V toggles 3D / 2D
uv run python scripts/bench_sim.py --data-dir tests/data/fixtures --stress
```
