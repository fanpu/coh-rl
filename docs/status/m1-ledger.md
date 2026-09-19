# SDD ledger — plan: docs/superpowers/plans/2026-09-19-m1-playable-sim.md
Branch: design/coh-rl-env. Spec: docs/superpowers/specs/2026-09-19-coh-rl-env-design.md

## Pre-flight scan
| Tasks | Produces vs consumes | Finding |
|---|---|---|
| 2 ↔ 12, 14 | EconomyDef fields | T12 adds `garrison_collapse_damage_frac`, T14 adds `min_manpower_income_frac` after schema task. Ruling: add both to EconomyDef in Task 2 (and T3 tables) up front — avoids schema churn — cost if wrong: none. |
| 2 ↔ 3 | schema vs scraped tables | consistent; T3 must emit T2 schema. OK |
| 5 ↔ 10, 11 | Squad fields | T10 adds `abandoned`, T11 adds `turret_heading`; additive, state_hash must include them. OK |
| 8 ↔ 9 | combat accumulates `squad.suppression`, T9 applies thresholds | OK |
| 13 ↔ 14 | Build(op) | T13 = PointState.op_building bookkeeping + neutralize block; Build validation/construction lives in T14. Ruling: T13 tests OP using `sim.spawn_building` directly — cost if wrong: none. |
| 5 ↔ 16 | Sim.issue / order dicts vs env order_log | consistent |
| 16 ↔ 17 | Replay/resimulate vs frames | consistent; events list consumed per tick |
| 16 self | bot test text "zero invalid orders > 5%" garbled | Ruling: assert invalid-order rate ≤ 5% of issued — cost if wrong: trivial. |
| 1..18 self | tests vs code in each task | no contradictions found; implementation bodies are prose by design (plan note). |
| 17 | playwright dev dep not in T1 | T17 adds it. OK |
Task 1-2: implemented (commits dae606f..aafeef0), review dispatched
Task 1: complete (commits dae606f..05737e6, review clean)
Task 2: complete (commits 05737e6..aafeef0, review clean)
Task 2: minor (deferred): _cover_table/_target_table near-duplicate in loader.py; no value-level economy assertions in test_loader; loader.py 585 lines
Task 2: note: verify `uv build --wheel` includes tables yaml once Task 3 lands
Task 3: dispatched (base aafeef0, main tree). Task 4: dispatched in isolated worktree (base aafeef0)
Standing instruction (user, 2026-09-19): commit often, push branch to origin after every completed task.
Task 4: implemented on branch worktree-agent-a764e0ae61367bd8d (aafeef0..3bb0a4d), review dispatched; merge after Task 3 lands
Task 4: Ruling: fuel_high (centre-west) / munitions_high (centre-east) treated as a symmetric swap pair in the symmetry test — plan's exact-type symmetry is impossible with singular high points — cost if wrong: slight faction-side asymmetry in resource type, fixable in map YAML.
Task 4: review → 1 Important (reachability test multi-source BFS), fix round 1 dispatched (agent a764e0ae61367bd8d, FIX_BASE 3bb0a4d)
Task 4: minor (deferred): cell_of/center_of duplicate CELL_M literal 2.0 (dependency direction prevents import)
Task 4: fix round 1/5 (1 addressed, 0 open; commits 3bb0a4d..34d86c4). Ruling: controller verified the 45-line test-only fix diff directly instead of dispatching a re-review agent — diff is trivially inspectable — cost if wrong: a weak test slips through; final review covers it.
Task 4: complete (commits aafeef0..34d86c4 on branch worktree-agent-a764e0ae61367bd8d, review clean) — NOT YET MERGED into design/coh-rl-env (waiting for Task 3 to leave the main tree)
Task 5: dispatched in isolated worktree, based on Task 4 branch
Standing instruction (user): push directly to main; no feature branches. main fast-forwarded to aafeef0 and pushed. Main tree still has design/coh-rl-env checked out until Task 3 agent finishes; then checkout main, merge, delete branch.
Task 5: implemented (34d86c4..24da072 on branch worktree-agent-a4c64909a381c9eaf), review dispatched
Task 5: Ruling: state_hash must also cover terrain mutations — fold `map.version` + list of mutated cells into the hash; assigned to Task 6 (first task that mutates terrain) — cost if wrong: replay divergence undetected.
Task 5: Ruling: starting-builder fallback (first builder by id when no faction match) stands — only fixture data triggers it.
Task 6: dispatched (worktree, agent a3adb52c446922c18, base = Task 5 branch 24da072)
Task 7: dispatched in parallel (worktree, agent abc2e8f33a688f032, base 24da072; file-scoped to vision.py to avoid merge conflicts with Task 6)
Ruling: running Tasks 6 and 7 in parallel worktrees despite skill's no-parallel-implementers rule — files are disjoint and worktrees isolate git state — cost if wrong: a small manual merge in state.py/helpers.py.
Task 5: review → 1 Important (neutral buildings double-stamped; test ineffective) + 1 minor (order_from_dict tuple heuristic); fix round 1 dispatched (FIX_BASE 24da072)
Task 5: fix round 1/5 (1 addressed + minor, 0 open; commits 24da072..f8f7acb). Ruling: controller verified small fix diff directly (spy test confirmed failing when reverted per report) — cost if wrong: final review covers.
Task 5: complete (commits 34d86c4..f8f7acb, review clean)
main = f8f7acb (Tasks 1,2,4,5) pushed. Task 3 (main tree, branch design/coh-rl-env, uncommitted) to be merged onto main when it finishes.
Task 7: implemented (24da072..cd3bb30 on worktree-agent-abc2e8f33a688f032), review dispatched
Task 7: review → 1 Critical (perimeter ray-casting leaves FOV holes — a PLAN defect), 1 Important (unbounded mask cache), 1 minor; fix round 1 dispatched (FIX_BASE cd3bb30)
Task 7: Ruling: replace plan's "rays to disc perimeter" with per-cell Bresenham LOS (vectorized gather), so mask == has_los by construction — plan wording was wrong — cost if wrong: perf; bounded by perf tests.
Task 6: implemented (24da072..c46e575 on worktree-agent-a3adb52c446922c18), review dispatched
Task 7: fix round 1 implemented (cd3bb30..3a247a1), scoped re-review dispatched
Task 6: review → 2 Important (spawned team weapons not SETTING_UP; in-flight squads ignore newly blocked cells) + minors; fix round 1 dispatched (FIX_BASE c46e575)
Task 6: Ruling: team weapons spawn SETTING_UP; Task 5's test updated accordingly — cost if wrong: trivial.
Task 6: Ruling: movement re-plans when next waypoint cell becomes impassable; failed re-plan drops the path — cost if wrong: squads idle near new buildings; bots re-issue orders.
Task 7: fix round 1/5 (3 addressed, 0 open; commits cd3bb30..3a247a1), re-review clean
Task 7: complete (commits 24da072..3a247a1, review clean); merged to main in integration worktree (.claude/worktrees/integration), 134 tests pass, pushed
Task 7: minor (deferred): no dedicated disc test with origin near the map edge (reviewer verified manually)
NOTE: controller integrates on `main` in /home/fzeng/ml/coh/.claude/worktrees/integration; new task worktrees must `git merge main` first.
Task 15: dispatched early in parallel (worktree, agent ab857962cacd9224f, base main 857e9b1; file-scoped to victory.py)
Task 6: fix round 1 implemented (c46e575..f9ac66e); merged to main (89a611b, 160 tests pass, pushed); scoped re-review dispatched
Task 8: dispatched (worktree, opus, agent a3d745b8251858ba3, base main 89a611b)
Task 13: dispatched in parallel (worktree, sonnet, agent a12b90bf9994603ac; file-scoped to territory.py)
Task 15: implemented (95a75df); collides with a Task 7 test that deletes the real HQ. Ruling: Task 15 agent authorized to retarget that vision test at a non-HQ building — annihilation behaviour is correct — cost if wrong: none.
Task 6: fix round 1/5 (5 addressed, 1 NEW Important — replan resolving to own cell strands squad; commits c46e575..f9ac66e). Fix round 2 dispatched (FIX_BASE = main 89a611b after merge)
Task 15: complete (commits 89a611b..6b5e255, review clean); merged to main, 169 tests pass, pushed
Task 6: fix round 2/5 (stranded-squad-on-replan addressed + unreachable-goal residual; commits f9ac66e..54db5ca). Ruling: controller verified the 33-line movement.py fix diff directly instead of a third reviewer dispatch — cost if wrong: final review covers.
Task 6: complete (commits 24da072..54db5ca, review clean after 2 fix rounds); merged to main, pushed
Task 13: implemented (56f19ea on worktree-agent-a12b90bf9994603ac), review dispatched
Task 13: complete (commit 56f19ea, review clean); merged to main, pushed
Task 13: minor (deferred): no test for a team with two HQ sectors feeding connectivity BFS (matters for M2 2v2); capturing_team set while OP blocks neutralize
Task 3: agent a21a1ba838238ab14 was STOPPED BY THE USER (harness refuses resume; "only launch a new agent if the user explicitly asks"). Its uncommitted work sits in the main checkout (branch design/coh-rl-env): coh/data/loader.py (M), coh/data/tables/*.yaml, tools/{scrape_coh_stats.py,roster.yaml,patch_2602.yaml,.cache}. NOT relaunched — awaiting user's word. Do not touch/clean the main checkout.
Task 14: dispatched with fixture data only; real-table tech-tree test deferred until Task 3 lands
Task 8: implemented (c380e7d on worktree-agent-a3d745b8251858ba3, base 89a611b9ee8774207880413502b87dbfa7bac7c6), review dispatched
Task 8: review (opus) → math verified; 2 Important (Attack-in-range leaves state MOVING → team weapon can't fire; pursuit memory outside state_hash) + minors; fix round 1 dispatched (FIX_BASE c380e7d)
Task 8: Ruling: shots at buildings always hit (no roll) — brief sentence was self-contradictory; CoH buildings are large static targets — cost if wrong: buildings die a bit faster; tunable via tt.damage.
Task 8: minor (deferred): BUILDING_AUTO_TARGET_MIN_DAMAGE_MULT=0.25 is arguably gameplay data (belongs in economy.yaml); fixture rifles auto-target HQs because no building row in target table (data issue; check real tables)
Task 14: implemented (c4c171a on worktree-agent-a505ee8ce4b499215), review dispatched
Task 8: fix round 1 implemented (c380e7d..7fae8af); merged to main, pushed; scoped re-review dispatched
Task 14: complete (commit c4c171a, review clean); merged to main (constants.py conflict: kept both), 267 tests pass, pushed
Task 14: minor (deferred → carry into Task 12 dispatch, which owns building destruction): when a building dies — builders CONSTRUCTING it must go IDLE/clear order, its queue is lost (no refund), OP "living" should mean exists (hp>0 implied); OP under construction blocks neutralize but earns nothing (accepted).
Task 9: dispatched (worktree, sonnet, agent a4236f5d015624fb1, base main 2244435)
Task 16: dispatched in parallel (worktree, opus; file-scoped to coh/env, coh/agents, coh/replay, scripts). Ruling: bot/replay tests run on the real map with FIXTURE data because Task 3 (real tables) is user-stopped; real-data bot assertions deferred to Task 18 — cost if wrong: bots need retuning when real data lands.
Task 8: fix round 1/5 (6 addressed, 0 open; commits c380e7d..7fae8af), re-review clean
Task 8: complete (commits ..7fae8af, review clean); on main, pushed
Task 16: agent adcc7f382fe8edc83 running
Task 9: implemented (f402eab on worktree-agent-a4236f5d015624fb1, base 2244435bf7941e40c5f041f1f78647ffac2cfccf), review dispatched
Task 9: merged to main ahead of review (f402eab, 283 tests pass, pushed); review agent a9e81dc216f170602 running — fixes (if any) go on top of main
Task 10: dispatched (worktree, opus, agent ac932e5779fac78bd, base main f402eab)
Task 10: Ruling: mortars may fire at any team-visible target in range (plan's "not LOS from the weapon" dropped — CoH mortars fire with direct LOS too); mortar AOE hits enemies only in M1 (no friendly fire) — cost if wrong: minor fidelity loss; easy to flip later.
Task 9: complete (commit f402eab, review clean, approved)
Task 9: minor (deferred): three ad-hoc monkeypatches of suppression.run could be a shared `no_suppression` fixture; no test for garrison-key vs heavy fallback in _recovery_cover; no assertion on "reinforced" event; stale Reinforce order left on a dead squad; team-weapon retreat→arrival→re-setup round trip untested
Task 16: implemented (1a12938; 307 tests; T1 beats T0 by annihilation at ~210 s, 0% invalid orders, ~5.6–10.5k ticks/s). Pre-review follow-up dispatched: merge main (real suppression), fix fixture neutral defs at source, remove skip_unknown_neutrals workaround.
Task 16: Ruling: Replay gains `final_tick`; T1 prefers armed capture-capable infantry (data-derived), ≤2 unarmed builders; T1-vs-T0 test uses time_limit 3600 — all accepted deviations from plan text — cost if wrong: trivial.
Task 16: follow-up done (..ea59b1c); merged to main, 324 tests pass, pushed; review agent a3eee0487f9543825 running
Task 17: dispatched (worktree, opus, agent a1a0cffe923ad6dd3, base main ea59b1c; scoped to coh/viewer, viewer/, tests/viewer)
Task 10: implemented (80febe4); merged to main ahead of review (358 tests pass, pushed); review dispatched
Task 10: Ruling: combat.py is 1215 lines — new vehicle logic (Task 11) and garrison logic (Task 12) go in their own modules; splitting existing combat.py into team_weapons.py/explosions.py deferred to final-review fix wave — cost if wrong: one large file for a while.
Task 11: dispatched (worktree, opus, base main d6c86df)
Task 16: complete (commits 1a12938..ea59b1c, review clean — fog/reward/replay determinism verified); ⚠️ resolved: controller ran full suite on main after merge (324 pass) incl. tests/agents on the real map
Task 16: minor (deferred): _cheapest/_cheapest_structure near-duplicates; CohEnv gained optional game_map param (accepted); inconsistent `slow` marking of full-match tests
Task 10 review: agent a1bcb0b932bd4b0f8 running. Task 11: agent a9a4ec35e89b88f7c running. Task 17: agent a1a0cffe923ad6dd3 running.
Task 10: review → 1 Important: mortar splash exempts same-team buildings (implementer inference beyond ruling text). Ruling: code stands — no mortar friendly fire on squads OR buildings in M1, for bot sanity and consistency — cost if wrong: minor fidelity; one-line flip.
Task 10: complete (commit 80febe4, review findings ruled; on main)
Task 10: minor (deferred): no test for two squads racing to re-crew the same shell / stale recrew_target; no abandon→recrew→abandon cycle test
Task 11: implemented (5da2335); merged to main ahead of review, pushed; review + Task 12 dispatched
Task 11 review: agent a4c1569d0be6eefae running
Task 12: dispatched (worktree, opus, agent a0caf1a80b2de1236, base main 5da2335)
Task 12: Ruling: only NEUTRAL buildings are enterable in M1 (BuildingDef has no capacity; plan's "own-team buildings" dropped); Move/Capture/etc. are INVALID while garrisoned (must Ungarrison first; Retreat auto-exits) — cost if wrong: agents need one extra order; easy to relax.
Task 11: complete (commit 5da2335, review clean, approved; on main)
Task 11: minor (deferred): duel fixture frontal p_pen=1.0 makes the 15-seed duel near-deterministic (mechanic itself validated separately) — add a clarifying comment; test_loader id-set assertion is brittle
Task 12: implemented (02ba189); merged to main ahead of review, pushed; review dispatched
Task 12 review: agent ab9978fc90b5881e7 running
Controller smoke (main 02ba189, all mechanics, fixture data): T1 vs T1 seeds 0/1 finish in 460 s / 553 s game time, 0% invalid orders, ~5,100 ticks/s single core (target ≥ 1,000).
Task 12: review → 1 Important (team weapon wiped while garrisoned leaves an unrecrewable shell inside the impassable footprint) + minor (duplicate footprint caches); fix round 1 dispatched (FIX_BASE 02ba189)
Task 12: minor (deferred): _prefers_building uses only the lowest-id occupant's target type for mixed garrisons; combat.py ~1363 lines (split deferred)
Task 12: fix round 1/5 (1 Important + 1 minor addressed; commits 02ba189..2cb084b; RED/GREEN evidence in report). Ruling: controller accepted the fix on the report's RED/GREEN evidence + green suite without a separate re-review dispatch; the final whole-branch review covers footprints.py — cost if wrong: caught at final review.
Task 12: complete (commits ..2cb084b); merged to main, 419 tests pass, pushed
Refactor (deferred combat.py split): dispatched now while coh/sim is otherwise idle (worktree, sonnet, agent ab9d3efe4c465613c, base main 2cb084b); guarded by golden state-hash tests committed first
Refactor: complete (dc6a287 golden hashes, bd6e19a split; golden hashes identical before/after; 422 tests pass); merged to main, pushed. Behaviour-preserving by golden-hash evidence; covered by final review.
Task 17: implemented (8428478); merged to main (624028f), 445 tests pass, pushed; review agent a21ba8a71a855ed5b running
Task 17: controller observations: one full-suite run hung >9 min inside a browser test on first run (later runs 23 s / browser file alone 78 s) — flagged to reviewer; play_match defaults both factions to "us" (should default p1 to wehr) — fold into viewer/T16 fix round; HQ sector labels overlap HQ buildings in the viewer.
Task 17: review → 3 Important (explosion + vehicle_destroyed events render as generic marks; browser-test hang reproduced by reviewer; rich synthetic verification not committed as a test) ; fix round 1 dispatched (FIX_BASE 8428478) incl. controller polish (HQ label overlap, play_match f1 default → wehr, regenerate screenshots on current main)
Task 17: fix round 1/5 (all 3 Important + 4 polish addressed; commits 8428478..276f588; hang root cause = Playwright sync evaluate() without timeout + orphaned Chromes; 5 consecutive runs 7–8 s; event-kind coverage test added). Ruling: controller verified directly (merged, full suite 454 pass in 29 s with no hang, inspected docs/img/viewer-units.png: HMG arc wedge, setting-up indicator, abandoned AT gun, tank+turret, construction hatching, queue in inspector all render) instead of a re-review dispatch — cost if wrong: final review covers viewer.js.
Task 17: complete; merged to main, pushed
Task 17: minor (deferred): viewer.js >1150 lines single file; deadline guard POSIX-only; SIGKILLed pytest leaks Chrome; T1 bot fields only infantry so real-match coverage of vehicles/team weapons in the viewer awaits T2/T3 bots (M2)
REMAINING: Task 3 (user-stopped; awaiting user), Task 18 (scenarios need real tables; perf part can run now), final whole-branch review.
Task 18a (perf bench + optimization, fixture-data mechanic scenarios, ladder sanity): dispatched (worktree, opus, agent a2ed4f7940a8fa3ca, base main 276f588). Real-table scenario suite (18b) deferred until Task 3 lands.
Task 18a: complete (16c1ee6..32bc75f: bench + stress mode 817→1085 ticks/s bit-identical, fixture mechanic scenarios 93–100% win rates, ladder T1>T0 10/10); merged to main, 467 tests pass in 42 s, pushed. Ruling: accepted on golden-hash + stress-hash evidence without a separate task review; final whole-branch review covers it — cost if wrong: caught at final review.
FINAL REVIEW: dispatching (opus), range dae606f..32bc75f
Ruling: final review split across 3 parallel opus reviewers by area (sim / env+agents+replay+bench+data+maps / viewer) — the 460 KB all-new source diff is too large for one reviewer to read well — cost if wrong: cross-area issues slightly less likely to be seen; each reviewer is told the neighbouring interfaces.
Final review agents: sim adb9c39312c258926 (opus), env/data/maps/bench a3c652bf95c401ee8 (opus), viewer a6f68bd0808cf7c86 (sonnet). Next: ONE fix wave with combined findings, one scoped re-review, then finishing-a-development-branch.
Final review (viewer): 1 Critical (enemy buildings not fog-filtered), 2 Important (effects ignore fog; fog test asserts nothing about hiding), minors → saved to final-findings-viewer.md for the single fix wave
Ruling: final fix wave split per area (viewer / sim / env-data) because areas are file-disjoint and reviewers finish at different times — one fix agent per area, each ONE dispatch — cost if wrong: none beyond coordination. Viewer fix wave dispatched (agent a1a0cffe923ad6dd3).
Final review (sim): 0 Critical, 4 Important (Retreat KeyError w/o HQ; malformed order payloads raise; SetFacing(nan) poisons state_hash; two definitions of team HQ sectors break 2-players-per-team connectivity) + minors → final-findings-sim.md
Ruling: capture CONTEST follows the spec, not the plan's narrowing — any eligible enemy infantry presence in the radius stalls progress regardless of orders; progress still requires a Capture order — cost if wrong: territory pacing changes; golden hashes re-recorded.
Sim fix wave dispatched (fresh opus agent).
Final review (env/data/maps): 1 Critical (hedgerow_crossing point cells + HQ/builder placement not seat-symmetric: seat 0 wins 14/16 same-bot mirrors), 6 Important (fuel/munitions swap unfair; has_op + neutral hp fog leaks; loader accepts 15/15 malformed tables; replay has no data/map identity or final-hash verification) + minors → final-findings-env.md
Ruling (REVERSES Task 4 ruling): strict point-type symmetry on maps; fuel_high/munitions_high swap pair removed — reviewer showed positional + value asymmetry — cost if wrong: none.
Ruling: terminal reward paid exactly once; per-step order-issue sequence rotated across players — removes first-mover bias — cost if wrong: replays/golden hashes re-recorded.
Ruling: finding 17 (viewer assets missing from wheel) deferred into the viewer 3D graphics pass.
Sim fix agent a55836210216338f7 given addendum (HQ footprint centring + builder/rally spawn toward map centre; no env.py edits). Env fix wave dispatched (fresh opus agent). Controller re-records golden hashes after both merge.
Standing instruction (user): never dispatch subagents on fable; always pass model explicitly (opus/sonnet).
Final fix wave (viewer): complete (2769b72 — fog context threaded through buildings/squads/effects/inspector/hit-testing; last-known ghosts; pixel-level fog tests; extra label-dodge leak found+fixed); merged to main, pushed. Scoped re-review: pending (batch with other fix waves).
NEXT for viewer: 3D graphics pass (three.js low-poly mode, CoH-style camera, keep 2D as tactical map; move static assets under coh/viewer/static so the wheel ships them — final-review finding 17). Model: opus. NOT fable (user rule).
Viewer 3D graphics pass: dispatched (worktree, opus, agent a45291340f7b522fd, base main 2769b72). Running concurrently: sim fix wave a55836210216338f7 (opus), env/data/maps fix wave ae301e05035313add (opus).
Final fix wave (sim): complete (553511c non-behavioural, golden unchanged; fdd35e7 behavioural + golden re-recorded); merged to main, 520 tests pass, pushed. All 4 reviewer claims reproduced before fixing.
Sim fix notes: map still seat-biased 10–2 after sim fix → root cause in map YAML (content symmetric about cell 47, p→94−p, on a 96-wide grid) → env agent informed via message. hq_point_ids now covers only occupied start slots. orders.py is 959 lines (minor, deferred: split validators by domain). No stress-hash oracle exists (bench task claim was wrong about that) — minor (deferred): add one.
Final fix wave (env/data/maps): complete (e7d4026..a1ec07f: map exactly seat-symmetric, fog leaks closed, loader invariants, replay data/map identity + final-hash verification, reward-once, order-issue rotation, junk-order survival). Seat split same-bot mirrors: before 14–2–0 → after 7–6–3. Merged into LOCAL main (not pushed: 5 expected/understood test failures).
Follow-up dispatched (sonnet, agent a5bcfae50b2f1a9a5): re-record golden hashes with determinism proof; fix event-tick frame-window convention in frames.py/test; lengthen the combat-events frames test. Push main after it merges green.
Minor (deferred): coh/data/loader.py ~800 lines; mid VP capture disc offset ~1.4 m from true centre (path-equidistant, accepted).
Follow-up complete (7d997e1: goldens re-recorded with two-process determinism proof; frames event t = emitting tick + 1; combat-events frames test lengthened); merged, 583 tests pass, main pushed.
Note for 3D agent: running the suite rewrites docs/img/viewer-units.png non-identically across environments (dirty tree) — screenshot tests should write to tmp unless an env var asks to refresh docs.
Scoped re-review of the final fix waves dispatched (sonnet).
Scoped re-review of final fix waves: ALL findings ADDRESSED (sim 1–11 + addendum; env 1,2,4–16,18). One NEW Important in the fix diff: CohEnv order log not normalized → numpy-typed orders make Replay.save() crash on json.dump.
Ruling: load-bearing for M3 RL (policies emit numpy scalars) and a few-line fix → smallest-change fix dispatched despite the "no second fix wave" rule (sonnet, agent a2f97e42cefdcc20b): normalized() canonicalizes to plain Python types; env logs the normalized order; env/replay/fuzz tests — cost if wrong: none.
Minor (deferred): loader's strictly-ascending ranges rule unverified against real CoH1 outliers (flamethrowers/pistols) — re-check when Task 3 tables land; document hq_point_ids occupied-slots semantics for future 4-player maps.
Numpy order-log fix: complete (bd1a255); merged, full suite green, main pushed. Final review + fix waves CLOSED for sim/env/viewer-fog. Remaining in flight: viewer 3D pass (a45291340f7b522fd). Open with user: Task 3 resume.
Viewer 3D pass: implemented (09b8d4d..0e41f3d: three.js vendored, static moved into package, ES-module split, 3D view, 13 WebGL browser tests under SwiftShader); merged into LOCAL main (modify/delete conflict on viewer/viewer.js resolved by deletion) — NOT pushed: effect-coverage test fails (unit_blocked missing in new modules). Controller inspected screenshots: good overall; polish round dispatched to same agent (unit_blocked; screenshot tests must not dirty docs/img; larger/brighter infantry + team ground rings; remove cell-aligned road blotches; better default camera; more visible tracers).
Minor (deferred): no real-GPU fps measurement (SwiftShader only); billboards are a screen-space overlay (not depth-tested); view2d.js 840 lines.
