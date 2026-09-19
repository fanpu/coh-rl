# Viewer — 3D battlefield view

- **Worktree:** `/home/fzeng/ml/coh/.claude/worktrees/agent-a45291340f7b522fd`
- **Branch:** `design/coh-rl-env` (the worktree's own checkout of it)
- `git merge --no-edit main` → "Already up to date"; `tests/viewer/test_viewer_fog.py`
  present. `uv sync` run once.

## Commits

| SHA | Subject |
|---|---|
| `09b8d4d` | `chore(viewer): vendor three.js and move the static page into the package` |
| `fe27d06` | `refactor(viewer): split the viewer into ES modules, 2D behaviour unchanged` |
| `cbe3b5e` | `feat(viewer): add a 3D battlefield view built on three.js` |
| `263c087` | `test(viewer): cover the 3D view in a real WebGL browser, and shoot the docs` |
| `0e41f3d` | `refactor(viewer): drop the viewer exports nothing consumes` |

## three.js

- **Version:** `0.169.0` (r169), MIT.
- **Fetched:** `curl -sSL https://registry.npmjs.org/three/-/three-0.169.0.tgz`,
  once, on the first attempt. Nothing is fetched at runtime.
- **Vendored:** `coh/viewer/static/vendor/three/three.module.min.js` (687 KB)
  plus `LICENSE` and a `README.md` recording the version and the refresh
  command. Only the build file — no `examples/jsm`; the camera controller is
  written locally (`view3d/camera.js`, 186 lines) because the wanted behaviour
  (pitch that flattens with zoom, zoom-toward-cursor, unit follow, map clamp)
  is not OrbitControls' behaviour.
- **Imported** through an import map in `index.html`
  (`"three": "./vendor/three/three.module.min.js"`), i.e. a relative
  ES-module specifier. The page works with no network at all.

## Module layout

Everything moved from a top-level `viewer/` directory to
`coh/viewer/static/`, inside the package.

| file | lines | responsibility |
|---|---:|---|
| `js/boot.js` | 40 | entry point; WebGL probe, then `import()` the 3D module |
| `js/state.js` | 40 | the one mutable app-state object |
| `js/util.js` | 106 | palette, colour maths, hashes, formatting |
| `js/effects.js` | 56 | the renderer-independent event catalogue |
| `js/data.js` | 444 | frames, interpolation, terrain deltas, **fog** + building memory |
| `js/hud.js` | 244 | HUD, banner, timeline, inspector, legend, notice |
| `js/view2d.js` | 840 | the canvas tactical map (behaviour unchanged) |
| `js/main.js` | 440 | boot, playback, input routing, view switching, test hook |
| `js/view3d/index.js` | 259 | the 3D view's public surface; render, pick, input |
| `js/view3d/scene.js` | 194 | renderer, sun + shadows, hemisphere, haze, sky, materials |
| `js/view3d/camera.js` | 186 | the CoH-style camera |
| `js/view3d/terrain.js` | 633 | ground texture, overlays, fog wash, instanced features |
| `js/view3d/models.js` | 414 | skin registry + the procedural default look |
| `js/view3d/geom.js` | 163 | part lists, primitives, `bake()` |
| `js/view3d/entities.js` | 578 | squads, buildings, points, arcs, selection, pick proxies |
| `js/view3d/effects.js` | 379 | tracers, blasts, dirt, smoke, rings, scorch, wrecks |
| `js/view3d/overlay.js` | 177 | screen-space billboards |
| | **5193** | (was 1761 in one file) |

The old 1761-line `viewer.js` is gone. The largest file is now `view2d.js` at
840, which is the old drawing code lifted verbatim.

**The fog is the reason for the split.** `data.js` owns every "may the watched
team see this?" decision — `visibleBuildings`, `visibleEffects`, `squadHidden`,
the per-team building memory — and both renderers consume its answers. Neither
view can grow its own rules. `effects.js` likewise says what an event *is* and
how long it lives; each view decides how to draw that.

## Packaging

`STATIC_DIR` is now `Path(__file__).resolve().parent / "static"`, and
`pyproject.toml` has `[tool.hatch.build.targets.wheel] include =
["coh/viewer/static/**"]`. Verified:

```
$ uv build --wheel -o /tmp/whl        # Successfully built coh-0.1.0-py3-none-any.whl
$ unzip -l /tmp/whl/*.whl | grep -c static/     # 22
$ python -m venv /tmp/wtest && /tmp/wtest/bin/pip install --no-deps /tmp/whl/*.whl
$ /tmp/wtest/bin/python -c "from coh.viewer.__main__ import STATIC_DIR; ..."
/tmp/wtest/lib/python3.14/site-packages/coh/viewer/static
is_dir: True
['index.html', 'js', 'style.css', 'vendor']
three.js bytes: 687458
```

The static handler also pins `.js` to `text/javascript`: ES-module imports
fail hard on a wrong `Content-Type`, and the stdlib takes `.js` from the host's
mimetypes registry.

## What was verified in a real browser, with which GL backend

**Backend: `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0)),
SwiftShader driver)` — a real WebGL2 context, software-rasterised.** Chrome
151 at `/usr/bin/google-chrome`, driven by Playwright 1.63.

Getting there was the fiddliest part of the task. Chrome's SwiftShader is a
Vulkan ICD, and ANGLE selects an X11/Vulkan display whenever `DISPLAY` is set;
that fails with `DisplayVkXcb: xcb_connect() failed` and
`eglInitialize SwANGLE failed`, and the page ends up with **no WebGL at all**.
I tried nine flag combinations (`--use-gl=swiftshader`, `--use-angle=swiftshader`,
`--use-angle=swiftshader-webgl`, `--use-angle=vulkan`, `--ozone-platform=headless`,
`--headless=new`, `--in-process-gpu`, `--disable-vulkan-surface`, headful) — none
helps while `DISPLAY` is set, and none is needed once it is cleared. The harness
therefore launches Chrome with `env=BROWSER_ENV` (the environment minus
`DISPLAY`) plus `--use-gl=angle --use-angle=swiftshader
--enable-unsafe-swiftshader`. The reasoning is written into `conftest.py`.

Driven headless and **looked at** (I read every PNG, iterating over six rounds):

- full-map overview of a real us-vs-wehr match (sectors, roads, tree lines,
  buildings, flags);
- the busiest moment of the match at operational, tactical and low-angle
  zooms — a red-roofed barn with Wehrmacht pioneers firing tracers at a US
  rifle squad crossing the field, two capture points with progress rings;
- the same moment through team 0's fog;
- the same moment as the 2D tactical map, to confirm it is unchanged;
- the synthetic showcase: tanks mid-duel, a halftrack, an HMG dug in behind a
  bocage run covering the road, a mortar team setting up, an abandoned AT gun,
  pinned/retreating/reinforcing/garrisoned infantry, a damaged tank depot, a
  barracks under construction, a cottage collapsing in a dust cloud, fireballs
  with dirt bursts and scorch marks, a burnt-out hull, an observation post, and
  an unknown unit kind falling back to a marker.

Console/page errors across all of it: **none**.

### Bugs the screenshots exposed (all fixed)

1. **`hash2`/`hash3` multiplied two 32-bit integers with `*`.** That overflows
   a double's 53-bit mantissa and silently drops the low bits, leaving the hash
   correlated along x+y. As per-cell tone variation in 2D it was invisible; as
   value noise for the 3D ground it produced broad diagonal bands across the
   whole map. Now `Math.imul`.
2. **The gable-roof geometry was wound inside-out**, so every roof normal
   pointed into the house and roofs rendered as unlit black slabs.
3. **The instanced effect and feature meshes used `vertexColors` materials on
   three.js primitives, which carry no colour attribute.** `USE_COLOR` is
   defined from the material alone, so the unbound attribute read as black and
   multiplied everything to zero — tracers, fire, smoke, dirt, walls, tree
   trunks and canopies were all invisible or black. They now use
   `instanceColor` on materials without `vertexColors`.
4. **Roads were ten cells wide and wandered.** The blur that softens road edges
   spills a cell either way; the threshold was below that spill, so a two-cell
   lane smeared into a brown river that followed the noise.
5. **`readPixels` returned zeros.** The default drawing buffer is not preserved
   across a composite, so a read from a later task is empty. `view3d.readPixels`
   now re-renders in the same task rather than paying for
   `preserveDrawingBuffer`.
6. **Terrain deltas never reached the 3D view** — `applyTerrainTo` was called
   inside `view2d.render`. It moved to `main.render`, so both views roll the
   deltas forward from one place.
7. **`?quality=low` was read too late.** The WebGL renderer is created while the
   3D module is imported, which is before the frame stream is even fetched, so
   the shadow-map budget was already fixed. Quality is now read in `state.js` at
   module load.

## Performance

Measured on **SwiftShader, i.e. a pure software rasteriser on a loaded box** —
there is no usable GPU path for a browser in this environment (no X server, no
`xvfb`, and headful Chrome will not start), so **I cannot report a real-hardware
frame rate.** What I can report:

| scene | draw calls | triangles | software fps @ 1600x950 |
|---|---:|---:|---:|
| real match, operational zoom, shadows on | **20–22** | 47.8 k | 11.1 |
| synthetic showcase (2 tanks, halftrack, 3 team weapons, 4 infantry squads, 6 buildings, 2 blasts, a collapse) | **71** | 65.9 k | — |
| real match, `?quality=low` | 20 | 47.8 k | — |

The draw-call budget is what the 60 fps target rests on, and it is met by
construction: the whole infantry of a faction is one `InstancedMesh`, every
hedgerow/wall/fence/trunk/canopy/crater-rim class is one more (530–596 instances
on `hedgerow_crossing`), and each building, tank, team weapon and flagpole is
baked into a single vertex-coloured geometry. Twenty draw calls and 48 k
triangles with one 2048² shadow map fitted to the visible area is a load
integrated graphics handles comfortably; 11 fps here is SwiftShader, not the
scene. `?quality=low` drops shadows, halves the ground texture (8 → 4 px/cell)
and turns off MSAA; `H` toggles shadows live; the pixel ratio is capped at 2.

I have flagged this as a concern below rather than claim a number I did not
measure.

## Payload

`coh/viewer/static/` total: **868 KB raw, 234 KB gzipped**, of which three.js
is 687 KB raw / ~191 KB gzipped. The app's own code is ~150 KB raw / ~40 KB
gzipped. three.js is only downloaded when the browser actually has WebGL —
`boot.js` probes first and `import()`s the 3D module conditionally, so a
machine without a GPU never pays for it.

Frame stream, for reference: the 283 s real match is 1133 frames, 0.16 MB
gzipped.

## Screenshots

| path | what |
|---|---|
| `docs/img/viewer-3d.png` | real us-vs-wehr match, the busiest moment, 3D |
| `docs/img/viewer-3d-showcase.png` | the synthetic showcase |
| `docs/img/viewer-3d-fog.png` | the same moment through team 0's fog |
| `docs/img/viewer.png` | the 2D tactical map (regenerated, inspector open) |
| `docs/img/viewer-units.png` | the 2D unit/effect showcase (test-regenerated) |

`docs/img/viewer-fog.png` was deleted: it was a stale 2D render nothing
referenced any more, and `viewer-3d-fog.png` tells that story now.

Regenerate with:

```
uv run python scripts/play_match.py --map hedgerow_crossing --p0 t1 --p1 t1 \
    --data-dir tests/data/fixtures --out /tmp/m.replay.json
uv run python -m tests.viewer.shots --replay /tmp/m.replay.json --out /tmp/shots --docs
```

## Tests

`uv run pytest -q` → **489 passed in 63 s**, output pristine.
`uv run pytest tests/viewer -q` → **54 passed in 33 s**.

`tests/viewer/test_viewer_3d.py` — 13 tests, **17–18 s** (budget 40 s), run
three times consecutively, all green, no stray Chrome afterwards:

- boots clean, is the default view, `#cv3` visible and `#cv` hidden;
- a real WebGL2 context renders a **non-blank** canvas — pixels read back out of
  the drawing buffer (variance > 50), plus `drawCalls > 5`, `triangles > 1000`
  and `features > 100`;
- `V` toggles the views and the hash carries `view=2d` / `view=3d`;
- stepping frames changes the rendered image (pixel digest);
- the camera: right-drag rotates, a held `Q` rotates, the wheel zooms in and
  steepens the pitch, `Home` restores the opening framing exactly;
- click selects and opens the same inspector as 2D;
- **team fog against the scene graph** — the rigged scenario from
  `test_viewer_fog.py` replayed: omniscient 4 buildings / 2 units; team 0
  2 live buildings, 1 ghost, 1 unit. The never-seen enemy building is not an
  object in the scene and not a proxy volume, so `pickAt` over it returns null,
  as does the out-of-vision enemy squad, while the own barracks still selects;
- the real match hides something in the 3D team view too;
- the showcase renders every unit kind and effect (10 units, ≥15 soldiers,
  6 buildings, >10 puffs, 1 wreck);
- a terrain delta crushes ten fence panels and grows six crater rims
  (`features` goes down 10 and up 6);
- `?quality=low` reports `shadows: false` and still renders;
- a browser stubbed to refuse WebGL falls back to the 2D map, shows the notice,
  and the tactical map really draws.

All pre-existing tests still pass. The changes they needed:

- `test_viewer_browser.py` and `test_viewer_fog.py` reached into page globals
  (`frames`, `D`, `allEvents`, `W`, `H`), which ES modules do not have; they now
  go through documented `window.__viewer` accessors (`frame()`, `data()`,
  `events()`, `mapSize()`).
- `open_page` takes the view through the same deep link a human would use and
  defaults to `view="2d"`, so the 2D fog **pixel** tests still assert against the
  2D canvas — `test_a_never_seen_enemy_building_leaves_no_pixels` is unchanged
  and still passes.
- `test_effect_coverage.py` reads the shared catalogue (`js/effects.js`,
  `EVENT_EFFECTS`) instead of the old monolith, so a new sim event kind still
  fails the build until *both* views have decided what to do with it.

Skip path re-verified: `PATH=/nonexistent PLAYWRIGHT_BROWSERS_PATH=/nonexistent`
→ **22 passed, 32 skipped in 1.3 s**.

## Self-review findings (fixed before reporting)

- The persistent scorch marks and wrecks were being laid down from the whole
  event log regardless of fog — a genuine leak, since a crater tells you a
  shell landed there. They now check the visibility bitmap **at the tick the
  blast happened**, and the whole set is rebuilt when the fog mode changes.
- Capture pulses were being fed the fog-filtered effect list, so they never
  drew; points get the unfiltered list (sector state is map knowledge in CoH,
  as in 2D).
- The camera test depended on `requestAnimationFrame` firing within a fixed
  250 ms and flaked once in a full-suite run. It now drives the synchronous
  right-drag path and polls for the held-key one.
- A batch of exports nothing consumed (`data.js` fog internals, `hud.js`
  pieces, geometry/colour/camera-limit constants) and a dead `wreckUpto`
  watermark were removed.
- The `docs/img/viewer.png` regeneration initially inherited the previous
  camera and framed the map off-centre; and `viewer-3d.png` inherited a sticky
  pitch offset from the low-angle preset. Both camera presets are now explicit.

## Concerns

1. **No real-hardware frame rate.** Everything here was rendered by SwiftShader;
   there is no X server, no `xvfb`, and headful Chrome will not start in this
   environment, so the "60 fps on integrated graphics" target is met by budget
   (20–71 draw calls, ≤66 k triangles, one fitted shadow map, capped pixel
   ratio) and **not** by measurement. Someone with a display should sanity-check
   it. 11 fps at 1600x950 under a software rasteriser is consistent with a
   comfortable margin on real hardware, but that is an inference.
2. **Key conflict, resolved by deviating from the brief.** `A` is a camera pan
   key (WASD), so the weapon-arc toggle is **`Z`** plus the toolbar button, not
   `A`. Arcs also always show for the selected team weapon. Documented in the
   help overlay and the README.
3. **Billboards are a screen-space overlay canvas, not scene sprites.** Health
   bars, suppression pips, retreat/reinforce icons, garrison badges and names
   are drawn on a transparent 2D canvas over the WebGL view at each anchor's
   projected position. Functionally identical to a camera-facing sprite (these
   are all screen-sized UI), and it keeps them crisp at any zoom for zero draw
   calls and zero texture memory — but it is a deviation from "billboards" read
   literally, and it means they are not depth-tested against the world.
4. **Fogged ground is washed, but objects standing on it are not.** The fog is a
   ground-plane overlay, as in 2D. Enemy units and buildings in fog are removed
   entirely (which is the part that matters), but a *neutral* barn in unseen
   ground is still lit normally. Darkening it would need a per-object fog
   lookup in the material.
5. **`hedgerow_crossing` has scattered single hedgerow cells, not long banks.**
   The bocage reads well where the map actually has runs (and in the showcase,
   which paints one deliberately), but an isolated `H` cell is a lump rather
   than a hedge. That is the map's shape, not the renderer's.
6. **Muzzle flashes have no point light.** Deliberate — the brief called it
   optional and "keep it cheap"; a per-shot light would force a shader
   recompile whenever the light count changes. Flashes are emissive geometry.
7. **`window.__viewer` has grown.** It now carries `scene3dStats`, `state3d`,
   `readPixels3d`, `camera3`, `reload`, `frame`, `data`, `events`, `mapSize`.
   All documented, all small, but it is test surface on the page.
8. **Building memory and the persistent scars are O(events) on a backward
   seek**, the same trade-off `data.js` already made for terrain deltas.
9. **`view2d.js` is still 840 lines.** It is the old code moved verbatim so that
   "2D unchanged" could be verified by the existing tests; splitting it further
   was out of scope for this task.
10. **The showcase asserts `units == 10` and `buildings == 6`**, which are counts
    of the scene it builds; if someone edits `showcase.py` they must edit the
    test. That is deliberate (it is the point of the assertion) but it is a
    coupling worth knowing about.

---

# Polish round — controller feedback

- **Worktree:** `/home/fzeng/ml/coh/.claude/worktrees/agent-a45291340f7b522fd`
- **Branch:** `worktree-agent-a45291340f7b522fd`
- `git merge --no-edit main` → clean (46 files; the sim/env wave, the new
  `coh/env/match.py`, hashing modules and the `viewer.js` modify/delete the
  controller had already resolved).

| SHA | Subject |
|---|---|
| `867610a` | `fix(viewer): handle the new unit_blocked event, and stop tests dirtying docs/img` |
| `c414880` | `feat(viewer): make the 3D battlefield legible at the zoom you actually play at` |

## 1. `unit_blocked`

`EVENT_EFFECTS` gained `unit_blocked: { ticks: 28, mode: 'badge', colour:
'#e2705f', text: 'blocked' }`. There is nothing to draw on the map — the event
*is* that nothing happened — so it is a word over the building that is stuck:
"blocked", blinking on a half-second cycle rather than fading, so it stays
readable to the end of its life rather than going faint exactly when you look
at it. The 2D map draws it in `drawEffects`; the 3D view draws it through the
screen-space overlay, **before** the other labels so it wins the label
collision test against the building's own name.

`test_effect_coverage.py` already read the new module location
(`coh/viewer/static/js/effects.js`, `EVENT_EFFECTS`) from the earlier work — it
was failing on content, not on path — and now passes.

Re-checked by hand against `grep -rhoE 'kind\s*=\s*"[a-z_]+"' coh/sim` plus the
`VEHICLE_DESTROYED` constant on merged main: **19 kinds, all with a deliberate
entry** — `shot`, `explosion`, `squad_destroyed`, `vehicle_destroyed`,
`building_destroyed`, `weapon_abandoned`, `weapon_recrewed`,
`garrison_entered`, `garrison_ejected`, `reinforced`, `unit_trained`,
`unit_blocked`, `construction_started`, `building_completed`,
`research_completed`, `upgrade_bought`, `point_captured`, `point_neutralized`,
`game_over`. The catalogue is shared, so each one is decided once for both
views.

`tests/viewer/showcase.py` fires a `unit_blocked` at the half-built barracks,
and a new test asserts the badge's own colour really lands on the overlay near
that building (sampling `#ov` for red-dominant pixels — the other labels are
cream, so this cannot pass by accident).

## 2. `pytest` no longer dirties the tree

`docs/img/viewer-units.png` was rewritten on every run and is never
byte-identical between environments (fonts, GL backend, driver). `conftest.py`
gained:

```python
REFRESH_DOCS_IMG = os.environ.get("COH_REFRESH_DOCS_IMG") == "1"

def doc_image(name, fallback):          # fallback is the test's tmp_path
    return DOC_IMAGES / name if REFRESH_DOCS_IMG else fallback / name
```

Tests write to `tmp_path`; `tests/viewer/shots.py --docs` honours the same
variable (and defaults to it). Verified:

```
$ uv run pytest -q
602 passed in 116.12s
$ git status --short
$            # empty
```

Run twice from a clean tree, empty both times.

## 3–6. Visual polish

All five points, each looked at in a regenerated screenshot and iterated.

**Infantry legibility.** Figures are drawn at `FIGURE_SCALE = 1.75`. Positions
and formation offsets stay true to the sim; only the model is exaggerated,
which is what every RTS does and what it takes for four men to read as four men
from a camera you would actually play at. The first attempt at the team colour
*lightened* it toward bone and produced pastel toy soldiers; deepening it
instead (`mix(colour, '#2b2b26', 0.34)`) keeps the contrast against grass
without leaving the WW2 palette. The uniform stays muted; the shoulders — the
widest surface seen from above — and a band on the helmet carry the accent,
over an olive-drab (US) or field-grey (Wehrmacht) helmet chosen by faction. An
early version put a large flat plate of accent colour on the helmet brim, which
made every soldier a coloured box; it is now a small dark brim disc.

**Ground rings.** Every squad, vehicle and team weapon stands on a thin
team-coloured ring (one `InstancedMesh`, so the whole field is one draw call).
From a pitched camera that ring is a stronger ownership cue than the unit
itself, half of which is in shadow — in the regenerated shot a US and a
Wehrmacht squad standing ten metres apart are unmistakable. The selection and
hover rings sit on top, brighter and thicker.

**No cell-shaped anything on the ground.** This was the real bug behind the
"dark rectangular blotches". Road wear, canopy shade, crater bowls and water
were all per-cell decisions (`if (c === 'H') out *= 0.7`), which stamps a hard
2 m rectangle on the texture — it reads as a hole and it hands the player the
sim's grid. Each is now a blurred scalar field sampled bilinearly per texel and
blended through `smoothstep`, with the blur width chosen per field: two passes
for a road (a wide soft verge), one for craters and water (two passes flatten a
lone shell crater to nothing — that was visible as donuts of rubble with no
bowl). The dry-grass and ploughed-field mixes are ramps rather than thresholds,
so a field has an edge of scrub rather than a seam where the noise happened to
cross a line. Sector borders stay as lines, drawn at a lower alpha since they
were reading as chalk on a pitch at close range.

**Opening camera.** `camera.opening(frame)` finds the watched team's HQ (or its
first building), places the target a third of the way from there toward the
middle of the map, faces along that axis, and sits at 76 m — about 58 m of
ground on screen, so a 1.8 m figure is ~20 px tall. `Home` still gives the
whole-map view, and `main.start` no longer overrides the 3D framing.

**Tracers.** Additive blending, a slightly thicker round, a short dimmer
afterglow behind it and a brighter muzzle flash. The first pass was too much —
15 m white rods across the field — so the streak and glow were shortened and
narrowed until they read as fast rounds rather than searchlights.

## Tests

`uv run pytest -q` → **602 passed in 116 s**, output pristine, `git status`
empty afterwards (checked twice).
`uv run pytest tests/viewer -q` → 56 passed in 39 s, twice, no stray Chrome.
`tests/viewer/test_viewer_3d.py` → **15 tests in 21.8 s** (budget 40 s).

New since the last report:

- `test_the_opening_shot_is_close_enough_to_see_individual_soldiers` — on its
  own page (the shared page's `Home` reset is exactly what it must not see):
  the opening distance is under half the whole-map distance, under 70 m of
  ground is on screen, a 1.8 m figure works out over 15 px tall, and the target
  is inside the map.
- `test_a_blocked_unit_is_announced_over_the_building` — the badge's own colour
  on the overlay near the stuck barracks.

Two harness races fixed (both were test bugs, not product bugs): the no-WebGL
fallback test read `stats()` before the page had rendered a frame, and the
opening-shot test originally reused the shared page and read its state before
`reload()` had finished, because `wait_loaded` was satisfied by the *previous*
payload.

## Screenshots

Regenerated with `COH_REFRESH_DOCS_IMG` / `--docs` from a fresh
`play_match.py` run on merged main (the old replay no longer resimulates —
the sim wave changed the state hash), and reviewed: `docs/img/viewer-3d.png`,
`viewer-3d-showcase.png`, `viewer-3d-fog.png`, `viewer.png`.

## Concerns carried forward

The concerns in the first report still stand, with two updates:

- the **draw-call budget** moved with the ground rings and the extra tracer
  segments: a real match is now ~36 calls (was 20–22) and the showcase ~75 (was
  71), 50–68 k triangles. Still small; still not measured on real hardware.
- `A` remains a camera pan key, so the arc toggle is still `Z`.
