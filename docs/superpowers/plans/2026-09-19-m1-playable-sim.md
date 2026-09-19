# M1 — Playable Sim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic CoH1-mechanics sim (US vs Wehrmacht) with real sourced stats, one 1v1 map, replays, a web viewer, and T0/T1 scripted bots — you can watch a full bot-vs-bot game.

**Architecture:** Hybrid world: static grid layers (2 m cells, NumPy) + continuous squad positions. Data-driven: all numbers in YAML under `coh/data/tables/`, mirroring CoH1's attribute structure (per-weapon target tables and cover tables, target types instead of armor). A fixed-order system pipeline ticks 8×/game-second. A minimal multi-agent env wraps the sim with fog-filtered observations; bots use only that. Replays are seed + order log; the viewer server re-simulates to frames.

**Tech Stack:** Python 3.12 (uv-managed), NumPy, PyYAML, pytest; BeautifulSoup + requests for the scraper only; viewer is dependency-free HTML/JS canvas served by `http.server`.

**Spec:** `docs/superpowers/specs/2026-09-19-coh-rl-env-design.md` (read §2–§4, §6–§8, §10). Data sources: `docs/data-sources.md`.

## Global Constraints

- `TICKS_PER_SECOND = 8`, `DT = 0.125`, `CELL_M = 2.0`. World coords are meters, x right, y down. Grid arrays are indexed `[cy, cx]`; `cell = (int(x // CELL_M), int(y // CELL_M))`.
- Determinism: the only randomness is `state.rng` (`numpy.random.default_rng(seed)`). Never iterate a `set` or rely on dict order other than insertion order; iterate entities in ascending id. No wall-clock, no `random` module, no hash-seed-dependent behaviour in `coh/sim`.
- Dependency rule: `coh/sim` imports only `coh/data`, `coh/maps`, numpy, stdlib. `coh/agents` imports only `coh/env`. Viewer consumes only frame JSON.
- No gameplay number literals in `coh/sim` — everything comes from `GameData`. (Structural constants like tick rate and cell size live in `coh/sim/constants.py`.)
- Every value in `coh/data/tables/*.yaml` sits in a file whose header records `source:`; individually estimated values are listed under that file's `estimated:` key with a reason.
- Invalid orders never raise: they return `OrderResult(ok=False, reason=...)` and are counted.
- Out of scope (do not build): doctrines, activated abilities, mines/sandbags/wire, vehicle criticals, veterancy, 2v2, tensor/JSON encoders, LLM adapters, live websocket viewing. (2v2 must not be *precluded*: teams are lists of players everywhere.)
- Commit after every task with a conventional-commit message ending in the `Co-Authored-By` trailer used in this repo's history.
- Run tests with `uv run pytest -q`. All tests must pass before each commit.

**Note on code in this plan:** interfaces, formulas and tests are exact and binding. Implementation bodies are described rather than fully written where they are mechanical; the implementer writes them TDD-style against the given tests and adds further unit tests for edge cases they encounter.

---

## File map

```
pyproject.toml
coh/__init__.py
coh/data/schema.py          dataclasses for all stat tables
coh/data/loader.py          YAML -> GameData, validation
coh/data/tables/            weapons.yaml squads.yaml buildings.yaml upgrades.yaml
                            economy.yaml neutral.yaml patch_2602.yaml
tools/scrape_coh_stats.py   mirror HTML -> tables/*.yaml (run once, output checked in)
tools/roster.yaml           which coh-stats pages are in scope, id mapping
coh/maps/format.py          GameMap dataclass, YAML/ASCII loader, validation
coh/maps/cover.py           directional cover lookup
coh/maps/library/hedgerow_crossing.yaml    first 1v1 map (Angoville-inspired)
coh/sim/constants.py
coh/sim/state.py            Member, Squad, Building, Player, PointState, GameState
coh/sim/orders.py           Order dataclasses, OrderResult, validation + application
coh/sim/sim.py              Sim: issue(), tick(), state_hash()
coh/sim/pathfinding.py      A* on grid (infantry / vehicle passability)
coh/sim/systems/production.py movement.py vision.py combat.py suppression.py
                garrison.py territory.py economy.py victory.py
coh/env/observation.py      fog-filtered Observation builder
coh/env/env.py              CohEnv (parallel multi-agent step loop)
coh/agents/base.py          Agent protocol
coh/agents/scripted.py      T0Idle, T1Capper
coh/replay/replay.py        Replay record/save/load/resimulate
coh/viewer/__main__.py      frames builder + http server
coh/viewer/frames.py
viewer/index.html viewer/viewer.js viewer/style.css
scripts/play_match.py       run bot vs bot, write replay
tests/...                   mirrors the package layout
```

---

### Task 1: Project scaffold

**Files:** Create `pyproject.toml`, `coh/__init__.py`, `coh/sim/constants.py`, `tests/test_smoke.py`, `.gitignore`, update `README.md`.

- [ ] **Step 1:** `uv init --bare --python 3.12`, then `uv add numpy pyyaml` and `uv add --dev pytest beautifulsoup4 requests`. Configure `[tool.pytest.ini_options] testpaths=["tests"]` and a setuptools/hatch build including the `coh` package with `coh/data/tables/*.yaml` and `coh/maps/library/*.yaml` as package data.
- [ ] **Step 2:** `coh/sim/constants.py`:

```python
TICKS_PER_SECOND = 8
DT = 1.0 / TICKS_PER_SECOND
CELL_M = 2.0
```

- [ ] **Step 3:** `tests/test_smoke.py` asserts `coh.sim.constants.DT == 0.125`. Run `uv run pytest -q` → PASS.
- [ ] **Step 4:** `.gitignore` (`.venv/`, `__pycache__/`, `*.replay.json`, `.pytest_cache/`, `tools/.cache/`). README: one paragraph + `uv sync`, `uv run pytest`, `uv run python scripts/play_match.py`, `uv run python -m coh.viewer <replay>`. Commit.

---

### Task 2: Data schema and loader

**Files:** Create `coh/data/schema.py`, `coh/data/loader.py`, `tests/data/test_loader.py`, `tests/data/fixtures/*.yaml` (tiny hand-written tables used by all sim tests — see Task 5 `tests/helpers.py`).

**Interfaces — Produces:**

```python
# schema.py — all @dataclass(frozen=True)
class Cost: manpower: float = 0; munitions: float = 0; fuel: float = 0
class TargetMods: accuracy=1.0; moving=1.0; damage=1.0; penetration=1.0; rear_penetration=1.0; suppression=1.0; priority=0
class CoverMods: accuracy=1.0; damage=1.0; suppression=1.0
COVER_TYPES = ("open", "light", "heavy", "negative", "garrison")   # "open" = no cover

class WeaponDef:
    id: str
    damage: float
    ranges: tuple[float, float, float]        # short, medium, long(=max)
    min_range: float
    accuracy: tuple[float, float, float]      # indexed by range band
    penetration: tuple[float, float, float]
    suppression: tuple[float, float, float]   # added to target per bullet
    cooldown: tuple[float, float]             # seconds min,max between shots/bursts
    cooldown_range_mult: tuple[float, float, float]
    burst: tuple[float, float] | None         # burst duration min,max seconds; None = single shot
    rate_of_fire: float                       # bullets/sec during a burst
    reload: tuple[float, float]
    reload_every: int                         # shots or bursts between reloads
    setup_time: float                         # 0 for non-team weapons
    arc_deg: float                            # 360 unless team weapon / hull gun
    moving_accuracy: float                    # 0 => cannot fire while moving
    nearby_suppression_mult: float
    nearby_suppression_radius: float
    aoe_radius: float
    deflection_damage_mult: float
    indirect: bool
    scatter_m: float                          # indirect only: stddev of impact offset at max range
    cover_table: dict[str, CoverMods]         # keys ⊆ COVER_TYPES; missing => CoverMods()
    target_table: dict[str, TargetMods]       # missing target type => TargetMods()

class SuppressionDef: suppress_at; suppress_recover; pin_at; pin_recover; recovery_per_s
class SquadDef:
    id: str; faction: str                     # "us" | "wehr"
    kind: str                                 # "infantry" | "team_weapon" | "vehicle"
    members: int; member_hp: float            # vehicles: members=1, member_hp = vehicle HP
    loadout: tuple[str, ...]                  # weapon id per member ("" = unarmed); vehicles: all weapons on the one member
    target_type: str
    cost: Cost; population: int; build_time: float; upkeep_per_min: float
    speed: float; rotation_deg_s: float       # rotation only meaningful for vehicles/team weapons
    sight: float; capture_rate: float         # 0 => cannot capture
    builds: tuple[str, ...]                   # building ids this squad can construct
    suppression: SuppressionDef | None        # None => immune (vehicles)
    reinforce_cost_mult: float; reinforce_time_mult: float
    retreat_received_accuracy: float
    upgrade_slots: int
    crushes_light_cover: bool; can_garrison: bool

class BuildingDef:
    id; faction; cost: Cost; build_time; hp; target_type
    footprint: tuple[int, int]                # cells w,h
    produces: tuple[str, ...]; researches: tuple[str, ...]
    requires: tuple[str, ...]                 # building or upgrade ids, all required
    is_hq: bool; reinforce_radius: float; sight: float
class UpgradeDef:                             # global research
    id; faction; cost: Cost; time: float; requires: tuple[str, ...]
    upkeep_mult: float = 1.0                  # Supply Yard levels
class SquadUpgradeDef:                        # per-squad purchase, e.g. BAR
    id; applies_to: tuple[str, ...]; cost: Cost; time: float
    requires: tuple[str, ...]; weapon: str; count: int   # replaces `count` member weapon slots
class NeutralBuildingDef: id; hp; target_type; capacity: int; footprint
class PointIncome: manpower; munitions; fuel; population
class EconomyDef:
    start_resources: Cost; base_income_per_min: Cost; base_population: int; max_population: int
    point_income: dict[str, PointIncome]      # keys: strategic, munitions_low/med/high, fuel_low/med/high, victory
    op_income_mult: float; op_building: dict[str, str]   # faction -> building id
    capture_time_s: float; neutralize_time_s: float; capture_radius: float
    tickets: int; ticket_interval_s: float; tickets_per_vp_lead: float
    retreat_speed_mult: float; suppressed_speed_mult: float
    suppressed_accuracy_mult: float; suppressed_cooldown_mult: float
    noncombat_recovery_delay_s: float; noncombat_recovery_mult: float
    cover_recovery_mult: dict[str, float]
    reinforce_base_cost_frac: float           # per-model cost = squad cost / members * frac * squad.reinforce_cost_mult
class GameData:
    weapons: dict[str, WeaponDef]; squads: dict[str, SquadDef]; buildings: dict[str, BuildingDef]
    upgrades: dict[str, UpgradeDef]; squad_upgrades: dict[str, SquadUpgradeDef]
    neutral: dict[str, NeutralBuildingDef]; economy: EconomyDef

# loader.py
def load_game_data(tables_dir: Path | None = None) -> GameData   # default: packaged tables
class DataError(ValueError)                                     # message includes file + key
```

Validation (fail fast, `DataError`): unknown keys; tuple lengths; every weapon id in loadouts/squad upgrades exists; every `produces`/`researches`/`requires`/`builds` id exists; cover-table keys ⊆ `COVER_TYPES`; exactly one `is_hq` building per faction.

- [ ] **Step 1:** Write fixtures: a tiny but complete data set — weapons `rifle`, `hmg`, `tank_gun`, `at_gun`, `mortar`; squads `rifles`, `engineers`, `hmg_team`, `at_team`, `mortar_team`, `tank`, `sniper`-less; buildings `hq_us`, `hq_wehr`, `barracks`, `op_us`, `op_wehr`; one upgrade, one squad upgrade, economy. Use round numbers so tests are easy to reason about (e.g. rifle accuracy `1.0/0.5/0.25`).
- [ ] **Step 2:** Failing tests: loads fixtures; `data.weapons["rifle"].accuracy == (1.0, 0.5, 0.25)`; missing target type falls back to `TargetMods()` via helper `weapon.vs(target_type)`; and one test per validation rule asserting `DataError` whose message contains the offending file name and key.
- [ ] **Step 3:** Implement; tests pass. Commit.

---

### Task 3: Scrape real CoH1 stats into the tables

**Files:** Create `tools/scrape_coh_stats.py`, `tools/roster.yaml`, `coh/data/tables/*.yaml`, `tests/data/test_real_tables.py`.

Source: `https://www.hq-coh.com/stats/coh-stats.com/` (see `docs/data-sources.md`). Cache fetched HTML under `tools/.cache/` and sleep 0.5 s between requests. Costs in the HTML are separated by resource icon `<img>` tags — parse by icon, never by stripped text.

**Roster (`tools/roster.yaml`)** — non-doctrinal base-game units only:
- US: Engineers, Riflemen, Jeep, .30 cal HMG team, 60 mm Mortar team, Sniper, 57 mm AT gun, M8 Greyhound, M3 Halftrack, M4 Sherman, M10, M4 Crocodile (flamethrower → skip if weapon model doesn't fit; record in `skipped:`). Buildings: HQ, Barracks, Weapons Support Center, Motor Pool, Tank Depot, Supply Yard, Observation Post. Upgrades: Supply Yard L1/L2; squad upgrade BAR (grenades/stickies are abilities → skipped).
- Wehrmacht: Pioneers, Volksgrenadiers, Motorcycle, MG42 HMG team, Sniper, Grenadiers, 8 cm Mortar team, Pak 38, Sd.Kfz 251 Halftrack, Puma, StuG IV, Panzer IV, Panther, Ostwind, Knights Cross Holders (non-doctrinal, Sturm Armory; Stormtroopers are doctrinal → excluded). Buildings: Reich HQ, Wehrmacht Quarters, Krieg Barracks, Sturm Armory, Panzer Command, Observation Post. Upgrades: Skirmish / Assault / Battle phase; squad upgrades MP40 (Volks), MG42 LMG and Panzerschreck (Grenadiers).

- [ ] **Step 1:** Write the scraper: index pages → roster-filtered unit/structure/upgrade pages → follow weapon links → emit YAML in the Task 2 schema. Map coh-stats target-type names through unchanged (they become our `target_type` strings). Keep only target-table rows for target types that exist in our roster plus building types. Cover-table rows map: `tp_light→light`, `tp_heavy→heavy`, `tp_negative→negative`, `tp_open→open`, `tp_garrison_cover→garrison` (inspect actual labels on the page and map accordingly).
- [ ] **Step 2:** `patch_2602.yaml` — list of `{table, id, field, old, new}` from the 2.602 notes that touch roster units and modelled fields (MG42 team 260→250 MP, build 40→35 s; Pak 38 310→290 MP; Engineer suppression thresholds = standard infantry; US mortar setup 2.4→1.5 s; .30 cal and 57 mm changes where the field is modelled). The scraper applies it and asserts each `old` matches what was scraped.
- [ ] **Step 3:** Infantry `speed`: fill from omgmod `entities.json` (`https://raw.githubusercontent.com/omgmod/omg_rails/master/lib/assets/stats/entities.json`, `speed_max`); if a unit is absent use 3.0 and list it under `estimated:`.
- [ ] **Step 4:** Hand-write `economy.yaml` and `neutral.yaml` using `docs/data-sources.md` values: point income 5/10/16, +3 MP per point, OP ×1.6, base pop 30 (+2/+3/+5/+7 per sector type, max 75), tickets 500, ticker 4 s × VP lead, HQ +5 fuel/min; **estimated:** base manpower 270/min with `income = base + points − Σupkeep`, capture 20 s, neutralize 10 s, capture radius 6 m, start resources 350 MP / 0 MU / 20 F, suppressed multipliers (speed 0.5, accuracy 0.5, cooldown 2.0), retreat speed ×1.5, reinforce fraction 0.5.
- [ ] **Step 5:** `tests/data/test_real_tables.py`: packaged tables load; spot checks against verified samples — Riflemen 6 × 55 HP, 270 MP; Garand accuracy `(0.75, 0.55, 0.35)` ranges `(8, 17, 35)`; MG42 team cost 250 (patched); Sherman 636 HP, 75 mm penetration `(1, 0.92, 0.83)`; Garand cover table heavy = `(0.5, 0.5, 0.1)`; every squad in every `produces` list exists; both factions can reach a tank through the tech tree (walk `requires`).
- [ ] **Step 6:** Run scraper, check in generated YAML, tests pass. Commit. If the mirror is unreachable: stop and report; do not invent numbers.

---

### Task 4: Map format, cover lookup, first map

**Files:** Create `coh/maps/format.py`, `coh/maps/cover.py`, `coh/maps/library/hedgerow_crossing.yaml`, `tests/maps/test_format.py`, `tests/maps/test_cover.py`.

**Map YAML:** `name`, `cell_m: 2.0`, `terrain:` (list of equal-length strings), `sectors:` (same shape; one char per sector), `points:` `[{id, name, type, cell:[cx,cy]}]`, `neutral_buildings:` `[{def, cell}]`, `starts:` `[{slot, team, hq_cell, sector}]`.

Terrain legend:

| ch | meaning | infantry | vehicle | LOS | cover |
|----|---------|----------|---------|-----|-------|
| `.` | open ground | ✓ | ✓ | – | open |
| `r` | road | ✓ | ✓ | – | negative |
| `c` | crater / rubble | ✓ | ✓ | – | heavy (area, non-directional) |
| `f` | fence / low hedge | ✓ | ✓ (crushed → `.`) | – | light object |
| `w` | stone wall | ✗ | ✗ | – | heavy object |
| `H` | bocage hedgerow | ✗ | ✗ | blocks | heavy object |
| `T` | trees | ✓ | ✗ | blocks | light (area) |
| `~` | water | ✗ | ✗ | – | – |

**Interfaces — Produces:**

```python
class GameMap:
    name: str; width: int; height: int                 # cells
    pass_inf: np.ndarray[bool]; pass_veh: np.ndarray[bool]; los_block: np.ndarray[bool]
    area_cover: np.ndarray[uint8]                      # index into COVER_TYPES
    cover_object: np.ndarray[uint8]                    # 0 none, else COVER_TYPES index (light/heavy)
    sector_id: np.ndarray[int16]
    sectors: dict[int, SectorDef]                      # SectorDef(id, point_id, neighbors: tuple[int,...])
    points: dict[str, PointDef]                        # PointDef(id, name, type, cell, sector)
    neutral_buildings: list[NeutralPlacement]; starts: list[StartDef]
def load_map(name_or_path) -> GameMap                  # raises MapError
def cell_of(pos) -> tuple[int,int];  def center_of(cell) -> np.ndarray

# cover.py
def cover_at(m: GameMap, cell: tuple[int,int], from_pos: np.ndarray) -> str
```

`cover_at`: if `area_cover[cell]` is light/heavy → that. Else look at the 8 neighbours of `cell`; a neighbour with `cover_object` counts if the angle between (cell→neighbour) and (cell→from_pos) is ≤ 60°; return the best (heavy > light). Else `negative` if road, else `open`. Sector adjacency is derived from 4-neighbour contact in the `sectors` layer. Neutral building footprints are stamped impassable + LOS-blocking at load.

Validation (`MapError`): rectangular layers; each sector has exactly one point and the point's cell is inside it; sector graph connected; HQ cells passable and infantry-pathable to every point (BFS); at least one victory point.

- [ ] **Step 1:** Tests with a 12×8 inline map: passability/LOS arrays match legend; adjacency correct; each validation rule raises. Cover tests: unit east of a wall shot from the west → `heavy`; same unit shot from the east → `open`; crater → `heavy` from any direction; road → `negative`.
- [ ] **Step 2:** Implement; pass.
- [ ] **Step 3:** Author `hedgerow_crossing`: 96×96 cells (192 m), two HQ sectors in opposite corners (NW/SE), 13 sectors, point mix: 3 victory (centre line), 2 high-ish: 1 fuel_high centre-west, 1 munitions_high centre-east, per side 1 fuel_low, 1 munitions_med, 1 strategic next to base; rotational symmetry (180°) for fairness; bocage lanes, a central farm cluster of 4 neutral buildings, roads crossing the middle, fences and craters for cover. Test: loads, validates, and is 180° rotationally symmetric in `pass_inf` and point types.
- [ ] **Step 4:** Commit.

---

### Task 5: Sim state, orders, tick loop, hashing

**Files:** Create `coh/sim/state.py`, `coh/sim/orders.py`, `coh/sim/sim.py`, `coh/sim/systems/__init__.py`, `tests/helpers.py`, `tests/sim/test_sim_core.py`.

**Interfaces — Produces:**

```python
# state.py (mutable dataclasses)
class Member: hp: float; weapon: str; next_ready_tick: int = 0; shots_since_reload: int = 0; burst_until_tick: int = 0
class SquadState(Enum): IDLE, MOVING, SETTING_UP, SET_UP, TEARING_DOWN, RETREATING, GARRISONED, CONSTRUCTING
class Squad:
    id: int; owner: int; def_id: str; pos: np.ndarray; heading: float   # radians
    members: list[Member]; state: SquadState
    order: Order | None; path: list[tuple[int,int]]; target_id: int | None
    suppression: float; suppressed: bool; pinned: bool; last_hit_tick: int
    garrison_in: int | None; facing: float | None        # team-weapon arc centre
    setup_done_tick: int; reinforce_done_tick: int; upgrades: list[str]
class Building:
    id: int; owner: int | None; def_id: str; cell: tuple[int,int]; hp: float
    progress: float                                       # 1.0 = complete
    queue: list[QueueItem]; garrison: list[int]; neutral: bool
class QueueItem: kind: str ("train"|"research"); item_id: str; remaining_s: float
class Player: id: int; team: int; faction: str; manpower: float; munitions: float; fuel: float
              upgrades: list[str]; hq_id: int; invalid_orders: int
class PointState: owner_team: int | None; progress: float; capturing_team: int | None; op_building: int | None
class GameState:
    tick: int; rng: np.random.Generator; players: dict[int, Player]
    squads: dict[int, Squad]; buildings: dict[int, Building]; points: dict[str, PointState]
    connected: dict[int, set[int]]                        # team -> connected sector ids (iterate sorted)
    tickets: dict[int, float]; visible: dict[int, np.ndarray]   # team -> bool grid
    ghosts: dict[int, dict[int, Ghost]]; winner: int | None; next_id: int
    events: list[Event]                                   # cleared each tick; consumed by viewer (shots, deaths, captures)
# entity ids: single counter shared by squads and buildings.

# orders.py (frozen dataclasses, all have `squad` or `building` int id)
Move(squad, cell) AttackMove(squad, cell) Attack(squad, target_id) Capture(squad, point_id)
Garrison(squad, building_id) Ungarrison(squad) Retreat(squad) Reinforce(squad) SetFacing(squad, direction_deg)
Build(squad, structure, cell) Train(building, unit) Research(building, upgrade) BuyUpgrade(squad, upgrade)
Stop(squad)
class OrderResult: ok: bool; reason: str = ""
def order_to_dict(o) -> dict;  def order_from_dict(d) -> Order      # {"type": "Move", ...}

# sim.py
class PlayerSetup: faction: str; team: int; start_slot: int
class SimConfig: time_limit_s: float = 2700.0
class Sim:
    def __init__(self, game_map: GameMap, players: list[PlayerSetup], data: GameData, seed: int, config: SimConfig = SimConfig())
    state: GameState; map: GameMap; data: GameData
    def issue(self, player_id: int, orders: list[Order]) -> list[OrderResult]
    def tick(self) -> None;  def run(self, ticks: int) -> None
    def state_hash(self) -> str          # sha256 over a canonical serialization (floats rounded to 1e-6)
    def spawn_squad(self, owner: int, def_id: str, pos) -> Squad      # used by production and by tests
    def spawn_building(self, owner, def_id, cell, complete=True) -> Building
```

`Sim.__init__` places each player's HQ at its start, stamps building footprints impassable, spawns one builder squad next to the HQ, sets start resources, marks the HQ sector owned, places neutral buildings. `tick()` calls systems in the spec §2.3 order; each system is `def run(sim: Sim) -> None` in its module. In this task, systems are empty stubs.

Order validation common rules: entity exists, belongs to player, squad not retreating (except no orders accepted), cell in bounds. Specific rules are added by the task that implements each order; until then the order is accepted and stored in `squad.order`.

`tests/helpers.py`: `make_sim(ascii_map: str | None = None, seed=0, **kw)` building a Sim from the Task 2 fixture data and a small inline map, plus `spawn(sim, owner, def_id, cell)`.

- [ ] **Step 1:** Tests: two sims with the same seed and same issued orders have equal `state_hash()` after 80 ticks; different seeds that consume RNG differ (use a stub that draws); order for another player's squad → `ok=False` and `invalid_orders == 1`; `order_from_dict(order_to_dict(o)) == o` for every order type; start state has HQ + builder per player.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 6: Pathfinding and movement

**Files:** Create `coh/sim/pathfinding.py`, `coh/sim/systems/movement.py`, `tests/sim/test_movement.py`.

**Interfaces:** `find_path(passable: np.ndarray, start: cell, goal: cell) -> list[cell] | None` — 8-connected A* with octile heuristic, no corner cutting, deterministic tie-break (lower f, then lower h, then insertion order); if `goal` is impassable, path to the nearest passable cell (BFS ring). Cache key `(kind, start, goal, map_version)`; `map_version` bumps when footprints or crushed cover change.

Movement system: for squads with a path, advance `pos` toward the next cell centre by `speed × DT × mults` (suppressed → `economy.suppressed_speed_mult`; pinned → 0; retreating → `retreat_speed_mult`). Infantry heading snaps to travel direction. Vehicles rotate heading toward travel direction at `rotation_deg_s` and move only when within 45° of it. Vehicles with `crushes_light_cover` entering an `f` cell clear its `cover_object` and bump `map_version`. `Move`/`AttackMove`/`Capture`/`Garrison`/`Build`/`Retreat` all resolve to a path; arrival sets `IDLE` (or the order's follow-up state). `AttackMove`: squad halts while it has a target in range (checked via `squad.target_id` set by combat) and resumes after. Team weapons given a move order first spend `setup_time` tearing down (`TEARING_DOWN`), and on arrival set up again facing their last `facing` (or travel direction).

- [ ] **Step 1:** Tests: path routes around a wall; returns `None` for enclosed goal; infantry with speed 3 m/s covers 30 m ± 1 cell in 80 ticks; vehicle cannot enter `T`; tank facing east ordered west takes ≥ 180/rotation seconds before displacement; tank drives over `f` and the cell's cover becomes none; pinned squad does not move; determinism hash test with movement.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 7: Vision and fog of war

**Files:** Create `coh/sim/systems/vision.py`, `tests/sim/test_vision.py`.

**Interfaces:** after the system runs, `state.visible[team]` is a bool grid; `is_visible(sim, team, entity) -> bool`; `has_los(m: GameMap, a_pos, b_pos) -> bool` (Bresenham over `los_block`, endpoints excluded; garrisoned squads see from the building centre). Ghosts: `state.ghosts[team][building_id] = Ghost(def_id, owner, cell, hp_frac, last_seen_tick)` updated while visible, removed when the cell is visible and the building is gone.

Implementation: per unit, precomputed disc offsets for its sight radius; cast rays to the disc's perimeter cells marking cells until a `los_block` cell (mark the blocker itself, then stop). Recompute every tick only for squads whose cell changed; buildings once. Cache per-squad masks.

- [ ] **Step 1:** Tests: cell behind a hedgerow is not visible, cell in front is; teammates share (construct a 2-players-one-team sim); enemy squad outside sight is not `is_visible`; ghost persists after losing sight and disappears after re-scouting a destroyed building.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 8: Combat — small arms, cover, deaths

**Files:** Create `coh/sim/systems/combat.py`, `tests/sim/test_combat.py`.

**Rules (binding):**
- Target acquisition each tick for squads not pinned/retreating/constructing: keep `target_id` if still visible to the team, alive, within max range of any member weapon and LOS; an explicit `Attack` order pins the target and the squad paths toward it until in range. Otherwise choose the visible enemy (squad or building) in range with LOS maximizing `(priority from the squad's first weapon's target table, −distance)`; ties → lower id. Buildings are auto-targeted only by weapons whose damage mod vs that building type ≥ 0.25.
- Per member with a weapon and `tick ≥ next_ready_tick`: `band = 0 if d ≤ r_short else 1 if d ≤ r_medium else 2`; out of range or `d < min_range` → no shot. Moving attacker: if `moving_accuracy == 0` no shot, else multiply.
- `p_hit = accuracy[band] × cover.accuracy × moving × tt.accuracy × (tt.moving if target moving else 1) × (suppressed_accuracy_mult if attacker suppressed else 1) × (retreat_received_accuracy if target retreating else 1)`, clamped to [0,1]. Cover = `cover_at(map, victim_member_cell, attacker_pos)`; garrisoned target → `"garrison"`.
- Bullets per firing: single shot → 1; burst → `round(rate_of_fire × burst_duration)` bullets resolved immediately, each with its own roll. Each bullet adds `suppression[band] × cover.suppression × tt.suppression` to the target squad regardless of hit (Task 9 consumes `squad.suppression`).
- On hit vs infantry: uniformly random living member takes `damage × tt.damage × cover.damage`. Members at ≤ 0 HP are removed; their weapon is lost unless it is the team weapon (Task 10). Empty squad → removed, `Event("squad_destroyed")`.
- After firing: `next_ready_tick = tick + ceil(uniform(cooldown) × cooldown_range_mult[band] × (suppressed_cooldown_mult if suppressed) × TPS)`; every `reload_every` firings add `uniform(reload)`.
- Buildings: hits always land (`p_hit` uses accuracy × tt only); damage `damage × tt.damage`; at 0 HP the building is removed, footprint freed, garrison ejected (Task 12).
- Member positions for cover: member *i* stands at `squad.pos + FORMATION[i]` (fixed offsets on a 1.5 m ring, rotated by heading); if that cell is impassable, use the squad cell.
- Emit `Event("shot", src_id, dst_id, hit)` per firing for the viewer.

- [ ] **Step 1:** Tests (fixture data, seeded, statistical where needed — 200 trials via fresh seeds): accuracy 1.0 rifle at short range always hits and removes expected HP; target out of range takes nothing; target behind a wall (from shooter's side) takes ≈ half the hits of one in the open (±15%) and the same target flanked takes full; no LOS through hedgerow → no fire; dead members removed and squad deleted when empty; burst weapon applies N bullets; moving shooter with `moving_accuracy 0` never fires; acquisition prefers higher priority, then nearer.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 9: Suppression, retreat, reinforce

**Files:** Create `coh/sim/systems/suppression.py`; modify `movement.py`, `orders.py`; `tests/sim/test_suppression.py`.

**Rules:** Squads with `suppression: None` ignore all of this. Thresholds from `SquadDef.suppression`: `suppressed` turns on at `suppress_at`, off below `suppress_recover`; `pinned` on at `pin_at`, off below `pin_recover`; value clamped to [0,1]. Nearby spill: when a bullet adds `s` to a squad, every other enemy-of-shooter squad within `nearby_suppression_radius` of the target gains `s × nearby_suppression_mult`. Recovery per second: `recovery_per_s × cover_recovery_mult[current cover vs last attacker direction, default open]`, and × `noncombat_recovery_mult` once `noncombat_recovery_delay_s` has passed since `last_hit_tick`. Garrisoned squads receive 0 suppression via the cover table.

`Retreat`: clears suppression flags and value, state `RETREATING`, paths to the nearest passable cell adjacent to own HQ, rejects all further orders until arrival (then `IDLE`). Vehicles and garrisoned squads: garrisoned squads exit first; vehicles → invalid. `Reinforce`: valid if squad is within `reinforce_radius` of a friendly building with `reinforce_radius > 0`, is missing members, not in combat state RETREATING, and the player can afford one model: cost `= squad.cost / members × reinforce_base_cost_frac × reinforce_cost_mult` (each resource); time per model `= build_time / members × reinforce_time_mult`. One model per order completion; the order repeats automatically until full or unaffordable. New member gets the def's default weapon for that slot.

- [ ] **Step 1:** Tests: HMG fire pins a rifle squad in the open within a few seconds, and the pinned squad neither moves nor fires; same squad behind heavy cover is not pinned in the same time; suppression decays and flags clear with hysteresis; spill suppresses a neighbour within the radius but not beyond; retreating squad ignores orders, reaches HQ, takes reduced hits; reinforce restores members, charges manpower, fails away from HQ and when broke.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 10: Team weapons and indirect fire

**Files:** Modify `combat.py`, `movement.py`, `orders.py`; `tests/sim/test_team_weapons.py`.

**Rules:** `kind == "team_weapon"`: the first loadout slot is the team weapon (`setup_time > 0`). It fires only in state `SET_UP`, and only at targets within `arc_deg/2` of `squad.facing`. `SetFacing` → tear down + set up (`2 × setup_time`) at the new facing; invalid for other kinds. A squad that stops moving auto-sets-up facing its last travel direction or its current target. If the target leaves the arc and there is no other target in arc, the squad re-faces automatically (paying setup time). Crew loss: when the gunner (member 0) dies, the next living member takes the weapon (team weapon is never lost while anyone lives). When the last member dies, leave a `Squad` shell with `members=[]`, `owner=None`, state `IDLE` flagged `abandoned=True` (add the field); any infantry squad with ≥ 2 members ordered `Move` onto its cell (within 2 m) re-crews it: the shell takes that squad's owner and members (capped at def `members`), the original squad is removed. Abandoned shells are not targets and do not block.

Indirect (`indirect: True`, mortar): requires target visible to team but **not** LOS from the weapon; `min_range` enforced; impact point `= target.pos + N(0, scatter_m × d / r_long)` per axis; all members of any squad within `aoe_radius` of the impact take `damage × tt.damage × cover.damage` (cover evaluated from the impact point; `garrison` if inside) with linear falloff to 50% at the edge; suppression applied to squads in radius. `Event("explosion", pos, radius)`.

- [ ] **Step 1:** Tests: HMG does not fire before setup completes; does not engage a target 120° off its facing until re-faced; re-facing costs time; killing the gunner keeps the weapon firing; wiping the crew leaves an abandoned shell and an enemy rifle squad can re-crew and fire it; mortar hits a squad behind a hedgerow that a teammate spots, and never fires inside `min_range`; AOE damages two adjacent squads.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 11: Vehicles — penetration, rear arcs, turrets

**Files:** Modify `combat.py`, `state.py`; `tests/sim/test_vehicles.py`.

**Rules:** For targets with `kind == "vehicle"`: after a hit, `rear = angle between target heading and (target→attacker) > 120°`; `p_pen = clamp(penetration[band] × tt.penetration × (tt.rear_penetration if rear else 1), 0, 1)`; penetrate → `damage × tt.damage`; else `damage × tt.damage × deflection_damage_mult`. Vehicle at 0 HP → destroyed (`Event("vehicle_destroyed")`), leaves a `c` crater cell (area heavy cover; bump `map_version`). Vehicle weapons: weapon index 0 is turreted (`arc_deg == 360` means turret) — add `Squad.turret_heading`, which rotates toward the target at 2× hull rotation and must be within 5° to fire; weapons with `arc_deg < 360` are hull-mounted and require the hull arc (vehicle rotates in place toward the target when idle). Vehicles can fire while moving subject to `moving_accuracy`. Vehicles cannot capture, garrison or retreat.

- [ ] **Step 1:** Tests (fixture: `tank_gun` pen 1.0 with `tt.penetration 0.5` vs `tank`, `rear_penetration 2.0`): frontal hits penetrate ≈ 50%, rear hits 100% (statistical); deflections deal `deflection_damage_mult`; rifles deal negligible damage to the tank per target table; AT gun beats tank frontally at long range in a seeded majority of trials and loses when the tank starts behind it; wreck leaves heavy cover; turret must traverse before first shot.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 12: Garrisons

**Files:** Create `coh/sim/systems/garrison.py`; modify `orders.py`, `combat.py`, `vision.py`; `tests/sim/test_garrison.py`.

**Rules:** `Garrison(squad, building)` valid for `can_garrison` squads on neutral or own-team buildings with remaining capacity (capacity counts squads; from `NeutralBuildingDef.capacity`); squad paths to the nearest footprint-adjacent cell then enters: state `GARRISONED`, `pos` = building centre, not pathing. Garrisoned squads: fire with range measured from the building centre, 360° (team weapons still need setup after entering, then ignore arcs), receive `"garrison"` cover mods, and see from the building. Attackers target the squad (small arms) or the building (weapons with building damage mod ≥ 0.25); occupants make an otherwise-neutral building a valid target. `Ungarrison` → exits to the nearest passable adjacent cell on the side away from the nearest visible enemy (ties → lowest cell). Building destroyed → occupants ejected the same way and each member takes 25% of max HP damage (`economy` field `garrison_collapse_damage_frac`, add to schema + tables as estimated). Retreat from a garrison = exit + retreat.

- [ ] **Step 1:** Tests: enter/exit and capacity; garrisoned rifles beat identical rifles in the open in a seeded majority; garrisoned squad gains no suppression from small arms; tank gun destroys the building and ejects damaged occupants; enemy cannot enter an occupied building.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 13: Territory — capture, supply, observation posts

**Files:** Create `coh/sim/systems/territory.py`; modify `orders.py`; `tests/sim/test_territory.py`.

**Rules:** Capturers at a point = squads with `capture_rate > 0`, a `Capture` order for that point, within `capture_radius`, not suppressed-pinned, not retreating, not garrisoned. If capturers of two teams are present → stalled. Else for team T: if point owned by another team → `progress -= Σcapture_rate × DT / neutralize_time_s`; at ≤ 0 owner becomes None (OP on it, if any, must be destroyed first: a point with a living OP cannot be neutralized). If neutral → `progress += Σrate × DT / capture_time_s`; at ≥ 1 owner = T. Progress without capturers decays toward the owner's resting value (0 neutral, 1 owned) at the capture rate 1. On any ownership change recompute `state.connected[team]`: BFS over `SectorDef.neighbors` from each of the team's HQ sectors through sectors whose point is owned by the team. HQ sectors are permanently owned by their team and cannot be captured. `Build(op)` by a builder squad on an owned, connected point without an OP constructs the faction's `op_building` on the point cell (uses Task 14's construction; here just validation + `PointState.op_building` bookkeeping, cleared when the building dies).

- [ ] **Step 1:** Tests: capture takes `capture_time_s / rate`; contested stalls; enemy point needs neutralize then capture; vehicles can't capture; cutting the middle sector of a chain removes the far sector from `connected` and restores when retaken; HQ sector uncapturable; point with OP cannot be neutralized until OP destroyed.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 14: Economy, construction, production, tech, squad upgrades

**Files:** Create `coh/sim/systems/economy.py`, `coh/sim/systems/production.py`; modify `orders.py`; `tests/sim/test_economy.py`, `tests/sim/test_production.py`.

**Economy rules (per tick, per player; teams' sectors are shared for connectivity but income is per player: each connected point owned by the team pays every player on the team):** `income/min = base_income + Σ point_income[type] × (op_income_mult if OP)` over connected team points; manpower additionally `− Σ upkeep_per_min × Π upkeep_mult(researched upgrades)`, floored at a minimum of 25% of base manpower (estimated; add `min_manpower_income_frac` to economy schema). Add `income × DT / 60`. Population: `used = Σ population` of living squads + queued trains; `cap = min(max_population, base_population + Σ point_income[type].population over connected points)`.

**Production rules:** `Build(squad, structure, cell)`: squad def lists the structure in `builds`; `requires` all satisfied (a required id is satisfied by a completed own building with that def id or a researched upgrade); footprint cells all passable for vehicles, unoccupied, inside a sector connected for the team (OPs: exactly the point cell of an owned connected point); affordable → deduct cost, place a `Building(progress=0, hp=10% of max)` footprint (impassable), squad paths adjacent then enters `CONSTRUCTING`; each constructing builder adds `DT / build_time` progress and proportional HP; builders leaving pauses it; re-issuing `Build` with the same structure on an unfinished building's cell resumes/assists at no cost. `Train(building, unit)`: building complete, unit in `produces`, affordable (units are gated only by their producing building existing; phase gating is expressed through building `requires`), pop available → deduct, enqueue; queue processes head only; on completion spawn at the nearest passable cell to the building's south side, `IDLE`. `Research`: same queue, upgrade in `researches`, not already owned/queued. Queue max length 5. `BuyUpgrade(squad, upgrade)`: squad def in `applies_to`, `requires` satisfied, squad in a connected friendly sector, free `upgrade_slots`, affordable → after `time` seconds replace the weapons of the first `count` members not already upgraded; reinforced members do not inherit upgrades until bought again (simplification, note in code). Wehrmacht phases are `UpgradeDef`s researched at the HQ that appear in later buildings' `requires`.

- [ ] **Step 1:** Economy tests: income accrues at the table rate over 60 s; cut-off sector stops paying and its pop is lost; OP multiplies; upkeep lowers manpower but not below the floor; Supply-Yard-style `upkeep_mult` applies.
- [ ] **Step 2:** Production tests: can't build without prerequisite / funds / in enemy or disconnected territory / on blocked cells; construction completes in `build_time` with one builder and ~half with two; pausing works; train deducts, respects pop cap and queue order, spawns the squad; research unlocks the next building tier (real tables: US Barracks → Motor Pool → Tank Depot → Sherman; Wehr phases → Panzer Command → Panzer IV, driven end-to-end with generous resources); BuyUpgrade swaps weapons and charges munitions.
- [ ] **Step 3:** Implement; pass; commit.

---

### Task 15: Victory

**Files:** Create `coh/sim/systems/victory.py`; `tests/sim/test_victory.py`.

**Rules:** Every `ticket_interval_s`: for each team, `lead = max over enemy teams (their VP count) − own VP count`; if `lead > 0`, `tickets −= tickets_per_vp_lead × lead` (VPs count regardless of supply connection). Tickets ≤ 0 → other team wins (2 teams only). All of a team's HQ buildings destroyed → other team wins. `tick × DT ≥ time_limit_s` → higher tickets wins; equal → `winner = -1` (draw). Once `winner` is set, `Sim.tick()` is a no-op.

- [ ] **Step 1:** Tests: 2–1 VP lead drains 1 ticket per interval from the trailing team only; equal VPs drain nothing; 0 tickets ends the game; HQ kill ends the game; time limit picks the ticket leader; ticking after the end changes nothing.
- [ ] **Step 2:** Implement; pass; commit.

---

### Task 16: Env, observations, agents, replay, match script

**Files:** Create `coh/env/observation.py`, `coh/env/env.py`, `coh/agents/base.py`, `coh/agents/scripted.py`, `coh/replay/replay.py`, `scripts/play_match.py`; tests `tests/env/test_env.py`, `tests/replay/test_replay.py`, `tests/agents/test_bots.py`.

**Interfaces:**

```python
# observation.py — plain dataclasses, JSON-serializable via asdict
class SquadView: id, def_id, owner, pos: tuple[float,float], cell, members: int, max_members: int,
                 hp_frac: float, suppressed: bool, pinned: bool, state: str, order: dict | None,
                 cover: str, in_reinforce_range: bool            # order/cover/in_reinforce_range None for enemies
class BuildingView: id, def_id, owner, cell, hp_frac, progress, queue: list[str], garrison_count: int
class GhostView: id, def_id, owner, cell, last_seen_s
class PointView: id, name, type, cell, owner_team, progress, connected: bool | None, has_op: bool
class Observation:
    player_id, team, faction, time_s, manpower, munitions, fuel, income: dict, pop_used, pop_cap
    upgrades: list[str]; tickets: dict[int, float]
    own_squads, ally_squads, enemy_squads: list[SquadView]
    own_buildings, ally_buildings, enemy_buildings: list[BuildingView]; ghosts: list[GhostView]
    neutral_buildings: list[BuildingView]; points: list[PointView]
    available: dict      # {"train": {building_id: [unit ids affordable+unlocked]}, "build": [...], "research": {...}, "squad_upgrades": {squad_id: [...]}}
def build_observation(sim: Sim, player_id: int) -> Observation

# env.py
class CohEnv:
    def __init__(self, map_name: str, players: list[PlayerSetup], seed: int = 0,
                 decision_interval_s: float = 2.0, config: SimConfig = SimConfig(), data: GameData | None = None)
    def reset(self) -> dict[int, Observation]
    def step(self, orders: dict[int, list[Order]]) -> tuple[dict[int, Observation], dict[int, float], bool, dict[int, dict]]
    sim: Sim; order_log: list[tuple[int, int, dict]]        # (tick, player_id, order dict)
# rewards: 0 until the end; +1 win / −1 loss / 0 draw. info: {"invalid_orders": n, "results": [OrderResult...]}
# Point ownership is public knowledge (as in CoH); `connected` is None for enemy-owned points.

# agents/base.py
class Agent(Protocol):
    def reset(self, player_id: int, map: GameMap, data: GameData) -> None     # static map + data are public
    def act(self, obs: Observation) -> list[Order]

# replay.py
class Replay: map_name; players: list[PlayerSetup]; seed; decision_interval_s; config; orders: list[tuple[int,int,dict]]; final_hash: str
def save(replay, path); def load(path) -> Replay
def resimulate(replay) -> Iterator[Sim]      # yields the sim after every tick, applying logged orders at their ticks
CohEnv.to_replay() -> Replay
```

**Bots:** `T0Idle.act → []`. `T1Capper`: (1) HQ/barracks-type building with empty queue and pop room → train the cheapest capture-capable infantry available; builder: if no infantry-producing building exists and one is affordable → `Build` it at the first valid site found by spiralling out from the HQ (validate by trying candidate cells against `obs.available["build"]` + map passability); (2) every idle capture-capable squad → `Capture` the nearest (path-agnostic Euclidean) point not owned by its team and not already targeted by a friendly squad; if none, `AttackMove` to the nearest enemy-owned or VP point; (3) squad with `members ≤ max_members/3` and enemies within 40 m → `Retreat`; squad at HQ missing members → `Reinforce`.

`scripts/play_match.py --map hedgerow_crossing --p0 t1 --p1 t0 --seed 0 --out match.replay.json` prints winner, tickets, duration.

- [ ] **Step 1:** Env tests: enemy squads outside vision are absent from the observation and appear when scouted; ghosts appear; `available` reflects affordability; step advances exactly 16 ticks; rewards at the end.
- [ ] **Step 2:** Replay tests: play 120 s of T1 vs T1, save, load, resimulate → final `state_hash` equals `final_hash`; tampering with one order changes it.
- [ ] **Step 3:** Bot tests (real data, real map): T1 vs T0 — T1 wins by tickets or annihilation before the time limit; T1 holds ≥ 6 points by minute 8; T1 vs T1 over 3 seeds finishes without exceptions and with zero invalid orders > 5% of issued.
- [ ] **Step 4:** Implement; pass; commit.

---

### Task 17: Viewer

**Files:** Create `coh/viewer/frames.py`, `coh/viewer/__main__.py`, `viewer/index.html`, `viewer/viewer.js`, `viewer/style.css`; `tests/viewer/test_frames.py`.

**Frames:** `build_frames(replay, every_ticks=2) -> dict` = `{"map": {name, width, height, cell_m, terrain: [rows], sectors: [rows], points: [...], }, "players": [...], "frames": [ {t, players:[{mp,mu,fu,pop,cap}], tickets, points:[{id,owner,progress,op}], squads:[{id,o,def,kind,x,y,h,th,n,max,hp,sup,st,g}], buildings:[{id,o,def,cx,cy,w,h,hp,prog,n}], events:[...], vis: {team: base64(packbits(visible))}, terrain_delta:[[cx,cy,ch]...] } ], "winner"}`. Events from all ticks since the previous frame are merged. `python -m coh.viewer path.replay.json [--port 8000]` builds frames (gzip JSON in memory), serves `/frames.json` and the static `viewer/` dir, prints the URL.

**Canvas app (no dependencies, no build):** fit-to-window with pan/zoom (wheel, drag). Layers bottom-up: terrain (grass `.`, tan road, dark-green hedgerow, grey wall, brown fence, crater, trees, water) → sector tint by owner (blue US / red-grey Wehr / none, 18% alpha) + sector borders → cover hint dots (green heavy / yellow light) toggle `C` → points (icon by type: star VP, fuel drum, ammo, flag; ring shows capture progress; square marks OP) → buildings (footprint rect, owner colour, progress bar, garrison count) → squads (infantry: one dot per member in formation; team weapon: dots + arc wedge; vehicle: rotated hull rect + turret line; health bar, yellow/red suppression pip, retreat icon) → tracers (thin line shooter→target, 120 ms; misses offset) and explosion rings → fog (dark 45% where the selected team's `vis` is false; key `F` cycles omniscient / team 0 / team 1) → HUD top bar per player (MP / MU / FU / pop) and centre VP tickers, bottom timeline with play/pause (space), speed 1×/2×/4×/8×, scrub. Click a squad/building → side panel with its frame fields. Interpolate positions between frames.

- [ ] **Step 1:** Test `build_frames` on a 30 s replay: frame count, required keys, `vis` decodes to the map shape, JSON-serializable. Implement `frames.py` + server.
- [ ] **Step 2:** Build the canvas app. Verify by running `scripts/play_match.py` then the viewer, and drive it with Playwright (Chromium at `/opt/pw-browsers`, install `playwright` python package as a dev dep without downloading browsers): load page, assert no console errors, canvas non-blank, press space, take screenshots at 3 timeline positions and visually inspect them (terrain readable, units visible, HUD populated, fog toggles). Fix what looks wrong.
- [ ] **Step 3:** Commit (include one screenshot at `docs/img/viewer.png`, referenced from README).

---

### Task 18: CoH-outcome scenario tests, performance, ladder sanity

**Files:** Create `tests/scenarios/test_coh_outcomes.py`, `tests/scenarios/test_perf.py`, `scripts/bench_sim.py`.

Scenarios use **real tables**, small purpose-built inline maps, 30 seeds each, assert the expected side wins ≥ 70%:
1. MG42 team set up in heavy cover vs one Riflemen squad attacking frontally across open ground → MG wins (riflemen pinned or wiped).
2. Same MG42 vs two Riflemen squads, one frontal, one arriving from behind the arc → Riflemen win.
3. Pak 38 set up facing a Sherman approaching frontally from max range → Pak wins; Sherman starting behind the Pak → Sherman wins.
4. Volksgrenadiers garrisoned vs Riflemen in the open at medium range → garrison wins.
5. Riflemen in heavy cover vs Volksgrenadiers in the open at long range → Riflemen win.
6. Rifles vs Sherman → Sherman loses < 5% HP in 60 s.

If a scenario fails, first suspect sim bugs (check the math against Task 8–11 rules); only then adjust values marked `estimated`. Never edit sourced values to make a scenario pass — if sourced values produce a non-CoH outcome, record it in `docs/data-sources.md` under "Fidelity notes" and mark the test `xfail` with the reason.

Perf: `scripts/bench_sim.py` runs T1 vs T1 on `hedgerow_crossing` for 10 game-minutes and reports ticks/s overall and per system (time each system with `perf_counter` in the script via a wrapper, not in sim code). Target ≥ 1,000 ticks/s single core. If below: profile, optimize the top system (vision mask caching, skip acquisition for squads with no enemy within max range using a coarse spatial hash, path cache), re-measure. `test_perf.py` asserts ≥ 300 ticks/s (loose floor so CI noise doesn't flake) and is marked `@pytest.mark.slow`.

- [ ] **Step 1:** Write scenarios; run; fix sim bugs / tune estimated values; record fidelity notes.
- [ ] **Step 2:** Bench + optimize to target or document the measured number and the bottleneck in README.
- [ ] **Step 3:** Full run: `uv run pytest -q`, a fresh `play_match.py` T1 vs T1, open in viewer, final screenshots. Commit.

---

## Self-review notes

- Spec coverage: §2 (Tasks 1, 4–7), §3.1 (13–15), §3.2 (8–12), §3.3 (14), §3.4 (7), §4 minimal env (16; encoders/masks are M2 by spec §9), §6 (17; live websocket + agent panel are M2), §7 (2, 4, 5, 16), §8 (every task + 18), §10 (3).
- Snipers: included through the roster as ordinary squads with their sourced weapon tables; no camouflage.
- Deferred to M2 by spec: T2/T3 bots, JSON/tensor encoders, adapters, runner/Elo, 2v2 map, veterancy, Kampfkraft.
