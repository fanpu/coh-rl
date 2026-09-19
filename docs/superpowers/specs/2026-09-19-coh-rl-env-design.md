# CoH-RL: a Company of Heroes clone as an RL / agent benchmark environment

Date: 2026-09-19
Status: design approved in brainstorming; pending written-spec review

## 1. Purpose

Build a headless, deterministic, mechanics-faithful 2D clone of Company of Heroes 1
(US vs Wehrmacht) and use it to:

1. Benchmark API-driven decision models — primarily TypeSafe AI's **Jev** (text/structured
   state in, schema-typed decisions out, 70–500 ms latency), and also general LLMs.
2. Train and benchmark RL policies on a modest compute budget (one NVIDIA GB10:
   20 ARM cores, 128 GB unified memory).

We are cloning mechanics, not wrapping the real game. The goal is "mechanics-faithful":
real CoH1 stat values where they can be sourced, the real rules for cover, suppression,
territory and tech; approximations for pathfinding, physics and map geometry. Balance
parity with the retail game is not a goal.

### In scope

- Factions: US, Wehrmacht (CoH1 base game).
- Modes: 1v1 and 2v2. One agent per player; teammates share vision.
- Maps: two 1v1 maps and one 2v2 map, original layouts inspired by Angoville, Semois and
  a classic 2v2 map.
- Mechanics: three-resource economy with upkeep and pop cap, sector territory with
  supply connection, capture points, observation posts, victory points, squads with
  per-model HP, range-banded accuracy, directional cover, garrisoning, suppression and
  pinning, retreat and reinforce, team weapons with setup and arcs, indirect fire,
  vehicle armor facing and penetration, construction, production queues, tech tree,
  global and squad upgrades, fog of war with building ghosts.
- A lightweight web viewer for replays and live games.
- Scripted bot ladder, benchmark runner, Jev/LLM adapters, RL training.

### Out of scope (for now)

Doctrines/commander trees; all activated abilities (grenades, satchels, sticky bombs,
panzerfausts, sniper camouflage, etc.); mines, sandbags, wire, tank traps; vehicle
criticals; British and Panzer Elite; 3v3+. Veterancy is deferred to milestone 2.
The order schema is designed so activated abilities can be added later as new typed
orders without changing the interface shape.

## 2. Architecture

### 2.1 World representation: hybrid

- **Static world on a grid** of ~2 m cells, stored as NumPy layers: passability
  (infantry / vehicle), cover type, LOS blocking, sector id.
- **Units have continuous positions and headings** (float arrays). Movement is
  `pos += v·dt` along an A* path computed on the grid; vehicles have hull turn rate and
  turret traverse.
- **The squad is the simulated entity**: position, heading, member list (each member:
  HP, weapon slot). Members sit at formation offsets from the squad position and take
  cover from the cell they stand on.
- Orders from agents target cells or entity ids, so the agent-facing interface is
  identical to what a pure-grid sim would expose.

### 2.2 Layout

```
coh/
  data/      YAML stat tables: units, weapons, target tables, buildings, upgrades,
             cover modifiers, economy constants. Every value carries provenance
             (`source: <url-or-dump>` or `estimated: <reason>`).
  sim/       deterministic core (see 2.3)
  maps/      map format + the three shipped maps + authoring tool
  env/       multi-agent env API, order schema, observation builder, encoders
  agents/    scripted bots, Jev adapter, generic LLM adapter, RL policy wrapper
  bench/     match runner, ladder eval, Elo, result tables
  replay/    replay = map + config + seed + per-tick order log
  viewer/    python server entry point (`python -m coh.viewer replay.json`)
viewer/      static single-page canvas app, no build step
train/       PPO training (milestone 3)
tests/
```

Dependency rule: `sim` imports only `data` and `maps`. Only `env` and `replay` touch
`sim`. Agents see only fog-filtered observations from `env`. The viewer consumes only
replay/state-stream JSON.

### 2.3 Sim core

- Fixed tick: 8 ticks per game-second. Single seeded RNG (`numpy.random.Generator`)
  owned by the sim; no other randomness source.
- State is a plain data structure (dataclasses + arrays) with a stable `state_hash()`.
- Systems run in fixed order each tick:
  1. **orders** — validate and apply newly issued orders
  2. **production/tech** — construction progress, train queues, research
  3. **movement** — pathing, steering, setup/teardown, retreat movement
  4. **vision** — per-team visibility grid, building ghosts
  5. **combat** — target acquisition, shots, damage, suppression, deaths
  6. **garrison** — enter/exit, building damage, forced ejection on destruction
  7. **territory** — capture progress, ownership, supply connectivity
  8. **economy** — income, upkeep, pop cap
  9. **victory** — VP ticket drain, annihilation check
- Performance: hot paths (visibility, LOS, target acquisition) are vectorized NumPy,
  with Numba as the first escalation. Target: ≥ 1,000 ticks/s per core on a mid-game
  1v1 state, so 20 cores give ≥ 2,500 game-seconds/s of experience. If not reached,
  profile and port only the hot system.

## 3. Game model

All numbers come from `coh/data/`. This section specifies rules only.

### 3.1 Economy and territory

- Resources: manpower (MP), munitions, fuel. MP has a base income reduced by upkeep
  per population point in use. Munitions and fuel come only from held sectors.
- The map is a graph of sectors; each has exactly one capture point of type strategic,
  munitions (low/med/high), fuel (low/med/high) or victory point.
- Capture: an infantry squad issued `Capture` stands in the point radius. Enemy-owned
  points must be neutralized first, then captured. Progress stalls while enemy infantry
  contest the radius. Vehicles and retreating squads cannot capture.
- **Supply**: a sector yields income and pop cap only if connected to the owner's HQ
  sector through a chain of owned sectors. Recomputed on every ownership change.
- Observation posts: built by engineers/pioneers on an owned point; raise income and
  make capture slower; destructible.
- Pop cap: base value plus a per-connected-sector bonus, up to a maximum.
- Victory: each team starts with 500 tickets. While a team holds fewer VPs than the
  enemy, it loses tickets at a rate proportional to the difference. A team at 0 tickets
  loses. A team whose HQs are all destroyed loses. A configurable time limit ends the
  game as a win for the ticket leader (draw if equal).

### 3.2 Squads and combat

- Shot resolution: `p_hit = accuracy[range_band] × cover_mod(target) ×
  moving_mod(attacker) × target_table_mod`. On hit, a random living member takes
  `damage × target_table_damage_mod × cover_damage_mod`. Weapons have cooldown, burst
  and reload cycles.
- Target acquisition: squads auto-engage the highest-priority visible enemy in range
  (per-weapon priority: AT weapons prefer vehicles, etc.) unless given `Attack`.
- **Cover**: per-cell type none / light / heavy / negative. Directional: a cover cell's
  modifier applies only if the cover-providing object lies between shooter and target
  (tested by the incoming-fire direction against the cover object's side). Garrisoned
  buildings apply their own modifier.
- **Garrison**: infantry enter an adjacent enterable building (capacity limit), fire
  from its sides, and are ejected with damage if the building is destroyed.
- **Suppression**: per-squad meter. Incoming fire adds suppression (weapon-specific,
  reduced by cover); it decays when not under fire. Above the *suppressed* threshold:
  slow movement, reduced accuracy and rate of fire. Above the *pinned* threshold: no
  movement or fire. Retreat clears it.
- **Retreat**: squad runs to its HQ, is uncontrollable en route, receives reduced
  damage, and ignores suppression. **Reinforce**: near the HQ (or other reinforce
  point), pay per-model cost and time to restore members.
- **Team weapons** (HMG, mortar, AT gun): setup and teardown times, firing arc set by
  `SetFacing`; if the crew dies the weapon remains and can be re-crewed by any infantry
  squad. Mortars fire indirectly at visible targets with scatter.
- **Vehicles**: front and rear armor. `p_pen = clamp(penetration[range_band] /
  armor[facing_hit])`. Non-penetrating hits deflect (data-defined reduced damage).
  Vehicles crush light cover objects they drive over. Snipers exist as a unit
  (long range, high single-model kill chance) without camouflage.

### 3.3 Production and tech

- Engineers (US) / Pioneers (Wehrmacht) construct buildings at a chosen valid location
  inside owned, connected territory. Construction takes time, pauses when the builder
  leaves, and can be assisted by multiple builders.
- Buildings train units through a FIFO queue, research global upgrades, and gate tech:
  - US: HQ → Barracks, Weapons Support Center → Motor Pool → Tank Depot; Supply Yard
    (upkeep reduction upgrades). Triage Center is omitted.
  - Wehrmacht: HQ phase escalations 1→4 gate Wehrmacht Quarters → Krieg Barracks →
    Sturm Armory → Panzer Command. Kampfkraft Center arrives with veterancy (M2).
- Squad weapon upgrades (e.g. BAR, MG42 LMG, Panzerschreck) are bought per squad with
  munitions in friendly territory and replace member weapon slots.

### 3.4 Vision

Per-unit sight radius, blocked by LOS-blocking cells. Visibility is per team (teammates
share). Enemy units are observable only while visible. Enemy buildings, once seen,
persist as last-known ghosts until re-observed.

## 4. Agent interface

### 4.1 API

PettingZoo-style parallel env.

```python
env = CohEnv(map="angoville", players=[...], decision_interval_s=2.0, seed=0)
obs = env.reset()                                  # {player_id: Observation}
obs, rewards, dones, infos = env.step({player_id: [Order, ...]})
```

`step` applies orders, then advances `decision_interval_s` of game time (configurable;
default 2 s = 16 ticks). Orders persist until completed or replaced; an empty order
list is valid.

### 4.2 Orders (single typed schema for all agents)

```
Move(squad, cell)            AttackMove(squad, cell)      Attack(squad, target_id)
Capture(squad, point_id)     Garrison(squad, building_id) Ungarrison(squad)
Retreat(squad)               Reinforce(squad)             SetFacing(squad, direction)
Build(builder, structure_type, cell)                      Train(building, unit_type)
Research(building, upgrade)  BuyUpgrade(squad, weapon_upgrade)
```

- The env exposes the current legal-order set (affordability, tech, pop cap, valid ids).
- Jev/LLM adapters receive the schema as JSON Schema with valid ids enumerated.
- RL agents get a multi-head action (order type → actor → target/cell) with invalid
  action masks; cell targets are over the map grid downsampled to a fixed resolution.
- Invalid orders are dropped and counted in `infos[player]["invalid_orders"]`.

### 4.3 Observations

One fog-filtered structured `Observation` per player: resources, income, pop, tech
state, own squads and buildings (HP, members, suppression, current order, cover state,
position), visible enemies, building ghosts, sector ownership and capture progress,
tickets, game time, legal orders. Two encoders:

- `to_json()` — compact text for Jev/LLMs, with named points ("north fuel") and coarse
  coordinates; size-bounded.
- `to_tensors()` — entity table (padded, masked) plus spatial planes (terrain, cover,
  territory, visibility, friendly/enemy presence).

### 4.4 Rewards

Default: +1 / −1 / 0 at game end. Optional shaping terms (ticket differential,
territory held, resource value destroyed minus lost), each with a configurable weight.
Benchmarks always report true outcomes.

## 5. Benchmark

- **Scripted ladder** (all use the public env API):
  `T0-idle`, `T1-capper`, `T2-standard` (build order, MGs in cover, retreat at low HP,
  tech to armor), `T3-aggressive` (combined arms, flanking, supply cuts, AT focus).
- **Runner**: N seeded matches per (agent, opponent, map, faction), sides swapped.
  Metrics: win rate, ticket margin, game length, invalid-order rate, decision latency,
  API cost. Head-to-head results feed an Elo table.
- **Timing modes**: *paused* (sim waits for the agent; default, removes network jitter)
  and *real-time budget* (a response slower than the decision interval forfeits those
  decision steps).
- 2v2: four independent agents; mixed teams allowed.

## 6. Viewer

Static single-page canvas app served by `python -m coh.viewer`. Plays a replay file or
follows a live game over a websocket. Renders territory tint by owner, cover zones
(green/yellow/red), buildings, squads as member dots with heading, vehicles with
turrets, tracers and explosions, health and suppression bars, capture progress, fog
overlay (per-player or omniscient toggle), resource and VP HUD. Controls: play/pause,
speed, timeline scrub, click to inspect. M2 adds a side panel showing the selected
agent's last observation and orders.

Replays store map, config, seed and the order log; the viewer server re-simulates to
produce frames, so replays are tiny and exact.

## 7. Error handling

- Invalid orders never raise; they are dropped, counted and reported in `infos`.
- Agent adapter failures (timeout, API error, schema violation) are treated as an empty
  order list for that step, counted per match, and retried with backoff only in paused
  mode. A match with adapter failure rate above a threshold is flagged in results.
- Data loading validates every stat table against a schema at import time and fails
  fast with the file and key.
- Map loading validates: every sector has exactly one point, the sector graph is
  connected, HQ positions are passable and pathable to every point.

## 8. Testing

- Unit tests per system (directional cover, supply cut-off stops income, rear-armor
  penetration, suppression thresholds, capture contest, pop cap, build prerequisites).
- Determinism: same map + seed + orders ⇒ identical `state_hash()` sequence; replay
  round-trip.
- Scenario tests with CoH-expected outcomes (statistical over seeds): HMG in cover
  beats a frontal rifle assault and loses to a flank; AT gun beats a Sherman frontally
  at range and loses when flanked; garrisoned squad beats the same squad in the open.
- Ladder ordering: T3 > T2 > T1 > T0 by win rate.
- Performance regression test against the ticks/s target.

## 9. Milestones

Each milestone gets its own implementation plan.

- **M1 — playable sim**: data tables, sim core, one 1v1 map, replay, viewer, `T0`/`T1`
  bots. Exit: watch a full bot-vs-bot game in the viewer; determinism and scenario
  tests pass.
- **M2 — benchmark**: env API + encoders, Jev and LLM adapters, `T2`/`T3` bots, runner
  and Elo, 2v2, remaining maps, veterancy, agent-inspection panel. Exit: a published
  results table for Jev and at least one LLM against the ladder.
- **M3 — RL**: vectorized multi-process rollout, PPO with action masking against the
  ladder and self-play, checkpoints entered into the benchmark.

## 10. Data sourcing

Stat values are imported from community-extracted CoH1 data where available; each value
records its provenance. Values that cannot be sourced are estimated, marked
`estimated`, and tuned using the scenario tests in section 8. A survey of available
sources is recorded in `docs/data-sources.md`.
