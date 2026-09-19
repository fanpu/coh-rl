/* The replay model: frame lookup, interpolation, terrain deltas, and fog.
 *
 * Both views consume this module and nothing else for "what is on the map and
 * may I see it?", so the fog rules cannot diverge between the 2D tactical map
 * and the 3D battlefield. If a squad is hidden here it is absent from both
 * renderers and unpickable in both.
 */

import { S } from './state.js';
import { NEUTRAL, TEAM_COLOURS, clamp, lerp, lerpAngle, mix } from './util.js';
import { DEFAULT_EFFECT_TICKS, EVENT_EFFECTS, MARKER_COLOURS, MAX_EFFECT_TICKS } from './effects.js';

/** The loaded replay. Mutated in place by `prepare` so importers keep a live view. */
export const R = {
  D: null,              // the whole frames payload
  W: 0, H: 0,           // map size in cells
  CELL: 2,              // metres per cell
  TPS: 8,
  frames: [],
  frameTicks: [],
  allEvents: [],
  markers: [],
  baseTerrain: null,
  terr: null,           // terrain chars with deltas up to `terrApplied` applied
  terrApplied: -1,
  terrDeltas: [],
  terrVersion: 0,       // bumped whenever `terr` changes, so views can rebuild
  sectorCells: null,    // Int16Array, sector id per cell
  sectorPoint: {}       // sector id -> point id
};

let visIndex = {};
let visCache = {};
let seenBuildings = {};   // team -> {building id: {b, t}} ever seen by that team
let seenUpto = {};        // team -> last frame index folded into seenBuildings

/** The current fog context, or null in omniscient mode. Set by `setFogFrame`. */
export let fog = null;

// ------------------------------------------------------------------ loading --

export function prepare(payload) {
  R.D = payload;
  R.frames = payload.frames;
  R.W = payload.map.width;
  R.H = payload.map.height;
  R.CELL = payload.map.cell_m || 2;
  R.TPS = (payload.meta && payload.meta.ticks_per_second) || 8;

  R.frameTicks = R.frames.map(function (f) { return f.t; });

  R.baseTerrain = new Array(R.W * R.H);
  for (let cy = 0; cy < R.H; cy++) {
    const row = payload.map.terrain[cy] || '';
    for (let cx = 0; cx < R.W; cx++) R.baseTerrain[cy * R.W + cx] = row.charAt(cx) || '.';
  }
  R.terr = R.baseTerrain.slice();
  R.terrApplied = -1;
  R.terrVersion++;

  R.terrDeltas = [];
  R.frames.forEach(function (f, i) {
    (f.terrain_delta || []).forEach(function (d) {
      R.terrDeltas.push({ i: i, cx: d[0], cy: d[1], ch: d[2] });
    });
  });

  R.sectorCells = new Int16Array(R.W * R.H);
  for (let y = 0; y < R.H; y++) {
    const srow = payload.map.sectors[y] || [];
    for (let x = 0; x < R.W; x++) R.sectorCells[y * R.W + x] = srow[x] || 0;
  }
  R.sectorPoint = {};
  payload.map.points.forEach(function (p) { R.sectorPoint[p.sector] = p.id; });

  seenBuildings = {};
  seenUpto = {};
  fog = null;
  visIndex = {};
  visCache = {};
  R.frames.forEach(function (f, i) {
    const v = f.vis || {};
    for (const team in v) {
      if (!visIndex[team]) visIndex[team] = [];
      visIndex[team].push(i);
    }
  });

  R.allEvents = [];
  R.markers = [];
  R.frames.forEach(function (f) {
    (f.events || []).forEach(function (e) {
      R.allEvents.push({ k: e.k, t: e.t, d: e.d || {} });
      const colour = MARKER_COLOURS[e.k];
      if (colour) R.markers.push({ t: e.t, c: colour });
    });
  });
  R.allEvents.sort(function (a, b) { return a.t - b.t; });
}

// -------------------------------------------------------------- team/player --

export function playerOf(id) { return (R.D && R.D.players && R.D.players[id]) || null; }

export function teamOf(playerId) {
  const p = playerOf(playerId);
  return p ? p.team : 0;
}

/** Player colour: team hue, desaturated toward field grey for Wehrmacht. */
export function playerColour(playerId) {
  const p = playerOf(playerId);
  if (!p) return NEUTRAL;
  const base = TEAM_COLOURS[p.team % TEAM_COLOURS.length] || NEUTRAL;
  return p.faction === 'wehr' ? mix(base, '#8d9196', 0.4) : base;
}

export function teamColour(team) {
  if (team === null || team === undefined) return NEUTRAL;
  return TEAM_COLOURS[team % TEAM_COLOURS.length] || NEUTRAL;
}

// --------------------------------------------------- terrain / vis lookups --

/** Roll `R.terr` forward (or rebuild it) so it matches frame `frameIdx`. */
export function applyTerrainTo(frameIdx) {
  if (frameIdx < R.terrApplied) {
    R.terr = R.baseTerrain.slice();
    R.terrApplied = -1;
    R.terrVersion++;
  }
  let changed = false;
  for (let n = 0; n < R.terrDeltas.length; n++) {
    const d = R.terrDeltas[n];
    if (d.i > frameIdx || d.i <= R.terrApplied) continue;
    if (d.cx >= 0 && d.cx < R.W && d.cy >= 0 && d.cy < R.H) {
      R.terr[d.cy * R.W + d.cx] = d.ch;
      changed = true;
    }
  }
  R.terrApplied = frameIdx;
  if (changed) R.terrVersion++;
}

function decodeBits(b64) {
  if (visCache[b64]) return visCache[b64];
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  visCache[b64] = bytes;
  return bytes;
}

/** Visibility grid for `team` at or before frame `idx`, or null. */
export function visAt(team, idx) {
  const list = visIndex[String(team)];
  if (!list || !list.length) return null;
  let lo = 0, hi = list.length - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid] <= idx) { best = list[mid]; lo = mid + 1; } else hi = mid - 1;
  }
  if (best < 0) best = list[0];
  return decodeBits(R.frames[best].vis[String(team)]);
}

export function visibleAt(bits, cx, cy) {
  if (!bits) return true;
  if (cx < 0 || cx >= R.W || cy < 0 || cy >= R.H) return false;
  const n = cy * R.W + cx;
  return (bits[n >> 3] & (128 >> (n & 7))) !== 0;
}

// ------------------------------------------------------------------- fog --

/* Everything a renderer needs to answer "can the team I am watching see
 * this?". `fog` is null in omniscient mode, and every helper below then says
 * yes, so the omniscient path costs nothing. */

/** Recompute the fog context for frame `idx`. Call once per rendered frame. */
export function setFogFrame(idx) {
  if (S.fogMode === 0) { fog = null; return fog; }
  const team = S.fogMode - 1;
  const bits = visAt(team, idx);
  fog = bits ? { team: team, bits: bits, idx: idx } : null;
  return fog;
}

export function isVisibleCell(cx, cy) {
  return fog === null || visibleAt(fog.bits, cx, cy);
}

export function isVisibleWorld(x, y) {
  if (fog === null) return true;
  return visibleAt(fog.bits, Math.floor(x / R.CELL), Math.floor(y / R.CELL));
}

/** A building counts as seen while any cell of its footprint is. */
export function isFootprintVisible(b) {
  if (fog === null) return true;
  return footprintSeen(fog.bits, b);
}

function footprintSeen(bits, b) {
  for (let cy = b.cy; cy < b.cy + b.h; cy++) {
    for (let cx = b.cx; cx < b.cx + b.w; cx++) {
      if (visibleAt(bits, cx, cy)) return true;
    }
  }
  return false;
}

/** Neutral entities (`owner === null`) belong to nobody and are never enemies. */
export function isEnemy(owner) {
  return fog !== null && owner !== null && owner !== undefined && teamOf(owner) !== fog.team;
}

/* Per-team memory of enemy buildings: once a team has seen one, it keeps
 * knowing where it was and what it looked like, even after it leaves vision
 * (or is destroyed). Folded in frame by frame and cached, and rebuilt from
 * scratch on a backward seek — the same trick `applyTerrainTo` uses. */
export function seenBuildingsFor(team, idx) {
  const key = String(team);
  if (seenUpto[key] === undefined || idx < seenUpto[key]) {
    seenBuildings[key] = {};
    seenUpto[key] = -1;
  }
  const seen = seenBuildings[key];
  for (let i = seenUpto[key] + 1; i <= idx; i++) {
    const bits = visAt(team, i);
    if (!bits) continue;
    const frame = R.frames[i];
    for (let j = 0; j < frame.buildings.length; j++) {
      const b = frame.buildings[j];
      if (b.nu || b.o === null || b.o === undefined || teamOf(b.o) === team) continue;
      if (!footprintSeen(bits, b)) continue;
      seen[b.id] = { b: b, t: frame.t };
    }
  }
  seenUpto[key] = idx;
  return seen;
}

/** The memory for the fog context currently in force, or null when omniscient. */
export function seenNow() {
  return fog ? seenBuildingsFor(fog.team, fog.idx) : null;
}

/** 'live' (draw it as it is), 'ghost' (last known) or 'hidden' (never seen). */
export function buildingFogMode(b, seen) {
  if (fog === null || b.nu || !isEnemy(b.o)) return 'live';
  if (isFootprintVisible(b)) return 'live';
  return seen && seen[b.id] ? 'ghost' : 'hidden';
}

export function squadHidden(s) {
  if (fog === null || !isEnemy(s.o)) return false;
  return !isVisibleCell(Math.floor(s.x / R.CELL), Math.floor(s.y / R.CELL));
}

/** A garrisoned squad is drawn as a badge on its building, not as a unit. */
export function squadOnMap(s) {
  return s.g === undefined && !squadHidden(s);
}

/** The buildings a renderer should actually draw, each tagged live or ghost.
 *
 *  One list, used by both views and by hit testing, so a building can never be
 *  visible in one place and hidden in another. */
export function visibleBuildings(frame, stats) {
  const seen = seenNow();
  const out = [];
  const drawn = {};
  frame.buildings.forEach(function (b) {
    const mode = buildingFogMode(b, seen);
    if (mode === 'hidden') { if (stats) stats.hiddenBuildings++; return; }
    drawn[b.id] = true;
    if (mode === 'ghost') {
      out.push({ b: seen[b.id].b, seenTick: seen[b.id].t });
      if (stats) stats.ghosts++;
    } else {
      out.push({ b: b, seenTick: null });
      if (stats) stats.buildings++;
    }
  });
  // A building the team saw and has since lost sight of — or that has been
  // destroyed behind its back — is still remembered where it stood.
  if (seen) {
    Object.keys(seen).forEach(function (id) {
      if (drawn[id]) return;
      out.push({ b: seen[id].b, seenTick: seen[id].t });
      if (stats) stats.ghosts++;
    });
  }
  return out;
}

export function emptyStats() {
  return {
    buildings: 0, ghosts: 0, hiddenBuildings: 0, squads: 0, hiddenSquads: 0,
    effects: 0, hiddenEffects: 0, badges: 0
  };
}

// ------------------------------------------------------------ frame lookup --

export function frameIndexFor(t) {
  let lo = 0, hi = R.frameTicks.length - 1, best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (R.frameTicks[mid] <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}

export function lastTick() { return R.frameTicks[R.frameTicks.length - 1] || 0; }

export function currentFrame() { return R.frames[frameIndexFor(S.tick)]; }

/** Positions/headings interpolated between the bracketing frames. */
export function interpolated(a, b, u) {
  const byId = {};
  (b.squads || []).forEach(function (s) { byId[s.id] = s; });
  return (a.squads || []).map(function (s) {
    const n = byId[s.id];
    if (!n || u <= 0) return s;
    const out = Object.create(s);
    out.x = lerp(s.x, n.x, u);
    out.y = lerp(s.y, n.y, u);
    out.h = lerpAngle(s.h, n.h, u);
    out.th = lerpAngle(s.th, n.th, u);
    // a moving squad bobs in 3D; the 2D map ignores this
    out.moving = Math.hypot(n.x - s.x, n.y - s.y) > 0.05;
    return out;
  });
}

/** The frame pair and blend factor for the current playhead. */
export function playhead() {
  const idx = frameIndexFor(S.tick);
  const a = R.frames[idx];
  const b = R.frames[Math.min(idx + 1, R.frames.length - 1)];
  const span = b.t - a.t;
  return { idx: idx, a: a, b: b, u: span > 0 ? clamp((S.tick - a.t) / span, 0, 1) : 0 };
}

// ---------------------------------------------------------------- effects --

/** Events whose effect is still alive at `t`, each tagged with age in 0..1. */
export function activeEffects(t) {
  const out = [];
  let lo = 0, hi = R.allEvents.length - 1, start = R.allEvents.length;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (R.allEvents[mid].t >= t - MAX_EFFECT_TICKS) { start = mid; hi = mid - 1; } else lo = mid + 1;
  }
  for (let i = start; i < R.allEvents.length && R.allEvents[i].t <= t; i++) {
    const e = R.allEvents[i];
    const spec = EVENT_EFFECTS[e.k];
    const span = (spec && spec.ticks) || DEFAULT_EFFECT_TICKS;
    const age = (t - e.t) / span;
    if (age >= 0 && age <= 1) out.push({ k: e.k, d: e.d, t: e.t, age: age, spec: spec });
  }
  return out;
}

export function entityPos(frame, id) {
  for (let i = 0; i < frame.squads.length; i++) {
    if (frame.squads[i].id === id) return [frame.squads[i].x, frame.squads[i].y];
  }
  for (let j = 0; j < frame.buildings.length; j++) {
    const b = frame.buildings[j];
    if (b.id === id) return [(b.cx + b.w / 2) * R.CELL, (b.cy + b.h / 2) * R.CELL];
  }
  return null;
}

export function eventPos(e, frame) {
  const d = e.d || {};
  if (Array.isArray(d.pos) && d.pos.length === 2) return [d.pos[0], d.pos[1]];
  if (Array.isArray(d.cell) && d.cell.length === 2) {
    return [(d.cell[0] + 0.5) * R.CELL, (d.cell[1] + 0.5) * R.CELL];
  }
  if (typeof d.point_id === 'string') {
    for (let i = 0; i < R.D.map.points.length; i++) {
      const p = R.D.map.points[i];
      if (p.id === d.point_id) return [(p.cell[0] + 0.5) * R.CELL, (p.cell[1] + 0.5) * R.CELL];
    }
  }
  if (typeof d.building === 'number') return entityPos(frame, d.building);
  if (typeof d.squad === 'number') return entityPos(frame, d.squad);
  if (typeof d.id === 'number') return entityPos(frame, d.id);
  return null;
}

/** Would the team being watched have seen this happen?
 *
 *  A shot counts if either end is visible — you see your own squad fire into
 *  the dark, and you see rounds arriving from an unseen shooter. Anything else
 *  needs its own position in vision. An event with no position at all is
 *  dropped under fog: there is nowhere to check, so it cannot be vouched for. */
export function effectVisible(e, frame) {
  if (fog === null) return true;
  const d = e.d || {};
  if (e.k === 'shot') {
    return (Array.isArray(d.src_pos) && isVisibleWorld(d.src_pos[0], d.src_pos[1])) ||
           (Array.isArray(d.dst_pos) && isVisibleWorld(d.dst_pos[0], d.dst_pos[1]));
  }
  const w = eventPos(e, frame);
  return w !== null && isVisibleWorld(w[0], w[1]);
}

/** The effects a renderer should actually draw, after the fog filter.
 *
 *  Kinds with `mode: null` are shown elsewhere and drop out before the
 *  visibility check, so point pulses stay omniscient (sector ownership is map
 *  knowledge in CoH). */
export function visibleEffects(effects, frame, stats) {
  const out = [];
  effects.forEach(function (e) {
    if (e.spec !== undefined && e.spec.mode === null) return;
    if (!effectVisible(e, frame)) { if (stats) stats.hiddenEffects++; return; }
    out.push(e);
    if (stats) stats.effects++;
  });
  return out;
}

// ---------------------------------------------------------------- lookups --

export function squadMeta(defId) {
  return (R.D.meta.squad_defs && R.D.meta.squad_defs[defId]) || {};
}

/** Fraction of a queued train/research item that is already done. */
export function queueFrac(item) {
  const meta = R.D.meta || {};
  const def = (meta.squad_defs || {})[item[1]] || (meta.building_defs || {})[item[1]];
  const total = (def && def.build_time) || (meta.research_times || {})[item[1]];
  if (!total) return 0;
  return clamp(1 - item[2] / total, 0, 1);
}

/** The last `game_over` event at or before the playhead, or null. */
export function gameOverAt(t) {
  for (let i = R.allEvents.length - 1; i >= 0; i--) {
    if (R.allEvents[i].k === 'game_over') return R.allEvents[i].t <= t ? R.allEvents[i] : null;
  }
  return null;
}
