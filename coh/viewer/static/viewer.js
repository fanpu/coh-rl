/* CoH replay viewer — a dependency-free canvas app over the frame JSON
 * produced by `coh.viewer.frames`.
 *
 * Layers, bottom up: terrain -> sector tint + borders -> cover hints ->
 * points -> buildings -> squads -> combat effects -> fog -> HUD.
 *
 * It never assumes more than the frame schema: unknown terrain characters,
 * squad kinds and event kinds all fall back to a generic marker rather than
 * throwing.
 */
'use strict';

// ---------------------------------------------------------------- palette --

var TEAM_COLOURS = ['#4f8fe0', '#d8503c'];
var NEUTRAL = '#b9b2a0';

var TERRAIN = {
  '.': { base: '#49612d', name: 'Grass' },
  'r': { base: '#a08e60', name: 'Road (negative cover)' },
  'c': { base: '#4a3d27', name: 'Crater (heavy cover)' },
  'f': { base: '#49612d', name: 'Fence (light cover)' },
  'w': { base: '#9d9887', name: 'Wall (heavy cover)' },
  'H': { base: '#1b2f12', name: 'Hedgerow' },
  'T': { base: '#284a1b', name: 'Trees (light cover)' },
  '~': { base: '#204a68', name: 'Water' }
};
var UNKNOWN_TERRAIN = { base: '#6a3a6a', name: 'Unknown' };

// cover class per terrain char: 2 = heavy, 1 = light, 0 = none/negative.
var COVER = { '.': 0, 'r': 0, 'c': 2, 'f': 1, 'w': 2, 'H': 2, 'T': 1, '~': 0 };

var ACRONYMS = { hmg: 'HMG', at: 'AT', mg: 'MG', hq: 'HQ', op: 'OP', us: 'US', vp: 'VP', aa: 'AA' };
var FACTION_NAMES = { us: 'US', wehr: 'Wehrmacht' };

var TILE = 8;          // offscreen terrain resolution, px per cell
var SPEEDS = [0.5, 1, 2, 4, 8, 16];

// ------------------------------------------------------------------ state --

var D = null;                 // the whole frames payload
var W = 0, H = 0, CELL = 2;   // map size in cells, metres per cell
var TPS = 8;

var cam = { x: 0, y: 0, scale: 5 };   // x/y = world metres at the viewport centre
var playing = false;
var speedIdx = 1;
var tick = 0;                 // playhead, in (fractional) sim ticks
var fogMode = 0;              // 0 omniscient, 1 team 0, 2 team 1
var showCover = false;
var showSectors = true;
var selection = null;         // {kind: 'squad'|'building', id}
var lastRAF = 0, rafPending = false;

var cv = document.getElementById('cv');
var ctx = cv.getContext('2d');
var dpr = Math.min(window.devicePixelRatio || 1, 2);

// Derived indexes, filled by prepare().
var frames = [], frameTicks = [], allEvents = [], markers = [];
var visIndex = {}, visCache = {};
var baseTerrain = null, terr = null, terrDeltas = [], terrApplied = -1;
var sectorPoint = {};         // sector id -> point id
var sectorCells = null;       // Int16Array sector id per cell
var layers = {};              // cached offscreen canvases
var labelBoxes = [];          // text boxes already placed this frame
var fog = null;               // the current fog context, recomputed each render
var seenBuildings = {};       // team -> {building id: {b, t}} ever seen by that team
var seenUpto = {};            // team -> last frame index folded into seenBuildings
var renderStats = emptyStats();

// ------------------------------------------------------------------ utils --

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

/** Pixels per metre for unit *shapes*: never below a floor, so squads stay
 *  readable when the whole 96x96 map is fitted to the window. */
function unitPx() { return Math.max(cam.scale, 6.5); }
function lerp(a, b, u) { return a + (b - a) * u; }

function lerpAngle(a, b, u) {
  var d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return a + d * u;
}

function title(id) {
  if (!id) return '';
  return String(id).split(/[_\s]+/).map(function (w) {
    if (!w) return '';
    if (ACRONYMS[w.toLowerCase()]) return ACRONYMS[w.toLowerCase()];
    return w.charAt(0).toUpperCase() + w.slice(1);
  }).join(' ');
}

function factionName(faction) {
  return FACTION_NAMES[faction] || title(faction) || '?';
}

function mmss(seconds) {
  var s = Math.max(0, Math.floor(seconds));
  return Math.floor(s / 60) + ':' + ('0' + (s % 60)).slice(-2);
}

/** Accepts "#rrggbb" or "rgb(r,g,b)" and returns [r, g, b]. */
function parseColour(c) {
  if (c.charAt(0) === '#') {
    var n = parseInt(c.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  var m = c.match(/-?\d+(\.\d+)?/g) || [0, 0, 0];
  return [+m[0] || 0, +m[1] || 0, +m[2] || 0];
}

function mix(colourA, colourB, u) {
  var a = parseColour(colourA), b = parseColour(colourB);
  return 'rgb(' + Math.round(lerp(a[0], b[0], u)) + ',' +
    Math.round(lerp(a[1], b[1], u)) + ',' + Math.round(lerp(a[2], b[2], u)) + ')';
}

/** `rgb(...)` -> `rgba(..., alpha)`; any colour string -> rgba. */
function alpha(colour, a) {
  var c = parseColour(colour);
  return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
}

function shade(hex, u) { return mix(hex, u < 0 ? '#000000' : '#ffffff', Math.abs(u)); }

/** Deterministic per-cell noise so terrain texture does not swim while panning. */
function hash2(cx, cy) {
  var h = (cx * 374761393 + cy * 668265263) | 0;
  h = (h ^ (h >> 13)) * 1274126177;
  return ((h ^ (h >> 16)) >>> 0) / 4294967296;
}

// -------------------------------------------------------------- team/player --

function playerOf(id) { return (D && D.players && D.players[id]) || null; }

function teamOf(playerId) {
  var p = playerOf(playerId);
  return p ? p.team : 0;
}

/** Player colour: team hue, desaturated toward field grey for Wehrmacht. */
function playerColour(playerId) {
  var p = playerOf(playerId);
  if (!p) return NEUTRAL;
  var base = TEAM_COLOURS[p.team % TEAM_COLOURS.length] || NEUTRAL;
  return p.faction === 'wehr' ? mix(base, '#8d9196', 0.4) : base;
}

function teamColour(team) {
  if (team === null || team === undefined) return NEUTRAL;
  return TEAM_COLOURS[team % TEAM_COLOURS.length] || NEUTRAL;
}

// ------------------------------------------------------------------ loading --

function fail(message) {
  document.getElementById('loading').hidden = true;
  document.getElementById('error').hidden = false;
  document.getElementById('error-msg').textContent = message;
}

function boot() {
  fetch('frames.json').then(function (r) {
    if (!r.ok) throw new Error('GET /frames.json -> HTTP ' + r.status + ' ' + r.statusText);
    return r.json();
  }).then(function (data) {
    if (!data || !data.frames || !data.frames.length) throw new Error('frame stream is empty');
    D = data;
    prepare();
    document.getElementById('loading').hidden = true;
    ['hud', 'bar'].forEach(function (id) { document.getElementById(id).hidden = false; });
    buildLegend();
    resize();
    resetCamera();
    // `#paused` opens on the first frame without starting playback, which is
    // what a deep link into a moment (and the headless browser check) wants.
    setPlaying(window.location.hash.indexOf('paused') < 0);
    requestRender();
  }).catch(function (err) {
    fail(String(err && err.message ? err.message : err));
  });
}

function prepare() {
  frames = D.frames;
  W = D.map.width; H = D.map.height; CELL = D.map.cell_m || 2;
  TPS = (D.meta && D.meta.ticks_per_second) || 8;

  frameTicks = frames.map(function (f) { return f.t; });

  baseTerrain = new Array(W * H);
  for (var cy = 0; cy < H; cy++) {
    var row = D.map.terrain[cy] || '';
    for (var cx = 0; cx < W; cx++) baseTerrain[cy * W + cx] = row.charAt(cx) || '.';
  }
  terr = baseTerrain.slice();
  terrApplied = -1;

  terrDeltas = [];
  frames.forEach(function (f, i) {
    (f.terrain_delta || []).forEach(function (d) {
      terrDeltas.push({ i: i, cx: d[0], cy: d[1], ch: d[2] });
    });
  });

  sectorCells = new Int16Array(W * H);
  for (var y = 0; y < H; y++) {
    var srow = D.map.sectors[y] || [];
    for (var x = 0; x < W; x++) sectorCells[y * W + x] = srow[x] || 0;
  }
  D.map.points.forEach(function (p) { sectorPoint[p.sector] = p.id; });

  seenBuildings = {};
  seenUpto = {};
  fog = null;

  visIndex = {};
  frames.forEach(function (f, i) {
    var v = f.vis || {};
    for (var team in v) {
      if (!visIndex[team]) visIndex[team] = [];
      visIndex[team].push(i);
    }
  });

  allEvents = [];
  markers = [];
  frames.forEach(function (f, i) {
    (f.events || []).forEach(function (e) {
      allEvents.push({ k: e.k, t: e.t, d: e.d || {}, i: i });
      var colour = MARKER_COLOURS[e.k];
      if (colour) markers.push({ t: e.t, c: colour });
    });
  });
  allEvents.sort(function (a, b) { return a.t - b.t; });

  layers = {};
  tick = 0;
}

// --------------------------------------------------- terrain / vis lookups --

function applyTerrainTo(frameIdx) {
  if (frameIdx < terrApplied) { terr = baseTerrain.slice(); terrApplied = -1; delete layers.terrain; delete layers.cover; }
  var changed = false;
  for (var n = 0; n < terrDeltas.length; n++) {
    var d = terrDeltas[n];
    if (d.i > frameIdx || d.i <= terrApplied) continue;
    if (d.cx >= 0 && d.cx < W && d.cy >= 0 && d.cy < H) { terr[d.cy * W + d.cx] = d.ch; changed = true; }
  }
  terrApplied = frameIdx;
  if (changed) { delete layers.terrain; delete layers.cover; }
}

function decodeBits(b64) {
  if (visCache[b64]) return visCache[b64];
  var bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  visCache[b64] = bytes;
  return bytes;
}

/** Visibility grid for `team` at or before frame `idx`, or null. */
function visAt(team, idx) {
  var list = visIndex[String(team)];
  if (!list || !list.length) return null;
  var lo = 0, hi = list.length - 1, best = -1;
  while (lo <= hi) {
    var mid = (lo + hi) >> 1;
    if (list[mid] <= idx) { best = list[mid]; lo = mid + 1; } else hi = mid - 1;
  }
  if (best < 0) best = list[0];
  return decodeBits(frames[best].vis[String(team)]);
}

function visibleAt(bits, cx, cy) {
  if (!bits) return true;
  if (cx < 0 || cx >= W || cy < 0 || cy >= H) return false;
  var n = cy * W + cx;
  return (bits[n >> 3] & (128 >> (n & 7))) !== 0;
}

// ------------------------------------------------------------------- fog --

/* Everything the renderer needs to answer "can the team I am watching see
 * this?". `fog` is null in omniscient mode, and every helper below then says
 * yes, so the omniscient path costs nothing. */

function fogContext(idx) {
  if (fogMode === 0) return null;
  var team = fogMode - 1;
  var bits = visAt(team, idx);
  return bits ? { team: team, bits: bits, idx: idx } : null;
}

function isVisibleCell(cx, cy) {
  return fog === null || visibleAt(fog.bits, cx, cy);
}

function isVisibleWorld(x, y) {
  if (fog === null) return true;
  return visibleAt(fog.bits, Math.floor(x / CELL), Math.floor(y / CELL));
}

/** A building counts as seen while any cell of its footprint is. */
function isFootprintVisible(b) {
  if (fog === null) return true;
  for (var cy = b.cy; cy < b.cy + b.h; cy++) {
    for (var cx = b.cx; cx < b.cx + b.w; cx++) {
      if (visibleAt(fog.bits, cx, cy)) return true;
    }
  }
  return false;
}

/** Neutral entities (`owner === null`) belong to nobody and are never enemies. */
function isEnemy(owner) {
  return fog !== null && owner !== null && owner !== undefined && teamOf(owner) !== fog.team;
}

/* Per-team memory of enemy buildings: once a team has seen one, it keeps
 * knowing where it was and what it looked like, even after it leaves vision
 * (or is destroyed). Folded in frame by frame and cached, and rebuilt from
 * scratch on a backward seek — the same trick `applyTerrainTo` uses. */
function seenBuildingsFor(team, idx) {
  var key = String(team);
  if (seenUpto[key] === undefined || idx < seenUpto[key]) {
    seenBuildings[key] = {};
    seenUpto[key] = -1;
  }
  var seen = seenBuildings[key];
  for (var i = seenUpto[key] + 1; i <= idx; i++) {
    var bits = visAt(team, i);
    if (!bits) continue;
    var frame = frames[i];
    for (var j = 0; j < frame.buildings.length; j++) {
      var b = frame.buildings[j];
      if (b.nu || b.o === null || b.o === undefined || teamOf(b.o) === team) continue;
      if (!footprintSeen(bits, b)) continue;
      seen[b.id] = { b: b, t: frame.t };
    }
  }
  seenUpto[key] = idx;
  return seen;
}

function footprintSeen(bits, b) {
  for (var cy = b.cy; cy < b.cy + b.h; cy++) {
    for (var cx = b.cx; cx < b.cx + b.w; cx++) {
      if (visibleAt(bits, cx, cy)) return true;
    }
  }
  return false;
}

/** 'live' (draw it as it is), 'ghost' (last known) or 'hidden' (never seen). */
function buildingFogMode(b, seen) {
  if (fog === null || b.nu || !isEnemy(b.o)) return 'live';
  if (isFootprintVisible(b)) return 'live';
  return seen && seen[b.id] ? 'ghost' : 'hidden';
}

function squadHidden(s) {
  if (fog === null || !isEnemy(s.o)) return false;
  return !isVisibleCell(Math.floor(s.x / CELL), Math.floor(s.y / CELL));
}

function emptyStats() {
  return { buildings: 0, ghosts: 0, hiddenBuildings: 0, squads: 0, hiddenSquads: 0,
           effects: 0, hiddenEffects: 0, badges: 0 };
}

// ----------------------------------------------------------------- camera --

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(window.innerWidth * dpr);
  cv.height = Math.round(window.innerHeight * dpr);
  var tl = document.getElementById('timeline');
  tl.width = Math.round(tl.clientWidth * dpr);
  tl.height = Math.round(tl.clientHeight * dpr);
}

function resetCamera() {
  var wm = W * CELL, hm = H * CELL;
  var padX = 40, padTop = 90, padBottom = 90;
  var sx = (window.innerWidth - 2 * padX) / wm;
  var sy = (window.innerHeight - padTop - padBottom) / hm;
  cam.scale = Math.max(0.5, Math.min(sx, sy));
  cam.x = wm / 2;
  cam.y = hm / 2 + (padTop - padBottom) / (2 * cam.scale);
}

function toScreen(x, y) {
  return [(x - cam.x) * cam.scale + window.innerWidth / 2, (y - cam.y) * cam.scale + window.innerHeight / 2];
}

function toWorld(px, py) {
  return [(px - window.innerWidth / 2) / cam.scale + cam.x, (py - window.innerHeight / 2) / cam.scale + cam.y];
}

// ---------------------------------------------------------- offscreen layers --

function offscreen(w, h) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

/* Terrain is drawn in two passes because per-cell canvas calls over a
 * 96x96 grid are ruinously slow on software rendering (~40 ms/cell-op in a
 * headless browser): first the flat base colours as one pixel-per-cell
 * ImageData scaled up, then detail shapes for the few hundred non-grass
 * cells. Grouping the details by character keeps canvas state changes down. */
function terrainLayer() {
  if (layers.terrain) return layers.terrain;

  var small = offscreen(W, H), sg = small.getContext('2d');
  var img = sg.createImageData(W, H), px = img.data;
  var byChar = {}, tone = {};
  for (var i = 0; i < W * H; i++) {
    var ch = terr[i];
    var n = hash2(i % W, (i / W) | 0);
    var step = Math.round(n * 7);
    var key = ch + step;
    var rgb = tone[key];
    if (!rgb) {
      rgb = tone[key] = parseColour(shade((TERRAIN[ch] || UNKNOWN_TERRAIN).base, (step / 7 - 0.5) * 0.16));
    }
    px[i * 4] = rgb[0]; px[i * 4 + 1] = rgb[1]; px[i * 4 + 2] = rgb[2]; px[i * 4 + 3] = 255;
    if (ch !== '.') (byChar[ch] || (byChar[ch] = [])).push(i);
  }
  sg.putImageData(img, 0, 0);

  var c = offscreen(W * TILE, H * TILE), g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.drawImage(small, 0, 0, W * TILE, H * TILE);

  function each(ch, draw) {
    var cells = byChar[ch];
    if (!cells) return;
    for (var k = 0; k < cells.length; k++) {
      var idx = cells[k], cx = idx % W, cy = (idx / W) | 0;
      draw(cx * TILE, cy * TILE, hash2(cx, cy));
    }
  }

  // hedgerow: a lighter crown on the dark mass, so the banks read as walls
  g.fillStyle = 'rgba(122,152,92,0.22)';
  each('H', function (x, y) { g.fillRect(x + 1, y + 1, TILE - 2, TILE - 2); });

  // trees: dark canopy blobs
  g.fillStyle = 'rgba(16,36,12,0.8)';
  g.beginPath();
  each('T', function (x, y, n) {
    g.moveTo(x + TILE * (0.63 + n * 0.4), y + TILE * (0.3 + n * 0.4));
    g.arc(x + TILE * (0.3 + n * 0.4), y + TILE * (0.3 + n * 0.4), TILE * 0.33, 0, 6.2832);
  });
  g.fill();

  // craters: a dark pit
  g.fillStyle = 'rgba(0,0,0,0.45)';
  g.beginPath();
  each('c', function (x, y) {
    g.moveTo(x + TILE * 0.92, y + TILE / 2);
    g.ellipse(x + TILE / 2, y + TILE / 2, TILE * 0.42, TILE * 0.34, 0, 0, 6.2832);
  });
  g.fill();

  // fences: a low brown rail
  g.strokeStyle = '#8a6437';
  g.lineWidth = Math.max(1, TILE * 0.2);
  g.beginPath();
  each('f', function (x, y) { g.moveTo(x, y + TILE / 2); g.lineTo(x + TILE, y + TILE / 2); });
  g.stroke();

  // walls: mortar lines
  g.strokeStyle = 'rgba(40,40,35,0.55)';
  g.lineWidth = 1;
  g.beginPath();
  each('w', function (x, y) { g.rect(x + 0.5, y + 0.5, TILE - 1, TILE - 1); });
  g.stroke();

  // water: a highlight ripple
  g.fillStyle = 'rgba(170,205,225,0.1)';
  g.beginPath();
  each('~', function (x, y) { g.rect(x, y + TILE * 0.55, TILE, TILE * 0.2); });
  g.fill();

  layers.terrain = c;
  return c;
}

function coverLayer() {
  if (layers.cover) return layers.cover;
  var c = offscreen(W * TILE, H * TILE), g = c.getContext('2d');
  [2, 1].forEach(function (level) {
    g.fillStyle = level === 2 ? 'rgba(90,225,110,0.6)' : 'rgba(235,205,70,0.55)';
    g.beginPath();
    for (var i = 0; i < W * H; i++) {
      if ((COVER[terr[i]] || 0) !== level) continue;
      var x = (i % W) * TILE + TILE / 2, y = ((i / W) | 0) * TILE + TILE / 2;
      g.moveTo(x + TILE * 0.3, y);
      g.arc(x, y, TILE * 0.3, 0, 6.2832);
    }
    g.fill();
  });
  layers.cover = c;
  return c;
}

/** Sector outlines, each drawn just inside its own sector in its owner's
 *  colour — the strongest cue for who holds what, CoH tactical-map style. */
function sectorLinesLayer(frame) {
  var key = frame.points.map(function (p) { return p.owner === null ? '-' : p.owner; }).join('');
  if (layers.linesKey === key && layers.lines) return layers.lines;

  var owners = {};
  frame.points.forEach(function (p) { owners[p.id] = p.owner; });
  function ownerColour(sector) {
    var pid = sectorPoint[sector];
    var owner = pid === undefined ? null : owners[pid];
    return owner === null || owner === undefined ? NEUTRAL : teamColour(owner);
  }

  var c = offscreen(W * TILE, H * TILE), g = c.getContext('2d');
  var inset = TILE * 0.35;
  var paths = {};
  function seg(colour, x1, y1, x2, y2) {
    if (!paths[colour]) paths[colour] = [];
    paths[colour].push([x1, y1, x2, y2]);
  }
  for (var cy = 0; cy < H; cy++) {
    for (var cx = 0; cx < W; cx++) {
      var s = sectorCells[cy * W + cx];
      var col = ownerColour(s);
      var x = cx * TILE, y = cy * TILE;
      if (cx + 1 >= W || sectorCells[cy * W + cx + 1] !== s) seg(col, x + TILE - inset, y, x + TILE - inset, y + TILE);
      if (cx - 1 < 0 || sectorCells[cy * W + cx - 1] !== s) seg(col, x + inset, y, x + inset, y + TILE);
      if (cy + 1 >= H || sectorCells[(cy + 1) * W + cx] !== s) seg(col, x, y + TILE - inset, x + TILE, y + TILE - inset);
      if (cy - 1 < 0 || sectorCells[(cy - 1) * W + cx] !== s) seg(col, x, y + inset, x + TILE, y + inset);
    }
  }
  g.lineWidth = Math.max(2, TILE * 0.3);
  g.lineCap = 'square';
  for (var colour in paths) {
    g.strokeStyle = colour;
    g.beginPath();
    paths[colour].forEach(function (p) { g.moveTo(p[0], p[1]); g.lineTo(p[2], p[3]); });
    g.stroke();
  }
  layers.lines = c; layers.linesKey = key;
  return c;
}

/** Sector ownership tint; rebuilt only when the owner set changes. */
function tintLayer(frame) {
  var key = frame.points.map(function (p) { return p.owner === null ? '-' : p.owner; }).join('');
  if (layers.tintKey === key && layers.tint) return layers.tint;
  var owners = {};
  frame.points.forEach(function (p) { owners[p.id] = p.owner; });

  var c = offscreen(W, H), g = c.getContext('2d');
  var img = g.createImageData(W, H), px = img.data;
  var rgb = {};
  for (var i = 0; i < W * H; i++) {
    var pid = sectorPoint[sectorCells[i]];
    var owner = pid === undefined ? null : owners[pid];
    if (owner === null || owner === undefined) continue;
    if (!rgb[owner]) {
      var hex = teamColour(owner);
      rgb[owner] = parseColour(hex);
    }
    px[i * 4] = rgb[owner][0]; px[i * 4 + 1] = rgb[owner][1]; px[i * 4 + 2] = rgb[owner][2];
    px[i * 4 + 3] = 30;
  }
  g.putImageData(img, 0, 0);
  layers.tint = c; layers.tintKey = key;
  return c;
}

function fogLayer(bits) {
  var c = layers.fog || (layers.fog = offscreen(W, H));
  if (layers.fogBits === bits) return c;
  layers.fogBits = bits;
  var g = c.getContext('2d');
  var img = g.createImageData(W, H), px = img.data;
  for (var i = 0; i < W * H; i++) {
    var seen = (bits[i >> 3] & (128 >> (i & 7))) !== 0;
    px[i * 4 + 3] = seen ? 0 : 150;
  }
  g.putImageData(img, 0, 0);
  return c;
}

function blitCells(layer, alpha, smooth) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = !!smooth;
  var o = toScreen(0, 0);
  ctx.drawImage(layer, o[0], o[1], W * CELL * cam.scale, H * CELL * cam.scale);
  ctx.restore();
}

// ------------------------------------------------------------ frame lookup --

function frameIndexFor(t) {
  var lo = 0, hi = frameTicks.length - 1, best = 0;
  while (lo <= hi) {
    var mid = (lo + hi) >> 1;
    if (frameTicks[mid] <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}

/** Positions/headings interpolated between the bracketing frames. */
function interpolated(a, b, u) {
  var byId = {};
  (b.squads || []).forEach(function (s) { byId[s.id] = s; });
  return (a.squads || []).map(function (s) {
    var n = byId[s.id];
    if (!n || u <= 0) return s;
    var out = Object.create(s);
    out.x = lerp(s.x, n.x, u);
    out.y = lerp(s.y, n.y, u);
    out.h = lerpAngle(s.h, n.h, u);
    out.th = lerpAngle(s.th, n.th, u);
    return out;
  });
}

function lastTick() { return frameTicks[frameTicks.length - 1]; }

// ------------------------------------------------------------------- draw --

function drawPoints(frame, now) {
  var footprints = buildingRects(frame);
  var pulses = {};
  now.forEach(function (e) {
    if (e.k === 'point_captured' || e.k === 'point_neutralized') pulses[e.d.point_id] = e.age;
  });
  var byId = {};
  frame.points.forEach(function (p) { byId[p.id] = p; });

  D.map.points.forEach(function (def) {
    var st = byId[def.id] || {};
    var p = toScreen((def.cell[0] + 0.5) * CELL, (def.cell[1] + 0.5) * CELL);
    var r = clamp(cam.scale * 2.4, 9, 26);
    var colour = st.owner === null || st.owner === undefined ? NEUTRAL : teamColour(st.owner);

    if (pulses[def.id] !== undefined) {
      var a = 1 - pulses[def.id];
      ctx.strokeStyle = 'rgba(255,240,180,' + (a * 0.9).toFixed(2) + ')';
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(p[0], p[1], r * (1 + (1 - a) * 2.4), 0, 6.2832); ctx.stroke();
    }

    ctx.save();
    ctx.translate(p[0], p[1]);
    ctx.fillStyle = 'rgba(12,14,10,0.72)';
    ctx.beginPath(); ctx.arc(0, 0, r, 0, 6.2832); ctx.fill();
    ctx.strokeStyle = colour; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.arc(0, 0, r, 0, 6.2832); ctx.stroke();

    // capture progress ring
    if (st.progress > 0 && st.progress < 1 && st.c !== null && st.c !== undefined) {
      ctx.strokeStyle = teamColour(st.c); ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(0, 0, r + 3, -Math.PI / 2, -Math.PI / 2 + 6.2832 * st.progress); ctx.stroke();
    }
    if (st.op) {
      ctx.strokeStyle = '#f0e2b0'; ctx.lineWidth = 1.5;
      ctx.strokeRect(-r - 6, -r - 6, 2 * r + 12, 2 * r + 12);
    }
    drawPointIcon(def.type, r * 0.62, colour);
    ctx.restore();

    if (showSectors && cam.scale > 2.2) {
      placeLabel(def.name || title(def.id), p[0], p[1], r, '#efe9d8',
                 '600 11px system-ui, sans-serif', footprints);
    }
  });
}

function drawPointIcon(type, s, colour) {
  ctx.fillStyle = colour;
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.6;
  if (type === 'victory') {
    ctx.beginPath();
    for (var i = 0; i < 10; i++) {
      var a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? s * 0.45 : s;
      ctx[i ? 'lineTo' : 'moveTo'](Math.cos(a) * rr, Math.sin(a) * rr);
    }
    ctx.closePath(); ctx.fill();
  } else if (type.indexOf('fuel') === 0) {
    ctx.fillRect(-s * 0.55, -s * 0.8, s * 1.1, s * 1.6);
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.beginPath();
    ctx.moveTo(-s * 0.55, -s * 0.3); ctx.lineTo(s * 0.55, -s * 0.3);
    ctx.moveTo(-s * 0.55, s * 0.3); ctx.lineTo(s * 0.55, s * 0.3);
    ctx.stroke();
  } else if (type.indexOf('munitions') === 0) {
    ctx.fillRect(-s * 0.85, -s * 0.6, s * 1.7, s * 1.2);
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.beginPath();
    ctx.moveTo(-s * 0.85, -s * 0.6); ctx.lineTo(s * 0.85, s * 0.6);
    ctx.moveTo(-s * 0.85, s * 0.6); ctx.lineTo(s * 0.85, -s * 0.6);
    ctx.stroke();
  } else if (type === 'hq') {
    ctx.fillRect(-s * 0.8, -s * 0.8, s * 1.6, s * 1.6);
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.strokeRect(-s * 0.4, -s * 0.4, s * 0.8, s * 0.8);
  } else {
    // strategic (and anything unknown): a flag
    ctx.beginPath();
    ctx.moveTo(-s * 0.4, s); ctx.lineTo(-s * 0.4, -s);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-s * 0.4, -s); ctx.lineTo(s * 0.8, -s * 0.55); ctx.lineTo(-s * 0.4, -s * 0.1);
    ctx.closePath(); ctx.fill();
  }
}

function drawBuildings(frame) {
  var seen = fog ? seenBuildingsFor(fog.team, fog.idx) : null;
  var drawn = {};

  frame.buildings.forEach(function (b) {
    var mode = buildingFogMode(b, seen);
    if (mode === 'hidden') { renderStats.hiddenBuildings++; return; }
    drawn[b.id] = true;
    if (mode === 'ghost') { drawBuilding(seen[b.id].b, seen[b.id].t); renderStats.ghosts++; }
    else { drawBuilding(b, null); renderStats.buildings++; }
  });

  // A building the team saw and has since lost sight of — or that has been
  // destroyed behind its back — is still remembered where it stood.
  if (seen) {
    Object.keys(seen).forEach(function (id) {
      if (drawn[id]) return;
      drawBuilding(seen[id].b, seen[id].t);
      renderStats.ghosts++;
    });
  }
}

/** `seenTick` null = live; otherwise this is a last-known ghost from that tick. */
function drawBuilding(b, seenTick) {
  var ghost = seenTick !== null && seenTick !== undefined;
  var o = toScreen(b.cx * CELL, b.cy * CELL);
  var w = b.w * CELL * cam.scale, h = b.h * CELL * cam.scale;
  var colour = ghost ? '#7d7a70' : (b.nu ? '#8a8171' : playerColour(b.o));
  var done = b.prog >= 1;

  ctx.save();
  ctx.fillStyle = ghost ? 'rgba(20,22,17,0.55)'
    : (done ? mix(colour, '#12140f', 0.55) : 'rgba(30,33,25,0.75)');
  ctx.fillRect(o[0], o[1], w, h);

  if (!done && !ghost) {
    ctx.save();
    ctx.beginPath(); ctx.rect(o[0], o[1], w, h); ctx.clip();
    ctx.strokeStyle = 'rgba(230,220,180,0.35)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var d = -h; d < w; d += 7) { ctx.moveTo(o[0] + d, o[1] + h); ctx.lineTo(o[0] + d + h, o[1]); }
    ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = colour;
  ctx.lineWidth = selection && selection.kind === 'building' && selection.id === b.id ? 3 : 1.8;
  if (ghost) ctx.setLineDash([5, 4]);
  ctx.strokeRect(o[0] + 0.5, o[1] + 0.5, w - 1, h - 1);
  ctx.setLineDash([]);

  if (ghost) {
    // No live bars, badge or queue: this is memory, not observation.
    if (cam.scale > 4.5) {
      label(title(b.def) + ' (' + mmss(seenTick / TPS) + ')', o[0] + w / 2, o[1] + h + 12,
            'rgba(190,185,172,0.85)');
    }
    ctx.restore();
    return;
  }

  // health / construction bar
  var frac = done ? b.hp : b.prog;
  if (frac < 0.999) {
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(o[0], o[1] - 6, w, 4);
    ctx.fillStyle = done ? barColour(b.hp) : '#d8b45a';
    ctx.fillRect(o[0], o[1] - 6, w * clamp(frac, 0, 1), 4);
  }

  // production queue progress (when inspected)
  if (selection && selection.kind === 'building' && selection.id === b.id && b.q && b.q.length) {
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(o[0], o[1] + h + 2, w, 4);
    ctx.fillStyle = '#9fc4e8';
    ctx.fillRect(o[0], o[1] + h + 2, w * queueFrac(b.q[0]), 4);
  }

  // A neutral building's garrison is only knowable while you can see it.
  if (b.n > 0 && (!b.nu || isFootprintVisible(b))) {
    badge(o[0] + w - 2, o[1] + 2, String(b.n), colour);
    renderStats.badges++;
  }

  if (cam.scale > 4.5) label(title(b.def), o[0] + w / 2, o[1] + h + 12);
  ctx.restore();
}

/** Fraction of a queued train/research item that is already done. */
function queueFrac(item) {
  var meta = D.meta || {};
  var def = (meta.squad_defs || {})[item[1]] || (meta.building_defs || {})[item[1]];
  var total = (def && def.build_time) || (meta.research_times || {})[item[1]];
  if (!total) return 0;
  return clamp(1 - item[2] / total, 0, 1);
}

function badge(x, y, text, colour) {
  ctx.save();
  ctx.fillStyle = colour;
  ctx.strokeStyle = 'rgba(0,0,0,0.7)';
  ctx.beginPath(); ctx.arc(x, y + 7, 8, 0, 6.2832); ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#12140f';
  ctx.font = '700 10px system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y + 7.5);
  ctx.restore();
}

/** Centred text with a dark halo so it stays legible over any terrain.
 *  Labels that would collide with one already placed this frame are dropped,
 *  which keeps a crowded map readable. */
function label(text, x, y, colour, font, avoid) {
  ctx.save();
  ctx.font = font || '10px system-ui, sans-serif';
  var w = ctx.measureText(text).width;
  var box = [x - w / 2 - 2, y - 10, x + w / 2 + 2, y + 3];
  var blockers = avoid ? labelBoxes.concat(avoid) : labelBoxes;
  for (var i = 0; i < blockers.length; i++) {
    var o = blockers[i];
    if (box[0] < o[2] && o[0] < box[2] && box[1] < o[3] && o[1] < box[3]) { ctx.restore(); return false; }
  }
  labelBoxes.push(box);
  ctx.textAlign = 'center';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(8,10,6,0.85)';
  ctx.strokeText(text, x, y);
  ctx.fillStyle = colour || 'rgba(238,233,220,0.92)';
  ctx.fillText(text, x, y);
  ctx.restore();
  return true;
}

/** Label a map feature, trying below / above / right / left of it until a
 *  position is free — an HQ point's name must not end up under the HQ. */
function placeLabel(text, x, y, r, colour, font, avoid) {
  ctx.save();
  ctx.font = font || '10px system-ui, sans-serif';
  var half = ctx.measureText(text).width / 2;
  ctx.restore();
  var spots = [
    [x, y + r + 14],
    [x, y - r - 7],
    [x + r + 8 + half, y + 4],
    [x - r - 8 - half, y + 4]
  ];
  for (var i = 0; i < spots.length; i++) {
    if (label(text, spots[i][0], spots[i][1], colour, font, avoid)) return true;
  }
  return false;
}

/** Screen-space footprint rectangles, used to keep labels off buildings.
 *  Only buildings that are actually drawn count: a label must not dodge
 *  something the viewer is hiding (and a hidden building must not be
 *  detectable by the way it nudges labels around). */
function buildingRects(frame) {
  var seen = fog ? seenBuildingsFor(fog.team, fog.idx) : null;
  var rects = [];
  function add(b) {
    var o = toScreen(b.cx * CELL, b.cy * CELL);
    rects.push([o[0] - 2, o[1] - 8, o[0] + b.w * CELL * cam.scale + 2, o[1] + b.h * CELL * cam.scale + 14]);
  }
  var drawn = {};
  frame.buildings.forEach(function (b) {
    var mode = buildingFogMode(b, seen);
    if (mode === 'hidden') return;
    drawn[b.id] = true;
    add(mode === 'ghost' ? seen[b.id].b : b);
  });
  if (seen) Object.keys(seen).forEach(function (id) { if (!drawn[id]) add(seen[id].b); });
  return rects;
}

function barColour(frac) {
  return frac > 0.6 ? '#7fbf6a' : (frac > 0.3 ? '#d8b45a' : '#e2705f');
}

function drawSquads(squads) {
  var offsets = (D.meta && D.meta.formation_offsets) || [[0, 0]];
  squads.forEach(function (s) {
    if (squadHidden(s)) { renderStats.hiddenSquads++; return; }
    if (s.g !== undefined) return;   // garrisoned: shown as a badge on the building
    renderStats.squads++;

    var meta = (D.meta.squad_defs && D.meta.squad_defs[s.def]) || {};
    var p = toScreen(s.x, s.y);
    var colour = s.ab ? '#8f8a7c' : playerColour(s.o);
    var sel = selection && selection.kind === 'squad' && selection.id === s.id;

    ctx.save();

    // owner glow so squads read at map zoom
    var gr = unitPx() * 2.9;
    var glow = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], gr);
    glow.addColorStop(0, alpha(mix(colour, '#000000', 0.55), 0.85));
    glow.addColorStop(0.55, alpha(mix(colour, '#000000', 0.5), 0.5));
    glow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(p[0], p[1], gr, 0, 6.2832); ctx.fill();

    if (s.kind === 'team_weapon') drawTeamWeapon(s, meta, p, colour, offsets);
    else if (s.kind === 'vehicle') drawVehicle(s, meta, p, colour);
    else if (s.kind === 'infantry') drawInfantry(s, p, colour, offsets);
    else drawGeneric(p, colour);

    if (sel) {
      ctx.strokeStyle = '#f2e9c8'; ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(12, cam.scale * 2.4), 0, 6.2832); ctx.stroke();
      ctx.setLineDash([]);
    }

    drawSquadStatus(s, p, colour);
    ctx.restore();
  });
}

function memberDot(p, ox, oy, cos, sin, r, colour, alive) {
  var k = unitPx();
  var x = p[0] + (ox * cos - oy * sin) * k;
  var y = p[1] + (ox * sin + oy * cos) * k;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 6.2832);
  ctx.fillStyle = alive ? colour : 'rgba(80,80,74,0.6)';
  ctx.fill();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = 'rgba(8,10,6,0.95)';
  ctx.stroke();
}

function drawInfantry(s, p, colour, offsets) {
  var r = clamp(unitPx() * 0.55, 3, 7);
  var cos = Math.cos(s.h), sin = Math.sin(s.h);
  for (var i = 0; i < s.n; i++) {
    var off = offsets[i % offsets.length];
    memberDot(p, off[0], off[1], cos, sin, r, colour, true);
  }
}

function drawTeamWeapon(s, meta, p, colour, offsets) {
  if (s.fa !== undefined) {
    var arc = ((meta.arc_deg || 90) * Math.PI / 180) / 2;
    var reach = (meta.range || 30) * cam.scale;
    var grad = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], reach);
    grad.addColorStop(0, alpha(mix(colour, '#ffffff', 0.3), 0.30));
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.arc(p[0], p[1], reach, s.fa - arc, s.fa + arc);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = alpha(mix(colour, '#ffffff', 0.35), 0.55);
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  var r = clamp(unitPx() * 0.55, 3, 7);
  var cos = Math.cos(s.h), sin = Math.sin(s.h);
  for (var i = 0; i < s.n; i++) {
    var off = offsets[i % offsets.length];
    memberDot(p, off[0], off[1], cos, sin, r, colour, true);
  }
  // the gun itself
  ctx.strokeStyle = s.ab ? '#6f6a5e' : '#14160f';
  ctx.lineWidth = Math.max(2.5, unitPx() * 0.4);
  var a = s.fa !== undefined ? s.fa : s.h;
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(p[0] + Math.cos(a) * unitPx() * 2.0, p[1] + Math.sin(a) * unitPx() * 2.0);
  ctx.stroke();

  if (s.setup !== undefined) {
    ctx.fillStyle = '#d8b45a';
    ctx.font = '700 10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    label('setting up', p[0], p[1] - unitPx() * 2.6, '#d8b45a', '700 10px system-ui, sans-serif');
  }
}

function drawVehicle(s, meta, p, colour) {
  var L = unitPx() * 3.4, Wd = unitPx() * 1.9;
  ctx.save();
  ctx.translate(p[0], p[1]);
  ctx.rotate(s.h);
  ctx.fillStyle = s.ab ? '#6f6a5e' : mix(colour, '#12140f', 0.35);
  ctx.strokeStyle = shade(colour, 0.25);
  ctx.lineWidth = 1.6;
  ctx.fillRect(-L / 2, -Wd / 2, L, Wd);
  ctx.strokeRect(-L / 2, -Wd / 2, L, Wd);
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.fillRect(-L / 2, -Wd / 2, L, Wd * 0.18);
  ctx.fillRect(-L / 2, Wd / 2 - Wd * 0.18, L, Wd * 0.18);
  ctx.restore();

  ctx.save();
  ctx.translate(p[0], p[1]);
  ctx.rotate(s.th);
  ctx.fillStyle = shade(colour, -0.2);
  ctx.beginPath(); ctx.arc(0, 0, Wd * 0.38, 0, 6.2832); ctx.fill();
  ctx.strokeStyle = '#14160f';
  ctx.lineWidth = Math.max(2.5, unitPx() * 0.34);
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(L * 0.8, 0); ctx.stroke();
  ctx.restore();
}

function drawGeneric(p, colour) {
  var r = unitPx() * 1.1;
  ctx.fillStyle = colour;
  ctx.strokeStyle = '#14160f';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(p[0], p[1] - r); ctx.lineTo(p[0] + r, p[1]); ctx.lineTo(p[0], p[1] + r); ctx.lineTo(p[0] - r, p[1]);
  ctx.closePath(); ctx.fill(); ctx.stroke();
}

function drawSquadStatus(s, p, colour) {
  var top = p[1] - unitPx() * 2.5;
  var w = unitPx() * 3.6;

  if (s.hp < 0.999 || cam.scale > 4) {
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(p[0] - w / 2, top, w, 3.5);
    ctx.fillStyle = barColour(s.hp);
    ctx.fillRect(p[0] - w / 2, top, w * clamp(s.hp, 0, 1), 3.5);
  }
  if (s.sup) {
    ctx.fillStyle = s.sup === 2 ? '#e2705f' : '#e8c84a';
    ctx.beginPath(); ctx.arc(p[0] - w / 2 - 5, top + 2, 3.2, 0, 6.2832); ctx.fill();
  }
  if (s.st === 'retreating') {
    ctx.fillStyle = '#f0e2b0';
    ctx.font = '700 11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('↑', p[0] + w / 2 + 6, top + 6);
  }
  if (s.re) {
    ctx.strokeStyle = '#7fbf6a';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(p[0] + w / 2 + 3, top + 3); ctx.lineTo(p[0] + w / 2 + 11, top + 3);
    ctx.moveTo(p[0] + w / 2 + 7, top - 1); ctx.lineTo(p[0] + w / 2 + 7, top + 7);
    ctx.stroke();
  }
  if (cam.scale > 5.5) {
    label(title(s.def) + ' ' + s.n + '/' + s.max, p[0], top - 5, 'rgba(238,233,220,0.9)');
  }
}

// ---------------------------------------------------------------- effects --

/* Every event kind `coh/sim/systems/*.py` emits gets a deliberate decision
 * here: an on-map effect, or `draw: null` when it is deliberately shown
 * somewhere else (points pulse themselves; `game_over` is the HUD banner).
 * A kind that is not listed — a newer sim than this page — falls through to
 * `genericMark`, so the viewer degrades instead of throwing. `ticks` is how
 * long the effect lives, in sim ticks. */
var EFFECTS = {
  shot: { ticks: 3, draw: function (e) { drawTracer(e); } },
  explosion: {
    ticks: 14,
    // the sim tells us the blast radius in metres; draw it at true size
    draw: function (e, f) { explosion(e, f, Math.max(Number(e.d.radius) || 0, 2)); }
  },
  squad_destroyed: { ticks: 20, draw: function (e, f) { explosion(e, f, 4.5); deathMark(e, f); } },
  vehicle_destroyed: { ticks: 28, draw: function (e, f) { explosion(e, f, 8); deathMark(e, f); } },
  building_destroyed: { ticks: 26, draw: function (e, f) { explosion(e, f, 9); deathMark(e, f); } },
  weapon_abandoned: { ticks: 18, draw: function (e, f) { ripple(e, f, '#8f8a7c', 3.5); } },
  weapon_recrewed: { ticks: 18, draw: function (e, f) { ripple(e, f, '#7fbf6a', 3.5); } },
  garrison_entered: { ticks: 14, draw: function (e, f) { ripple(e, f, '#9fc4e8', 2.5); } },
  garrison_ejected: { ticks: 20, draw: function (e, f) { explosion(e, f, 3); deathMark(e, f); } },
  reinforced: { ticks: 14, draw: function (e, f) { ripple(e, f, '#7fbf6a', 2.5); } },
  unit_trained: { ticks: 16, draw: function (e, f) { ripple(e, f, '#9fc4e8', 4); } },
  construction_started: { ticks: 16, draw: function (e, f) { ripple(e, f, '#d8b45a', 4); } },
  building_completed: { ticks: 16, draw: function (e, f) { ripple(e, f, '#7fbf6a', 6); } },
  research_completed: { ticks: 16, draw: function (e, f) { ripple(e, f, '#9fc4e8', 5); } },
  upgrade_bought: { ticks: 14, draw: function (e, f) { ripple(e, f, '#d8b45a', 2.5); } },
  point_captured: { ticks: 16, draw: null },     // pulse drawn with the point
  point_neutralized: { ticks: 16, draw: null },  // pulse drawn with the point
  game_over: { ticks: 1, draw: null }            // winner banner, see updateHud
};
var DEFAULT_EFFECT_TICKS = 16;
var MAX_EFFECT_TICKS = 30;

/** Timeline tick marks, by event kind (kinds not listed are not marked). */
var MARKER_COLOURS = {
  point_captured: '#d8b45a',
  point_neutralized: '#d8b45a',
  squad_destroyed: '#e2705f',
  vehicle_destroyed: '#e2705f',
  building_destroyed: '#e2705f',
  game_over: '#f2e9c8'
};

/** Events whose effect is still alive at `t`, each tagged with age in 0..1. */
function activeEffects(t) {
  var out = [];
  var lo = 0, hi = allEvents.length - 1, start = allEvents.length;
  while (lo <= hi) {
    var mid = (lo + hi) >> 1;
    if (allEvents[mid].t >= t - MAX_EFFECT_TICKS) { start = mid; hi = mid - 1; } else lo = mid + 1;
  }
  for (var i = start; i < allEvents.length && allEvents[i].t <= t; i++) {
    var e = allEvents[i];
    var span = (EFFECTS[e.k] && EFFECTS[e.k].ticks) || DEFAULT_EFFECT_TICKS;
    var age = (t - e.t) / span;
    if (age >= 0 && age <= 1) out.push({ k: e.k, d: e.d, age: age });
  }
  return out;
}

function entityPos(frame, id) {
  for (var i = 0; i < frame.squads.length; i++) if (frame.squads[i].id === id) return [frame.squads[i].x, frame.squads[i].y];
  for (var j = 0; j < frame.buildings.length; j++) {
    var b = frame.buildings[j];
    if (b.id === id) return [(b.cx + b.w / 2) * CELL, (b.cy + b.h / 2) * CELL];
  }
  return null;
}

function eventPos(e, frame) {
  var d = e.d || {};
  if (Array.isArray(d.pos) && d.pos.length === 2) return [d.pos[0], d.pos[1]];
  if (Array.isArray(d.cell) && d.cell.length === 2) return [(d.cell[0] + 0.5) * CELL, (d.cell[1] + 0.5) * CELL];
  if (typeof d.point_id === 'string') {
    for (var i = 0; i < D.map.points.length; i++) {
      var p = D.map.points[i];
      if (p.id === d.point_id) return [(p.cell[0] + 0.5) * CELL, (p.cell[1] + 0.5) * CELL];
    }
  }
  if (typeof d.building === 'number') return entityPos(frame, d.building);
  if (typeof d.squad === 'number') return entityPos(frame, d.squad);
  if (typeof d.id === 'number') return entityPos(frame, d.id);
  return null;
}

function drawEffects(effects, frame) {
  ctx.save();
  effects.forEach(function (e) {
    try {
      var spec = EFFECTS[e.k];
      if (spec !== undefined && !spec.draw) return;    // null = shown elsewhere
      if (!effectVisible(e, frame)) { renderStats.hiddenEffects++; return; }
      if (spec === undefined) genericMark(e, frame);   // a kind this page predates
      else spec.draw(e, frame);
      renderStats.effects++;
    } catch (err) { /* an effect must never break the frame */ }
  });
  ctx.restore();
}

/** Would the team being watched have seen this happen?
 *
 *  A shot counts if either end is visible — you see your own squad fire into
 *  the dark, and you see rounds arriving from an unseen shooter. Anything else
 *  needs its own position in vision. An event with no position at all is
 *  dropped under fog: there is nowhere to check, so it cannot be vouched for. */
function effectVisible(e, frame) {
  if (fog === null) return true;
  var d = e.d || {};
  if (e.k === 'shot') {
    return (Array.isArray(d.src_pos) && isVisibleWorld(d.src_pos[0], d.src_pos[1])) ||
           (Array.isArray(d.dst_pos) && isVisibleWorld(d.dst_pos[0], d.dst_pos[1]));
  }
  var w = eventPos(e, frame);
  return w !== null && isVisibleWorld(w[0], w[1]);
}

function drawTracer(e) {
  var d = e.d;
  if (!Array.isArray(d.src_pos) || !Array.isArray(d.dst_pos)) return;
  var a = toScreen(d.src_pos[0], d.src_pos[1]);
  var b = toScreen(d.dst_pos[0], d.dst_pos[1]);
  if (!d.hit) {
    // a miss lands short and wide of the target
    var dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
    b = [a[0] + dx * 0.93 - dy / len * 9, a[1] + dy * 0.93 + dx / len * 9];
  }
  var fade = 1 - e.age;
  ctx.strokeStyle = d.hit
    ? 'rgba(255,238,170,' + (0.95 * fade).toFixed(2) + ')'
    : 'rgba(255,225,190,' + (0.35 * fade).toFixed(2) + ')';
  ctx.lineWidth = d.hit ? 1.8 : 1.0;
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  if (d.hit) {
    ctx.fillStyle = 'rgba(255,230,140,' + (0.8 * fade).toFixed(2) + ')';
    ctx.beginPath(); ctx.arc(b[0], b[1], 2.4, 0, 6.2832); ctx.fill();
  }
}

/** Expanding ring that reaches `metres` (the real blast radius) at full age,
 *  plus a brief flash at the impact point. */
function explosion(e, frame, metres) {
  var w = eventPos(e, frame);
  if (!w) return;
  var p = toScreen(w[0], w[1]);
  var full = Math.max(metres * cam.scale, 6);
  var r = full * (0.15 + 0.85 * e.age);
  ctx.strokeStyle = 'rgba(255,180,90,' + ((1 - e.age) * 0.85).toFixed(2) + ')';
  ctx.lineWidth = Math.max(1.5, 4 * (1 - e.age));
  ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 6.2832); ctx.stroke();
  if (e.age < 0.3) {
    var flash = 1 - e.age / 0.3;
    var grad = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], full * 0.6);
    grad.addColorStop(0, 'rgba(255,240,190,' + (0.85 * flash).toFixed(2) + ')');
    grad.addColorStop(1, 'rgba(255,150,60,0)');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(p[0], p[1], full * 0.6, 0, 6.2832); ctx.fill();
  }
}

function deathMark(e, frame) {
  var w = eventPos(e, frame);
  if (!w) return;
  var p = toScreen(w[0], w[1]);
  var r = Math.max(4, cam.scale * 1.1);
  ctx.strokeStyle = 'rgba(40,40,38,' + ((1 - e.age) * 0.9).toFixed(2) + ')';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(p[0] - r, p[1] - r); ctx.lineTo(p[0] + r, p[1] + r);
  ctx.moveTo(p[0] + r, p[1] - r); ctx.lineTo(p[0] - r, p[1] + r);
  ctx.stroke();
}

function ripple(e, frame, colour, metres) {
  var w = eventPos(e, frame);
  if (!w) return;
  var p = toScreen(w[0], w[1]);
  ctx.strokeStyle = colour;
  ctx.globalAlpha = 1 - e.age;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(p[0], p[1], metres * cam.scale * (0.3 + e.age), 0, 6.2832); ctx.stroke();
  ctx.globalAlpha = 1;
}

function genericMark(e, frame) {
  var w = eventPos(e, frame);
  if (!w) return;
  var p = toScreen(w[0], w[1]);
  ctx.strokeStyle = 'rgba(240,226,176,' + ((1 - e.age) * 0.7).toFixed(2) + ')';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(4, cam.scale * 0.8), 0, 6.2832); ctx.stroke();
}

// ------------------------------------------------------------------ render --

function render() {
  var idx = frameIndexFor(tick);
  var a = frames[idx], b = frames[Math.min(idx + 1, frames.length - 1)];
  var span = b.t - a.t;
  var u = span > 0 ? clamp((tick - a.t) / span, 0, 1) : 0;
  applyTerrainTo(idx);
  fog = fogContext(idx);

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  ctx.fillStyle = '#0c0e0a';
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  blitCells(terrainLayer(), 1, cam.scale > 6);
  blitCells(tintLayer(a), 1, false);
  if (showSectors) blitCells(sectorLinesLayer(a), 0.9, true);
  if (showCover) blitCells(coverLayer(), 0.9, true);

  labelBoxes = [];
  renderStats = emptyStats();
  var effects = activeEffects(tick);
  // Sector ownership and capture progress are map knowledge in CoH, so points
  // (and their pulses) stay omniscient; units, buildings and effects do not.
  drawPoints(a, effects);
  drawBuildings(a);
  drawSquads(interpolated(a, b, u));
  drawEffects(effects, a);
  if (fog) blitCells(fogLayer(fog.bits), 1, true);

  updateHud(a);
  drawTimeline();
  if (selection) updatePanel(a);
}

// --------------------------------------------------------------------- HUD --

function updateHud(frame) {
  (D.players || []).forEach(function (p, i) {
    var el = document.getElementById('p' + i);
    if (!el) return;
    var r = frame.players[i] || { mp: 0, mu: 0, fu: 0, pop: 0, cap: 0 };
    el.style.setProperty('--team', playerColour(i));
    var over = r.pop > r.cap ? ' over' : '';
    el.innerHTML =
      '<div class="who">Player ' + i + ' <span class="fac">' + factionName(p.faction) +
      ' &middot; team ' + p.team + '</span></div>' +
      '<div class="res">' +
      '<span class="mp"><i>MP</i><b>' + Math.round(r.mp) + '</b></span>' +
      '<span class="mu"><i>MU</i><b>' + Math.round(r.mu) + '</b></span>' +
      '<span class="fu"><i>FU</i><b>' + Math.round(r.fu) + '</b></span>' +
      '<span class="pop' + over + '"><i>POP</i><b>' + r.pop + '/' + r.cap + '</b></span>' +
      '</div>';
  });

  document.getElementById('clock').textContent = mmss(frame.t / TPS);
  document.getElementById('tickets').innerHTML = (frame.tickets || []).map(function (v, team) {
    return '<span style="color:' + teamColour(team) + '">' + Math.round(v) + '</span>';
  }).join('<span style="color:#6b6656">/</span>');

  document.getElementById('mapname').textContent = title(D.map.name);
  updateBanner();
}

/** The `game_over` event, shown as a banner once the playhead reaches it. */
function updateBanner() {
  var banner = document.getElementById('banner');
  var over = null;
  for (var i = allEvents.length - 1; i >= 0; i--) {
    if (allEvents[i].k === 'game_over') { over = allEvents[i]; break; }
  }
  if (over === null || tick < over.t) { banner.hidden = true; return; }

  var winner = over.d.winner;
  if (winner === null || winner === undefined) winner = D.winner;
  var drawn = winner === -1 || winner === null || winner === undefined;
  banner.hidden = false;
  banner.style.setProperty('--team', drawn ? NEUTRAL : teamColour(winner));
  banner.innerHTML =
    '<b>' + (drawn ? 'Draw' : 'Team ' + winner + ' wins') + '</b>' +
    (over.d.reason ? '<span>' + title(over.d.reason) + '</span>' : '');
}

function drawTimeline() {
  var tl = document.getElementById('timeline');
  var g = tl.getContext('2d');
  var w = tl.clientWidth, h = tl.clientHeight;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  var total = lastTick() || 1;
  g.fillStyle = 'rgba(255,255,255,0.08)';
  g.fillRect(0, h / 2 - 4, w, 8);
  g.fillStyle = 'rgba(160,190,140,0.30)';
  g.fillRect(0, h / 2 - 4, w * clamp(tick / total, 0, 1), 8);

  markers.forEach(function (m) {
    g.fillStyle = m.c;
    g.fillRect(clamp(m.t / total, 0, 1) * w - 0.5, h / 2 - 9, 1.4, 18);
  });

  var x = clamp(tick / total, 0, 1) * w;
  g.fillStyle = '#f2e9c8';
  g.fillRect(x - 1.5, 1, 3, h - 2);

  g.fillStyle = 'rgba(200,192,170,0.75)';
  g.font = '10px system-ui, sans-serif';
  g.textAlign = 'right';
  g.fillText(mmss(total / TPS), w - 3, h - 2);
}

// ------------------------------------------------------------------ panel --

function closePanel() { selection = null; document.getElementById('panel').hidden = true; }

function updatePanel(frame) {
  var el = document.getElementById('panel');
  var body = document.getElementById('panel-body');
  var entity = null;
  var seenTick = null;
  if (selection.kind === 'squad') {
    entity = frame.squads.filter(function (s) { return s.id === selection.id; })[0];
    // An enemy squad out of vision is not reported on at all: the inspector
    // must never leak live state the watched team cannot observe.
    if (entity && squadHidden(entity)) {
      showHiddenPanel(el, 'Enemy squad', 'out of vision');
      return;
    }
  } else {
    entity = frame.buildings.filter(function (b) { return b.id === selection.id; })[0];
    var seen = fog ? seenBuildingsFor(fog.team, fog.idx) : null;
    var mode = entity ? buildingFogMode(entity, seen) : (seen && seen[selection.id] ? 'ghost' : 'live');
    if (mode === 'hidden') {
      showHiddenPanel(el, 'Enemy building', 'never seen');
      return;
    }
    if (mode === 'ghost' || (!entity && seen && seen[selection.id])) {
      entity = seen[selection.id].b;
      seenTick = seen[selection.id].t;
    }
  }

  if (!entity) {
    el.hidden = false;
    document.getElementById('panel-title').textContent = title(selection.kind) + ' ' + selection.id;
    body.innerHTML = '<p style="color:var(--dim)">not present in this frame</p>';
    return;
  }

  el.hidden = false;
  var swatch = seenTick !== null ? '#7d7a70' : (entity.nu ? NEUTRAL : playerColour(entity.o));
  document.getElementById('panel-title').innerHTML =
    '<span style="color:' + swatch + '">■</span> ' + title(entity.def) +
    ' <span style="color:var(--dim);font-weight:400">#' + entity.id +
    (seenTick !== null ? ' · last known' : '') + '</span>';

  var rows = [];
  function kv(k, v) { rows.push('<dt>' + k + '</dt><dd>' + v + '</dd>'); }

  if (seenTick !== null) {
    // Last known: position and identity only, never live HP, progress or queue.
    kv('owner', entity.o === null || entity.o === undefined ? 'neutral'
       : 'player ' + entity.o + ' (team ' + teamOf(entity.o) + ')');
    kv('cell', entity.cx + ', ' + entity.cy);
    kv('footprint', entity.w + ' × ' + entity.h + ' cells');
    kv('last seen', mmss(seenTick / TPS));
    body.innerHTML = '<dl class="kv">' + rows.join('') + '</dl>' +
      '<p style="color:var(--dim);margin:9px 0 0">Out of vision — showing the last ' +
      'thing team ' + fog.team + ' saw here.</p>';
    return;
  }

  if (selection.kind === 'squad') {
    var meta = (D.meta.squad_defs && D.meta.squad_defs[entity.def]) || {};
    kv('owner', entity.o === null ? 'neutral' : 'player ' + entity.o + ' (team ' + teamOf(entity.o) + ')');
    kv('kind', title(entity.kind));
    kv('state', title(entity.st));
    if (entity.ord) kv('order', entity.ord);
    kv('position', entity.x.toFixed(1) + ', ' + entity.y.toFixed(1) + ' m');
    kv('heading', Math.round(entity.h * 180 / Math.PI) + '°');
    if (entity.kind === 'vehicle') kv('turret', Math.round(entity.th * 180 / Math.PI) + '°');
    if (entity.fa !== undefined) kv('arc centre', Math.round(entity.fa * 180 / Math.PI) + '° ±' + ((meta.arc_deg || 0) / 2) + '°');
    kv('members', entity.n + ' / ' + entity.max);
    kv('health', Math.round(entity.hp * 100) + '%');
    kv('suppression', ['none', 'suppressed', 'pinned'][entity.sup] || '?');
    if (entity.tg !== undefined) kv('target', '#' + entity.tg);
    if (entity.g !== undefined) kv('garrisoned in', '#' + entity.g);
    if (entity.ab) kv('crew', 'abandoned');
    if (entity.re) kv('reinforcing', 'yes');
    if (entity.setup !== undefined) kv('setting up', (entity.setup / TPS).toFixed(1) + ' s left');
    if (meta.sight) kv('sight', meta.sight + ' m');
    if (meta.range) kv('range', meta.range + ' m');

    var hp = (entity.mhp || []).map(function (v, i) {
      var name = (entity.w || [])[i];
      return '<div>' + (name ? title(name) : 'unarmed') + ' &mdash; ' + Math.round(v) + ' hp</div>';
    }).join('') || '<div style="color:var(--dim)">none</div>';
    body.innerHTML = '<dl class="kv">' + rows.join('') + '</dl>' +
      '<div class="sect">Members</div>' + hp;
  } else {
    kv('owner', entity.o === null || entity.o === undefined ? 'neutral' : 'player ' + entity.o + ' (team ' + teamOf(entity.o) + ')');
    kv('cell', entity.cx + ', ' + entity.cy);
    kv('footprint', entity.w + ' × ' + entity.h + ' cells');
    kv('health', Math.round(entity.hp * 100) + '%');
    kv('construction', Math.round(entity.prog * 100) + '%');
    kv('garrison', entity.n);
    var queue = (entity.q || []).map(function (q) {
      return '<div>' + title(q[1]) + ' <span style="color:var(--dim)">(' + q[0] + ')</span>' +
        '<div class="bar"><i style="width:' + (queueFrac(q) * 100).toFixed(0) + '%"></i></div>' +
        q[2].toFixed(1) + ' s left</div>';
    }).join('') || '<div style="color:var(--dim)">empty</div>';
    body.innerHTML = '<dl class="kv">' + rows.join('') + '</dl>' +
      '<div class="sect">Production queue</div>' + queue;
  }
}

/** The panel for something the watched team cannot report on. */
function showHiddenPanel(el, what, why) {
  el.hidden = false;
  document.getElementById('panel-title').textContent = what;
  document.getElementById('panel-body').innerHTML =
    '<p style="color:var(--dim)">Hidden from team ' + (fog ? fog.team : '?') + ' (' + why + ').</p>';
}

function buildLegend() {
  var items = [];
  Object.keys(TERRAIN).forEach(function (ch) {
    items.push(['<s style="background:' + TERRAIN[ch].base + '"></s>', TERRAIN[ch].name]);
  });
  (D.players || []).forEach(function (p, i) {
    items.push(['<s style="background:' + playerColour(i) + '"></s>', 'Player ' + i + ' (' + p.faction + ', team ' + p.team + ')']);
  });
  items.push(['<s style="background:#5ae16e"></s>', 'Heavy cover']);
  items.push(['<s style="background:#ebcd46"></s>', 'Light cover / suppressed']);
  items.push(['<s style="background:#e2705f"></s>', 'Pinned / destroyed']);
  document.getElementById('legend').innerHTML = items.map(function (it) {
    return '<div>' + it[0] + '<span>' + it[1] + '</span></div>';
  }).join('');
}

// ------------------------------------------------------------------ input --

function setPlaying(on) {
  playing = on;
  lastRAF = performance.now();
  document.getElementById('btn-play').innerHTML = on ? '&#10073;&#10073;' : '&#9654;';
  requestRender();
}

function setSpeed(i) {
  speedIdx = clamp(i, 0, SPEEDS.length - 1);
  document.getElementById('speed').innerHTML = SPEEDS[speedIdx] + '&times;';
  requestRender();
}

function step(delta) {
  var idx = clamp(frameIndexFor(tick) + delta, 0, frames.length - 1);
  tick = frames[idx].t;
  setPlaying(false);
}

function cycleFog() {
  fogMode = (fogMode + 1) % 3;
  var b = document.getElementById('btn-fog');
  b.textContent = 'Fog: ' + (fogMode === 0 ? 'all' : 'team ' + (fogMode - 1));
  b.classList.toggle('on', fogMode > 0);
  requestRender();
}

/** Hit test in world space. Anything the watched team cannot see is not
 *  clickable either — otherwise the fog would only be skin deep. */
function pick(px, py) {
  var frame = frames[frameIndexFor(tick)];
  var seen = fog ? seenBuildingsFor(fog.team, fog.idx) : null;
  var w = toWorld(px, py);
  var best = null, bestD = Infinity;

  frame.squads.forEach(function (s) {
    if (squadHidden(s) || s.g !== undefined) return;
    var d = Math.hypot(s.x - w[0], s.y - w[1]);
    if (d < Math.max(3, 16 / cam.scale) && d < bestD) { bestD = d; best = { kind: 'squad', id: s.id }; }
  });
  if (best) return best;

  function hits(b) {
    return w[0] >= b.cx * CELL && w[0] <= (b.cx + b.w) * CELL &&
           w[1] >= b.cy * CELL && w[1] <= (b.cy + b.h) * CELL;
  }
  for (var i = frame.buildings.length - 1; i >= 0; i--) {
    var b = frame.buildings[i];
    if (buildingFogMode(b, seen) === 'hidden') continue;
    if (hits(b)) return { kind: 'building', id: b.id };
  }
  if (seen) {   // remembered ghosts stay clickable, live or not
    var ids = Object.keys(seen);
    for (var j = ids.length - 1; j >= 0; j--) {
      if (hits(seen[ids[j]].b)) return { kind: 'building', id: seen[ids[j]].b.id };
    }
  }
  return null;
}

function install() {
  window.addEventListener('resize', function () { resize(); requestRender(); });

  document.getElementById('btn-play').onclick = function () { setPlaying(!playing); };
  document.getElementById('btn-back').onclick = function () { step(-1); };
  document.getElementById('btn-fwd').onclick = function () { step(1); };
  document.getElementById('btn-slower').onclick = function () { setSpeed(speedIdx - 1); };
  document.getElementById('btn-faster').onclick = function () { setSpeed(speedIdx + 1); };
  document.getElementById('btn-fog').onclick = cycleFog;
  document.getElementById('btn-cover').onclick = function () {
    showCover = !showCover;
    this.classList.toggle('on', showCover);
    requestRender();
  };
  document.getElementById('btn-sectors').onclick = function () {
    showSectors = !showSectors;
    this.classList.toggle('on', showSectors);
    requestRender();
  };
  document.getElementById('btn-help').onclick = toggleHelp;
  document.getElementById('panel-close').onclick = closePanelAndRender;
  document.getElementById('btn-sectors').classList.add('on');

  document.addEventListener('keydown', function (e) {
    if (e.key === ' ') { setPlaying(!playing); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') step(-1);
    else if (e.key === 'ArrowRight') step(1);
    else if (e.key === '+' || e.key === '=') setSpeed(speedIdx + 1);
    else if (e.key === '-' || e.key === '_') setSpeed(speedIdx - 1);
    else if (e.key === 'f' || e.key === 'F') cycleFog();
    else if (e.key === 'c' || e.key === 'C') document.getElementById('btn-cover').click();
    else if (e.key === 'g' || e.key === 'G') document.getElementById('btn-sectors').click();
    else if (e.key === '?' || e.key === '/') toggleHelp();
    else if (e.key === 'Home') { resetCamera(); requestRender(); }
    else if (e.key === 'Escape') {
      if (!document.getElementById('help').hidden) toggleHelp();
      else closePanelAndRender();
    }
  });

  // pan / zoom / pick
  var drag = null, moved = 0;
  cv.addEventListener('mousedown', function (e) { drag = [e.clientX, e.clientY]; moved = 0; cv.classList.add('dragging'); });
  window.addEventListener('mousemove', function (e) {
    if (!drag) return;
    cam.x -= (e.clientX - drag[0]) / cam.scale;
    cam.y -= (e.clientY - drag[1]) / cam.scale;
    moved += Math.abs(e.clientX - drag[0]) + Math.abs(e.clientY - drag[1]);
    drag = [e.clientX, e.clientY];
    requestRender();
  });
  window.addEventListener('mouseup', function (e) {
    if (drag && moved < 4) {
      var hit = pick(e.clientX, e.clientY);
      if (hit) selection = hit; else closePanel();
      requestRender();
    }
    drag = null;
    cv.classList.remove('dragging');
    requestRender();
  });
  cv.addEventListener('wheel', function (e) {
    e.preventDefault();
    var before = toWorld(e.clientX, e.clientY);
    cam.scale = clamp(cam.scale * Math.pow(1.0015, -e.deltaY), 0.8, 60);
    var after = toWorld(e.clientX, e.clientY);
    cam.x += before[0] - after[0];
    cam.y += before[1] - after[1];
    requestRender();
  }, { passive: false });

  // timeline scrub
  var tl = document.getElementById('timeline');
  var scrubbing = false;
  function scrub(e) {
    var r = tl.getBoundingClientRect();
    tick = clamp((e.clientX - r.left) / r.width, 0, 1) * lastTick();
    requestRender();
  }
  tl.addEventListener('mousedown', function (e) { scrubbing = true; setPlaying(false); scrub(e); });
  window.addEventListener('mousemove', function (e) { if (scrubbing) scrub(e); });
  window.addEventListener('mouseup', function () { scrubbing = false; });
}

function toggleHelp() {
  var h = document.getElementById('help');
  h.hidden = !h.hidden;
}

function closePanelAndRender() { closePanel(); requestRender(); }

// ------------------------------------------------------------------- loop --

/* The page renders on demand rather than on a permanent animation loop: while
 * paused nothing moves, so repainting would only burn CPU (and keeps the page
 * from ever going idle, which headless screenshot capture waits for). */
function requestRender() {
  if (rafPending || !D) return;
  rafPending = true;
  requestAnimationFrame(loop);
}

function loop(now) {
  rafPending = false;
  var dt = Math.min((now - lastRAF) / 1000, 0.2);
  lastRAF = now;
  if (playing) {
    tick += dt * TPS * SPEEDS[speedIdx];
    if (tick >= lastTick()) { tick = lastTick(); playing = false; setPlaying(false); }
  }
  try {
    render();
  } catch (err) {
    console.error('render failed', err);
  }
  if (playing) requestRender();
}

install();
setSpeed(1);
boot();

// Test hook: everything the headless browser check needs to drive the app.
window.__viewer = {
  inject: function (data) {
    D = data;
    visCache = {};
    sectorPoint = {};
    prepare();
    document.getElementById('loading').hidden = true;
    document.getElementById('error').hidden = true;
    ['hud', 'bar'].forEach(function (id) { document.getElementById(id).hidden = false; });
    buildLegend();
    resetCamera();
    setPlaying(false);
    requestRender();
  },
  seek: function (fraction) { tick = clamp(fraction, 0, 1) * lastTick(); requestRender(); },
  seekTick: function (t) { tick = clamp(t, 0, lastTick()); requestRender(); },
  camera: function (x, y, scale) { cam.x = x; cam.y = y; cam.scale = scale; requestRender(); },
  select: function (kind, id) { selection = { kind: kind, id: id }; requestRender(); },
  state: function () {
    return {
      tick: tick, fogMode: fogMode, showCover: showCover, showSectors: showSectors,
      frames: frames.length, playing: playing, speed: SPEEDS[speedIdx],
      selection: selection, winner: D ? D.winner : null
    };
  },
  /** Draw one frame synchronously (headless capture must not wait on rAF). */
  redraw: function () { render(); },
  /** What the last render actually drew — the fog assertions read this. */
  stats: function () { return renderStats; },
  /** Screen pixel for a world position, so a test can sample a known spot. */
  screenOf: function (x, y) { return toScreen(x, y); },
  /** Checksum of the pixels in a box around a world position. */
  sampleWorld: function (x, y, radiusPx) {
    var p = toScreen(x, y);
    var cv = document.getElementById('cv');
    var g = cv.getContext('2d');
    var x0 = Math.max(0, Math.round((p[0] - radiusPx) * dpr));
    var y0 = Math.max(0, Math.round((p[1] - radiusPx) * dpr));
    var size = Math.max(1, Math.round(radiusPx * 2 * dpr));
    var w = Math.min(size, cv.width - x0), h = Math.min(size, cv.height - y0);
    if (w <= 0 || h <= 0) return 0;
    var data = g.getImageData(x0, y0, w, h).data;
    var sum = 2166136261;
    for (var i = 0; i < data.length; i += 4) {
      sum ^= data[i] + data[i + 1] * 3 + data[i + 2] * 7;
      sum = (sum * 16777619) | 0;
    }
    return sum;
  },
  /** Which of 'live' / 'ghost' / 'hidden' a building is under the current fog. */
  buildingFog: function (id) {
    var frame = frames[frameIndexFor(tick)];
    var seen = fog ? seenBuildingsFor(fog.team, fog.idx) : null;
    var b = frame.buildings.filter(function (x) { return x.id === id; })[0];
    if (!b) return seen && seen[id] ? 'ghost' : 'gone';
    return buildingFogMode(b, seen);
  },
  /** What `pick()` would select at a screen point (null when nothing is clickable). */
  pickAt: function (px, py) { return pick(px, py); },
  setPlaying: setPlaying,
  cycleFog: cycleFog
};
