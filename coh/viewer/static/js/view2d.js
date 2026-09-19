/* The 2D tactical map — the original canvas renderer, behaviour unchanged.
 *
 * Layers, bottom up: terrain -> sector tint + borders -> cover hints ->
 * points -> buildings -> squads -> combat effects -> fog.
 *
 * It never assumes more than the frame schema: unknown terrain characters,
 * squad kinds and event kinds all fall back to a generic marker rather than
 * throwing. Fog decisions all come from `data.js`, so this view and the 3D
 * battlefield hide exactly the same things.
 */

import { S } from './state.js';
import {
  COVER, NEUTRAL, TERRAIN, UNKNOWN_TERRAIN, alpha, clamp, hash2, mix, mmss,
  offscreen, parseColour, shade, title
} from './util.js';
import {
  R, buildingFogMode, emptyStats, eventPos, fog, frameIndexFor,
  isFootprintVisible, playerColour, queueFrac, seenNow, squadHidden, squadMeta,
  teamColour, visibleBuildings, visibleEffects
} from './data.js';
import { blastRadius } from './effects.js';

const TILE = 8;   // offscreen terrain resolution, px per cell

let cv = null, ctx = null;
let layers = {};          // cached offscreen canvases
let labelBoxes = [];      // text boxes already placed this frame
let terrVersion = -1;     // the `R.terrVersion` the cached layers were built at
let stats = emptyStats();

export function init(canvas) {
  cv = canvas;
  ctx = cv.getContext('2d');
}

export function reset() { layers = {}; terrVersion = -1; }

export function resize() {
  if (!cv) return;
  cv.width = Math.round(window.innerWidth * S.dpr);
  cv.height = Math.round(window.innerHeight * S.dpr);
}

export function lastStats() { return stats; }

// ----------------------------------------------------------------- camera --

export function resetCamera() {
  const wm = R.W * R.CELL, hm = R.H * R.CELL;
  const padX = 40, padTop = 90, padBottom = 90;
  const sx = (window.innerWidth - 2 * padX) / wm;
  const sy = (window.innerHeight - padTop - padBottom) / hm;
  S.cam.scale = Math.max(0.5, Math.min(sx, sy));
  S.cam.x = wm / 2;
  S.cam.y = hm / 2 + (padTop - padBottom) / (2 * S.cam.scale);
}

export function toScreen(x, y) {
  return [(x - S.cam.x) * S.cam.scale + window.innerWidth / 2,
          (y - S.cam.y) * S.cam.scale + window.innerHeight / 2];
}

export function toWorld(px, py) {
  return [(px - window.innerWidth / 2) / S.cam.scale + S.cam.x,
          (py - window.innerHeight / 2) / S.cam.scale + S.cam.y];
}

/** Pixels per metre for unit *shapes*: never below a floor, so squads stay
 *  readable when the whole 96x96 map is fitted to the window. */
function unitPx() { return Math.max(S.cam.scale, 6.5); }

// ------------------------------------------------------- offscreen layers --

/* Terrain is drawn in two passes because per-cell canvas calls over a
 * 96x96 grid are ruinously slow on software rendering (~40 ms/cell-op in a
 * headless browser): first the flat base colours as one pixel-per-cell
 * ImageData scaled up, then detail shapes for the few hundred non-grass
 * cells. Grouping the details by character keeps canvas state changes down. */
function terrainLayer() {
  if (layers.terrain) return layers.terrain;
  const W = R.W, H = R.H, terr = R.terr;

  const small = offscreen(W, H), sg = small.getContext('2d');
  const img = sg.createImageData(W, H), px = img.data;
  const byChar = {}, tone = {};
  for (let i = 0; i < W * H; i++) {
    const ch = terr[i];
    const n = hash2(i % W, (i / W) | 0);
    const step = Math.round(n * 7);
    const key = ch + step;
    let rgb = tone[key];
    if (!rgb) {
      rgb = tone[key] = parseColour(shade((TERRAIN[ch] || UNKNOWN_TERRAIN).base, (step / 7 - 0.5) * 0.16));
    }
    px[i * 4] = rgb[0]; px[i * 4 + 1] = rgb[1]; px[i * 4 + 2] = rgb[2]; px[i * 4 + 3] = 255;
    if (ch !== '.') (byChar[ch] || (byChar[ch] = [])).push(i);
  }
  sg.putImageData(img, 0, 0);

  const c = offscreen(W * TILE, H * TILE), g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.drawImage(small, 0, 0, W * TILE, H * TILE);

  function each(ch, draw) {
    const cells = byChar[ch];
    if (!cells) return;
    for (let k = 0; k < cells.length; k++) {
      const idx = cells[k], cx = idx % W, cy = (idx / W) | 0;
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
  const W = R.W, H = R.H;
  const c = offscreen(W * TILE, H * TILE), g = c.getContext('2d');
  [2, 1].forEach(function (level) {
    g.fillStyle = level === 2 ? 'rgba(90,225,110,0.6)' : 'rgba(235,205,70,0.55)';
    g.beginPath();
    for (let i = 0; i < W * H; i++) {
      if ((COVER[R.terr[i]] || 0) !== level) continue;
      const x = (i % W) * TILE + TILE / 2, y = ((i / W) | 0) * TILE + TILE / 2;
      g.moveTo(x + TILE * 0.3, y);
      g.arc(x, y, TILE * 0.3, 0, 6.2832);
    }
    g.fill();
  });
  layers.cover = c;
  return c;
}

function ownerKey(frame) {
  return frame.points.map(function (p) { return p.owner === null ? '-' : p.owner; }).join('');
}

/** Sector outlines, each drawn just inside its own sector in its owner's
 *  colour — the strongest cue for who holds what, CoH tactical-map style. */
function sectorLinesLayer(frame) {
  const key = ownerKey(frame);
  if (layers.linesKey === key && layers.lines) return layers.lines;
  const W = R.W, H = R.H;

  const owners = {};
  frame.points.forEach(function (p) { owners[p.id] = p.owner; });
  function ownerColour(sector) {
    const pid = R.sectorPoint[sector];
    const owner = pid === undefined ? null : owners[pid];
    return owner === null || owner === undefined ? NEUTRAL : teamColour(owner);
  }

  const c = offscreen(W * TILE, H * TILE), g = c.getContext('2d');
  const inset = TILE * 0.35;
  const paths = {};
  function seg(colour, x1, y1, x2, y2) {
    if (!paths[colour]) paths[colour] = [];
    paths[colour].push([x1, y1, x2, y2]);
  }
  for (let cy = 0; cy < H; cy++) {
    for (let cx = 0; cx < W; cx++) {
      const s = R.sectorCells[cy * W + cx];
      const col = ownerColour(s);
      const x = cx * TILE, y = cy * TILE;
      if (cx + 1 >= W || R.sectorCells[cy * W + cx + 1] !== s) seg(col, x + TILE - inset, y, x + TILE - inset, y + TILE);
      if (cx - 1 < 0 || R.sectorCells[cy * W + cx - 1] !== s) seg(col, x + inset, y, x + inset, y + TILE);
      if (cy + 1 >= H || R.sectorCells[(cy + 1) * W + cx] !== s) seg(col, x, y + TILE - inset, x + TILE, y + TILE - inset);
      if (cy - 1 < 0 || R.sectorCells[(cy - 1) * W + cx] !== s) seg(col, x, y + inset, x + TILE, y + inset);
    }
  }
  g.lineWidth = Math.max(2, TILE * 0.3);
  g.lineCap = 'square';
  for (const colour in paths) {
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
  const key = ownerKey(frame);
  if (layers.tintKey === key && layers.tint) return layers.tint;
  const W = R.W, H = R.H;
  const owners = {};
  frame.points.forEach(function (p) { owners[p.id] = p.owner; });

  const c = offscreen(W, H), g = c.getContext('2d');
  const img = g.createImageData(W, H), px = img.data;
  const rgb = {};
  for (let i = 0; i < W * H; i++) {
    const pid = R.sectorPoint[R.sectorCells[i]];
    const owner = pid === undefined ? null : owners[pid];
    if (owner === null || owner === undefined) continue;
    if (!rgb[owner]) rgb[owner] = parseColour(teamColour(owner));
    px[i * 4] = rgb[owner][0]; px[i * 4 + 1] = rgb[owner][1]; px[i * 4 + 2] = rgb[owner][2];
    px[i * 4 + 3] = 30;
  }
  g.putImageData(img, 0, 0);
  layers.tint = c; layers.tintKey = key;
  return c;
}

function fogLayer(bits) {
  const c = layers.fog || (layers.fog = offscreen(R.W, R.H));
  if (layers.fogBits === bits) return c;
  layers.fogBits = bits;
  const g = c.getContext('2d');
  const img = g.createImageData(R.W, R.H), px = img.data;
  for (let i = 0; i < R.W * R.H; i++) {
    const seen = (bits[i >> 3] & (128 >> (i & 7))) !== 0;
    px[i * 4 + 3] = seen ? 0 : 150;
  }
  g.putImageData(img, 0, 0);
  return c;
}

function blitCells(layer, a, smooth) {
  ctx.save();
  ctx.globalAlpha = a;
  ctx.imageSmoothingEnabled = !!smooth;
  const o = toScreen(0, 0);
  ctx.drawImage(layer, o[0], o[1], R.W * R.CELL * S.cam.scale, R.H * R.CELL * S.cam.scale);
  ctx.restore();
}

// ------------------------------------------------------------------- draw --

function drawPoints(frame, now) {
  const footprints = buildingRects(frame);
  const pulses = {};
  now.forEach(function (e) {
    if (e.k === 'point_captured' || e.k === 'point_neutralized') pulses[e.d.point_id] = e.age;
  });
  const byId = {};
  frame.points.forEach(function (p) { byId[p.id] = p; });

  R.D.map.points.forEach(function (def) {
    const st = byId[def.id] || {};
    const p = toScreen((def.cell[0] + 0.5) * R.CELL, (def.cell[1] + 0.5) * R.CELL);
    const r = clamp(S.cam.scale * 2.4, 9, 26);
    const colour = st.owner === null || st.owner === undefined ? NEUTRAL : teamColour(st.owner);

    if (pulses[def.id] !== undefined) {
      const a = 1 - pulses[def.id];
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

    if (S.showSectors && S.cam.scale > 2.2) {
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
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? s * 0.45 : s;
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
  visibleBuildings(frame, stats).forEach(function (entry) {
    drawBuilding(entry.b, entry.seenTick);
  });
}

/** `seenTick` null = live; otherwise this is a last-known ghost from that tick. */
function drawBuilding(b, seenTick) {
  const ghost = seenTick !== null && seenTick !== undefined;
  const o = toScreen(b.cx * R.CELL, b.cy * R.CELL);
  const w = b.w * R.CELL * S.cam.scale, h = b.h * R.CELL * S.cam.scale;
  const colour = ghost ? '#7d7a70' : (b.nu ? '#8a8171' : playerColour(b.o));
  const done = b.prog >= 1;

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
    for (let d = -h; d < w; d += 7) { ctx.moveTo(o[0] + d, o[1] + h); ctx.lineTo(o[0] + d + h, o[1]); }
    ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = colour;
  ctx.lineWidth = S.selection && S.selection.kind === 'building' && S.selection.id === b.id ? 3 : 1.8;
  if (ghost) ctx.setLineDash([5, 4]);
  ctx.strokeRect(o[0] + 0.5, o[1] + 0.5, w - 1, h - 1);
  ctx.setLineDash([]);

  if (ghost) {
    // No live bars, badge or queue: this is memory, not observation.
    if (S.cam.scale > 4.5) {
      label(title(b.def) + ' (' + mmss(seenTick / R.TPS) + ')', o[0] + w / 2, o[1] + h + 12,
            'rgba(190,185,172,0.85)');
    }
    ctx.restore();
    return;
  }

  // health / construction bar
  const frac = done ? b.hp : b.prog;
  if (frac < 0.999) {
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(o[0], o[1] - 6, w, 4);
    ctx.fillStyle = done ? barColour(b.hp) : '#d8b45a';
    ctx.fillRect(o[0], o[1] - 6, w * clamp(frac, 0, 1), 4);
  }

  // production queue progress (when inspected)
  if (S.selection && S.selection.kind === 'building' && S.selection.id === b.id && b.q && b.q.length) {
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(o[0], o[1] + h + 2, w, 4);
    ctx.fillStyle = '#9fc4e8';
    ctx.fillRect(o[0], o[1] + h + 2, w * queueFrac(b.q[0]), 4);
  }

  // A neutral building's garrison is only knowable while you can see it.
  if (b.n > 0 && (!b.nu || isFootprintVisible(b))) {
    badge(o[0] + w - 2, o[1] + 2, String(b.n), colour);
    stats.badges++;
  }

  if (S.cam.scale > 4.5) label(title(b.def), o[0] + w / 2, o[1] + h + 12);
  ctx.restore();
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
  const w = ctx.measureText(text).width;
  const box = [x - w / 2 - 2, y - 10, x + w / 2 + 2, y + 3];
  const blockers = avoid ? labelBoxes.concat(avoid) : labelBoxes;
  for (let i = 0; i < blockers.length; i++) {
    const o = blockers[i];
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
  const half = ctx.measureText(text).width / 2;
  ctx.restore();
  const spots = [
    [x, y + r + 14],
    [x, y - r - 7],
    [x + r + 8 + half, y + 4],
    [x - r - 8 - half, y + 4]
  ];
  for (let i = 0; i < spots.length; i++) {
    if (label(text, spots[i][0], spots[i][1], colour, font, avoid)) return true;
  }
  return false;
}

/** Screen-space footprint rectangles, used to keep labels off buildings.
 *  Only buildings that are actually drawn count: a label must not dodge
 *  something the viewer is hiding (and a hidden building must not be
 *  detectable by the way it nudges labels around). */
function buildingRects(frame) {
  return visibleBuildings(frame, null).map(function (entry) {
    const b = entry.b;
    const o = toScreen(b.cx * R.CELL, b.cy * R.CELL);
    return [o[0] - 2, o[1] - 8,
            o[0] + b.w * R.CELL * S.cam.scale + 2, o[1] + b.h * R.CELL * S.cam.scale + 14];
  });
}

function barColour(frac) {
  return frac > 0.6 ? '#7fbf6a' : (frac > 0.3 ? '#d8b45a' : '#e2705f');
}

function drawSquads(squads) {
  const offsets = (R.D.meta && R.D.meta.formation_offsets) || [[0, 0]];
  squads.forEach(function (s) {
    if (squadHidden(s)) { stats.hiddenSquads++; return; }
    if (s.g !== undefined) return;   // garrisoned: shown as a badge on the building
    stats.squads++;

    const meta = squadMeta(s.def);
    const p = toScreen(s.x, s.y);
    const colour = s.ab ? '#8f8a7c' : playerColour(s.o);
    const sel = S.selection && S.selection.kind === 'squad' && S.selection.id === s.id;

    ctx.save();

    // owner glow so squads read at map zoom
    const gr = unitPx() * 2.9;
    const glow = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], gr);
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
      ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(12, S.cam.scale * 2.4), 0, 6.2832); ctx.stroke();
      ctx.setLineDash([]);
    }

    drawSquadStatus(s, p, colour);
    ctx.restore();
  });
}

function memberDot(p, ox, oy, cos, sin, r, colour, alive) {
  const k = unitPx();
  const x = p[0] + (ox * cos - oy * sin) * k;
  const y = p[1] + (ox * sin + oy * cos) * k;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 6.2832);
  ctx.fillStyle = alive ? colour : 'rgba(80,80,74,0.6)';
  ctx.fill();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = 'rgba(8,10,6,0.95)';
  ctx.stroke();
}

function drawInfantry(s, p, colour, offsets) {
  const r = clamp(unitPx() * 0.55, 3, 7);
  const cos = Math.cos(s.h), sin = Math.sin(s.h);
  for (let i = 0; i < s.n; i++) {
    const off = offsets[i % offsets.length];
    memberDot(p, off[0], off[1], cos, sin, r, colour, true);
  }
}

function drawTeamWeapon(s, meta, p, colour, offsets) {
  if (s.fa !== undefined) {
    const arc = ((meta.arc_deg || 90) * Math.PI / 180) / 2;
    const reach = (meta.range || 30) * S.cam.scale;
    const grad = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], reach);
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
  const r = clamp(unitPx() * 0.55, 3, 7);
  const cos = Math.cos(s.h), sin = Math.sin(s.h);
  for (let i = 0; i < s.n; i++) {
    const off = offsets[i % offsets.length];
    memberDot(p, off[0], off[1], cos, sin, r, colour, true);
  }
  // the gun itself
  ctx.strokeStyle = s.ab ? '#6f6a5e' : '#14160f';
  ctx.lineWidth = Math.max(2.5, unitPx() * 0.4);
  const a = s.fa !== undefined ? s.fa : s.h;
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(p[0] + Math.cos(a) * unitPx() * 2.0, p[1] + Math.sin(a) * unitPx() * 2.0);
  ctx.stroke();

  if (s.setup !== undefined) {
    label('setting up', p[0], p[1] - unitPx() * 2.6, '#d8b45a', '700 10px system-ui, sans-serif');
  }
}

function drawVehicle(s, meta, p, colour) {
  const L = unitPx() * 3.4, Wd = unitPx() * 1.9;
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
  const r = unitPx() * 1.1;
  ctx.fillStyle = colour;
  ctx.strokeStyle = '#14160f';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(p[0], p[1] - r); ctx.lineTo(p[0] + r, p[1]);
  ctx.lineTo(p[0], p[1] + r); ctx.lineTo(p[0] - r, p[1]);
  ctx.closePath(); ctx.fill(); ctx.stroke();
}

function drawSquadStatus(s, p, colour) {
  const top = p[1] - unitPx() * 2.5;
  const w = unitPx() * 3.6;

  if (s.hp < 0.999 || S.cam.scale > 4) {
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
  if (S.cam.scale > 5.5) {
    label(title(s.def) + ' ' + s.n + '/' + s.max, p[0], top - 5, 'rgba(238,233,220,0.9)');
  }
}

// ---------------------------------------------------------------- effects --

function drawEffects(effects, frame) {
  ctx.save();
  effects.forEach(function (e) {
    try {
      const spec = e.spec;
      if (spec === undefined) { genericMark(e, frame); return; }
      if (spec.mode === 'tracer') drawTracer(e);
      else if (spec.mode === 'blast') {
        explosion(e, frame, blastRadius(e, spec));
        if (spec.death) deathMark(e, frame);
      } else if (spec.mode === 'ripple') ripple(e, frame, spec.colour, spec.radius);
      else genericMark(e, frame);
    } catch (err) { /* an effect must never break the frame */ }
  });
  ctx.restore();
}

function drawTracer(e) {
  const d = e.d;
  if (!Array.isArray(d.src_pos) || !Array.isArray(d.dst_pos)) return;
  const a = toScreen(d.src_pos[0], d.src_pos[1]);
  let b = toScreen(d.dst_pos[0], d.dst_pos[1]);
  if (!d.hit) {
    // a miss lands short and wide of the target
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
    b = [a[0] + dx * 0.93 - dy / len * 9, a[1] + dy * 0.93 + dx / len * 9];
  }
  const fade = 1 - e.age;
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
  const w = eventPos(e, frame);
  if (!w) return;
  const p = toScreen(w[0], w[1]);
  const full = Math.max(metres * S.cam.scale, 6);
  const r = full * (0.15 + 0.85 * e.age);
  ctx.strokeStyle = 'rgba(255,180,90,' + ((1 - e.age) * 0.85).toFixed(2) + ')';
  ctx.lineWidth = Math.max(1.5, 4 * (1 - e.age));
  ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 6.2832); ctx.stroke();
  if (e.age < 0.3) {
    const flash = 1 - e.age / 0.3;
    const grad = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], full * 0.6);
    grad.addColorStop(0, 'rgba(255,240,190,' + (0.85 * flash).toFixed(2) + ')');
    grad.addColorStop(1, 'rgba(255,150,60,0)');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(p[0], p[1], full * 0.6, 0, 6.2832); ctx.fill();
  }
}

function deathMark(e, frame) {
  const w = eventPos(e, frame);
  if (!w) return;
  const p = toScreen(w[0], w[1]);
  const r = Math.max(4, S.cam.scale * 1.1);
  ctx.strokeStyle = 'rgba(40,40,38,' + ((1 - e.age) * 0.9).toFixed(2) + ')';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(p[0] - r, p[1] - r); ctx.lineTo(p[0] + r, p[1] + r);
  ctx.moveTo(p[0] + r, p[1] - r); ctx.lineTo(p[0] - r, p[1] + r);
  ctx.stroke();
}

function ripple(e, frame, colour, metres) {
  const w = eventPos(e, frame);
  if (!w) return;
  const p = toScreen(w[0], w[1]);
  ctx.strokeStyle = colour;
  ctx.globalAlpha = 1 - e.age;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(p[0], p[1], metres * S.cam.scale * (0.3 + e.age), 0, 6.2832); ctx.stroke();
  ctx.globalAlpha = 1;
}

function genericMark(e, frame) {
  const w = eventPos(e, frame);
  if (!w) return;
  const p = toScreen(w[0], w[1]);
  ctx.strokeStyle = 'rgba(240,226,176,' + ((1 - e.age) * 0.7).toFixed(2) + ')';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(4, S.cam.scale * 0.8), 0, 6.2832); ctx.stroke();
}

// ------------------------------------------------------------------ render --

/** Draw one frame. `head` is `data.playhead()`; `effects` is `activeEffects()`. */
export function render(head, squads, effects) {
  // `main.js` has already rolled the terrain deltas forward; this only has to
  // notice that they moved and drop the cached layers.
  if (terrVersion !== R.terrVersion) {
    terrVersion = R.terrVersion;
    delete layers.terrain;
    delete layers.cover;
  }

  ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  ctx.fillStyle = '#0c0e0a';
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  blitCells(terrainLayer(), 1, S.cam.scale > 6);
  blitCells(tintLayer(head.a), 1, false);
  if (S.showSectors) blitCells(sectorLinesLayer(head.a), 0.9, true);
  if (S.showCover) blitCells(coverLayer(), 0.9, true);

  labelBoxes = [];
  stats = emptyStats();
  // Sector ownership and capture progress are map knowledge in CoH, so points
  // (and their pulses) stay omniscient; units, buildings and effects do not.
  drawPoints(head.a, effects);
  drawBuildings(head.a);
  drawSquads(squads);
  drawEffects(visibleEffects(effects, head.a, stats), head.a);
  if (fog) blitCells(fogLayer(fog.bits), 1, true);
}

/** Hit test in world space. Anything the watched team cannot see is not
 *  clickable either — otherwise the fog would only be skin deep. */
export function pick(px, py) {
  const frame = R.frames[frameIndexFor(S.tick)];
  const seen = seenNow();
  const w = toWorld(px, py);
  let best = null, bestD = Infinity;

  frame.squads.forEach(function (s) {
    if (squadHidden(s) || s.g !== undefined) return;
    const d = Math.hypot(s.x - w[0], s.y - w[1]);
    if (d < Math.max(3, 16 / S.cam.scale) && d < bestD) { bestD = d; best = { kind: 'squad', id: s.id }; }
  });
  if (best) return best;

  function hits(b) {
    return w[0] >= b.cx * R.CELL && w[0] <= (b.cx + b.w) * R.CELL &&
           w[1] >= b.cy * R.CELL && w[1] <= (b.cy + b.h) * R.CELL;
  }
  for (let i = frame.buildings.length - 1; i >= 0; i--) {
    const b = frame.buildings[i];
    if (buildingFogMode(b, seen) === 'hidden') continue;
    if (hits(b)) return { kind: 'building', id: b.id };
  }
  if (seen) {   // remembered ghosts stay clickable, live or not
    const ids = Object.keys(seen);
    for (let j = ids.length - 1; j >= 0; j--) {
      if (hits(seen[ids[j]].b)) return { kind: 'building', id: seen[ids[j]].b.id };
    }
  }
  return null;
}

