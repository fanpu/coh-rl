/* Billboards: health bars, suppression pips, retreat/reinforce icons,
 * garrison count badges, names and the "setting up" caption.
 *
 * These are drawn on a transparent 2D canvas over the WebGL view, at the
 * projected screen position of each anchor, rather than as textured sprites
 * in the scene. It is the same thing a camera-facing sprite would give you —
 * these are all screen-facing, screen-sized UI — but it stays crisp at any
 * zoom, costs no draw calls and no texture memory, and reuses the tactical
 * map's label collision logic so a crowded field stays readable.
 */

import * as THREE from 'three';
import { S } from '../state.js';
import { R, eventPos, isFootprintVisible, playerColour, squadHidden, teamColour } from '../data.js';
import { clamp, title } from '../util.js';

let cv = null, ctx = null, width = 0, height = 0;
let boxes = [];
let badges = 0;

const _v = new THREE.Vector3();

export function init(canvas) {
  cv = canvas;
  ctx = cv.getContext('2d');
}

export function resize(w, h) {
  width = w; height = h;
  cv.width = Math.round(w * S.dpr);
  cv.height = Math.round(h * S.dpr);
}

export function badgeCount() { return badges; }

/** Project a world point to screen pixels, or null if it is behind the camera. */
function project(camera, x, y, z) {
  _v.set(x, y, z).project(camera);
  if (_v.z > 1) return null;
  return [(_v.x * 0.5 + 0.5) * width, (-_v.y * 0.5 + 0.5) * height];
}

function barColour(frac) {
  return frac > 0.6 ? '#7fbf6a' : (frac > 0.3 ? '#d8b45a' : '#e2705f');
}

export function draw(camera, frame, squads, buildings, effects) {
  ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  boxes = [];
  badges = 0;

  // Events with no visual of their own — a trained unit with nowhere to stand
  // — say so in words over the thing that is stuck.
  (effects || []).forEach(function (e) {
    if (!e.spec || e.spec.mode !== 'badge') return;
    if (e.age > 0.5 && Math.floor(e.age * 8) % 2 === 0) return;
    const w = eventPos(e, frame);
    if (!w) return;
    const p = project(camera, w[0], 4.4, w[1]);
    if (p) label(e.spec.text, p[0], p[1], e.spec.colour, '700 11px system-ui, sans-serif');
  });

  // Farthest first, so a nearer unit's bar wins the collision test.
  const ordered = squads
    .filter(function (s) { return !squadHidden(s) && s.g === undefined; })
    .map(function (s) {
      const p = project(camera, s.x, s.kind === 'vehicle' ? 2.6 : 2.1, s.y);
      return p ? { s: s, p: p, d: _v.z } : null;
    })
    .filter(Boolean)
    .sort(function (a, b) { return b.d - a.d; });

  ordered.forEach(function (o) { squadBadge(o.s, o.p); });

  buildings.forEach(function (entry) {
    const b = entry.b;
    const p = project(camera, (b.cx + b.w / 2) * R.CELL, 4.6, (b.cy + b.h / 2) * R.CELL);
    if (!p) return;
    buildingBadge(b, entry.seenTick, p);
  });

  R.D.map.points.forEach(function (def) {
    const st = (frame.points || []).filter(function (x) { return x.id === def.id; })[0] || {};
    const p = project(camera, (def.cell[0] + 0.5) * R.CELL, def.type === 'victory' ? 8.4 : 6.4,
                      (def.cell[1] + 0.5) * R.CELL);
    if (!p) return;
    const colour = st.owner === null || st.owner === undefined ? '#d8d2c0' : teamColour(st.owner);
    if (S.showSectors) label(def.name || title(def.id), p[0], p[1], colour, '600 11px system-ui, sans-serif');
  });
}

function squadBadge(s, p) {
  const colour = s.ab ? '#8f8a7c' : playerColour(s.o);
  const w = 34;
  const top = p[1];

  if (s.hp < 0.999) {
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(p[0] - w / 2, top, w, 4);
    ctx.fillStyle = barColour(s.hp);
    ctx.fillRect(p[0] - w / 2, top, w * clamp(s.hp, 0, 1), 4);
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.lineWidth = 1;
    ctx.strokeRect(p[0] - w / 2 - 0.5, top - 0.5, w + 1, 5);
  }
  // an owner pip, so you can tell sides apart at a glance from above
  ctx.fillStyle = colour;
  ctx.beginPath(); ctx.arc(p[0] - w / 2 - 6, top + 2, 3, 0, 6.2832); ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.7)'; ctx.lineWidth = 1; ctx.stroke();

  if (s.sup) {
    ctx.fillStyle = s.sup === 2 ? '#e2705f' : '#e8c84a';
    ctx.beginPath(); ctx.arc(p[0] + w / 2 + 6, top + 2, 3.2, 0, 6.2832); ctx.fill();
  }
  if (s.st === 'retreating') {
    ctx.fillStyle = '#f0e2b0';
    ctx.font = '700 12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('↑', p[0] + w / 2 + 14, top + 7);
  }
  if (s.re) {
    ctx.strokeStyle = '#7fbf6a';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(p[0] - w / 2 - 16, top + 2); ctx.lineTo(p[0] - w / 2 - 8, top + 2);
    ctx.moveTo(p[0] - w / 2 - 12, top - 2); ctx.lineTo(p[0] - w / 2 - 12, top + 6);
    ctx.stroke();
  }
  if (s.setup !== undefined) {
    label('setting up', p[0], top - 16, '#d8b45a', '700 10px system-ui, sans-serif');
  } else if (S.cam3.dist < 110) {
    label(title(s.def) + ' ' + s.n + '/' + s.max, p[0], top - 6, 'rgba(238,233,220,0.92)');
  }
}

function buildingBadge(b, seenTick, p) {
  if (seenTick !== null && seenTick !== undefined) {
    label(title(b.def) + ' (last seen)', p[0], p[1], 'rgba(185,180,168,0.85)');
    return;
  }
  const colour = b.nu ? '#b9b2a0' : playerColour(b.o);
  const w = 40;
  const done = b.prog >= 1;
  const frac = done ? b.hp : b.prog;
  if (frac < 0.999) {
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(p[0] - w / 2, p[1], w, 4);
    ctx.fillStyle = done ? barColour(b.hp) : '#d8b45a';
    ctx.fillRect(p[0] - w / 2, p[1], w * clamp(frac, 0, 1), 4);
  }
  // A neutral building's garrison is only knowable while you can see it.
  if (b.n > 0 && (!b.nu || isFootprintVisible(b))) {
    badges++;
    ctx.fillStyle = colour;
    ctx.strokeStyle = 'rgba(0,0,0,0.75)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(p[0] + w / 2 + 9, p[1] + 2, 8, 0, 6.2832); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#12140f';
    ctx.font = '700 10px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(b.n), p[0] + w / 2 + 9, p[1] + 2.5);
    ctx.textBaseline = 'alphabetic';
  }
  if (S.cam3.dist < 150) label(title(b.def), p[0], p[1] - 6, 'rgba(238,233,220,0.9)');
}

/** Centred text with a dark halo; dropped when it would collide. */
function label(text, x, y, colour, font) {
  ctx.save();
  ctx.font = font || '10px system-ui, sans-serif';
  const w = ctx.measureText(text).width;
  const box = [x - w / 2 - 2, y - 10, x + w / 2 + 2, y + 3];
  for (let i = 0; i < boxes.length; i++) {
    const o = boxes[i];
    if (box[0] < o[2] && o[0] < box[2] && box[1] < o[3] && o[1] < box[3]) { ctx.restore(); return false; }
  }
  boxes.push(box);
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
