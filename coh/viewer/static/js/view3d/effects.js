/* Combat effects: tracers, muzzle flashes, blasts, dirt, smoke, dust, rings,
 * scorch decals and wrecks.
 *
 * Everything transient is drawn from three InstancedMeshes (tracers, fire,
 * smoke) plus one for dirt clods, so a firefight with fifty rounds in the air
 * still costs four draw calls. Rings come from a small pool of real meshes,
 * because each needs its own fading opacity and there are never many.
 *
 * Two things are *persistent* rather than transient — scorch decals and
 * vehicle wrecks. They accumulate as the playhead moves forward and are
 * rebuilt from scratch on a backward seek, the same trick `data.js` uses for
 * terrain deltas and building memory.
 */

import * as THREE from 'three';
import { R, eventPos, fog, frameIndexFor, visAt, visibleAt } from '../data.js';
import { EVENT_EFFECTS, blastRadius } from '../effects.js';
import { hash3, lerp } from '../util.js';
import { Parts, bake, box, sphere } from './geom.js';
import { materials } from './scene.js';

const MAX_TRACERS = 220;
const MAX_PUFFS = 320;
const MAX_DIRT = 260;
const MAX_RINGS = 12;

let group = null;
let tracers = null, fire = null, smoke = null, dirt = null;
let rings = [];
let decalCanvas = null, decalCtx = null, decalMesh = null;
let wrecks = null, wreckSeen = {};
let decalUpto = -1;

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _s = new THREE.Vector3();
const _c = new THREE.Color();
const _up = new THREE.Vector3(0, 1, 0);
const _dir = new THREE.Vector3();

const FIRE_HOT = new THREE.Color('#ffc560');
const FIRE_MID = new THREE.Color('#d1571a');
const SMOKE_A = new THREE.Color('#8c867c');
const SMOKE_B = new THREE.Color('#54504a');

export function build(scene) {
  group = new THREE.Group();
  group.name = 'effects';

  tracers = instanced(box(1, 1, 1), MAX_TRACERS, materials.tracer);
  fire = instanced(sphere(1, 1), MAX_PUFFS, materials.fire);
  smoke = instanced(sphere(1, 1), MAX_PUFFS, materials.smoke);
  dirt = instanced(box(0.45, 0.45, 0.45), MAX_DIRT, materials.dirt);
  dirt.castShadow = false;
  [tracers, fire, smoke, dirt].forEach(function (m) { group.add(m); });

  const ringGeo = new THREE.RingGeometry(0.86, 1.0, 40);
  ringGeo.rotateX(-Math.PI / 2);
  for (let i = 0; i < MAX_RINGS; i++) {
    const mesh = new THREE.Mesh(ringGeo, materials.ring.clone());
    mesh.visible = false;
    rings.push(mesh);
    group.add(mesh);
  }

  // scorch decals: one canvas over the whole map, composited just above the
  // ground so a shell leaves a mark that outlives its fireball
  const per = 4;
  decalCanvas = document.createElement('canvas');
  decalCanvas.width = R.W * per; decalCanvas.height = R.H * per;
  decalCtx = decalCanvas.getContext('2d');
  const tex = new THREE.CanvasTexture(decalCanvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  decalMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(R.W * R.CELL, R.H * R.CELL),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false })
  );
  decalMesh.rotation.x = -Math.PI / 2;
  decalMesh.position.set(R.W * R.CELL / 2, 0.02, R.H * R.CELL / 2);
  decalMesh.renderOrder = 2;
  decalMesh.matrixAutoUpdate = false;
  decalMesh.updateMatrix();
  group.add(decalMesh);

  wrecks = new THREE.Group();
  group.add(wrecks);

  scene.add(group);
  reset();
  return group;
}

function instanced(geometry, count, material) {
  const m = new THREE.InstancedMesh(geometry, material, count);
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  m.frustumCulled = false;
  m.count = 0;
  return m;
}

export function reset() {
  decalUpto = -1;
  wreckSeen = {};
  wreckSeen = {};
  if (decalCtx) decalCtx.clearRect(0, 0, decalCanvas.width, decalCanvas.height);
  if (wrecks) {
    while (wrecks.children.length) {
      const c = wrecks.children.pop();
      if (c.geometry) c.geometry.dispose();
    }
  }
}

export function dispose(scene) {
  if (group) {
    scene.remove(group);
    group.traverse(function (o) { if (o.geometry) o.geometry.dispose(); });
  }
  group = tracers = fire = smoke = dirt = decalMesh = wrecks = null;
  rings = [];
}

// ------------------------------------------------------------- placement --

let nT = 0, nF = 0, nS = 0, nD = 0, nR = 0;

function begin() { nT = nF = nS = nD = nR = 0; }

function put(inst, n, max, x, y, z, sx, sy, sz, quat, colour) {
  if (n >= max) return n;
  _v.set(x, y, z); _s.set(sx, sy, sz);
  _m.compose(_v, quat || _q.identity(), _s);
  inst.setMatrixAt(n, _m);
  inst.setColorAt(n, colour);
  return n + 1;
}

function puff(hot, x, y, z, r, colour) {
  if (hot) nF = put(fire, nF, MAX_PUFFS, x, y, z, r, r, r, null, colour);
  else nS = put(smoke, nS, MAX_PUFFS, x, y, z, r, r, r, null, colour);
}

function clod(x, y, z, s, colour) {
  nD = put(dirt, nD, MAX_DIRT, x, y, z, s, s, s, null, colour);
}

function ring(x, z, radius, colour, opacity) {
  if (nR >= MAX_RINGS) return;
  const m = rings[nR++];
  m.visible = true;
  m.position.set(x, 0.09, z);
  m.scale.set(radius, 1, radius);
  m.material.color.set(colour);
  m.material.opacity = opacity;
}

function finish() {
  tracers.count = nT; fire.count = nF; smoke.count = nS; dirt.count = nD;
  [tracers, fire, smoke, dirt].forEach(function (m) {
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  });
  for (let i = nR; i < MAX_RINGS; i++) rings[i].visible = false;
}

// ---------------------------------------------------------------- drawing --

/** Draw the fog-filtered effects for this frame. */
export function update(effects, frame, tick) {
  begin();
  syncPersistent(effects, frame, tick);

  for (let i = 0; i < effects.length; i++) {
    const e = effects[i];
    try {
      const spec = e.spec;
      if (spec === undefined) { marker(e, frame); continue; }
      if (spec.mode === 'tracer') tracer(e);
      else if (spec.mode === 'blast') blast(e, frame, spec);
      else if (spec.mode === 'ripple') rippleRing(e, frame, spec);
      // 'badge' is a word, not a thing in the world: the overlay draws it
      else if (spec.mode !== 'badge') marker(e, frame);
    } catch (err) { /* one bad event must never break the frame */ }
  }
  finish();
}

/** Capture pulses are drawn with their point, and are deliberately omniscient. */
export function capturePulse(x, z, age) {
  ring(x, z, 4 + age * 14, '#ffe8a0', (1 - age) * 0.85);
}

function tracer(e) {
  const d = e.d;
  if (!Array.isArray(d.src_pos) || !Array.isArray(d.dst_pos)) return;
  const a = d.src_pos, b0 = d.dst_pos;
  let bx = b0[0], bz = b0[1];
  if (!d.hit) {
    // a miss overshoots and goes wide
    const dx = bx - a[0], dz = bz - a[1], len = Math.hypot(dx, dz) || 1;
    bx = a[0] + dx * 1.07 - dz / len * 2.2;
    bz = a[1] + dz * 1.07 + dx / len * 2.2;
  }
  // the round travels the line over the effect's short life
  const u = Math.min(1, e.age * 1.25);
  const head = 0.12 + u * 0.88;
  const tail = Math.max(0, head - 0.13);
  const hx = lerp(a[0], bx, head), hz = lerp(a[1], bz, head);
  const tx = lerp(a[0], bx, tail), tz = lerp(a[1], bz, tail);
  const y = 1.05;

  _dir.set(hx - tx, 0, hz - tz);
  const len = _dir.length();
  if (len < 0.001) return;
  _dir.normalize();
  _q.setFromUnitVectors(_up, _dir);
  const fade = 1 - e.age;
  _c.setStyle(d.hit ? '#ffe9a8' : '#e8cfa6').multiplyScalar(d.hit ? 1 : 0.6);
  nT = put(tracers, nT, MAX_TRACERS, (hx + tx) / 2, y, (hz + tz) / 2,
           0.115, len, 0.115, _q, _c);

  /* A short afterglow behind the round: dimmer, longer and a touch wider.
   * On grass a single thin streak is easy to miss, and the trail is what
   * makes the direction of fire readable. */
  const gTail = Math.max(0, tail - 0.17);
  const gx = lerp(a[0], bx, gTail), gz = lerp(a[1], bz, gTail);
  _dir.set(tx - gx, 0, tz - gz);
  const glowLen = _dir.length();
  if (glowLen > 0.001) {
    _dir.normalize();
    _q.setFromUnitVectors(_up, _dir);
    _c.multiplyScalar(0.22 * fade);
    nT = put(tracers, nT, MAX_TRACERS, (tx + gx) / 2, y, (tz + gz) / 2,
             0.19, glowLen, 0.19, _q, _c);
  }

  // muzzle flash for the first instant of the shot, additive like the round
  if (e.age < 0.4) {
    const k = 1 - e.age / 0.4;
    _c.copy(FIRE_HOT).multiplyScalar(0.45 + k * 0.55);
    _q.identity();
    const flash = 0.28 + k * 0.5;
    nT = put(tracers, nT, MAX_TRACERS, a[0], y, a[1], flash, flash, flash, _q, _c);
  }
  if (d.hit && e.age > 0.5) {
    puff(false, bx, 0.4, bz, 0.25 * fade + 0.1, _c.copy(SMOKE_A));
  }
}

function blast(e, frame, spec) {
  const w = eventPos(e, frame);
  if (!w) return;
  const r = blastRadius(e, spec);
  const age = e.age;
  const x = w[0], z = w[1];
  const seed = Math.round(x * 7) + Math.round(z * 13);

  // fireball -> smoke column, expanding and lifting
  const n = Math.min(12, 5 + Math.round(r));
  for (let i = 0; i < n; i++) {
    const a = hash3(seed, i, 1), b = hash3(seed, i, 2), c = hash3(seed, i, 3);
    const spread = r * (0.18 + a * 0.42) * (0.3 + age * 1.15);
    const px = x + Math.cos(b * 6.283) * spread;
    const pz = z + Math.sin(b * 6.283) * spread;
    const py = 0.4 + age * (1.6 + c * 2.8) * (r / 8 + 0.5) + i * 0.09;
    const size = r * (0.13 + a * 0.11) * (0.45 + age * 0.85);
    if (age < 0.4) {
      const k = 1 - age / 0.4;
      // the core stays hot; the outer puffs are already going sooty
      const hot = k * (0.25 + (1 - a) * 0.75);
      _c.copy(FIRE_MID).lerp(FIRE_HOT, hot);
      if (a > 0.72) _c.lerp(SMOKE_B, 0.55);
      puff(true, px, py, pz, size * (0.75 + k * 0.4), _c);
    } else {
      const k = (age - 0.4) / 0.6;
      _c.copy(SMOKE_A).lerp(SMOKE_B, k * 0.8);
      puff(false, px, py, pz, size * (1.0 + k * 1.1), _c);
    }
  }

  // dirt burst: clods thrown up and falling back
  if (age < 0.75) {
    const clods = Math.min(14, 5 + Math.round(r * 1.2));
    for (let i = 0; i < clods; i++) {
      const a = hash3(seed, i, 11), b = hash3(seed, i, 12);
      const t = age / 0.75;
      const reach = r * (0.5 + a * 1.1) * t;
      const py = Math.max(0.05, (2.6 + b * 3.2) * (t * 1.8 - t * t * 2.1));
      _c.setStyle(a > 0.5 ? '#54452f' : '#3f3526');
      clod(x + Math.cos(b * 6.283) * reach, py, z + Math.sin(b * 6.283) * reach,
           0.6 + a * 0.9, _c);
    }
  }

  // a bright ground flash for the first instant
  if (age < 0.25) ring(x, z, r * (0.4 + age * 2.2), '#ffcf7a', (1 - age / 0.25) * 0.75);

  if (spec.death === 'building') {
    // collapse dust rolls outward along the ground rather than up
    for (let i = 0; i < 9; i++) {
      const a = hash3(seed, i, 21), b = hash3(seed, i, 22);
      const reach = r * (0.35 + a * 0.7) * (0.25 + age * 1.25);
      _c.copy(SMOKE_A).lerp(SMOKE_B, age * 0.7);
      puff(false, x + Math.cos(b * 6.283) * reach, 0.6 + a * 1.1 + age * 0.8,
           z + Math.sin(b * 6.283) * reach, r * (0.16 + a * 0.2) * (0.7 + age), _c);
    }
  }
}

function rippleRing(e, frame, spec) {
  const w = eventPos(e, frame);
  if (!w) return;
  ring(w[0], w[1], spec.radius * (0.35 + e.age * 1.1), spec.colour, (1 - e.age) * 0.7);
}

function marker(e, frame) {
  const w = eventPos(e, frame);
  if (!w) return;
  ring(w[0], w[1], 1.5 + e.age * 3, '#f0e2b0', (1 - e.age) * 0.6);
}

// ------------------------------------------------- persistent battle scars --

/* Scorch marks and burnt-out hulls stay where they happened. They are folded
 * in as the playhead advances and rebuilt on a backward seek. */
function syncPersistent(effects, frame, tick) {
  // A scar is only laid down if the watched team was looking at the time, so
  // switching fog mode rebuilds them from scratch.
  const team = fog ? fog.team : -1;
  if (tick < decalUpto || team !== scarTeam) { reset(); scarTeam = team; }
  const from = decalUpto;
  decalUpto = tick;
  if (from >= tick) return;

  for (let i = 0; i < R.allEvents.length; i++) {
    const e = R.allEvents[i];
    if (e.t <= from || e.t > tick) continue;
    const spec = EVENT_EFFECTS[e.k];
    if (!spec || spec.mode !== 'blast') continue;
    const w = eventPos(e, frame) || (Array.isArray(e.d.pos) ? e.d.pos : null);
    if (!w || !witnessed(team, e.t, w)) continue;
    scorch(w[0], w[1], blastRadius(e, spec));
    if (e.k === 'vehicle_destroyed') addWreck(e, w);
  }
}

let scarTeam = -1;

/** Could `team` see this spot at the tick it happened? -1 means omniscient. */
function witnessed(team, tick, w) {
  if (team < 0) return true;
  const bits = visAt(team, frameIndexFor(tick));
  if (!bits) return true;
  return visibleAt(bits, Math.floor(w[0] / R.CELL), Math.floor(w[1] / R.CELL));
}

function scorch(x, z, radius) {
  const per = 4;
  const px = x / R.CELL * per, pz = z / R.CELL * per;
  const r = Math.max(2, radius / R.CELL * per * 0.8);
  const grad = decalCtx.createRadialGradient(px, pz, 0, px, pz, r);
  grad.addColorStop(0, 'rgba(30,26,21,0.62)');
  grad.addColorStop(0.5, 'rgba(46,39,30,0.34)');
  grad.addColorStop(1, 'rgba(56,47,36,0)');
  decalCtx.fillStyle = grad;
  decalCtx.beginPath();
  decalCtx.ellipse(px, pz, r, r * 0.92, 0, 0, 6.2832);
  decalCtx.fill();
  decalMesh.material.map.needsUpdate = true;
}

/** A charred hull that stays on the field for the rest of the match. */
function addWreck(e, w) {
  if (wreckSeen[e.t + ':' + (e.d.id || 0)]) return;
  wreckSeen[e.t + ':' + (e.d.id || 0)] = true;
  const seed = Math.round(w[0] * 3 + w[1] * 5);
  const p = new Parts();
  p.box(4.6, 0.8, 2.4, '#2a2622', { y: 0.45, rz: 0.05 });
  p.box(4.8, 0.22, 2.6, '#1e1b18', { y: 0.12 });
  p.box(2.0, 0.6, 1.9, '#332e28', { x: -0.5, y: 1.1, ry: 0.5 });
  p.box(2.2, 0.16, 0.16, '#241f1b', { x: 1.6, y: 1.2, rz: -0.35 });   // bent barrel
  p.box(4.9, 0.5, 0.5, '#211e1a', { y: 0.3, z: 1.15 });
  p.box(4.9, 0.5, 0.5, '#211e1a', { y: 0.3, z: -1.15 });
  const mesh = new THREE.Mesh(bake(p), materials.prop);
  mesh.position.set(w[0], 0, w[1]);
  mesh.rotation.y = hash3(seed, 0, 5) * 6.283;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  wrecks.add(mesh);
}

export function counts() {
  return {
    tracers: nT, fire: nF, smoke: nS, dirt: nD, rings: nR,
    wrecks: wrecks ? wrecks.children.length : 0
  };
}
