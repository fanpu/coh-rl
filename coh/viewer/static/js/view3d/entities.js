/* Everything in the frame that stands on the ground: squads, buildings and
 * capture points, plus the selection ring, the weapon arcs and the invisible
 * proxy volumes clicks are tested against.
 *
 * Infantry is instanced — one InstancedMesh per faction colour — so every
 * rifleman on a 96x96 map is two draw calls. Vehicles, team weapons,
 * buildings and flagpoles are baked into one mesh each and cached until the
 * thing they depict actually changes shape.
 *
 * Fog is not decided here: `data.js` hands over the squads and buildings that
 * may be shown, and anything it withheld never reaches the scene graph (and
 * so is not in `pickRoot` either — it cannot be clicked).
 */

import * as THREE from 'three';
import { S } from '../state.js';
import {
  R, isFootprintVisible, playerColour, squadHidden, squadMeta, teamColour
} from '../data.js';
import { clamp, hash2, mix, shade } from '../util.js';
import { Parts, bake, box, cyl } from './geom.js';
import * as models from './models.js';
import { materials } from './scene.js';

let root = null;          // everything that renders
const pickRoot = new THREE.Group();   // proxy volumes, never rendered

const soldierPools = new Map();   // colour -> InstancedMesh
let props = null;                 // per-entity baked meshes, keyed below
const cache = new Map();          // key -> {node, key}
let selectionRing = null, hoverRing = null, footRings = null;
let footCount = 0;
let arcs = [];                    // pooled arc wedges
let proxyPool = [];
let stats = { units: 0, buildings: 0, ghosts: 0, points: 0, soldiers: 0 };

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _v = new THREE.Vector3();
const _s = new THREE.Vector3();
const _q2 = new THREE.Quaternion();
const _col = new THREE.Color();

const MAX_ARCS = 24;
const MAX_PROXIES = 400;
const SOLDIER_CAP = 600;
const RING_CAP = 200;

/* Figures are drawn larger than life. A 1.8 m man on a 192 m map is a few
 * pixels at any camera height you would actually play at, so — like every RTS
 * — the models are exaggerated until a squad reads as individual soldiers
 * rather than a smudge. Positions and formation offsets stay true to the sim;
 * only the model is scaled. */
const FIGURE_SCALE = 1.75;

export function build(scene) {
  root = new THREE.Group();
  root.name = 'entities';
  props = new THREE.Group();
  root.add(props);

  /* Every squad stands on a thin team-coloured ring. It is the single
   * strongest ownership cue from a pitched camera — better than unit colour,
   * which is half in shadow — and it is what makes a firefight legible at the
   * default zoom. The selection ring below is the same idea, brighter. */
  const footGeo = new THREE.RingGeometry(0.80, 1.0, 28);
  footGeo.rotateX(-Math.PI / 2);
  footRings = new THREE.InstancedMesh(footGeo, new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide, fog: false
  }), RING_CAP);
  footRings.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  footRings.frustumCulled = false;
  footRings.renderOrder = 3;
  footRings.count = 0;
  root.add(footRings);

  const ringGeo = new THREE.RingGeometry(0.88, 1.06, 36);
  ringGeo.rotateX(-Math.PI / 2);
  selectionRing = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
    color: 0xf2e9c8, transparent: true, opacity: 0.9, depthWrite: false, fog: false
  }));
  hoverRing = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false, fog: false
  }));
  selectionRing.visible = hoverRing.visible = false;
  selectionRing.renderOrder = 4;
  hoverRing.renderOrder = 4;
  root.add(selectionRing);
  root.add(hoverRing);

  scene.add(root);
  return root;
}

export function dispose(scene) {
  if (root) {
    scene.remove(root);
    root.traverse(function (o) { if (o.geometry) o.geometry.dispose(); });
  }
  root = props = footRings = null;
  soldierPools.clear();
  cache.clear();
  arcs = [];
  proxyPool = [];
  pickRoot.clear();
}

export function reset() {
  cache.forEach(function (entry) {
    if (entry.node.parent) entry.node.parent.remove(entry.node);
    entry.node.traverse(function (o) { if (o.geometry) o.geometry.dispose(); });
  });
  cache.clear();
}

export function lastStats() { return stats; }

// ---------------------------------------------------------- node caching --

/* A baked mesh is expensive to make and cheap to keep, so each entity's node
 * is cached under a key that encodes everything its *shape* depends on. When
 * a building finishes building or a tank dies, the key changes and the node
 * is rebuilt; when it merely moves, it is not. */
const alive = new Set();

function node(key, make) {
  alive.add(key);
  let entry = cache.get(key);
  if (!entry) {
    entry = { node: make() };
    cache.set(key, entry);
    props.add(entry.node);
  }
  entry.node.visible = true;
  return entry.node;
}

function sweep() {
  cache.forEach(function (entry, key) {
    if (alive.has(key)) return;
    props.remove(entry.node);
    entry.node.traverse(function (o) { if (o.geometry) o.geometry.dispose(); });
    cache.delete(key);
  });
  alive.clear();
}

// ----------------------------------------------------------- pick proxies --

let proxyCount = 0;

function proxy(kind, id, x, y, z, sx, sy, sz) {
  if (proxyCount >= MAX_PROXIES) return;
  let mesh = proxyPool[proxyCount];
  if (!mesh) {
    mesh = new THREE.Mesh(box(1, 1, 1), new THREE.MeshBasicMaterial());
    proxyPool.push(mesh);
    pickRoot.add(mesh);
  }
  mesh.visible = true;
  mesh.position.set(x, y, z);
  mesh.scale.set(sx, sy, sz);
  mesh.userData.hit = { kind: kind, id: id };
  mesh.updateMatrixWorld(true);
  proxyCount++;
}

// ---------------------------------------------------------------- update --

/** Rebuild the scene for one frame. `squads` and `buildings` are already
 *  fog-filtered by `data.js`. */
export function update(frame, squads, buildings, effects, allEffects, time) {
  stats = { units: 0, buildings: 0, ghosts: 0, points: 0, soldiers: 0 };
  proxyCount = 0;
  footCount = 0;
  soldierPools.forEach(function (m) { m.userData.n = 0; });
  let arcCount = 0;

  garrisonOwner = {};
  frame.squads.forEach(function (s) {
    if (s.g !== undefined && garrisonOwner[s.g] === undefined) garrisonOwner[s.g] = s.o;
  });

  buildings.forEach(function (entry) {
    drawBuilding(entry.b, entry.seenTick);
    if (entry.seenTick === null) stats.buildings++; else stats.ghosts++;
  });

  // Points (and their capture pulses) are map knowledge in CoH and stay
  // omniscient, so they read the unfiltered list.
  drawPoints(frame, allEffects);

  const recoil = recoilBySquad(effects);
  squads.forEach(function (s) {
    if (squadHidden(s) || s.g !== undefined) return;
    stats.units++;
    drawSquad(s, time, recoil[s.id] || 0);
    if (arcCount < MAX_ARCS && s.kind === 'team_weapon' && s.fa !== undefined) {
      const selected = S.selection && S.selection.kind === 'squad' && S.selection.id === s.id;
      if (S.showArcs || selected) { drawArc(s, arcCount++); }
    }
  });

  soldierPools.forEach(function (mesh) {
    mesh.count = mesh.userData.n;
    stats.soldiers += mesh.count;
    mesh.instanceMatrix.needsUpdate = true;
  });
  footRings.count = footCount;
  footRings.instanceMatrix.needsUpdate = true;
  if (footRings.instanceColor) footRings.instanceColor.needsUpdate = true;
  for (let i = arcCount; i < arcs.length; i++) arcs[i].visible = false;
  for (let i = proxyCount; i < proxyPool.length; i++) proxyPool[i].visible = false;

  markSelection(frame, squads, buildings);
  sweep();
}

// --------------------------------------------------------------- squads --

function soldierPool(colour, faction) {
  let mesh = soldierPools.get(colour);
  if (!mesh) {
    const geo = models.buildGeometry('soldier', { colour: colour, faction: faction });
    mesh = new THREE.InstancedMesh(geo, materials.unit, SOLDIER_CAP);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    mesh.userData.n = 0;
    soldierPools.set(colour, mesh);
    root.add(mesh);
  }
  return mesh;
}

/** One figure, posed for the squad's suppression state. */
function soldier(colour, faction, x, z, heading, pose, bob) {
  const mesh = soldierPool(colour, faction);
  const n = mesh.userData.n;
  if (n >= SOLDIER_CAP) return;
  let y = 0, pitch = 0, scaleY = 1, ry = -heading;
  if (pose === 2) {
    // pinned: flat on the ground, facing where the squad faces
    pitch = Math.PI / 2 * 0.92;
    y = 0.28;
  } else if (pose === 1) {
    // suppressed: down on one knee
    scaleY = 0.72;
    y = 0;
  } else {
    y = bob;
  }
  _e.set(0, ry, 0);
  _q.setFromEuler(_e);
  if (pitch) {
    _e.set(0, 0, -pitch);
    _q.multiply(_q2.setFromEuler(_e));
  }
  _v.set(x, y * FIGURE_SCALE, z);
  _s.set(FIGURE_SCALE, scaleY * FIGURE_SCALE, FIGURE_SCALE);
  _m.compose(_v, _q, _s);
  mesh.setMatrixAt(n, _m);
  mesh.userData.n = n + 1;
}

/** The team ring this squad stands on. */
function footRing(s, colour) {
  if (footCount >= RING_CAP) return;
  const r = s.kind === 'vehicle' ? 3.4 : (s.kind === 'team_weapon' ? 2.6 : 2.4);
  _e.set(0, 0, 0);
  _q.setFromEuler(_e);
  _v.set(s.x, 0.08, s.y);
  _s.set(r, 1, r);
  _m.compose(_v, _q, _s);
  footRings.setMatrixAt(footCount, _m);
  footRings.setColorAt(footCount, _col.setStyle(mix(colour, '#171713', 0.18)));
  footCount++;
}

function drawSquad(s, time, recoil) {
  const colour = s.ab ? '#8f8a7c' : playerColour(s.o);
  const player = (R.D.players || [])[s.o];
  const faction = player ? player.faction : '';
  footRing(s, colour);
  const offsets = (R.D.meta && R.D.meta.formation_offsets) || [[0, 0]];
  const cos = Math.cos(s.h), sin = Math.sin(s.h);
  const pose = s.sup || 0;
  const moving = s.moving && !pose;

  if (s.kind === 'vehicle') {
    drawVehicle(s, colour, recoil);
    proxy('squad', s.id, s.x, 1.4, s.y, 5.6, 2.8, 3.2);
    return;
  }

  if (s.kind === 'team_weapon') {
    const weapon = models.weaponKind(s.def);
    const key = 'w' + s.id + '|' + s.def + '|' + weapon + '|' + (s.ab ? 'a' : 'c') + '|' + colour;
    const gun = node(key, function () {
      return models.build('weapon', {
        def: s.def, colour: colour, weapon: weapon, abandoned: !!s.ab
      }, materials.prop);
    });
    gun.position.set(s.x, 0, s.y);
    gun.rotation.set(0, -(s.fa !== undefined ? s.fa : s.h), 0);
    // an abandoned gun sits askew, barrel down
    gun.rotation.z = s.ab ? 0.16 : 0;
    gun.position.y = s.ab ? -0.06 : 0;
  } else if (s.kind !== 'infantry' && s.n > 0) {
    // an unknown kind: a plain marker rather than nothing
    const key = 'u' + s.id + '|' + s.def + '|' + colour;
    const marker = node(key, function () {
      const p = new Parts();
      p.add(cyl(0, 0.9, 1.8, 6), colour, { y: 1.5 });
      p.add(cyl(0.9, 0, 1.8, 6), shade(colour, -0.2), { y: 0.9 });
      const mesh = new THREE.Mesh(bake(p), materials.unit);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      return mesh;
    });
    marker.position.set(s.x, 0, s.y);
    marker.rotation.y = -s.h;
  }

  for (let i = 0; i < s.n; i++) {
    const off = offsets[i % offsets.length];
    const ox = off[0] * cos - off[1] * sin;
    const oz = off[0] * sin + off[1] * cos;
    const bob = moving ? Math.abs(Math.sin(time * 7 + i * 1.7 + s.id)) * 0.09 : 0;
    soldier(colour, faction, s.x + ox, s.y + oz, s.h, pose, bob);
  }
  proxy('squad', s.id, s.x, 1.0, s.y, 4.4, 2.2, 4.4);
}

function drawVehicle(s, colour, recoil) {
  const chassis = models.vehicleChassis(s.def);
  const key = 'v' + s.id + '|' + s.def + '|' + chassis + '|' + colour + '|' + (s.ab ? 'a' : 'c');
  const group = node(key, function () {
    const built = models.buildVehicle({ def: s.def, colour: colour, chassis: chassis });
    const g = new THREE.Group();
    const hull = new THREE.Mesh(bake(built.hull), materials.unit);
    hull.castShadow = true; hull.receiveShadow = true;
    g.add(hull);
    const turret = new THREE.Mesh(bake(built.turret), materials.unit);
    turret.castShadow = true; turret.receiveShadow = true;
    turret.position.y = built.turretY;
    g.add(turret);
    g.userData.turret = turret;
    return g;
  });
  group.position.set(s.x, 0, s.y);
  group.rotation.y = -s.h;
  const turret = group.userData.turret;
  // the turret turns in world space, so undo the hull's own yaw
  turret.rotation.y = -(s.th - s.h);
  // barrel recoil: the whole turret rocks back briefly after a shot
  turret.position.x = -recoil * 0.55;
  turret.rotation.z = recoil * 0.06;
}

/** How far into a recoil each vehicle is, from this frame's shot events. */
function recoilBySquad(effects) {
  const out = {};
  for (let i = 0; i < effects.length; i++) {
    const e = effects[i];
    if (e.k !== 'shot') continue;
    const src = e.d.src;
    if (typeof src !== 'number') continue;
    const k = Math.max(0, 1 - e.age * 1.6);
    if (!(out[src] > k)) out[src] = k;
  }
  return out;
}

/** The ground wedge a set-up team weapon covers. */
function drawArc(s, index) {
  const meta = squadMeta(s.def);
  const deg = clamp(meta.arc_deg || 90, 5, 360);
  const range = meta.range || 30;
  let arc = arcs[index];
  if (!arc) {
    const mat = new THREE.MeshBasicMaterial({
      transparent: true, opacity: 0.16, depthWrite: false,
      side: THREE.DoubleSide, fog: false
    });
    arc = new THREE.Mesh(wedgeGeometry(deg), mat);
    arc.renderOrder = 3;
    arcs[index] = arc;
    root.add(arc);
  }
  if (arc.userData.deg !== deg) {
    arc.geometry.dispose();
    arc.geometry = wedgeGeometry(deg);
    arc.userData.deg = deg;
  }
  arc.visible = true;
  arc.position.set(s.x, 0.12, s.y);
  arc.scale.set(range, range, 1);
  // The wedge opens about +X in its own XY plane. With the default XYZ Euler
  // order, rotating -fa about Z and then -90 about X sends local +X to
  // (cos fa, 0, sin fa) — which is exactly where the sim's `facing` points.
  arc.rotation.set(-Math.PI / 2, 0, -s.fa);
  arc.material.color.set(mix(playerColour(s.o), '#ffffff', 0.35));
}

const _wedges = new Map();
function wedgeGeometry(deg) {
  const key = Math.round(deg);
  let g = _wedges.get(key);
  if (!g) {
    const half = key * Math.PI / 360;
    g = new THREE.CircleGeometry(1, Math.max(6, Math.round(key / 6)), -half, half * 2);
    _wedges.set(key, g);
  }
  return g;
}

// ------------------------------------------------------------- buildings --

let garrisonOwner = {};

function drawBuilding(b, seenTick) {
  const ghost = seenTick !== null && seenTick !== undefined;
  const wm = b.w * R.CELL, dm = b.h * R.CELL;
  const cx = (b.cx + b.w / 2) * R.CELL;
  const cz = (b.cy + b.h / 2) * R.CELL;
  const family = models.buildingFamily(b.def);
  const progressBucket = b.prog >= 1 ? 4 : Math.round(b.prog * 4);
  const ruined = b.hp <= 0;
  const colour = b.nu ? '#8a8171' : playerColour(b.o);

  if (ghost) {
    // last known: a translucent grey volume, no detail, no live state at all
    const key = 'G' + b.id + '|' + b.w + 'x' + b.h;
    const g = node(key, function () {
      const mesh = new THREE.Mesh(box(wm * 0.94, 3.4, dm * 0.94), materials.ghost);
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      return mesh;
    });
    g.position.set(cx, 1.7, cz);
    proxy('building', b.id, cx, 1.7, cz, wm, 3.4, dm);
    return;
  }

  const key = 'B' + b.id + '|' + b.def + '|' + b.w + 'x' + b.h + '|' + progressBucket +
              '|' + (ruined ? 'r' : 'o') + '|' + colour;
  const mesh = node(key, function () {
    return models.build('building', {
      def: b.def, colour: colour, family: family, neutral: !!b.nu,
      wm: wm, dm: dm, cx: b.cx, cy: b.cy,
      hash: hash2(b.cx, b.cy), progress: ruined ? 1 : b.prog, ruined: ruined
    }, materials.prop);
  });
  mesh.position.set(cx, 0, cz);
  // damage darkens the whole structure; the smoke comes from `effects`
  const dark = ruined ? 0.45 : (b.hp < 0.5 ? 0.55 + b.hp * 0.9 : 1);
  mesh.material = dark < 1 ? damagedMaterial(dark) : materials.prop;

  // a garrison lights the windows in the occupying team's colour. A neutral
  // building's occupants are only knowable while the team can see it.
  if (b.n > 0 && (!b.nu || isFootprintVisible(b))) {
    lightWindows(b, cx, cz, wm, dm, garrisonOwner[b.id]);
  }

  proxy('building', b.id, cx, 1.8, cz, wm, 3.6, dm);
}

const _damaged = new Map();
function damagedMaterial(k) {
  const key = Math.round(k * 10);
  let m = _damaged.get(key);
  if (!m) {
    m = materials.prop.clone();
    m.color = new THREE.Color(k, k, k);
    _damaged.set(key, m);
  }
  return m;
}


function lightWindows(b, cx, cz, wm, dm, owner) {
  const colour = owner === null || owner === undefined ? '#e8d9a8' : teamColour(owner);
  const key = 'L' + b.id + '|' + colour;
  const glow = node(key, function () {
    const p = new Parts();
    const y = 2.1;
    [[-0.28, 1], [0.28, 1], [-0.28, -1], [0.28, -1]].forEach(function (o) {
      p.box(0.62, 0.62, 0.08, colour, { x: o[0] * wm, y: y, z: o[1] * dm * 0.49 });
    });
    const mesh = new THREE.Mesh(bake(p), materials.fire);
    mesh.renderOrder = 3;
    return mesh;
  });
  glow.position.set(cx, 0, cz);
}

// ----------------------------------------------------------------- points --

function drawPoints(frame, effects) {
  const byId = {};
  frame.points.forEach(function (p) { byId[p.id] = p; });
  const pulses = {};
  effects.forEach(function (e) {
    if (e.k === 'point_captured' || e.k === 'point_neutralized') pulses[e.d.point_id] = e.age;
  });

  R.D.map.points.forEach(function (def) {
    const st = byId[def.id] || {};
    const x = (def.cell[0] + 0.5) * R.CELL, z = (def.cell[1] + 0.5) * R.CELL;
    const victory = def.type === 'victory';
    const owner = st.owner === null || st.owner === undefined ? null : st.owner;
    const colour = owner === null ? '#c9c2ae' : teamColour(owner);
    stats.points++;

    const key = 'P' + def.id + '|' + colour + '|' + (victory ? 'v' : 's');
    const g = node(key, function () {
      const built = models.buildPoint({ def: def.type, colour: colour, victory: victory });
      const group = new THREE.Group();
      const pole = new THREE.Mesh(bake(built.pole), materials.prop);
      pole.castShadow = true; pole.receiveShadow = true;
      group.add(pole);

      // the flag is its own node so capture progress can raise it
      const fp = new Parts();
      const fw = victory ? 2.0 : 1.5, fh = victory ? 1.2 : 0.9;
      fp.box(fw, fh, 0.06, colour, { x: fw / 2 });
      fp.box(fw, 0.1, 0.09, shade(colour, -0.35), { x: fw / 2, y: fh / 2 });
      if (victory) {
        // a star marks a victory point's flag
        fp.add(cyl(fh * 0.3, fh * 0.3, 0.1, 5), '#f0e2b0', { x: fw * 0.5, z: 0.07 });
      }
      const flag = new THREE.Mesh(bake(fp), materials.prop);
      flag.castShadow = true;
      group.add(flag);
      group.userData.flag = flag;
      group.userData.height = built.height;

      // ground ring showing capture progress
      const ringGeo = new THREE.RingGeometry(victory ? 3.4 : 2.6, victory ? 3.9 : 3.0, 40);
      ringGeo.rotateX(-Math.PI / 2);
      const ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false,
        side: THREE.DoubleSide, fog: false
      }));
      ring.position.y = 0.1;
      ring.renderOrder = 3;
      group.add(ring);
      group.userData.ring = ring;
      return group;
    });

    g.position.set(x, 0, z);
    const height = g.userData.height;
    // a flag being taken is halfway up the pole; a held one is at the top
    const raised = st.progress !== undefined && st.progress > 0 && st.progress < 1
      ? 0.25 + st.progress * 0.7 : 1;
    g.userData.flag.position.y = 0.4 + height * (0.45 + raised * 0.5);

    const ring = g.userData.ring;
    const capturing = st.c === null || st.c === undefined ? null : st.c;
    ring.material.color.set(capturing !== null ? teamColour(capturing) : colour);
    ring.material.opacity = st.progress > 0 && st.progress < 1 ? 0.8 : 0.22;
    ring.scale.setScalar(st.progress > 0 && st.progress < 1 ? 0.75 + st.progress * 0.35 : 1);

    if (pulses[def.id] !== undefined) pulseRing(x, z, pulses[def.id], victory);
  });
}

let pulseSink = null;
export function setPulseSink(fn) { pulseSink = fn; }
function pulseRing(x, z, age, big) {
  if (pulseSink) pulseSink(x, z, age, big);
}

// ------------------------------------------------------------ selection --

function markSelection(frame, squads, buildings) {
  selectionRing.visible = false;
  hoverRing.visible = false;
  place(selectionRing, S.selection, squads, buildings);
  if (S.hover && (!S.selection || S.hover.id !== S.selection.id || S.hover.kind !== S.selection.kind)) {
    place(hoverRing, S.hover, squads, buildings);
  }
}

function place(ring, sel, squads, buildings) {
  if (!sel) return;
  if (sel.kind === 'squad') {
    for (let i = 0; i < squads.length; i++) {
      const s = squads[i];
      if (s.id !== sel.id || squadHidden(s) || s.g !== undefined) continue;
      ring.visible = true;
      ring.position.set(s.x, 0.11, s.y);
      ring.scale.setScalar(s.kind === 'vehicle' ? 3.6 : 2.8);
      return;
    }
    return;
  }
  for (let i = 0; i < buildings.length; i++) {
    const b = buildings[i].b;
    if (b.id !== sel.id) continue;
    ring.visible = true;
    ring.position.set((b.cx + b.w / 2) * R.CELL, 0.11, (b.cy + b.h / 2) * R.CELL);
    ring.scale.setScalar(Math.max(b.w, b.h) * R.CELL * 0.72);
    return;
  }
}

export function sceneStats() {
  return {
    units: stats.units, soldiers: stats.soldiers, buildings: stats.buildings,
    ghosts: stats.ghosts, points: stats.points, nodes: cache.size,
    pickables: proxyCount
  };
}

export function pickables() {
  const out = [];
  for (let i = 0; i < proxyCount; i++) out.push(proxyPool[i]);
  return out;
}
