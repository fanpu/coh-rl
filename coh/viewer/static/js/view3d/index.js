/* The 3D battlefield view.
 *
 * Presents the same interface to `main.js` as `view2d.js` does — `render`,
 * `resize`, `resetCamera`, `pick`, `screenOf`, `lastStats` — so the two are
 * interchangeable, and takes every "may this be shown?" decision from
 * `data.js` rather than making its own.
 */

import * as THREE from 'three';
import { S } from '../state.js';
import {
  R, emptyStats, fog, squadHidden, visibleBuildings, visibleEffects
} from '../data.js';
import * as scene from './scene.js';
import * as camera from './camera.js';
import * as terrain from './terrain.js';
import * as entities from './entities.js';
import * as effects from './effects.js';
import * as overlay from './overlay.js';
import { register } from './models.js';

const $ = function (id) { return document.getElementById(id); };

let canvas = null, ov = null, built = false;
let three = null;
let stats = emptyStats();
let clock = 0;
let ray = null;

export { register };

// ------------------------------------------------------------------ setup --

canvas = $('cv3');
ov = $('ov');
overlay.init(ov);
three = scene.create(canvas);
camera.create(window.innerWidth / Math.max(1, window.innerHeight));
ray = new THREE.Raycaster();
installPointer();

/** Build (or rebuild) the scene for the loaded replay. */
export function reset() {
  if (!R.D) return;
  if (built) {
    entities.dispose(three.scene);
    effects.dispose(three.scene);
    terrain.dispose(three.scene);
  }
  terrain.reset();
  terrain.build(three.scene, scene.materials);
  entities.build(three.scene);
  effects.build(three.scene);
  entities.setPulseSink(function (x, z, age) { effects.capturePulse(x, z, age); });
  built = true;
  resize();
  camera.reset();
}

export function resetCamera() {
  if (built) camera.reset();
}

export function lookAt(x, y, scale) {
  if (built) camera.lookAt(x, y, scale);
}

export function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  scene.resize(w, h);
  camera.setAspect(w / Math.max(1, h));
  overlay.resize(w, h);
  if (built) camera.apply();
}

export function applyShadows() { scene.applyShadows(); }

/** Push `S.cam3` into the camera (the test hook writes the state directly). */
export function applyCamera(pitchOffset) {
  if (!built) return;
  if (pitchOffset !== undefined && pitchOffset !== null) camera.setPitchOffset(pitchOffset);
  camera.apply();
}

export function lastStats() { return stats; }

/** Pixels out of the WebGL canvas. Renders first, in this same task: the
 *  drawing buffer is discarded on composite, so a bare read comes back zeroed. */
export function readPixels(x, y, w, h) {
  if (!built) return new Uint8Array(w * h * 4);
  scene.render(camera.get());
  return scene.readPixels(x, y, w, h);
}

// ----------------------------------------------------------------- render --

export function render(head, squads, activeEffectsList) {
  if (!built) reset();
  if (!built) return;

  stats = emptyStats();
  squads.forEach(function (s) { if (squadHidden(s)) stats.hiddenSquads++; });

  const buildings = visibleBuildings(head.a, stats);
  const shown = visibleEffects(activeEffectsList, head.a, stats);

  terrain.update(three.scene, head.a, fog, R.terrVersion);
  entities.update(head.a, squads, buildings, shown, activeEffectsList, clock);
  effects.update(shown, head.a, S.tick);

  const cam = camera.get();
  camera.trackFollowed(squads, 1 / 60);
  scene.fitShadows(cam);
  scene.render(cam);
  overlay.draw(cam, head.a, squads, buildings);

  // the 2D view's `stats()` contract, filled from what actually got drawn
  const ent = entities.lastStats();
  stats.squads = ent.units;
  stats.badges = overlay.badgeCount();
}

/** Keep animating while something is moving under its own steam. */
export function needsFrame() {
  return S.cam3.follow !== null || camera.keysHeld();
}

export function animate(dt) {
  clock += dt;
  camera.step(dt);
}

// -------------------------------------------------------------- hit test --

function pickAt(px, py) {
  if (!built) return null;
  const ndc = new THREE.Vector2(
    (px / window.innerWidth) * 2 - 1,
    -(py / window.innerHeight) * 2 + 1
  );
  ray.setFromCamera(ndc, camera.get());
  const hits = ray.intersectObjects(entities.pickables(), false);
  for (let i = 0; i < hits.length; i++) {
    const hit = hits[i].object.userData.hit;
    if (hit) return { kind: hit.kind, id: hit.id };
  }
  return null;
}

export function pick(px, py) { return pickAt(px, py); }

/** Screen pixel for a world position, so a test can sample a known spot. */
export function screenOf(x, y) {
  const v = new THREE.Vector3(x, 1.0, y).project(camera.get());
  return [(v.x * 0.5 + 0.5) * window.innerWidth, (-v.y * 0.5 + 0.5) * window.innerHeight];
}

/** Where the camera is, so a test can assert it moved. */
export function cameraState() {
  return {
    x: S.cam3.x, y: S.cam3.y, dist: S.cam3.dist, yaw: S.cam3.yaw,
    pitch: S.cam3.pitch, follow: S.cam3.follow
  };
}

export function sceneStats() {
  const s = entities.sceneStats();
  const fx = effects.counts();
  const info = scene.info();
  return {
    units: s.units, soldiers: s.soldiers, buildings: s.buildings, ghosts: s.ghosts,
    points: s.points, nodes: s.nodes, pickables: s.pickables,
    features: terrain.featureCount(),
    tracers: fx.tracers, puffs: fx.fire + fx.smoke, rings: fx.rings, wrecks: fx.wrecks,
    drawCalls: info.calls, triangles: info.triangles,
    shadows: S.shadows && S.quality !== 'low'
  };
}

// ----------------------------------------------------------------- input --

/** Keys this view handles that `main.js` does not. Returns true if consumed. */
export function key(e) {
  if (!built) return false;
  return camera.keyDown(e);
}

export function keyUp(e) { camera.keyUp(e); }

/* Left-drag pans, right-drag rotates, the wheel zooms toward the cursor,
 * a click selects and a double-click follows. `main.js` installs the 2D
 * canvas's own handlers; these are ours and only fire over `#cv3`. */
function installPointer() {
  let drag = null, button = 0, moved = 0;

  canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });

  canvas.addEventListener('mousedown', function (e) {
    drag = [e.clientX, e.clientY];
    button = e.button;
    moved = 0;
    canvas.classList.add('dragging');
  });

  window.addEventListener('mousemove', function (e) {
    if (!drag) {
      if (S.view === '3d' && built) hover(e);
      return;
    }
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
    moved += Math.abs(dx) + Math.abs(dy);
    if (button === 2) camera.rotate(-dx * 0.006, -dy * 0.004);
    else camera.pan(dx, dy, window.innerHeight);
    drag = [e.clientX, e.clientY];
    request();
  });

  window.addEventListener('mouseup', function (e) {
    if (drag && moved < 4 && button === 0 && S.view === '3d') {
      select(pickAt(e.clientX, e.clientY));
    }
    drag = null;
    canvas.classList.remove('dragging');
  });

  canvas.addEventListener('dblclick', function (e) {
    const hit = pickAt(e.clientX, e.clientY);
    if (hit && hit.kind === 'squad') camera.follow(hit.id);
    request();
  });

  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    camera.zoom(e.deltaY, e.clientX, e.clientY, window.innerWidth, window.innerHeight);
    request();
  }, { passive: false });
}

function hover(e) {
  const hit = pickAt(e.clientX, e.clientY);
  const same = (hit === null) === (S.hover === null) &&
               (!hit || (S.hover && hit.id === S.hover.id && hit.kind === S.hover.kind));
  canvas.classList.toggle('picking', !!hit);
  if (same) return;
  S.hover = hit;
  request();
}

/* `main.js` owns the render loop; rather than import it (which would make the
 * dependency circular) the view asks for a frame through a callback it
 * installs. */
let requestFn = function () {};
let selectFn = function () {};
export function wire(onRequestRender, onSelect) {
  requestFn = onRequestRender;
  selectFn = onSelect;
}
function request() { requestFn(); }
function select(hit) { selectFn(hit); }
