/* A Company of Heroes-style camera, written here rather than pulled in from
 * three.js's `examples/jsm` — the behaviour wanted is specific enough
 * (pitch that flattens as you zoom out, zoom toward the cursor, unit follow,
 * clamped to the map) that OrbitControls would have been fought rather than
 * used, and keeping it local means one vendored file instead of two.
 *
 * The state is a ground `target`, a `dist` from it, a `yaw` and a `pitch`
 * from the horizon; it lives in `S.cam3` so a test can read and set it.
 */

import * as THREE from 'three';
import { S } from '../state.js';
import { R } from '../data.js';
import { clamp, lerp } from '../util.js';

const MIN_DIST = 18;
const MAX_DIST = 320;
const FOV = 42;

/* Pitch by zoom: close in you look along the ground like an infantryman's
 * commander; pulled back it flattens toward a map view. 55 degrees is the
 * CoH default and is what the mid range settles on. */
const NEAR_PITCH = 0.74;   // ~42 deg
const FAR_PITCH = 1.17;    // ~67 deg
const PITCH_MIN = 0.42;
const PITCH_MAX = 1.35;

let camera = null;
let pitchOffset = 0;       // how far the user has pitched away from the default
const keys = Object.create(null);

export function create(aspect) {
  camera = new THREE.PerspectiveCamera(FOV, aspect, 0.8, 1400);
  return camera;
}

export function get() { return camera; }

export function setAspect(aspect) {
  camera.aspect = aspect;
  camera.updateProjectionMatrix();
}

/** The pitch the current zoom level wants, before the user's own offset. */
function defaultPitch() {
  const u = clamp((S.cam3.dist - MIN_DIST) / (MAX_DIST - MIN_DIST), 0, 1);
  return lerp(NEAR_PITCH, FAR_PITCH, Math.pow(u, 0.7));
}

export function reset() {
  const wm = R.W * R.CELL, hm = R.H * R.CELL;
  S.cam3.x = wm / 2;
  S.cam3.y = hm / 2;
  S.cam3.yaw = 0;
  S.cam3.dist = clamp(Math.max(wm, hm) * 0.95, MIN_DIST, MAX_DIST);
  S.cam3.follow = null;
  pitchOffset = 0;
  apply();
}

/** Frame a world position at roughly the 2D view's zoom level. */
export function lookAt(x, y, scale) {
  S.cam3.x = x;
  S.cam3.y = y;
  if (scale) S.cam3.dist = clamp(520 / scale, MIN_DIST, MAX_DIST);
  S.cam3.follow = null;
  apply();
}

export function follow(id) { S.cam3.follow = id; }

/** Recompute the camera transform from `S.cam3`. */
export function apply() {
  const c = S.cam3;
  c.dist = clamp(c.dist, MIN_DIST, MAX_DIST);
  const margin = 40;
  c.x = clamp(c.x, -margin, R.W * R.CELL + margin);
  c.y = clamp(c.y, -margin, R.H * R.CELL + margin);
  c.pitch = clamp(defaultPitch() + pitchOffset, PITCH_MIN, PITCH_MAX);

  const cosP = Math.cos(c.pitch), sinP = Math.sin(c.pitch);
  camera.position.set(
    c.x - Math.sin(c.yaw) * cosP * c.dist,
    Math.max(2.5, sinP * c.dist),
    c.y - Math.cos(c.yaw) * cosP * c.dist
  );
  camera.lookAt(c.x, 0, c.y);
  camera.updateMatrixWorld();
}

// ------------------------------------------------------------------ input --

/** The ground point (world x, sim y) under a screen position. */
export function groundAt(px, py, width, height) {
  const ndc = new THREE.Vector2((px / width) * 2 - 1, -(py / height) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const hit = new THREE.Vector3();
  if (!ray.ray.intersectPlane(plane, hit)) return null;
  return [hit.x, hit.z];
}

export function pan(dxPx, dyPx, height) {
  // Convert a screen drag into ground metres: at the camera's distance, the
  // visible ground height is roughly 2*dist*tan(fov/2) / sin(pitch).
  const perPx = (2 * S.cam3.dist * Math.tan(FOV * Math.PI / 360)) / height;
  const forward = perPx / Math.max(0.35, Math.sin(S.cam3.pitch));
  const sy = Math.sin(S.cam3.yaw), cy = Math.cos(S.cam3.yaw);
  // screen right is (cos yaw, -sin yaw); screen up-the-map is (sin yaw, cos yaw)
  S.cam3.x -= dxPx * perPx * cy + dyPx * forward * sy;
  S.cam3.y += dxPx * perPx * sy - dyPx * forward * cy;
  S.cam3.follow = null;
  apply();
}

export function setPitchOffset(v) { pitchOffset = clamp(v, -0.5, 0.5); }

export function rotate(dYaw, dPitch) {
  S.cam3.yaw += dYaw;
  pitchOffset = clamp(pitchOffset + dPitch, -0.5, 0.5);
  apply();
}

/** Wheel zoom that keeps the ground point under the cursor under the cursor. */
export function zoom(deltaY, px, py, width, height) {
  const before = groundAt(px, py, width, height);
  S.cam3.dist = clamp(S.cam3.dist * Math.pow(1.0016, deltaY), MIN_DIST, MAX_DIST);
  apply();
  const after = groundAt(px, py, width, height);
  if (before && after) {
    S.cam3.x += before[0] - after[0];
    S.cam3.y += before[1] - after[1];
    S.cam3.follow = null;
    apply();
  }
}

export function keyDown(e) {
  const k = e.key.toLowerCase();
  if ('wasdqe'.indexOf(k) < 0) return false;
  keys[k] = true;
  return true;
}

export function keyUp(e) { keys[e.key.toLowerCase()] = false; }

export function keysHeld() {
  return !!(keys.w || keys.a || keys.s || keys.d || keys.q || keys.e);
}

/** Continuous keyboard pan/rotate. Returns true if the camera moved. */
export function step(dt) {
  if (!keysHeld()) return false;
  const speed = clamp(S.cam3.dist * 0.9, 20, 160) * dt;
  const sy = Math.sin(S.cam3.yaw), cy = Math.cos(S.cam3.yaw);
  let dx = 0, dz = 0;
  if (keys.w) { dx += sy; dz += cy; }
  if (keys.s) { dx -= sy; dz -= cy; }
  if (keys.a) { dx -= cy; dz += sy; }
  if (keys.d) { dx += cy; dz -= sy; }
  if (dx || dz) {
    S.cam3.x += dx * speed;
    S.cam3.y += dz * speed;
    S.cam3.follow = null;
  }
  if (keys.q) S.cam3.yaw -= dt * 1.5;
  if (keys.e) S.cam3.yaw += dt * 1.5;
  apply();
  return true;
}

/** Ease the camera onto a followed squad, if there is one. */
export function trackFollowed(squads, dt) {
  if (S.cam3.follow === null) return false;
  for (let i = 0; i < squads.length; i++) {
    if (squads[i].id !== S.cam3.follow) continue;
    const k = Math.min(1, dt * 6);
    S.cam3.x = lerp(S.cam3.x, squads[i].x, k);
    S.cam3.y = lerp(S.cam3.y, squads[i].y, k);
    apply();
    return true;
  }
  S.cam3.follow = null;   // it died or left the frame
  return false;
}

export { MIN_DIST, MAX_DIST };
