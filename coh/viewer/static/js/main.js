/* Boot, playback, input routing and the test hook.
 *
 * `main.js` owns the writes to `state.js`; the data layer and the two
 * renderers only read. Rendering is on demand rather than a permanent
 * animation loop: while paused nothing moves, so repainting would only burn
 * CPU (and would keep the page from ever going idle, which headless
 * screenshot capture waits for).
 */

import { S } from './state.js';
import { SPEEDS, clamp } from './util.js';
import {
  R, activeEffects, applyTerrainTo, buildingFogMode, currentFrame, frameIndexFor,
  interpolated, lastTick, playhead, prepare, seenNow, setFogFrame
} from './data.js';
import * as hud from './hud.js';
import * as view2d from './view2d.js';

const $ = function (id) { return document.getElementById(id); };

const cv = $('cv');
let lastRAF = 0, rafPending = false;

view2d.init(cv);

// ------------------------------------------------------------------- hash --

/* The URL hash carries the view choice and whether playback is paused, so a
 * link points at a moment *and* at a way of looking at it. Tokens are
 * `&`-separated: `#paused&view=2d`. */

function readHash() {
  const tokens = window.location.hash.replace(/^#/, '').split('&').filter(Boolean);
  const out = { paused: false, view: null };
  tokens.forEach(function (t) {
    if (t === 'paused') out.paused = true;
    else if (t.indexOf('view=') === 0) out.view = t.slice(5);
  });
  return out;
}

function writeHash() {
  const tokens = [];
  if (!S.playing) tokens.push('paused');
  tokens.push('view=' + S.view);
  const next = '#' + tokens.join('&');
  if (window.location.hash !== next) {
    // replaceState, not `location.hash =`, so the back button is not filled
    // with every pause and view toggle.
    window.history.replaceState(null, '', window.location.pathname + window.location.search + next);
  }
}

// ------------------------------------------------------------------ views --

/** The renderer module currently in charge. */
let view = view2d;

function applyView() {
  const threeD = S.view === '3d' && S.webgl && view3d !== null;
  if (!threeD) S.view = '2d';
  cv.hidden = threeD;
  $('cv3').hidden = !threeD;
  $('ov').hidden = !threeD;
  $('btn-view').textContent = threeD ? '3D' : '2D';
  $('btn-view').classList.toggle('on', threeD);
  document.body.classList.toggle('view-3d', threeD);
  view = threeD ? view3d : view2d;
  view.resize();
  writeHash();
}

/* Filled in by `install3d()` once the 3D module has loaded; until then (and
 * for ever, if WebGL is missing) `view` stays on the tactical map. */
let view3d = null;

export function install3d(module) {
  view3d = module;
}

function toggleView() {
  if (!view3d || !S.webgl) {
    hud.notice('3D needs WebGL, which this browser did not provide — showing the tactical map.');
    return;
  }
  S.view = S.view === '3d' ? '2d' : '3d';
  applyView();
  requestRender();
}

// ------------------------------------------------------------------ boot --

function fail(message) {
  $('loading').hidden = true;
  $('error').hidden = false;
  $('error-msg').textContent = message;
}

function boot() {
  fetch('frames.json').then(function (r) {
    if (!r.ok) throw new Error('GET /frames.json -> HTTP ' + r.status + ' ' + r.statusText);
    return r.json();
  }).then(function (payload) {
    if (!payload || !payload.frames || !payload.frames.length) throw new Error('frame stream is empty');
    start(payload);
  }).catch(function (err) {
    fail(String(err && err.message ? err.message : err));
  });
}

function start(payload) {
  prepare(payload);
  S.tick = 0;
  view2d.reset();
  if (view3d) view3d.reset();

  $('loading').hidden = true;
  $('error').hidden = true;
  ['hud', 'bar'].forEach(function (id) { $(id).hidden = false; });
  hud.buildLegend();

  const hash = readHash();
  if (hash.view === '2d' || hash.view === '3d') S.view = hash.view;
  if (view3d === null || !S.webgl) {
    if (S.view === '3d') hud.notice('WebGL is not available here — showing the 2D tactical map.');
    S.view = '2d';
  }

  resize();
  applyView();
  // Only the tactical map is framed here: the 3D view picks its own opening
  // shot (over the watched team's HQ, looking toward the middle) in its
  // `reset()` above, and `Home` is what gives the whole-map view.
  view2d.resetCamera();
  // `#paused` opens on the first frame without starting playback, which is
  // what a deep link into a moment (and the headless browser check) wants.
  setPlaying(!hash.paused);
  requestRender();
}

// ---------------------------------------------------------------- camera --

function resize() {
  S.dpr = Math.min(window.devicePixelRatio || 1, 2);
  view2d.resize();
  if (view3d) view3d.resize();
  const tl = $('timeline');
  tl.width = Math.round(tl.clientWidth * S.dpr);
  tl.height = Math.round(tl.clientHeight * S.dpr);
}

function resetCamera() {
  view2d.resetCamera();
  if (view3d) view3d.resetCamera();
}

// ---------------------------------------------------------------- render --

function render() {
  const head = playhead();
  applyTerrainTo(head.idx);
  setFogFrame(head.idx);
  const effects = activeEffects(S.tick);
  const squads = interpolated(head.a, head.b, head.u);
  view.render(head, squads, effects);
  hud.updateChrome(head.a);
}

export function requestRender() {
  if (rafPending || !R.D) return;
  rafPending = true;
  requestAnimationFrame(loop);
}

function loop(now) {
  rafPending = false;
  const dt = Math.min((now - lastRAF) / 1000, 0.2);
  lastRAF = now;
  if (playing()) {
    S.tick += dt * R.TPS * SPEEDS[S.speedIdx];
    if (S.tick >= lastTick()) { S.tick = lastTick(); setPlaying(false); }
  }
  if (view.animate) view.animate(dt);
  try {
    render();
  } catch (err) {
    console.error('render failed', err);
  }
  if (playing() || (view.needsFrame && view.needsFrame())) requestRender();
}

function playing() { return S.playing; }

// ----------------------------------------------------------------- input --

function setPlaying(on) {
  S.playing = on;
  lastRAF = performance.now();
  $('btn-play').innerHTML = on ? '&#10073;&#10073;' : '&#9654;';
  writeHash();
  requestRender();
}

function setSpeed(i) {
  S.speedIdx = clamp(i, 0, SPEEDS.length - 1);
  $('speed').innerHTML = SPEEDS[S.speedIdx] + '&times;';
  requestRender();
}

function step(delta) {
  const idx = clamp(frameIndexFor(S.tick) + delta, 0, R.frames.length - 1);
  S.tick = R.frames[idx].t;
  setPlaying(false);
}

function cycleFog() {
  S.fogMode = (S.fogMode + 1) % 3;
  const b = $('btn-fog');
  b.textContent = 'Fog: ' + (S.fogMode === 0 ? 'all' : 'team ' + (S.fogMode - 1));
  b.classList.toggle('on', S.fogMode > 0);
  requestRender();
}

function toggle(id, key) {
  S[key] = !S[key];
  const b = $(id);
  if (b) b.classList.toggle('on', S[key]);
  requestRender();
}

function closePanelAndRender() { hud.closePanel(); requestRender(); }

function install() {
  window.addEventListener('resize', function () { resize(); requestRender(); });
  window.addEventListener('hashchange', function () {
    const hash = readHash();
    if ((hash.view === '2d' || hash.view === '3d') && hash.view !== S.view && view3d) {
      S.view = hash.view;
      applyView();
      requestRender();
    }
  });

  $('btn-play').onclick = function () { setPlaying(!S.playing); };
  $('btn-back').onclick = function () { step(-1); };
  $('btn-fwd').onclick = function () { step(1); };
  $('btn-slower').onclick = function () { setSpeed(S.speedIdx - 1); };
  $('btn-faster').onclick = function () { setSpeed(S.speedIdx + 1); };
  $('btn-fog').onclick = cycleFog;
  $('btn-view').onclick = toggleView;
  $('btn-cover').onclick = function () { toggle('btn-cover', 'showCover'); };
  $('btn-sectors').onclick = function () { toggle('btn-sectors', 'showSectors'); };
  $('btn-arcs').onclick = function () { toggle('btn-arcs', 'showArcs'); };
  $('btn-shadows').onclick = function () {
    toggle('btn-shadows', 'shadows');
    if (view3d) view3d.applyShadows();
  };
  $('btn-help').onclick = hud.toggleHelp;
  $('panel-close').onclick = closePanelAndRender;
  $('btn-sectors').classList.add('on');
  $('btn-shadows').classList.toggle('on', S.shadows);
  $('notice-close').onclick = function () { $('notice').hidden = true; };

  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    // The active view gets first refusal: the 3D camera owns W/A/S/D/Q/E, and
    // a global single-letter binding must not shadow them.
    if (view.key && view.key(e)) { requestRender(); return; }
    const k = e.key;
    if (k === ' ') { setPlaying(!S.playing); e.preventDefault(); }
    else if (k === 'ArrowLeft') step(-1);
    else if (k === 'ArrowRight') step(1);
    else if (k === '+' || k === '=') setSpeed(S.speedIdx + 1);
    else if (k === '-' || k === '_') setSpeed(S.speedIdx - 1);
    else if (k === 'f' || k === 'F') cycleFog();
    else if (k === 'c' || k === 'C') $('btn-cover').click();
    else if (k === 'g' || k === 'G') $('btn-sectors').click();
    else if (k === 'v' || k === 'V') toggleView();
    else if (k === 'z' || k === 'Z') $('btn-arcs').click();
    else if (k === 'h' || k === 'H') $('btn-shadows').click();
    else if (k === 'Home') { resetCamera(); requestRender(); }
    else if (k === '?' || k === '/') hud.toggleHelp();
    else if (k === 'Escape') {
      if (!$('help').hidden) hud.toggleHelp();
      else closePanelAndRender();
    }
  });
  document.addEventListener('keyup', function (e) {
    if (view.keyUp) view.keyUp(e);
  });

  installPointer(cv, view2d);

  // timeline scrub
  const tl = $('timeline');
  let scrubbing = false;
  function scrub(e) {
    const r = tl.getBoundingClientRect();
    S.tick = clamp((e.clientX - r.left) / r.width, 0, 1) * lastTick();
    requestRender();
  }
  tl.addEventListener('mousedown', function (e) { scrubbing = true; setPlaying(false); scrub(e); });
  window.addEventListener('mousemove', function (e) { if (scrubbing) scrub(e); });
  window.addEventListener('mouseup', function () { scrubbing = false; });
}

/** Pan / zoom / pick on the 2D tactical map. The 3D view installs its own. */
function installPointer(canvas, v) {
  let drag = null, moved = 0;
  canvas.addEventListener('mousedown', function (e) {
    drag = [e.clientX, e.clientY]; moved = 0; canvas.classList.add('dragging');
  });
  window.addEventListener('mousemove', function (e) {
    if (!drag) return;
    S.cam.x -= (e.clientX - drag[0]) / S.cam.scale;
    S.cam.y -= (e.clientY - drag[1]) / S.cam.scale;
    moved += Math.abs(e.clientX - drag[0]) + Math.abs(e.clientY - drag[1]);
    drag = [e.clientX, e.clientY];
    requestRender();
  });
  window.addEventListener('mouseup', function (e) {
    if (drag && moved < 4) select(v.pick(e.clientX, e.clientY));
    drag = null;
    canvas.classList.remove('dragging');
    requestRender();
  });
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    const before = v.toWorld(e.clientX, e.clientY);
    S.cam.scale = clamp(S.cam.scale * Math.pow(1.0015, -e.deltaY), 0.8, 60);
    const after = v.toWorld(e.clientX, e.clientY);
    S.cam.x += before[0] - after[0];
    S.cam.y += before[1] - after[1];
    requestRender();
  }, { passive: false });
}

export function select(hit) {
  if (hit) S.selection = hit; else hud.closePanel();
  requestRender();
}

// ------------------------------------------------------------- test hook --

/* Everything the headless browser check needs to drive the app. Small, and
 * documented, but it is API surface that exists for the tests. */
window.__viewer = {
  inject: function (payload) { window.__viewerInjected = true; start(payload); },
  /** Re-fetch and reload the served stream — undoes an `inject`. */
  reload: function () { window.__viewerInjected = false; boot(); },
  seek: function (fraction) { S.tick = clamp(fraction, 0, 1) * lastTick(); requestRender(); },
  seekTick: function (t) { S.tick = clamp(t, 0, lastTick()); requestRender(); },
  camera: function (x, y, scale) {
    S.cam.x = x; S.cam.y = y; S.cam.scale = scale;
    if (view3d) view3d.lookAt(x, y, scale);
    requestRender();
  },
  /** Place the 3D camera directly: nulls keep the current target. */
  camera3: function (x, y, dist, yaw, pitch) {
    if (x !== null && x !== undefined) S.cam3.x = x;
    if (y !== null && y !== undefined) S.cam3.y = y;
    if (dist !== null && dist !== undefined) S.cam3.dist = dist;
    if (yaw !== null && yaw !== undefined) S.cam3.yaw = yaw;
    if (view3d) view3d.applyCamera(pitch);
    requestRender();
  },
  select: function (kind, id) { S.selection = { kind: kind, id: id }; requestRender(); },
  state: function () {
    return {
      tick: S.tick, fogMode: S.fogMode, showCover: S.showCover, showSectors: S.showSectors,
      showArcs: S.showArcs, shadows: S.shadows, view: S.view, quality: S.quality,
      webgl: S.webgl, frames: R.frames.length, playing: S.playing,
      speed: SPEEDS[S.speedIdx], selection: S.selection, winner: R.D ? R.D.winner : null,
      hash: window.location.hash
    };
  },
  /** Draw one frame synchronously (headless capture must not wait on rAF). */
  redraw: function () { render(); },
  /** What the last render actually drew — the fog assertions read this. */
  stats: function () { return view.lastStats(); },
  /** Scene-graph counts for the 3D view; null when 3D never loaded. */
  scene3dStats: function () { return view3d ? view3d.sceneStats() : null; },
  /** The 3D camera's state, for the camera tests. */
  state3d: function () { return view3d ? view3d.cameraState() : null; },
  /** Raw pixels out of the WebGL canvas (stays inside the page). */
  readPixels3d: function (x, y, w, h) { return view3d ? view3d.readPixels(x, y, w, h) : null; },
  setView: function (which) {
    if (which !== S.view) toggleView();
    return S.view;
  },
  /** Screen pixel for a world position, so a test can sample a known spot. */
  screenOf: function (x, y) { return view.screenOf ? view.screenOf(x, y) : view.toScreen(x, y); },
  /** Checksum of the pixels in a box around a world position. */
  sampleWorld: function (x, y, radiusPx) {
    const p = window.__viewer.screenOf(x, y);
    return sampleBox(p[0] - radiusPx, p[1] - radiusPx, radiusPx * 2, radiusPx * 2);
  },
  /** Which of 'live' / 'ghost' / 'hidden' a building is under the current fog. */
  buildingFog: function (id) {
    const frame = currentFrame();
    const seen = seenNow();
    const b = frame.buildings.filter(function (x) { return x.id === id; })[0];
    if (!b) return seen && seen[id] ? 'ghost' : 'gone';
    return buildingFogMode(b, seen);
  },
  /** What a click at a screen point would select (null when nothing is clickable). */
  pickAt: function (px, py) { return view.pick(px, py); },
  /** The current frame, for tests that need to build on real data. */
  frame: function () { return currentFrame(); },
  data: function () { return R.D; },
  mapSize: function () { return { W: R.W, H: R.H, cell: R.CELL }; },
  events: function () { return R.allEvents; },
  setPlaying: setPlaying,
  cycleFog: cycleFog
};

function sampleBox(px, py, w, h) {
  const canvas = S.view === '3d' && S.webgl ? $('cv3') : cv;
  const x0 = Math.max(0, Math.round(px * S.dpr));
  const y0 = Math.max(0, Math.round(py * S.dpr));
  const ww = Math.min(Math.max(1, Math.round(w * S.dpr)), canvas.width - x0);
  const hh = Math.min(Math.max(1, Math.round(h * S.dpr)), canvas.height - y0);
  if (ww <= 0 || hh <= 0) return 0;
  const data = readPixels(canvas, x0, y0, ww, hh);
  let sum = 2166136261;
  for (let i = 0; i < data.length; i += 4) {
    sum ^= data[i] + data[i + 1] * 3 + data[i + 2] * 7;
    sum = (sum * 16777619) | 0;
  }
  return sum;
}

function readPixels(canvas, x, y, w, h) {
  if (canvas === cv) return cv.getContext('2d').getImageData(x, y, w, h).data;
  return view3d.readPixels(x, y, w, h);
}

/** Start the app. `boot.js` calls this after deciding whether 3D is available. */
export function run() {
  install();
  setSpeed(1);
  boot();
}
