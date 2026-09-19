/* The ground: one textured mesh, three blended overlay layers, and the
 * vertical features that grow out of the terrain characters.
 *
 * The ground texture is painted procedurally into a canvas at several pixels
 * per cell, with noise that does not line up with the cell grid — that is
 * what stops a 96x96 array of characters from looking like a 96x96 array of
 * characters. Terrain deltas repaint only the cells that changed.
 *
 * Everything repeated (hedgerow banks, walls, fence panels, tree trunks and
 * canopies, crater rims) is an InstancedMesh, so a map with two thousand
 * props still costs a handful of draw calls.
 */

import * as THREE from 'three';
import { S } from '../state.js';
import { COVER, hash2, hash3, mix, parseColour } from '../util.js';
import { R, teamColour } from '../data.js';
import { Parts, bake, box, cyl, lumpyBox, sphere } from './geom.js';

// Ground tones, muted and a touch desaturated from the tactical-map palette:
// at ground level a saturated green reads as a golf course.
const GRASS = [94, 110, 66];
const GRASS_DRY = [124, 124, 82];
const FIELD = [130, 120, 80];
const DIRT = [132, 112, 80];
const MUD = [96, 78, 54];
const WATER = [52, 82, 102];
const SCORCH = [42, 36, 30];

let px = 8;            // ground-texture pixels per cell
let canvas = null, g2 = null, tex = null;
let ground = null, tint = null, cover = null, fogPlane = null, surround = null;
let roadField = null;  // per-cell road coverage, smoothed
let features = null;   // the InstancedMesh group
let tintKey = null, coverBuilt = false, fogBits = null;
let materials = null;

export function dispose(scene) {
  [ground, tint, cover, fogPlane, surround, features].forEach(function (o) {
    if (o) { scene.remove(o); o.traverse && o.traverse(function (c) { if (c.geometry) c.geometry.dispose(); }); }
  });
  if (tex) tex.dispose();
  canvas = g2 = tex = ground = tint = cover = fogPlane = surround = features = null;
  tintKey = null; coverBuilt = false; fogBits = null;
}

// --------------------------------------------------------- ground texture --

function noise(x, y, scale, salt) {
  // value noise, bilinearly interpolated — cheap and good enough for grass
  const fx = x / scale, fy = y / scale;
  const ix = Math.floor(fx), iy = Math.floor(fy);
  const tx = fx - ix, ty = fy - iy;
  const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
  const a = hash3(ix, iy, salt), b = hash3(ix + 1, iy, salt);
  const c = hash3(ix, iy + 1, salt), d = hash3(ix + 1, iy + 1, salt);
  return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
}

function ch(cx, cy) {
  if (cx < 0 || cy < 0 || cx >= R.W || cy >= R.H) return '.';
  return R.terr[cy * R.W + cx];
}

/** Per-cell road coverage, blurred one cell out so road edges are soft. */
function buildRoadField() {
  const W = R.W, H = R.H;
  const raw = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) raw[i] = R.terr[i] === 'r' ? 1 : 0;
  const out = new Float32Array(W * H);
  for (let cy = 0; cy < H; cy++) {
    for (let cx = 0; cx < W; cx++) {
      let sum = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          const w = (dx === 0 && dy === 0) ? 3 : 1;
          sum += raw[y * W + x] * w; n += w;
        }
      }
      out[cy * W + cx] = sum / n;
    }
  }
  roadField = out;
}

function roadAt(fx, fy) {
  // bilinear sample of the smoothed field, in cell coordinates
  const x = fx - 0.5, y = fy - 0.5;
  const ix = Math.floor(x), iy = Math.floor(y);
  const tx = x - ix, ty = y - iy;
  function at(cx, cy) {
    if (cx < 0 || cy < 0 || cx >= R.W || cy >= R.H) return 0;
    return roadField[cy * R.W + cx];
  }
  return (at(ix, iy) * (1 - tx) + at(ix + 1, iy) * tx) * (1 - ty) +
         (at(ix, iy + 1) * (1 - tx) + at(ix + 1, iy + 1) * tx) * ty;
}

const _rgb = [0, 0, 0];

/** The ground colour for one texel, in cell coordinates. */
function groundTexel(fx, fy, out) {
  const cx = Math.floor(fx), cy = Math.floor(fy);
  const c = ch(cx, cy);

  // large slow variation: dry patches and ploughed fields
  const broad = noise(fx, fy, 11, 3) * 0.65 + noise(fx, fy, 31, 5) * 0.35;
  const field = noise(fx, fy, 23, 9);
  const fine = noise(fx, fy, 2.6, 17) * 0.45 + noise(fx, fy, 0.8, 23) * 0.35 +
               noise(fx, fy, 0.3, 29) * 0.20;

  // Grass dominates; dry ground and ploughed fields are the exception, not
  // the rule, or the whole map reads as desert rather than Normandy.
  const ploughed = field > 0.78;
  let mixB = GRASS_DRY;
  let k = Math.min(0.42, Math.max(0, broad - 0.56) * 1.2);
  if (ploughed) { mixB = FIELD; k = Math.min(0.62, (field - 0.78) * 2.6); }

  for (let i = 0; i < 3; i++) out[i] = GRASS[i] + (mixB[i] - GRASS[i]) * k;

  // Fine grain only. A periodic furrow pattern was tried here and had to go:
  // at any distance its ~1-cell period aliased through the mipmaps into
  // map-wide diagonal stripes.
  const grain = (fine - 0.5) * 21;
  for (let i = 0; i < 3; i++) out[i] += grain;

  /* Roads: dirt with soft, noise-jittered edges and darker ruts. The
   * threshold has to sit well above the blur's spill, or a two-cell lane
   * smears into a ten-cell brown river that wanders with the noise. */
  const road = roadAt(fx, fy) + (fine - 0.5) * 0.08;
  if (road > 0.34) {
    const t = Math.min(1, (road - 0.34) / 0.3);
    // Mottled wear rather than a regular rut pattern: anything periodic here
    // comes out as a ladder of sleepers at map zoom.
    const wear = noise(fx * 2.2, fy * 0.5, 1.4, 37);
    const dirt = wear > 0.62 ? MUD : DIRT;
    for (let i = 0; i < 3; i++) out[i] += (dirt[i] + (fine - 0.5) * 16 - out[i]) * t;
  }

  if (c === '~') {
    const ripple = noise(fx, fy, 2.5, 41);
    for (let i = 0; i < 3; i++) out[i] = WATER[i] + (ripple - 0.5) * 18;
  } else if (c === 'c') {
    // crater: a scorched bowl, darkest in the middle
    const dx = fx - cx - 0.5, dy = fy - cy - 0.5;
    const r = Math.min(1, Math.hypot(dx, dy) / 0.55);
    const t = (1 - r * r) * 0.85;
    for (let i = 0; i < 3; i++) out[i] += (SCORCH[i] - out[i]) * t;
  } else if (c === 'H' || c === 'T') {
    // shaded, leaf-littered ground under the canopy
    for (let i = 0; i < 3; i++) out[i] *= 0.7;
  } else if (c === 'w' || c === 'f') {
    for (let i = 0; i < 3; i++) out[i] *= 0.88;
  }
  return out;
}

/** Paint (or repaint) a rectangle of cells into the ground texture. */
function paintCells(x0, y0, x1, y1) {
  x0 = Math.max(0, x0); y0 = Math.max(0, y0);
  x1 = Math.min(R.W, x1); y1 = Math.min(R.H, y1);
  if (x1 <= x0 || y1 <= y0) return;
  const w = (x1 - x0) * px, h = (y1 - y0) * px;
  const img = g2.createImageData(w, h);
  const data = img.data;
  for (let j = 0; j < h; j++) {
    const fy = y0 + (j + 0.5) / px;
    for (let i = 0; i < w; i++) {
      const fx = x0 + (i + 0.5) / px;
      groundTexel(fx, fy, _rgb);
      const o = (j * w + i) * 4;
      data[o] = _rgb[0] < 0 ? 0 : (_rgb[0] > 255 ? 255 : _rgb[0]);
      data[o + 1] = _rgb[1] < 0 ? 0 : (_rgb[1] > 255 ? 255 : _rgb[1]);
      data[o + 2] = _rgb[2] < 0 ? 0 : (_rgb[2] > 255 ? 255 : _rgb[2]);
      data[o + 3] = 255;
    }
  }
  g2.putImageData(img, x0 * px, y0 * px);
}

/** Repaint the cells touched by a terrain delta, plus a one-cell margin. */
function repaintRegion(cells) {
  if (!canvas || !cells.length) return;
  buildRoadField();
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  cells.forEach(function (c) {
    x0 = Math.min(x0, c[0] - 1); y0 = Math.min(y0, c[1] - 1);
    x1 = Math.max(x1, c[0] + 2); y1 = Math.max(y1, c[1] + 2);
  });
  paintCells(x0, y0, x1, y1);
  tex.needsUpdate = true;
}

// ------------------------------------------------------------ overlay maps --

function overlayCanvas(perCell) {
  const c = document.createElement('canvas');
  c.width = R.W * perCell; c.height = R.H * perCell;
  return c;
}

function planeFor(canvasEl, y, opts) {
  const t = new THREE.CanvasTexture(canvasEl);
  t.colorSpace = THREE.SRGBColorSpace;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = opts && opts.nearest ? THREE.NearestFilter : THREE.LinearFilter;
  t.anisotropy = 4;
  const geo = new THREE.PlaneGeometry(R.W * R.CELL, R.H * R.CELL);
  const mat = new THREE.MeshBasicMaterial({
    map: t, transparent: true, depthWrite: false, opacity: (opts && opts.opacity) || 1,
    fog: !(opts && opts.noFog)
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(R.W * R.CELL / 2, y, R.H * R.CELL / 2);
  mesh.renderOrder = Math.round(y * 100);
  mesh.matrixAutoUpdate = false;
  mesh.updateMatrix();
  return mesh;
}

function paintTint(frame) {
  const owners = {};
  frame.points.forEach(function (p) { owners[p.id] = p.owner; });
  const per = 4;
  const g = tint.userData.ctx;
  const W = R.W, H = R.H;
  g.clearRect(0, 0, W * per, H * per);

  // ownership wash
  const img = g.createImageData(W, H);
  const data = img.data;
  const rgb = {};
  for (let i = 0; i < W * H; i++) {
    const pid = R.sectorPoint[R.sectorCells[i]];
    const owner = pid === undefined ? null : owners[pid];
    if (owner === null || owner === undefined) continue;
    if (!rgb[owner]) rgb[owner] = parseColour(teamColour(owner));
    data[i * 4] = rgb[owner][0]; data[i * 4 + 1] = rgb[owner][1];
    data[i * 4 + 2] = rgb[owner][2]; data[i * 4 + 3] = 58;
  }
  const small = document.createElement('canvas');
  small.width = W; small.height = H;
  small.getContext('2d').putImageData(img, 0, 0);
  g.drawImage(small, 0, 0, W * per, H * per);

  // sector borders in the owner's colour, drawn just inside each sector
  function ownerColour(sector) {
    const pid = R.sectorPoint[sector];
    const owner = pid === undefined ? null : owners[pid];
    if (owner === null || owner === undefined) return '#b3ae9e';
    // pulled well toward bone: a saturated line at ground level looks like neon
    return mix(teamColour(owner), '#d6d2c4', 0.45);
  }
  const inset = per * 0.35;
  const paths = {};
  function seg(colour, x1, y1, x2, y2) {
    (paths[colour] || (paths[colour] = [])).push([x1, y1, x2, y2]);
  }
  for (let cy = 0; cy < H; cy++) {
    for (let cx = 0; cx < W; cx++) {
      const s = R.sectorCells[cy * W + cx];
      const col = ownerColour(s);
      const x = cx * per, y = cy * per;
      if (cx + 1 >= W || R.sectorCells[cy * W + cx + 1] !== s) seg(col, x + per - inset, y, x + per - inset, y + per);
      if (cx - 1 < 0 || R.sectorCells[cy * W + cx - 1] !== s) seg(col, x + inset, y, x + inset, y + per);
      if (cy + 1 >= H || R.sectorCells[(cy + 1) * W + cx] !== s) seg(col, x, y + per - inset, x + per, y + per - inset);
      if (cy - 1 < 0 || R.sectorCells[(cy - 1) * W + cx] !== s) seg(col, x, y + inset, x + per, y + inset);
    }
  }
  g.globalAlpha = 0.38;
  g.lineWidth = Math.max(1.3, per * 0.3);
  g.lineCap = 'square';
  for (const colour in paths) {
    g.strokeStyle = colour;
    g.beginPath();
    paths[colour].forEach(function (p) { g.moveTo(p[0], p[1]); g.lineTo(p[2], p[3]); });
    g.stroke();
  }
  g.globalAlpha = 1;
  tint.material.map.needsUpdate = true;
}

function paintCover() {
  const per = 4;
  const g = cover.userData.ctx;
  g.clearRect(0, 0, R.W * per, R.H * per);
  [2, 1].forEach(function (level) {
    g.fillStyle = level === 2 ? 'rgba(90,200,110,0.42)' : 'rgba(215,190,80,0.38)';
    g.beginPath();
    for (let i = 0; i < R.W * R.H; i++) {
      if ((COVER[R.terr[i]] || 0) !== level) continue;
      const x = (i % R.W) * per + per / 2, y = ((i / R.W) | 0) * per + per / 2;
      g.moveTo(x + per * 0.34, y);
      g.arc(x, y, per * 0.34, 0, 6.2832);
    }
    g.fill();
  });
  cover.material.map.needsUpdate = true;
  coverBuilt = true;
}

function paintFog(bits) {
  const g = fogPlane.userData.ctx;
  const img = g.createImageData(R.W, R.H);
  const data = img.data;
  for (let i = 0; i < R.W * R.H; i++) {
    const seen = (bits[i >> 3] & (128 >> (i & 7))) !== 0;
    if (seen) continue;
    // a cold, desaturated wash rather than pure black: unseen ground should
    // read as "not observed", not as "night"
    data[i * 4] = 30; data[i * 4 + 1] = 35; data[i * 4 + 2] = 42; data[i * 4 + 3] = 150;
  }
  g.putImageData(img, 0, 0);
  fogPlane.material.map.needsUpdate = true;
  fogBits = bits;
}

// ------------------------------------------------------ vertical features --

/** Rebuild every instanced prop from the current terrain characters. */
function buildFeatures(scene) {
  if (features) {
    scene.remove(features);
    features.traverse(function (o) { if (o.geometry) o.geometry.dispose(); });
  }
  features = new THREE.Group();
  features.name = 'features';

  const W = R.W, H = R.H, CELL = R.CELL;
  const by = { H: [], w: [], f: [], T: [], c: [] };
  for (let i = 0; i < W * H; i++) {
    const c = R.terr[i];
    if (by[c]) by[c].push(i);
  }

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const v = new THREE.Vector3();
  const sc = new THREE.Vector3();
  const col = new THREE.Color();

  function place(inst, n, x, y, z, ry, sx, sy, sz, colour) {
    e.set(0, ry, 0); q.setFromEuler(e);
    v.set(x, y, z); sc.set(sx, sy, sz);
    m.compose(v, q, sc);
    inst.setMatrixAt(n, m);
    if (colour) inst.setColorAt(n, col.setStyle(colour));
  }

  function makeInstanced(geometry, count, material) {
    const inst = new THREE.InstancedMesh(geometry, material, count);
    inst.castShadow = true;
    inst.receiveShadow = true;
    inst.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    return inst;
  }

  function centre(i) { return [(i % W + 0.5) * CELL, ((i / W | 0) + 0.5) * CELL]; }
  function cellOf(i) { return [i % W, (i / W) | 0]; }

  // Hedgerows: lumpy earth banks, 2.5 m of them, which really do block a view.
  if (by.H.length) {
    const inst = makeInstanced(hedgeGeometry(CELL), by.H.length, materials.feature);
    by.H.forEach(function (i, n) {
      const p = centre(i), c = cellOf(i);
      const a = hash2(c[0], c[1]), b = hash3(c[0], c[1], 3);
      place(inst, n, p[0], 1.25 + a * 0.25, p[1], a * 6.283,
            1, 0.85 + b * 0.35, 1,
            '#' + hedgeColour(a, b));
    });
    inst.instanceColor.needsUpdate = true;
    features.add(inst);
  }

  // Low stone walls, aligned with the run of wall cells.
  if (by.w.length) {
    const inst = makeInstanced(box(CELL * 1.02, 1.0, 0.62), by.w.length, materials.feature);
    by.w.forEach(function (i, n) {
      const p = centre(i), c = cellOf(i);
      const a = hash2(c[0], c[1]);
      place(inst, n, p[0], 0.5, p[1], runAngle(c[0], c[1], 'w'), 1, 0.85 + a * 0.3, 1,
            a > 0.5 ? '#9a9486' : '#8b857a');
    });
    inst.instanceColor.needsUpdate = true;
    features.add(inst);
  }

  // Fences and low hedges: thin, crushable, and removed by a terrain delta.
  if (by.f.length) {
    const inst = makeInstanced(fenceGeometry(CELL), by.f.length, materials.feature);
    by.f.forEach(function (i, n) {
      const p = centre(i), c = cellOf(i);
      const a = hash2(c[0], c[1]);
      place(inst, n, p[0], 0, p[1], runAngle(c[0], c[1], 'f'), 1, 0.9 + a * 0.25, 1,
            a > 0.5 ? '#7a6244' : '#6a5238');
    });
    inst.instanceColor.needsUpdate = true;
    features.add(inst);
  }

  // Tree clusters: one trunk and two or three blobby canopies per cell,
  // scaled and turned by a hash so no two look alike.
  if (by.T.length) {
    const trunks = makeInstanced(cyl(0.16, 0.26, 3.0, 6), by.T.length, materials.feature);
    let canopies = 0;
    by.T.forEach(function (i) { canopies += hash2(i % W, (i / W) | 0) > 0.45 ? 3 : 2; });
    const leaves = makeInstanced(sphere(1.35, 0), canopies, materials.feature);
    let k = 0;
    by.T.forEach(function (i, n) {
      const p = centre(i), c = cellOf(i);
      const a = hash2(c[0], c[1]);
      const lean = (hash3(c[0], c[1], 13) - 0.5) * 0.12;
      place(trunks, n, p[0], 1.5, p[1], a * 6.283, 1, 0.85 + a * 0.4, 1, '#4a3a28');
      const count = a > 0.45 ? 3 : 2;
      for (let j = 0; j < count; j++) {
        const b = hash3(c[0], c[1], 20 + j), d = hash3(c[0], c[1], 30 + j);
        place(leaves, k++,
              p[0] + (b - 0.5) * 1.5 + lean * 4, 2.9 + d * 1.5, p[1] + (d - 0.5) * 1.5,
              b * 6.283, 0.7 + b * 0.55, 0.62 + d * 0.5, 0.7 + d * 0.5,
              '#' + canopyColour(b, d));
      }
    });
    trunks.instanceColor.needsUpdate = true;
    leaves.instanceColor.needsUpdate = true;
    features.add(trunks);
    features.add(leaves);
  }

  // Craters: the bowl is painted into the ground; this is the thrown-up rim.
  if (by.c.length) {
    const inst = makeInstanced(craterRim(CELL), by.c.length, materials.feature);
    by.c.forEach(function (i, n) {
      const p = centre(i), c = cellOf(i);
      const a = hash2(c[0], c[1]);
      place(inst, n, p[0], 0, p[1], a * 6.283, 1, 0.7 + a * 0.6, 1, '#4a3f30');
    });
    inst.instanceColor.needsUpdate = true;
    features.add(inst);
  }

  scene.add(features);
}

function hedgeColour(a, b) {
  return hex(Math.round(64 + a * 26), Math.round(102 + b * 30), Math.round(52 + a * 18));
}

function canopyColour(a, b) {
  return hex(Math.round(70 + a * 30), Math.round(112 + b * 34), Math.round(56 + a * 20));
}

function hex(r, g, b) {
  return ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/** Which way a run of like cells goes, so walls and fences line up with it. */
function runAngle(cx, cy, want) {
  const horiz = ch(cx - 1, cy) === want || ch(cx + 1, cy) === want;
  const vert = ch(cx, cy - 1) === want || ch(cx, cy + 1) === want;
  if (horiz && !vert) return 0;
  if (vert && !horiz) return Math.PI / 2;
  return hash2(cx, cy) > 0.5 ? 0 : Math.PI / 2;
}

let _hedge = null;
/** An earth bank with a crown of scrub: 2.5 m of very solid Normandy bocage. */
function hedgeGeometry(cell) {
  if (_hedge) return _hedge;
  const p = new Parts();
  p.add(lumpyBox(cell * 1.18, 1.5, cell * 1.05, 5), '#ffffff', { y: 0.75 });
  for (let i = 0; i < 4; i++) {
    const a = hash3(i, 3, 9), b = hash3(i, 7, 11);
    p.sphere(0.95, '#ffffff', {
      x: (i / 3 - 0.5) * cell * 0.95, y: 1.75 + a * 0.45, z: (b - 0.5) * cell * 0.4,
      sx: 0.95 + a * 0.4, sy: 0.7 + b * 0.35, sz: 0.85 + b * 0.35, ry: a * 6.283
    });
  }
  _hedge = bake(p);
  return _hedge;
}

let _fence = null;
function fenceGeometry(cell) {
  if (_fence) return _fence;
  const p = new Parts();
  for (let i = 0; i < 3; i++) {
    p.box(0.1, 0.95, 0.1, '#ffffff', { x: -cell / 2 + cell * i / 2, y: 0.48 });
  }
  p.box(cell, 0.08, 0.08, '#ffffff', { y: 0.82 });
  p.box(cell, 0.08, 0.08, '#ffffff', { y: 0.45 });
  _fence = bake(p);
  return _fence;
}

let _rim = null;
function craterRim(cell) {
  if (_rim) return _rim;
  const p = new Parts();
  for (let i = 0; i < 7; i++) {
    const a = i / 7 * 6.283;
    const r = cell * 0.46;
    p.sphere(0.34, '#ffffff', {
      x: Math.cos(a) * r, y: 0.06, z: Math.sin(a) * r,
      sx: 1.3, sy: 0.55, sz: 1.3, ry: a
    });
  }
  _rim = bake(p);
  return _rim;
}

// ------------------------------------------------------------------ build --

export function build(scene, mats) {
  materials = mats;
  px = S.quality === 'low' ? 4 : 8;

  buildRoadField();
  canvas = document.createElement('canvas');
  canvas.width = R.W * px; canvas.height = R.H * px;
  g2 = canvas.getContext('2d');
  paintCells(0, 0, R.W, R.H);

  tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = S.quality === 'low' ? 1 : 8;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;

  const geo = new THREE.PlaneGeometry(R.W * R.CELL, R.H * R.CELL, 1, 1);
  ground = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    map: tex, roughness: 0.96, metalness: 0
  }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(R.W * R.CELL / 2, 0, R.H * R.CELL / 2);
  ground.receiveShadow = true;
  ground.matrixAutoUpdate = false;
  ground.updateMatrix();
  scene.add(ground);

  /* The map is a hard-edged rectangle; without something beyond it the
   * overview zoom shows the battlefield floating in the page background.
   * This is a big, plain apron in the average field colour, sunk a hair so
   * it never z-fights with the map itself. */
  const apron = new THREE.Mesh(
    new THREE.PlaneGeometry(R.W * R.CELL * 9, R.H * R.CELL * 9),
    new THREE.MeshStandardMaterial({ color: 0x6d7a4e, roughness: 1 })
  );
  apron.rotation.x = -Math.PI / 2;
  apron.position.set(R.W * R.CELL / 2, -0.06, R.H * R.CELL / 2);
  apron.receiveShadow = false;
  apron.matrixAutoUpdate = false;
  apron.updateMatrix();
  scene.add(apron);
  surround = apron;

  const tintCanvas = overlayCanvas(4);
  tint = planeFor(tintCanvas, 0.04, { opacity: 0.55 });
  tint.userData.ctx = tintCanvas.getContext('2d');
  scene.add(tint);

  const coverCanvas = overlayCanvas(4);
  cover = planeFor(coverCanvas, 0.07, { opacity: 1 });
  cover.userData.ctx = coverCanvas.getContext('2d');
  cover.visible = false;
  scene.add(cover);

  const fogCanvas = overlayCanvas(1);
  fogPlane = planeFor(fogCanvas, 0.5, { opacity: 1, noFog: true });
  fogPlane.userData.ctx = fogCanvas.getContext('2d');
  fogPlane.visible = false;
  scene.add(fogPlane);

  buildFeatures(scene);
}

/** Per-frame update: overlays, fog wash, and a rebuild when terrain changed. */
export function update(scene, frame, fogCtx, terrVersion) {
  if (terrain_version !== terrVersion) {
    if (terrain_version !== -1) {
      // a fence was crushed or a crater was blown: repaint and re-grow
      repaintRegion(changedCells());
      buildFeatures(scene);
      coverBuilt = false;
    }
    terrain_version = terrVersion;
    snapshotTerrain();
  }

  const key = frame.points.map(function (p) { return p.owner === null ? '-' : p.owner; }).join('');
  if (tintKey !== key) { tintKey = key; paintTint(frame); }
  tint.visible = S.showSectors;

  if (S.showCover && !coverBuilt) paintCover();
  cover.visible = S.showCover;

  if (fogCtx) {
    if (fogBits !== fogCtx.bits) paintFog(fogCtx.bits);
    fogPlane.visible = true;
  } else {
    fogPlane.visible = false;
  }
}

/* The terrain character array is mutated in place by `data.js`, so to know
 * *which* cells a delta touched we keep our own copy of what we last painted
 * and diff against it. The diff is only ever taken when `terrVersion` moved,
 * which is at most a handful of times per match. */
let terrain_version = -1;
let painted = null;

function snapshotTerrain() { painted = R.terr.slice(); }

function changedCells() {
  const out = [];
  if (!painted || painted.length !== R.terr.length) return out;
  for (let i = 0; i < R.terr.length; i++) {
    if (painted[i] !== R.terr[i]) out.push([i % R.W, (i / R.W) | 0]);
  }
  return out;
}

export function featureCount() {
  let n = 0;
  if (features) features.children.forEach(function (c) { n += c.count || 0; });
  return n;
}

export function reset() { terrain_version = -1; painted = null; tintKey = null; fogBits = null; }
