/* Small geometry toolkit for the procedural models.
 *
 * Everything a model factory builds is a *part list* — geometry, a transform
 * and a colour — which `bake()` merges into one vertex-coloured
 * BufferGeometry. That keeps a whole farmhouse or a whole tank to a single
 * draw call, which is what makes a 96x96 map with sixty squads cheap enough
 * for integrated graphics.
 *
 * World mapping: the sim is planar with x right and y *down*; here that is
 * x right, z down-screen and y up. A model's local forward is +X, so a sim
 * heading `h` becomes `rotation.y = -h`.
 */

import * as THREE from 'three';
import { hash3 } from '../util.js';

const geoCache = new Map();

function cached(key, make) {
  let g = geoCache.get(key);
  if (!g) { g = make(); geoCache.set(key, g); }
  return g;
}

export function box(w, h, d) {
  return cached('b' + w + ',' + h + ',' + d, function () { return new THREE.BoxGeometry(w, h, d); });
}

export function cyl(rTop, rBottom, h, seg) {
  seg = seg || 8;
  return cached('c' + rTop + ',' + rBottom + ',' + h + ',' + seg, function () {
    return new THREE.CylinderGeometry(rTop, rBottom, h, seg);
  });
}

export function sphere(r, detail) {
  return cached('s' + r + ',' + (detail || 0), function () {
    return new THREE.IcosahedronGeometry(r, detail || 0);
  });
}

/** A gable roof: a triangular prism `w` x `d` at the base, `h` to the ridge. */
export function gable(w, h, d) {
  return cached('g' + w + ',' + h + ',' + d, function () {
    const hw = w / 2, hd = d / 2;
    // ridge runs along +X
    const v = [
      [-hw, 0, -hd], [hw, 0, -hd], [hw, 0, hd], [-hw, 0, hd],   // eaves
      [-hw, h, 0], [hw, h, 0]                                    // ridge
    ];
    // Wound counter-clockwise seen from *outside*: get this backwards and
    // every normal points into the roof, which renders as an unlit black slab.
    const tris = [
      [5, 1, 0], [4, 5, 0],       // -z slope
      [4, 3, 2], [5, 4, 2],       // +z slope
      [3, 4, 0],                  // -x gable end
      [5, 2, 1]                   // +x gable end
    ];
    const pos = new Float32Array(tris.length * 9);
    tris.forEach(function (t, i) {
      for (let k = 0; k < 3; k++) {
        pos[i * 9 + k * 3] = v[t[k]][0];
        pos[i * 9 + k * 3 + 1] = v[t[k]][1];
        pos[i * 9 + k * 3 + 2] = v[t[k]][2];
      }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.computeVertexNormals();
    return g;
  });
}

/** A box with its vertices pushed around by a hash — for hedgerow banks. */
export function lumpyBox(w, h, d, salt) {
  return cached('l' + w + ',' + h + ',' + d + ',' + salt, function () {
    const g = new THREE.BoxGeometry(w, h, d, 3, 2, 3);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      const n = hash3(Math.round(x * 97), Math.round(z * 97), salt + Math.round(y * 31));
      const k = (n - 0.5);
      // the top lumps most, the base stays planted on the ground
      const up = (y + h / 2) / h;
      p.setXYZ(i, x + k * 0.5 * w * 0.35, y + k * 0.5 * h * up, z + k * 0.5 * d * 0.35);
    }
    g.computeVertexNormals();
    return g;
  });
}

const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _v = new THREE.Vector3();
const _s = new THREE.Vector3();

/** A part list under construction. `add` takes a geometry plus a transform. */
export class Parts {
  constructor() { this.list = []; }

  /** `t`: {x,y,z, rx,ry,rz, sx,sy,sz} — all optional. */
  add(geometry, colour, t) {
    t = t || {};
    _e.set(t.rx || 0, t.ry || 0, t.rz || 0);
    _q.setFromEuler(_e);
    _v.set(t.x || 0, t.y || 0, t.z || 0);
    _s.set(t.sx === undefined ? 1 : t.sx, t.sy === undefined ? 1 : t.sy, t.sz === undefined ? 1 : t.sz);
    this.list.push({ g: geometry, m: new THREE.Matrix4().compose(_v, _q, _s), c: toColour(colour) });
    return this;
  }

  box(w, h, d, colour, t) { return this.add(box(w, h, d), colour, t); }
  cyl(rt, rb, h, colour, t, seg) { return this.add(cyl(rt, rb, h, seg), colour, t); }
  sphere(r, colour, t, detail) { return this.add(sphere(r, detail), colour, t); }
  gable(w, h, d, colour, t) { return this.add(gable(w, h, d), colour, t); }

  get length() { return this.list.length; }
}

const _colourCache = new Map();

function toColour(c) {
  if (c && c.isColor) return c;
  let out = _colourCache.get(c);
  if (!out) { out = new THREE.Color().setStyle(String(c)); _colourCache.set(c, out); }
  return out;
}

/** Merge a part list into one vertex-coloured geometry. */
export function bake(parts) {
  const list = parts.list || parts;
  let total = 0;
  const prepared = [];
  for (let i = 0; i < list.length; i++) {
    const p = list[i];
    let g = p.g.index ? p.g.toNonIndexed() : p.g.clone();
    g.applyMatrix4(p.m);
    if (!g.attributes.normal) g.computeVertexNormals();
    total += g.attributes.position.count;
    prepared.push({ g: g, c: p.c });
  }
  const pos = new Float32Array(total * 3);
  const nor = new Float32Array(total * 3);
  const col = new Float32Array(total * 3);
  let o = 0;
  for (let i = 0; i < prepared.length; i++) {
    const g = prepared[i].g, c = prepared[i].c;
    const n = g.attributes.position.count;
    pos.set(g.attributes.position.array, o * 3);
    nor.set(g.attributes.normal.array, o * 3);
    for (let k = 0; k < n; k++) {
      col[(o + k) * 3] = c.r; col[(o + k) * 3 + 1] = c.g; col[(o + k) * 3 + 2] = c.b;
    }
    o += n;
    g.dispose();
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  out.setAttribute('color', new THREE.BufferAttribute(col, 3));
  out.computeBoundingSphere();
  return out;
}
