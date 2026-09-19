/* The model registry and the procedural default skin.
 *
 * ## Skin hook
 *
 *     import * as models from './js/view3d/models.js';
 *     models.register('building', /^hq/, function (ctx) { ... });
 *
 * `register(kind, pattern, factory)` adds an alternative look without
 * touching the renderer. `kind` is one of:
 *
 *   'building'  a player structure or a neutral house
 *   'vehicle'   a tank, halftrack or armoured car
 *   'weapon'    a crewed team weapon (HMG, mortar, AT gun)
 *   'soldier'   the single infantry figure, instanced per living member
 *   'point'     a capture point's flagpole
 *
 * `pattern` matches the def id (or, for points, the point type): a string is
 * an exact match, a RegExp is tested against it, a function gets the whole
 * context object. The *last* matching registration wins, so a pack loaded
 * later overrides the built-in look.
 *
 * `factory(ctx)` returns a `Parts` list (see `geom.js`), which the renderer
 * bakes into one vertex-coloured mesh — one draw call per object. `ctx`
 * carries at least `{ def, colour, hash }` plus kind-specific fields
 * documented at each default factory below.
 *
 * Only the procedural default skin ships with the repo. No third-party art.
 */

import * as THREE from 'three';
import { Parts, bake, box, cyl } from './geom.js';
import { hash3, mix, shade } from '../util.js';

// ------------------------------------------------------------- registry --

const registry = { building: [], vehicle: [], weapon: [], soldier: [], point: [] };

export function register(kind, pattern, factory) {
  if (!registry[kind]) throw new Error('unknown model kind: ' + kind);
  registry[kind].push({ pattern: pattern, factory: factory });
}

function matches(pattern, ctx) {
  if (typeof pattern === 'function') return !!pattern(ctx);
  if (pattern instanceof RegExp) return pattern.test(ctx.def || '');
  return pattern === (ctx.def || '');
}

/** The factory for `ctx`, latest registration first, else the built-in default. */
function factoryFor(kind, ctx) {
  const list = registry[kind];
  for (let i = list.length - 1; i >= 0; i--) {
    if (matches(list[i].pattern, ctx)) return list[i].factory;
  }
  return DEFAULTS[kind];
}

/** Build a baked, vertex-coloured mesh for `ctx`. */
export function build(kind, ctx, material) {
  const parts = factoryFor(kind, ctx)(ctx);
  if (parts && parts.isObject3D) return parts;       // a factory may return a node
  const mesh = new THREE.Mesh(bake(parts), material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** Just the baked geometry — for the instanced soldier, which shares one mesh. */
export function buildGeometry(kind, ctx) {
  return bake(factoryFor(kind, ctx)(ctx));
}

// ---------------------------------------------------- shared muted palette --

const KHAKI = '#7d7a58';
const OLIVE = '#59603a';
const STEEL = '#4b5148';
const RUST = '#5a4632';
const TIMBER = '#6b563c';
const PLASTER = '#a9a08b';
const SLATE = '#5d5a55';
const CHARCOAL = '#25231f';

/** Faction colour, pulled well down so the field reads as WW2 rather than neon. */
function warpaint(colour) { return mix(colour, '#4c4c46', 0.78); }

/* The uniform is muted, but a squad has to be *identifiable* from the default
 * camera height, so the shoulders and a helmet band carry a deep, saturated
 * version of the team colour. Deep, not bright: lightening a saturated hue
 * gives pastel toy soldiers, while darkening it keeps the contrast against
 * grass without leaving the WW2 palette. */
function accent(colour) { return mix(colour, '#2b2b26', 0.34); }

/** The steel itself: US olive drab, Wehrmacht field grey, faintly team-tinted. */
function helmetSteel(colour, faction) {
  return mix(colour, faction === 'wehr' ? '#585b52' : '#5c6040', 0.7);
}

// ------------------------------------------------------------- buildings --

/* `ctx` for a building:
 *   { def, colour, hash, wm, dm, neutral, family, progress, damaged, ruined } */

/** Which structure family a def id belongs to, inferred from the id. */
export function buildingFamily(defId) {
  const id = String(defId || '').toLowerCase();
  if (/hq|headquarter/.test(id)) return 'hq';
  if (/barrack|quarter|kaserne|infantry/.test(id)) return 'barracks';
  if (/motor|depot|panzer|tank|vehicle|werk/.test(id)) return 'motorpool';
  if (/supply|resource|cache|fuel|munition/.test(id)) return 'supply';
  if (/observ|\bop\b|op_|_op|tower|post/.test(id)) return 'op';
  if (/barn|farm|shed|stable/.test(id)) return 'barn';
  if (/church|chapel/.test(id)) return 'church';
  return 'house';
}

function neutralHouse(ctx) {
  const p = new Parts();
  const w = ctx.wm, d = ctx.dm;
  const hash = ctx.hash;
  const tallBarn = ctx.family === 'barn';
  const wallH = tallBarn ? 4.2 : 3.2;
  // slight colour variety by cell hash, but all within plaster/timber
  const wallColour = tallBarn ? mix(TIMBER, RUST, hash * 0.6)
                              : shade(PLASTER, (hash - 0.5) * 0.22);
  const roofColour = shade(mix('#7a3b2c', SLATE, hash3(ctx.cx | 0, ctx.cy | 0, 7)), (hash - 0.5) * 0.18);

  p.box(w * 0.94, wallH, d * 0.94, wallColour, { y: wallH / 2 });
  // stone plinth
  p.box(w * 0.98, 0.45, d * 0.98, shade(SLATE, -0.1), { y: 0.22 });
  // pitched roof, ridge along the longer side
  const alongX = w >= d;
  const ridgeH = tallBarn ? 3.1 : 2.4;
  p.gable(alongX ? w * 1.06 : d * 1.06, ridgeH, alongX ? d * 1.06 : w * 1.06, roofColour,
          { y: wallH, ry: alongX ? 0 : Math.PI / 2 });
  // chimney, off to one side
  const cx = (hash - 0.5) * w * 0.5;
  p.box(0.55, 1.5, 0.55, shade('#6d4a3d', (hash - 0.5) * 0.2),
        { x: cx, y: wallH + ridgeH * 0.55, z: (hash3(ctx.cx | 0, ctx.cy | 0, 11) - 0.5) * d * 0.4 });
  // a door and two windows as recessed dark panels, so the walls are not blank
  const doorZ = d * 0.48;
  p.box(0.9, 1.9, 0.12, CHARCOAL, { y: 0.95, z: doorZ });
  p.box(0.7, 0.7, 0.12, CHARCOAL, { x: -w * 0.28, y: 2.1, z: doorZ });
  p.box(0.7, 0.7, 0.12, CHARCOAL, { x: w * 0.28, y: 2.1, z: doorZ });
  return p;
}

function playerStructure(ctx) {
  const p = new Parts();
  const w = ctx.wm, d = ctx.dm;
  const body = warpaint(ctx.colour);
  const trim = shade(body, -0.25);
  const fam = ctx.family;

  if (fam === 'op') {
    // a small flagged tower that sits on the point it watches
    p.box(1.6, 0.4, 1.6, SLATE, { y: 0.2 });
    p.cyl(0.35, 0.5, 4.2, TIMBER, { y: 2.3 }, 6);
    p.box(1.9, 1.0, 1.9, body, { y: 4.7 });
    p.box(2.2, 0.22, 2.2, trim, { y: 5.25 });
    p.cyl(0.06, 0.06, 2.4, STEEL, { y: 6.4 }, 5);
    p.box(1.1, 0.7, 0.05, ctx.colour, { x: 0.55, y: 7.1 });
    return p;
  }

  const wallH = fam === 'hq' ? 4.0 : 3.2;
  p.box(w * 0.92, wallH, d * 0.92, body, { y: wallH / 2 });
  p.box(w * 0.98, 0.4, d * 0.98, SLATE, { y: 0.2 });

  if (fam === 'hq') {
    // a blockhouse with a stepped tower and a flag
    p.box(w * 0.55, 1.6, d * 0.55, shade(body, 0.08), { y: wallH + 0.8 });
    p.box(w * 0.6, 0.2, d * 0.6, trim, { y: wallH + 1.7 });
    p.cyl(0.07, 0.07, 3.0, STEEL, { y: wallH + 3.2 }, 5);
    p.box(1.4, 0.85, 0.06, ctx.colour, { x: 0.7, y: wallH + 4.2 });
    p.box(w * 1.0, 0.35, d * 1.0, trim, { y: wallH });
  } else if (fam === 'barracks') {
    // a long hut with a curved roof and a sandbag line
    p.add(cyl(d * 0.5, d * 0.5, w * 0.95, 10), shade(body, 0.1),
          { y: wallH, rz: Math.PI / 2, sy: 1, sx: 0.55 });
    p.box(w * 0.9, 0.55, 0.7, KHAKI, { y: 0.28, z: d * 0.46 });
    p.box(0.95, 2.0, 0.12, CHARCOAL, { y: 1.0, z: d * 0.47 });
  } else if (fam === 'motorpool') {
    // an open-fronted vehicle hall: two bays and a gantry
    p.box(w * 0.92, 1.2, d * 0.92, shade(body, -0.12), { y: wallH + 0.6 });
    p.box(w * 0.3, 2.6, 0.2, CHARCOAL, { x: -w * 0.24, y: 1.3, z: d * 0.46 });
    p.box(w * 0.3, 2.6, 0.2, CHARCOAL, { x: w * 0.24, y: 1.3, z: d * 0.46 });
    p.box(0.3, 0.3, d * 1.1, STEEL, { y: wallH + 1.5 });
    p.cyl(0.5, 0.5, 0.35, RUST, { x: w * 0.3, y: 0.18, z: -d * 0.3, rx: Math.PI / 2 }, 8);
  } else if (fam === 'supply') {
    // stacked crates and a fuel drum, low and wide
    p.box(w * 0.5, 1.0, d * 0.5, TIMBER, { x: -w * 0.18, y: wallH + 0.5 });
    p.box(w * 0.35, 0.8, d * 0.35, shade(TIMBER, -0.15), { x: w * 0.22, y: wallH + 0.4, z: d * 0.1 });
    p.cyl(0.55, 0.55, 1.2, OLIVE, { x: w * 0.25, y: wallH + 0.6, z: -d * 0.22 }, 8);
    p.box(w * 1.0, 0.3, d * 1.0, trim, { y: wallH });
  } else {
    p.gable(w * 1.02, 1.6, d * 1.02, trim, { y: wallH });
  }
  return p;
}

function DEFAULT_BUILDING(ctx) {
  if (ctx.ruined) return rubble(ctx);
  const parts = ctx.neutral ? neutralHouse(ctx) : playerStructure(ctx);
  if (ctx.progress < 1) return scaffold(ctx, parts);
  return parts;
}

/** Under construction: the real shape, cut down to its progress, in a frame. */
function scaffold(ctx, parts) {
  const p = new Parts();
  const w = ctx.wm, d = ctx.dm;
  const grow = Math.max(0.08, ctx.progress);
  // foundation slab, always there
  p.box(w, 0.3, d, shade(SLATE, -0.2), { y: 0.15 });
  // the structure so far, squashed down to `grow` of its height
  parts.list.forEach(function (part) {
    p.list.push({ g: part.g, m: part.m.clone().premultiply(scaleY(grow)), c: part.c });
  });
  // scaffold poles and planks around it
  const hw = w * 0.52, hd = d * 0.52, poleH = 1.2 + 3.2 * grow;
  [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]].forEach(function (c) {
    p.cyl(0.07, 0.07, poleH, TIMBER, { x: c[0], y: poleH / 2, z: c[1] }, 5);
  });
  p.box(w * 1.06, 0.1, 0.22, TIMBER, { y: poleH * 0.75, z: -hd });
  p.box(w * 1.06, 0.1, 0.22, TIMBER, { y: poleH * 0.75, z: hd });
  return p;
}

const _scaleY = new THREE.Matrix4();
function scaleY(k) { return _scaleY.makeScale(1, k, 1); }

/** Destroyed: a low heap of broken masonry, kept as a permanent scar. */
function rubble(ctx) {
  const p = new Parts();
  const w = ctx.wm, d = ctx.dm;
  p.box(w, 0.22, d, '#4a4038', { y: 0.11 });
  for (let i = 0; i < 9; i++) {
    const a = hash3(ctx.cx | 0, ctx.cy | 0, i);
    const b = hash3(ctx.cx | 0, ctx.cy | 0, i + 40);
    const s = 0.35 + a * 0.7;
    p.box(s, s * 0.6, s, shade(mix(PLASTER, CHARCOAL, 0.55 + a * 0.3), (b - 0.5) * 0.25), {
      x: (a - 0.5) * w * 0.8, y: s * 0.3 + b * 0.2, z: (b - 0.5) * d * 0.8,
      ry: a * 6.28, rz: (b - 0.5) * 0.5
    });
  }
  return p;
}

// --------------------------------------------------------------- soldier --

/* `ctx` for a soldier: `{ colour, faction }`. One baked geometry is shared by
 * every figure on the field and drawn with an InstancedMesh, so the whole
 * infantry of both sides costs one draw call per faction colour. The figure
 * stands on the origin, 1.8 m tall, facing +X. */
function DEFAULT_SOLDIER(ctx) {
  const p = new Parts();
  const coat = warpaint(ctx.colour);
  const flash = accent(ctx.colour);
  const steel = helmetSteel(ctx.colour, ctx.faction);
  p.box(0.36, 0.55, 0.28, shade(coat, -0.26), { y: 0.3 });                // legs
  p.box(0.46, 0.64, 0.34, coat, { y: 0.93 });                             // torso
  // team-coloured shoulders: the widest thing on the figure seen from above
  p.box(0.62, 0.15, 0.36, flash, { y: 1.19 });
  p.box(0.13, 0.5, 0.13, shade(coat, -0.12), { x: 0.16, y: 0.95, rz: -0.35 });  // arms
  p.box(0.13, 0.5, 0.13, shade(coat, -0.12), { x: -0.07, y: 0.95, rz: 0.2 });
  p.sphere(0.13, '#9a7a5e', { y: 1.36 });                                 // head
  p.add(cyl(0.21, 0.24, 0.17, 8), steel, { y: 1.47 });                    // helmet
  p.add(cyl(0.3, 0.3, 0.05, 8), shade(steel, -0.18), { y: 1.39 });        // its brim
  p.box(0.26, 0.09, 0.26, flash, { y: 1.55 });                            // team band
  p.box(0.84, 0.06, 0.06, '#33261b', { x: 0.3, y: 1.02, rz: -0.12 });     // rifle
  p.box(0.24, 0.1, 0.06, '#4a3826', { x: 0.02, y: 1.05, rz: -0.12 });
  p.box(0.32, 0.36, 0.18, shade(OLIVE, -0.1), { x: -0.26, y: 0.99 });     // pack
  return p;
}

// ---------------------------------------------------------- team weapons --

/* `ctx` for a team weapon: `{ def, colour, weapon }` where `weapon` is
 * 'hmg' | 'mortar' | 'at' | 'gun'. The gun points along +X. */
export function weaponKind(defId) {
  const id = String(defId || '').toLowerCase();
  if (/mortar|granatwerfer/.test(id)) return 'mortar';
  if (/\bat\b|at_|_at|pak|anti_tank|atg/.test(id)) return 'at';
  if (/hmg|mmg|\bmg\b|machine|maxim|browning/.test(id)) return 'hmg';
  return 'gun';
}

function DEFAULT_WEAPON(ctx) {
  const p = new Parts();
  const metal = ctx.abandoned ? '#6a675f' : shade(STEEL, -0.05);
  const kit = ctx.abandoned ? '#6f6a5e' : warpaint(ctx.colour);

  if (ctx.weapon === 'mortar') {
    p.add(cyl(0.6, 0.6, 0.12, 8), shade(metal, -0.15), { y: 0.06 });        // baseplate
    p.cyl(0.09, 0.11, 1.15, metal, { x: -0.1, y: 0.62, rz: -0.55 }, 8);     // tube
    p.box(0.06, 0.8, 0.06, metal, { x: 0.32, y: 0.4, rz: 0.4 });            // bipod
    p.box(0.06, 0.8, 0.06, metal, { x: 0.32, y: 0.4, rz: -0.1, rx: 0.4 });
    p.box(0.34, 0.26, 0.34, kit, { x: -0.7, y: 0.14 });                     // ammo crate
    return p;
  }
  if (ctx.weapon === 'at') {
    p.box(1.25, 0.95, 0.09, shade(kit, -0.1), { x: 0.15, y: 0.7, rz: 0.06 });  // shield
    p.cyl(0.075, 0.09, 2.3, metal, { x: 1.25, y: 0.85, rz: Math.PI / 2 }, 8);  // barrel
    p.box(0.1, 0.1, 0.1, metal, { x: 2.3, y: 0.85, sx: 1.6, sy: 2.0, sz: 2.0 });
    p.box(0.5, 0.3, 0.5, metal, { x: -0.1, y: 0.62 });                         // breech
    p.box(1.5, 0.11, 0.11, metal, { x: -0.9, y: 0.3, rz: 0.13 });              // trails
    p.box(1.5, 0.11, 0.11, metal, { x: -0.9, y: 0.3, rz: 0.13, ry: 0.35 });
    p.box(1.5, 0.11, 0.11, metal, { x: -0.9, y: 0.3, rz: 0.13, ry: -0.35 });
    p.add(cyl(0.42, 0.42, 0.16, 10), CHARCOAL, { x: 0.1, y: 0.42, z: 0.62, rz: Math.PI / 2 });
    p.add(cyl(0.42, 0.42, 0.16, 10), CHARCOAL, { x: 0.1, y: 0.42, z: -0.62, rz: Math.PI / 2 });
    return p;
  }
  // HMG on a tripod
  p.box(0.07, 0.8, 0.07, metal, { x: -0.3, y: 0.34, rz: 0.35 });
  p.box(0.07, 0.8, 0.07, metal, { x: 0.22, y: 0.34, rz: -0.3, rx: 0.35 });
  p.box(0.07, 0.8, 0.07, metal, { x: 0.22, y: 0.34, rz: -0.3, rx: -0.35 });
  p.box(0.34, 0.16, 0.2, metal, { x: 0.0, y: 0.72 });                      // receiver
  p.cyl(0.045, 0.05, 0.95, metal, { x: 0.6, y: 0.75, rz: Math.PI / 2 }, 6);  // barrel
  p.box(0.2, 0.2, 0.06, shade(metal, 0.1), { x: 0.16, y: 0.9 });           // ammo drum
  p.box(0.3, 0.2, 0.26, kit, { x: -0.55, y: 0.12 });                       // ammo box
  return p;
}

// -------------------------------------------------------------- vehicles --

/* `ctx` for a vehicle: `{ def, colour, chassis }`. The hull points along +X;
 * the turret is returned separately so it can rotate to `turret_heading`. */
export function vehicleChassis(defId) {
  const id = String(defId || '').toLowerCase();
  if (/halftrack|half_track|sdkfz|carrier|truck|transport/.test(id)) return 'halftrack';
  if (/car|puma|greyhound|scout|recon/.test(id)) return 'car';
  return 'tank';
}

function DEFAULT_VEHICLE(ctx) {
  const hull = new Parts(), turret = new Parts();
  const body = warpaint(ctx.colour);
  const dark = shade(body, -0.25);
  const chassis = ctx.chassis;

  if (chassis === 'car') {
    hull.box(4.2, 0.85, 2.0, body, { y: 0.95 });
    hull.box(3.0, 0.55, 1.85, shade(body, 0.06), { x: -0.2, y: 1.6 });
    hull.box(4.4, 0.25, 2.1, dark, { y: 0.5 });
    wheels(hull, [1.45, 0.0, -1.5], 0.55, dark);
    turret.add(cyl(0.72, 0.8, 0.6, 8), shade(body, 0.1), { y: 0.3 });
    turret.cyl(0.075, 0.085, 1.7, dark, { x: 1.0, y: 0.34, rz: Math.PI / 2 }, 6);
    return { hull: hull, turret: turret, turretY: 1.95 };
  }
  if (chassis === 'halftrack') {
    hull.box(5.0, 0.95, 2.2, body, { y: 1.0 });
    hull.box(1.9, 0.7, 2.0, shade(body, 0.05), { x: 1.5, y: 1.7 });   // cab
    hull.box(2.6, 0.55, 2.15, dark, { x: -1.1, y: 1.6 });             // open bed sides
    tracks(hull, -0.9, 2.4, 0.55, dark);
    wheels(hull, [2.0], 0.5, dark);
    turret.box(0.5, 0.3, 0.5, dark, { y: 0.15 });
    turret.cyl(0.05, 0.06, 1.0, dark, { x: 0.5, y: 0.35, rz: Math.PI / 2 }, 6);
    return { hull: hull, turret: turret, turretY: 1.95 };
  }
  // tank
  hull.box(5.4, 0.95, 2.7, body, { y: 1.05 });
  hull.box(4.4, 0.5, 2.5, shade(body, 0.07), { x: -0.2, y: 1.72 });   // glacis deck
  hull.box(1.2, 0.45, 2.55, dark, { x: 2.3, y: 1.35, rz: -0.42 });    // sloped front
  tracks(hull, 0, 5.6, 0.62, dark);
  hull.box(5.2, 0.12, 0.2, shade(dark, 0.15), { y: 1.55, z: 1.4 });   // fenders
  hull.box(5.2, 0.12, 0.2, shade(dark, 0.15), { y: 1.55, z: -1.4 });
  hull.box(0.7, 0.35, 0.35, dark, { x: -2.6, y: 1.5 });               // stowage box

  turret.add(cyl(1.15, 1.35, 0.85, 8), shade(body, 0.1), { y: 0.42 });
  turret.box(1.0, 0.6, 1.6, shade(body, 0.1), { x: 0.6, y: 0.42 });
  turret.cyl(0.11, 0.13, 3.1, dark, { x: 1.9, y: 0.45, rz: Math.PI / 2 }, 8);
  turret.add(cyl(0.2, 0.2, 0.35, 8), dark, { x: 3.3, y: 0.45, rz: Math.PI / 2 });
  turret.add(cyl(0.42, 0.42, 0.22, 8), shade(body, 0.14), { x: -0.4, y: 0.92 });  // cupola
  turret.cyl(0.04, 0.04, 0.9, dark, { x: -0.4, y: 1.35, rz: 1.2 }, 5);            // aerial
  return { hull: hull, turret: turret, turretY: 2.0 };
}

function tracks(p, x, len, r, colour) {
  [-1, 1].forEach(function (side) {
    p.box(len, r * 2, 0.6, colour, { x: x, y: r, z: side * 1.25 });
    for (let i = 0; i < 5; i++) {
      p.add(cyl(r * 0.62, r * 0.62, 0.5, 8), shade(colour, 0.12),
            { x: x - len / 2 + len * (i + 0.5) / 5, y: r, z: side * 1.25, rx: Math.PI / 2 });
    }
  });
}

function wheels(p, xs, r, colour) {
  xs.forEach(function (x) {
    [-1, 1].forEach(function (side) {
      p.add(cyl(r, r, 0.4, 10), colour, { x: x, y: r, z: side * 0.95, rx: Math.PI / 2 });
    });
  });
}

// ----------------------------------------------------------------- point --

/* `ctx` for a point: `{ def (the point type), colour, victory }`. The flag
 * itself is a separate node so it can be raised with capture progress. */
function DEFAULT_POINT(ctx) {
  const pole = new Parts();
  const big = ctx.victory;
  const h = big ? 7.0 : 5.2;
  pole.box(1.7, 0.25, 1.7, SLATE, { y: 0.12 });
  pole.add(cyl(0.6, 0.75, 0.5, 8), shade(SLATE, -0.12), { y: 0.3 });
  pole.cyl(0.08, 0.1, h, TIMBER, { y: h / 2 + 0.4 }, 6);
  if (big) {
    // a star finial marks a victory point
    pole.add(cyl(0.3, 0.3, 0.1, 5), '#d8c27a', { y: h + 0.55, rx: Math.PI / 2 });
  }
  return { pole: pole, height: h, big: big };
}

// -------------------------------------------------------------- defaults --

const DEFAULTS = {
  building: DEFAULT_BUILDING,
  vehicle: DEFAULT_VEHICLE,
  weapon: DEFAULT_WEAPON,
  soldier: DEFAULT_SOLDIER,
  point: DEFAULT_POINT
};

/** Vehicles and points return two part lists, so they get their own entry points. */
export function buildVehicle(ctx) { return factoryFor('vehicle', ctx)(ctx); }
export function buildPoint(ctx) { return factoryFor('point', ctx)(ctx); }

/* Re-exported for skin authors, who build their parts with the same
 * primitives the default look does. */
export { box, cyl, bake, Parts };
