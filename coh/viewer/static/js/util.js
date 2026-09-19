/* Palette, colour maths and small formatting helpers.
 *
 * Nothing here knows about the replay or about a renderer, so both the 2D
 * tactical map and the 3D battlefield pull their colours from the same place
 * and cannot drift apart.
 */

export const TEAM_COLOURS = ['#4f8fe0', '#d8503c'];
export const NEUTRAL = '#b9b2a0';

export const TERRAIN = {
  '.': { base: '#49612d', name: 'Grass' },
  'r': { base: '#a08e60', name: 'Road (negative cover)' },
  'c': { base: '#4a3d27', name: 'Crater (heavy cover)' },
  'f': { base: '#49612d', name: 'Fence (light cover)' },
  'w': { base: '#9d9887', name: 'Wall (heavy cover)' },
  'H': { base: '#1b2f12', name: 'Hedgerow' },
  'T': { base: '#284a1b', name: 'Trees (light cover)' },
  '~': { base: '#204a68', name: 'Water' }
};
export const UNKNOWN_TERRAIN = { base: '#6a3a6a', name: 'Unknown' };

/** Cover class per terrain char: 2 = heavy, 1 = light, 0 = none/negative. */
export const COVER = { '.': 0, 'r': 0, 'c': 2, 'f': 1, 'w': 2, 'H': 2, 'T': 1, '~': 0 };

const ACRONYMS = { hmg: 'HMG', at: 'AT', mg: 'MG', hq: 'HQ', op: 'OP', us: 'US', vp: 'VP', aa: 'AA' };
const FACTION_NAMES = { us: 'US', wehr: 'Wehrmacht' };

export const SPEEDS = [0.5, 1, 2, 4, 8, 16];

export function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
export function lerp(a, b, u) { return a + (b - a) * u; }

export function lerpAngle(a, b, u) {
  const d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return a + d * u;
}

export function title(id) {
  if (!id) return '';
  return String(id).split(/[_\s]+/).map(function (w) {
    if (!w) return '';
    if (ACRONYMS[w.toLowerCase()]) return ACRONYMS[w.toLowerCase()];
    return w.charAt(0).toUpperCase() + w.slice(1);
  }).join(' ');
}

export function factionName(faction) {
  return FACTION_NAMES[faction] || title(faction) || '?';
}

export function mmss(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return Math.floor(s / 60) + ':' + ('0' + (s % 60)).slice(-2);
}

/** Accepts "#rrggbb" or "rgb(r,g,b)" and returns [r, g, b]. */
export function parseColour(c) {
  if (c.charAt(0) === '#') {
    const n = parseInt(c.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const m = c.match(/-?\d+(\.\d+)?/g) || [0, 0, 0];
  return [+m[0] || 0, +m[1] || 0, +m[2] || 0];
}

export function mix(colourA, colourB, u) {
  const a = parseColour(colourA), b = parseColour(colourB);
  return 'rgb(' + Math.round(lerp(a[0], b[0], u)) + ',' +
    Math.round(lerp(a[1], b[1], u)) + ',' + Math.round(lerp(a[2], b[2], u)) + ')';
}

/** Any colour string -> `rgba(..., a)`. */
export function alpha(colour, a) {
  const c = parseColour(colour);
  return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
}

export function shade(hex, u) { return mix(hex, u < 0 ? '#000000' : '#ffffff', Math.abs(u)); }

/* Deterministic per-cell noise, so terrain texture does not swim while
 * panning. `Math.imul` matters here: a plain `*` on two 32-bit integers
 * overflows a double's 53-bit mantissa and quietly drops the low bits, which
 * leaves the hash correlated along x+y — as broad diagonal bands across the
 * whole map once it is used as value noise for the ground. */
export function hash2(cx, cy) {
  let h = (Math.imul(cx, 374761393) + Math.imul(cy, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** A second independent stream from the same cell, for varying a second trait. */
export function hash3(cx, cy, salt) {
  let h = (Math.imul(cx, 2654435761) + Math.imul(cy, 2246822519) + Math.imul(salt, 3266489917)) | 0;
  h = Math.imul(h ^ (h >>> 15), 2246822519);
  h = Math.imul(h ^ (h >>> 13), 3266489917);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

export function offscreen(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}
