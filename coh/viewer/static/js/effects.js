/* The catalogue of what each sim event kind means on screen.
 *
 * This table is renderer-independent on purpose: it says how long an effect
 * lives and *what it is*, and each view decides how to draw that. The 2D
 * tactical map draws a ring; the 3D battlefield throws fire, smoke and dirt.
 * Neither may invent its own list, so a new `Event(kind=...)` in the sim
 * lands in both views (or is deliberately skipped in both).
 *
 * `mode: null` means "deliberately shown somewhere else" — points pulse
 * themselves, `game_over` is the HUD banner. A kind that is not in the table
 * at all is a sim newer than this page: both views fall back to a generic
 * marker rather than throwing.
 *
 * `tests/viewer/test_effect_coverage.py` reads the keys of both objects below
 * and compares them against every `kind=` literal in `coh/sim`.
 */
export const EVENT_EFFECTS = {
  shot: { ticks: 3, mode: 'tracer' },
  // the sim tells us the blast radius in metres; both views use it at true size
  explosion: { ticks: 14, mode: 'blast', radius: null },
  squad_destroyed: { ticks: 20, mode: 'blast', radius: 4.5, death: 'infantry' },
  vehicle_destroyed: { ticks: 28, mode: 'blast', radius: 8, death: 'vehicle' },
  building_destroyed: { ticks: 26, mode: 'blast', radius: 9, death: 'building' },
  weapon_abandoned: { ticks: 18, mode: 'ripple', colour: '#8f8a7c', radius: 3.5 },
  weapon_recrewed: { ticks: 18, mode: 'ripple', colour: '#7fbf6a', radius: 3.5 },
  garrison_entered: { ticks: 14, mode: 'ripple', colour: '#9fc4e8', radius: 2.5 },
  garrison_ejected: { ticks: 20, mode: 'blast', radius: 3, death: 'infantry' },
  reinforced: { ticks: 14, mode: 'ripple', colour: '#7fbf6a', radius: 2.5 },
  unit_trained: { ticks: 16, mode: 'ripple', colour: '#9fc4e8', radius: 4 },
  construction_started: { ticks: 16, mode: 'ripple', colour: '#d8b45a', radius: 4 },
  building_completed: { ticks: 16, mode: 'ripple', colour: '#7fbf6a', radius: 6 },
  research_completed: { ticks: 16, mode: 'ripple', colour: '#9fc4e8', radius: 5 },
  upgrade_bought: { ticks: 14, mode: 'ripple', colour: '#d8b45a', radius: 2.5 },
  point_captured: { ticks: 16, mode: null },     // pulse drawn with the point
  point_neutralized: { ticks: 16, mode: null },  // pulse drawn with the point
  game_over: { ticks: 1, mode: null }            // winner banner, see hud.js
};

/** Timeline tick marks, by event kind (kinds not listed are not marked). */
export const MARKER_COLOURS = {
  point_captured: '#d8b45a',
  point_neutralized: '#d8b45a',
  squad_destroyed: '#e2705f',
  vehicle_destroyed: '#e2705f',
  building_destroyed: '#e2705f',
  game_over: '#f2e9c8'
};

export const DEFAULT_EFFECT_TICKS = 16;
export const MAX_EFFECT_TICKS = 30;

/** Blast radius in metres for an effect, preferring the sim's own number. */
export function blastRadius(effect, spec) {
  if (spec && spec.radius) return spec.radius;
  return Math.max(Number((effect.d || {}).radius) || 0, 2);
}
