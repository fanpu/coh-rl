/* The one mutable app-state object, shared by every module.
 *
 * `main.js` owns the writes (playback, input); the renderers and the data
 * layer only read. Keeping it in its own module means `data.js` can ask
 * "which fog mode is on?" without importing a renderer, and both views can
 * read the same selection and playhead without importing each other.
 */

export const S = {
  /** '3d' (the battlefield) or '2d' (the tactical map). */
  view: '3d',
  /** 'high' or 'low' — `?quality=low` drops shadows and texture resolution. */
  quality: 'high',
  webgl: true,          // cleared when a WebGL context cannot be created
  dpr: 1,               // device pixel ratio, capped at 2

  // --- playback ---
  playing: false,
  speedIdx: 1,
  tick: 0,              // playhead, in (fractional) sim ticks

  // --- overlays ---
  fogMode: 0,           // 0 omniscient, 1 team 0, 2 team 1
  showCover: false,
  showSectors: true,
  showArcs: false,      // team-weapon arcs for every weapon, not just the selection
  shadows: true,

  selection: null,      // {kind: 'squad'|'building', id}
  hover: null,          // same shape; 3D only

  /** 2D tactical-map camera: world metres at the viewport centre. */
  cam: { x: 0, y: 0, scale: 5 },

  /** 3D camera: a ground target plus an orbit. `pitch` is from the horizon. */
  cam3: { x: 0, y: 0, dist: 90, yaw: 0, pitch: 0.96, follow: null }
};
