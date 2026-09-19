/* Entry point: decide whether the 3D battlefield can run, then start.
 *
 * The 3D module (and with it three.js) is only imported when the browser can
 * actually give us a WebGL context, so a machine without one never downloads
 * or parses 700 KB it cannot use — it falls back to the 2D tactical map with
 * a one-line notice.
 */

import { S } from './state.js';
import * as main from './main.js';

function hasWebGL() {
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    if (!gl) return false;
    // Release it at once: browsers cap the number of live contexts and the
    // real renderer is about to ask for one of its own.
    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
    return true;
  } catch (e) {
    return false;
  }
}

if (hasWebGL()) {
  try {
    const view3d = await import('./view3d/index.js');
    view3d.wire(main.requestRender, main.select);
    main.install3d(view3d);
  } catch (err) {
    console.warn('3D view unavailable, falling back to the tactical map:', err);
    S.webgl = false;
  }
} else {
  S.webgl = false;
}

main.run();
