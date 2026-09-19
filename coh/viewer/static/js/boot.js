/* Entry point.
 *
 * Kept separate from `main.js` so that deciding *which* renderer to load can
 * happen before the app starts. For now there is only the 2D tactical map; a
 * later commit adds the 3D battlefield here behind a WebGL probe.
 */

import { S } from './state.js';
import * as main from './main.js';

S.webgl = false;

main.run();
