/* Renderer, lighting, sky and the shared materials.
 *
 * One warm directional sun with a soft (PCF) shadow map that is re-fitted
 * each frame to whatever the camera can actually see, a hemisphere fill so
 * shadowed sides are not black, a light haze toward the horizon and ACES tone
 * mapping into sRGB. `?quality=low` drops shadows and halves the texture
 * budget for machines that cannot afford them.
 */

import * as THREE from 'three';
import { S } from '../state.js';
import { clamp } from '../util.js';

export const materials = {};

let renderer = null, scene = null, sun = null;

const SKY_TOP = new THREE.Color('#8fadc8');
const SKY_HORIZON = new THREE.Color('#d9d4c2');
const HAZE = 0xcdc9b8;

export function create(canvas) {
  renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: S.quality !== 'low',
    powerPreference: 'high-performance'
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.enabled = S.quality !== 'low' && S.shadows;

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(HAZE, 150, 620);

  sun = new THREE.DirectionalLight(0xfff2de, 2.3);
  sun.castShadow = true;
  const size = S.quality === 'low' ? 1024 : 2048;
  sun.shadow.mapSize.set(size, size);
  sun.shadow.bias = -0.0012;
  sun.shadow.normalBias = 0.05;
  scene.add(sun);
  scene.add(sun.target);

  const hemi = new THREE.HemisphereLight(0xbcd4ea, 0x6d6748, 1.35);
  scene.add(hemi);
  // a dim fill from the opposite side stops silhouettes going flat black
  const fill = new THREE.DirectionalLight(0xb6c6d8, 0.4);
  fill.position.set(60, 40, 80);
  scene.add(fill);

  buildSky();
  buildMaterials();
  return { renderer: renderer, scene: scene };
}

function buildSky() {
  const geo = new THREE.SphereGeometry(900, 24, 12);
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      top: { value: SKY_TOP },
      horizon: { value: SKY_HORIZON }
    },
    vertexShader: [
      'varying float vH;',
      'void main() {',
      '  vH = normalize(position).y;',
      '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
      '}'
    ].join('\n'),
    fragmentShader: [
      'uniform vec3 top;',
      'uniform vec3 horizon;',
      'varying float vH;',
      'void main() {',
      '  float k = clamp(vH * 1.6 + 0.12, 0.0, 1.0);',
      '  gl_FragColor = vec4(mix(horizon, top, sqrt(k)), 1.0);',
      '}'
    ].join('\n')
  });
  const sky = new THREE.Mesh(geo, mat);
  sky.renderOrder = -1;
  sky.frustumCulled = false;
  scene.add(sky);
}

function buildMaterials() {
  function std(opts) {
    return new THREE.MeshStandardMaterial(Object.assign({
      vertexColors: true, roughness: 0.85, metalness: 0.0
    }, opts || {}));
  }
  // One material per family; every baked model carries its colours in the
  // vertex stream, so a whole army shares these few programs.
  materials.prop = std();
  materials.unit = std({ roughness: 0.8 });
  materials.foliage = std({ roughness: 1.0, flatShading: true });
  materials.stone = std({ roughness: 0.95, flatShading: true });
  materials.timber = std({ roughness: 0.95 });
  materials.bark = std({ roughness: 1.0 });
  materials.earth = std({ roughness: 1.0, flatShading: true });
  /* Instanced meshes colour themselves per instance, and several of them are
   * drawn from raw three.js primitives that carry no colour attribute. A
   * `vertexColors` material would multiply that missing attribute in and come
   * out black, so these deliberately leave it off and let `instanceColor`
   * supply the colour on its own. */
  materials.feature = new THREE.MeshStandardMaterial({
    roughness: 0.95, metalness: 0, flatShading: true
  });
  materials.dirt = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0, flatShading: true });
  materials.ghost = new THREE.MeshStandardMaterial({
    color: 0x8d8b82, transparent: true, opacity: 0.3, roughness: 1,
    depthWrite: false, flatShading: true
  });
  // `toneMapped: false` keeps a muzzle flash and a fireball genuinely hot:
  // pushed through ACES at this exposure they wash out to pale peach.
  materials.fire = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.88, depthWrite: false, fog: false, toneMapped: false
  });
  materials.smoke = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.55, depthWrite: false
  });
  materials.tracer = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.95, depthWrite: false, fog: false, toneMapped: false
  });
  materials.ring = new THREE.MeshBasicMaterial({
    color: 0xffe8b0, transparent: true, opacity: 0.8, depthWrite: false,
    side: THREE.DoubleSide, fog: false
  });
}

export function applyShadows() {
  const on = S.shadows && S.quality !== 'low';
  renderer.shadowMap.enabled = on;
  sun.castShadow = on;
  // materials compiled against the old shadow setting must be rebuilt
  scene.traverse(function (o) {
    if (o.material) {
      const list = Array.isArray(o.material) ? o.material : [o.material];
      list.forEach(function (m) { m.needsUpdate = true; });
    }
  });
}

export function resize(width, height) {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height, false);
}

/* The shadow map covers what the camera can see and no more: a single 2048
 * map stretched over a 192 m map would be mush, so it follows the camera
 * target and sizes itself to the zoom level. */
export function fitShadows(camera) {
  const radius = clamp(S.cam3.dist * 0.75, 28, 150);
  const cam = sun.shadow.camera;
  cam.left = -radius; cam.right = radius;
  cam.top = radius; cam.bottom = -radius;
  cam.near = 1; cam.far = radius * 4 + 120;
  cam.updateProjectionMatrix();

  // a low afternoon sun from the north-west: long shadows that ground things
  const height = radius * 1.7 + 40;
  sun.target.position.set(S.cam3.x, 0, S.cam3.y);
  sun.position.set(S.cam3.x - radius * 1.1, height, S.cam3.y - radius * 0.75);
  sun.target.updateMatrixWorld();
  sun.updateMatrixWorld();
}

export function render(camera) {
  renderer.render(scene, camera);
}

/** Read back a box of rendered pixels, for the tests' pixel assertions.
 *
 *  The drawing buffer is not preserved across a composite (turning that on
 *  costs a full-screen copy every frame), so the caller has to re-render in
 *  the same task before reading — see `view3d/index.js`. */
export function readPixels(x, y, w, h) {
  const gl = renderer.getContext();
  const buf = new Uint8Array(w * h * 4);
  // WebGL's origin is bottom-left; the callers think top-left
  gl.readPixels(x, renderer.domElement.height - y - h, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
  return buf;
}

export function info() {
  const i = renderer.info.render;
  return { calls: i.calls, triangles: i.triangles };
}
