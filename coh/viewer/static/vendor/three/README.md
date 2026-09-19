# Vendored three.js

- **Package:** `three`
- **Version:** `0.169.0` (pinned)
- **Licence:** MIT — see `LICENSE` in this directory
- **Upstream:** <https://github.com/mrdoob/three.js>

Only `build/three.module.min.js` is vendored. Nothing else from the package is
used: the viewer writes its own camera controller (`static/js/view3d/camera.js`)
rather than pulling in `examples/jsm/controls/OrbitControls.js`, which keeps the
payload to one file with no transitive imports.

Refreshed with:

```
curl -sSL -o three.tgz https://registry.npmjs.org/three/-/three-0.169.0.tgz
tar xzf three.tgz
cp package/build/three.module.min.js package/LICENSE coh/viewer/static/vendor/three/
```

The page imports it through the import map in `static/index.html`
(`"three": "./vendor/three/three.module.min.js"`), so it is a relative
ES-module import and the viewer works with no network access at all. Do not
replace this with a CDN reference.
