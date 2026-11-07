// gl.js - a small, dependency-free WebGL renderer for the 3D digital twin.
//
// Provides what the twin needs and nothing more: box / cylinder / sphere meshes
// with simple directional lighting, line edges, transparency, an orbit camera
// (rotate / pan / zoom with damping), ray picking against axis-aligned boxes and
// world-to-screen projection for HTML labels.
//
// Why not Three.js: see README ("3D technology choice"). The scene code in
// scene.js only uses the small API below, so it could be ported to Three.js
// (Mesh, BoxGeometry, OrbitControls, Raycaster) without changing app.js.

// ---------------------------------------------------------------- math
export const vec3 = {
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
};

const mat4 = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  },
  lookAt(eye, c, up) {
    const z = vec3.norm(vec3.sub(eye, c));
    const x = vec3.norm(vec3.cross(up, z));
    const y = vec3.cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -vec3.dot(x, eye), -vec3.dot(y, eye), -vec3.dot(z, eye), 1]);
  },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  },
  invert(m) {
    const inv = new Float32Array(16);
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    det = det ? 1 / det : 0;
    for (let i = 0; i < 16; i++) inv[i] *= det;
    return inv;
  },
  xform(m, v) {
    return [0, 1, 2, 3].map((r) => m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3]);
  },
};

export function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
}
export function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

// ---------------------------------------------------------------- geometry
function boxGeometry() {
  const p = [], n = [], idx = [];
  const faces = [
    [[1, 0, 0], [[.5, -.5, -.5], [.5, .5, -.5], [.5, .5, .5], [.5, -.5, .5]]],
    [[-1, 0, 0], [[-.5, -.5, .5], [-.5, .5, .5], [-.5, .5, -.5], [-.5, -.5, -.5]]],
    [[0, 1, 0], [[-.5, .5, -.5], [-.5, .5, .5], [.5, .5, .5], [.5, .5, -.5]]],
    [[0, -1, 0], [[-.5, -.5, .5], [-.5, -.5, -.5], [.5, -.5, -.5], [.5, -.5, .5]]],
    [[0, 0, 1], [[.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5], [-.5, -.5, .5]]],
    [[0, 0, -1], [[-.5, -.5, -.5], [-.5, .5, -.5], [.5, .5, -.5], [.5, -.5, -.5]]],
  ];
  faces.forEach(([nn, vs], f) => {
    vs.forEach((v) => { p.push(...v); n.push(...nn); });
    const b = f * 4; idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
  });
  return { p, n, idx };
}

function boxEdges() {
  const c = [[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]];
  const e = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7];
  const p = [], n = [];
  c.forEach((v) => { p.push(...v); n.push(0, 1, 0); });
  return { p, n, idx: e };
}

function cylinderGeometry(seg = 16) {
  // unit cylinder: radius 0.5, y from -0.5 to 0.5
  const p = [], n = [], idx = [];
  for (let i = 0; i <= seg; i++) {
    const a = (i / seg) * Math.PI * 2, x = Math.cos(a) * 0.5, z = Math.sin(a) * 0.5;
    p.push(x, -0.5, z, x, 0.5, z); n.push(x * 2, 0, z * 2, x * 2, 0, z * 2);
  }
  for (let i = 0; i < seg; i++) { const b = i * 2; idx.push(b, b + 1, b + 3, b, b + 3, b + 2); }
  for (const [y, ny] of [[0.5, 1], [-0.5, -1]]) {
    const c = p.length / 3; p.push(0, y, 0); n.push(0, ny, 0);
    for (let i = 0; i <= seg; i++) { const a = (i / seg) * Math.PI * 2; p.push(Math.cos(a) * 0.5, y, Math.sin(a) * 0.5); n.push(0, ny, 0); }
    for (let i = 0; i < seg; i++) ny > 0 ? idx.push(c, c + i + 2, c + i + 1) : idx.push(c, c + i + 1, c + i + 2);
  }
  return { p, n, idx };
}

function sphereGeometry(lat = 10, lon = 16) {
  const p = [], n = [], idx = [];
  for (let i = 0; i <= lat; i++) {
    const th = (i / lat) * Math.PI;
    for (let j = 0; j <= lon; j++) {
      const ph = (j / lon) * Math.PI * 2;
      const x = Math.sin(th) * Math.cos(ph), y = Math.cos(th), z = Math.sin(th) * Math.sin(ph);
      p.push(x * 0.5, y * 0.5, z * 0.5); n.push(x, y, z);
    }
  }
  for (let i = 0; i < lat; i++) for (let j = 0; j < lon; j++) {
    const a = i * (lon + 1) + j, b = a + lon + 1;
    idx.push(a, a + 1, b, b, a + 1, b + 1);
  }
  return { p, n, idx };
}

// ---------------------------------------------------------------- renderer
const VS = `
attribute vec3 aPos; attribute vec3 aNormal;
uniform mat4 uModel; uniform mat4 uViewProj; uniform mat3 uNormalMat;
varying vec3 vN;
void main(){ vN = normalize(uNormalMat * aNormal); gl_Position = uViewProj * uModel * vec4(aPos,1.0); }`;
const FS = `
precision mediump float;
uniform vec4 uColor; uniform vec3 uEmissive; uniform float uUnlit;
uniform vec3 uL1; uniform vec3 uL2; uniform float uAmbient;
varying vec3 vN;
void main(){
  vec3 n = normalize(vN);
  float d = uAmbient + 0.62*max(dot(n,uL1),0.0) + 0.22*max(dot(n,uL2),0.0);
  vec3 c = uUnlit > 0.5 ? uColor.rgb : uColor.rgb*d + uEmissive;
  gl_FragColor = vec4(min(c, vec3(1.0)), uColor.a);
}`;

export class Mesh {
  constructor(kind, opts = {}) {
    this.kind = kind;                      // box | cyl | sphere
    this.pos = opts.pos || [0, 0, 0];
    this.size = opts.size || [1, 1, 1];
    this.rotY = opts.rotY || 0;
    this.color = opts.color || [0.8, 0.8, 0.8];
    this.opacity = opts.opacity ?? 1;
    this.emissive = opts.emissive || [0, 0, 0];
    this.edges = opts.edges || null;       // [r,g,b] to draw box edges
    this.floor = opts.floor ?? null;       // floor group for explode / visibility
    this.visible = opts.visible ?? true;
    this.unlit = !!opts.unlit;
    this.tag = opts.tag || null;
  }
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl", { antialias: true, alpha: false }) || canvas.getContext("experimental-webgl");
    if (!gl) throw new Error("WebGL is not available in this browser.");
    this.gl = gl;
    this.prog = this._program(VS, FS);
    this.loc = {};
    for (const n of ["aPos", "aNormal"]) this.loc[n] = gl.getAttribLocation(this.prog, n);
    for (const n of ["uModel", "uViewProj", "uNormalMat", "uColor", "uEmissive", "uUnlit", "uL1", "uL2", "uAmbient"])
      this.loc[n] = gl.getUniformLocation(this.prog, n);
    this.geo = { box: this._upload(boxGeometry()), cyl: this._upload(cylinderGeometry()), sphere: this._upload(sphereGeometry()), boxEdges: this._upload(boxEdges()) };
    this.meshes = [];
    this.floorOffset = {};       // floor -> extra y offset
    this.floorVisible = {};      // floor -> bool
    this.background = [0.93, 0.95, 0.97];
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
  }

  _program(vs, fs) {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  _upload(g) {
    const gl = this.gl;
    const mk = (target, data) => { const b = gl.createBuffer(); gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW); return b; };
    return { pos: mk(gl.ARRAY_BUFFER, new Float32Array(g.p)), nrm: mk(gl.ARRAY_BUFFER, new Float32Array(g.n)),
      idx: mk(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(g.idx)), count: g.idx.length };
  }

  add(mesh) { this.meshes.push(mesh); return mesh; }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.gl.viewport(0, 0, w, h);
  }

  meshVisible(m) { return m.visible && (m.floor === null || this.floorVisible[m.floor] !== false); }

  worldPos(m) { return [m.pos[0], m.pos[1] + (m.floor !== null ? (this.floorOffset[m.floor] || 0) : 0), m.pos[2]]; }

  _model(m) {
    const [px, py, pz] = this.worldPos(m), [sx, sy, sz] = m.size, c = Math.cos(m.rotY), s = Math.sin(m.rotY);
    const model = new Float32Array([c * sx, 0, -s * sx, 0, 0, sy, 0, 0, s * sz, 0, c * sz, 0, px, py, pz, 1]);
    const nm = new Float32Array([c / sx, 0, -s / sx, 0, 1 / sy, 0, s / sz, 0, c / sz]);
    return [model, nm];
  }

  _bind(g) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, g.pos); gl.enableVertexAttribArray(this.loc.aPos); gl.vertexAttribPointer(this.loc.aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, g.nrm); gl.enableVertexAttribArray(this.loc.aNormal); gl.vertexAttribPointer(this.loc.aNormal, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, g.idx);
  }

  _draw(m, viewProj) {
    const gl = this.gl, L = this.loc;
    const [model, nm] = this._model(m);
    gl.uniformMatrix4fv(L.uModel, false, model);
    gl.uniformMatrix3fv(L.uNormalMat, false, nm);
    gl.uniform4f(L.uColor, m.color[0], m.color[1], m.color[2], m.opacity);
    gl.uniform3fv(L.uEmissive, m.emissive);
    gl.uniform1f(L.uUnlit, m.unlit ? 1 : 0);
    const g = this.geo[m.kind];
    this._bind(g);
    gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_SHORT, 0);
  }

  render(camera) {
    const gl = this.gl;
    this.resize();
    const aspect = this.canvas.width / this.canvas.height;
    const viewProj = camera.viewProj(aspect);
    this.viewProj = viewProj;
    gl.clearColor(...this.background, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.loc.uViewProj, false, viewProj);
    gl.uniform3fv(this.loc.uL1, vec3.norm([0.45, 0.85, 0.35]));
    gl.uniform3fv(this.loc.uL2, vec3.norm([-0.6, 0.4, -0.7]));
    gl.uniform1f(this.loc.uAmbient, 0.42);
    const vis = this.meshes.filter((m) => this.meshVisible(m));
    // opaque
    gl.disable(gl.BLEND); gl.depthMask(true);
    for (const m of vis) if (m.opacity >= 1) this._draw(m, viewProj);
    // edges
    const eg = this.geo.boxEdges; this._bind(eg);
    gl.uniform1f(this.loc.uUnlit, 1); gl.uniform3fv(this.loc.uEmissive, [0, 0, 0]);
    for (const m of vis) {
      if (!m.edges) continue;
      const [model, nm] = this._model(m);
      gl.uniformMatrix4fv(this.loc.uModel, false, model); gl.uniformMatrix3fv(this.loc.uNormalMat, false, nm);
      gl.uniform4f(this.loc.uColor, m.edges[0], m.edges[1], m.edges[2], 1);
      gl.drawElements(gl.LINES, eg.count, gl.UNSIGNED_SHORT, 0);
    }
    // transparent, back to front
    const eye = camera.eye();
    const tr = vis.filter((m) => m.opacity < 1)
      .map((m) => ({ m, d: vec3.len(vec3.sub(this.worldPos(m), eye)) })).sort((a, b) => b.d - a.d);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
    for (const { m } of tr) this._draw(m, viewProj);
    gl.depthMask(true); gl.disable(gl.BLEND);
  }

  project(p) {
    // world -> CSS pixel coordinates of the canvas; null if behind the camera
    if (!this.viewProj) return null;
    const c = mat4.xform(this.viewProj, [p[0], p[1], p[2], 1]);
    if (c[3] <= 0.01) return null;
    const x = c[0] / c[3], y = c[1] / c[3];
    return { x: (x * 0.5 + 0.5) * this.canvas.clientWidth, y: (-y * 0.5 + 0.5) * this.canvas.clientHeight, depth: c[2] / c[3] };
  }

  ray(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    const x = ((clientX - r.left) / r.width) * 2 - 1, y = -(((clientY - r.top) / r.height) * 2 - 1);
    const inv = mat4.invert(this.viewProj);
    const a = mat4.xform(inv, [x, y, -1, 1]), b = mat4.xform(inv, [x, y, 1, 1]);
    const o = [a[0] / a[3], a[1] / a[3], a[2] / a[3]], f = [b[0] / b[3], b[1] / b[3], b[2] / b[3]];
    return { o, d: vec3.norm(vec3.sub(f, o)) };
  }

  dispose() {
    const gl = this.gl;
    for (const g of Object.values(this.geo)) { gl.deleteBuffer(g.pos); gl.deleteBuffer(g.nrm); gl.deleteBuffer(g.idx); }
    gl.deleteProgram(this.prog);
    const ext = gl.getExtension("WEBGL_lose_context"); if (ext) ext.loseContext();
    this.meshes = [];
  }
}

export function rayBox(ray, min, max) {
  let tmin = -Infinity, tmax = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(ray.d[i]) < 1e-9) { if (ray.o[i] < min[i] || ray.o[i] > max[i]) return null; continue; }
    let t1 = (min[i] - ray.o[i]) / ray.d[i], t2 = (max[i] - ray.o[i]) / ray.d[i];
    if (t1 > t2) [t1, t2] = [t2, t1];
    tmin = Math.max(tmin, t1); tmax = Math.min(tmax, t2);
    if (tmin > tmax) return null;
  }
  return tmax < 0 ? null : Math.max(tmin, 0);
}

// ---------------------------------------------------------------- orbit camera
export class OrbitCamera {
  constructor(target, distance, azimuth, elevation) {
    this.home = { target: [...target], distance, azimuth, elevation };
    this.cur = { target: [...target], distance, azimuth, elevation };
    this.goal = { target: [...target], distance, azimuth, elevation };
    this.fov = 45 * Math.PI / 180;
    this.minDist = 8; this.maxDist = 90;
  }
  eye(s = this.cur) {
    const ce = Math.cos(s.elevation);
    return [s.target[0] + s.distance * ce * Math.sin(s.azimuth), s.target[1] + s.distance * Math.sin(s.elevation),
      s.target[2] + s.distance * ce * Math.cos(s.azimuth)];
  }
  viewProj(aspect) {
    const proj = mat4.perspective(this.fov, aspect, 0.3, 400);
    return mat4.mul(proj, mat4.lookAt(this.eye(), this.cur.target, [0, 1, 0]));
  }
  rotate(dx, dy) {
    this.goal.azimuth -= dx * 0.008;
    this.goal.elevation = Math.min(Math.PI / 2 - 0.05, Math.max(0.06, this.goal.elevation + dy * 0.006));
  }
  pan(dx, dy) {
    const eye = this.eye(this.goal), fwd = vec3.norm(vec3.sub(this.goal.target, eye));
    const right = vec3.norm(vec3.cross(fwd, [0, 1, 0])), up = vec3.cross(right, fwd);
    const k = this.goal.distance * 0.0015;
    this.goal.target = vec3.add(this.goal.target, vec3.add(vec3.scale(right, -dx * k), vec3.scale(up, dy * k)));
    this.goal.target[1] = Math.min(20, Math.max(-2, this.goal.target[1]));
  }
  zoom(factor) { this.goal.distance = Math.min(this.maxDist, Math.max(this.minDist, this.goal.distance * factor)); }
  focus(target, distance) { this.goal.target = [...target]; if (distance) this.goal.distance = distance; }
  reset() { this.goal = { target: [...this.home.target], distance: this.home.distance, azimuth: this.home.azimuth, elevation: this.home.elevation }; }
  update() {
    // damping; returns true while still moving
    let moving = false;
    const f = 0.22;
    for (const k of ["distance", "azimuth", "elevation"]) {
      const d = this.goal[k] - this.cur[k];
      if (Math.abs(d) > 1e-4) { this.cur[k] += d * f; moving = true; } else this.cur[k] = this.goal[k];
    }
    for (let i = 0; i < 3; i++) {
      const d = this.goal.target[i] - this.cur.target[i];
      if (Math.abs(d) > 1e-4) { this.cur.target[i] += d * f; moving = true; } else this.cur.target[i] = this.goal.target[i];
    }
    return moving;
  }
}

export function attachOrbitControls(canvas, camera, onChange) {
  let drag = null;
  const down = (e) => {
    if (e.button !== 0 && e.button !== 2) return;
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, moved: 0, id: e.pointerId };
    canvas.setPointerCapture(e.pointerId);
  };
  const move = (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.pan) camera.pan(dx, dy); else camera.rotate(dx, dy);
    onChange();
  };
  const up = (e) => { if (drag) { try { canvas.releasePointerCapture(drag.id); } catch (_) { /* ignore */ } } drag = null; };
  const wheel = (e) => { e.preventDefault(); camera.zoom(e.deltaY > 0 ? 1.1 : 1 / 1.1); onChange(); };
  const ctx = (e) => e.preventDefault();
  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("wheel", wheel, { passive: false });
  canvas.addEventListener("contextmenu", ctx);
  return {
    wasDrag: () => false,
    dragDistance: () => (drag ? drag.moved : 0),
    dispose() {
      canvas.removeEventListener("pointerdown", down); canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up); canvas.removeEventListener("pointercancel", up);
      canvas.removeEventListener("wheel", wheel); canvas.removeEventListener("contextmenu", ctx);
    },
  };
}
