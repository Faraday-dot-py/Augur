import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { lut, fmt } from "./colors.js";

THREE.ColorManagement.enabled = false;

const BG = 0x06080c, MAXN = 128, POOL = 16;
const EDGE = new THREE.Color(0x2b3a52), HOT = new THREE.Color(0x6fc3ff);
const BODY = new THREE.Color(0x62d6a4), BODY_HOT = new THREE.Color(0xffffff);

export class Block {
  constructor(id, o) {
    Object.assign(this, { id, kind: o.kind, H: o.H, W: o.W || o.H, n: o.n || 1, size: o.size, sizeY: o.sizeY || o.size, title: o.title, C: o.C || 1, gap: o.gap || 0, thick: o.thick || 0, base: o.opacity || 1, rest: o.rest === undefined ? 1 : o.rest });
    this.kinds = o.kinds || null; this.sub = null;
    this.op = 1; this.hl = 0; this.chan = 0; this.src = []; this.planes = [];
    this.group = new THREE.Group();
    this.group.position.set(...o.pos);
    const geo = new THREE.PlaneGeometry(this.size, this.sizeY);
    for (let k = 0; k < this.n; k++) {
      const data = new Uint8Array(this.W * this.H * 4);
      for (let i = 0; i < data.length; i += 4) { data[i] = 22; data[i + 1] = 26; data[i + 2] = 36; data[i + 3] = 255; }
      const map = new THREE.DataTexture(data, this.W, this.H, THREE.RGBAFormat);
      map.magFilter = map.minFilter = THREE.NearestFilter;
      map.needsUpdate = true;
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = -k * this.gap;
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ transparent: true }));
      mesh.add(edge);
      mesh.userData = { block: this, k };
      this.group.add(mesh);
      this.planes.push({ mesh, map, data, edge });
    }
    if (this.thick) {
      this.box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(this.size, this.thick, this.sizeY)), new THREE.LineBasicMaterial({ transparent: true }));
      this.group.add(this.box);
    }
    this.setChannel(0);
  }

  paint(k, arr, off = 0, scale = 0, center = false) {
    const W = this.W, H = this.H, n = W * H, p = this.planes[k];
    let med = 0;
    if (center) med = arr.slice(off, off + n).sort()[n >> 1];
    let s = scale;
    if (!s) {
      const m = W >= 16 ? 2 : 0;
      for (let y = m; y < H - m; y++) for (let x = m; x < W - m; x++) { const a = Math.abs(arr[off + y * W + x] - med); if (a > s) s = a; }
      s = s || 1e-12;
    }
    const L = lut(this.kinds ? this.kinds[k] : this.kind), d = p.data;
    for (let i = 0; i < n; i++) {
      const t = (arr[off + i] - med) / s;
      const q = 3 * (((t < -1 ? -1 : t > 1 ? 1 : t) * 127.5 + 127.5) | 0);
      d[4 * i] = L[q]; d[4 * i + 1] = L[q + 1]; d[4 * i + 2] = L[q + 2];
    }
    p.map.needsUpdate = true;
    this.src[k] = { arr, off, s };
  }

  setTensor(arr) {
    this.tensor = arr;
    this.setChannel(this.chan);
  }

  setChannel(c) {
    this.chan = Math.min(c, this.C - 1);
    if (!this.tensor) return;
    this.paint(0, this.tensor, this.chan * this.W * this.H, 0, true);
    if (this.thick) this.planes[0].mesh.position.y = (this.C > 1 ? this.chan / (this.C - 1) - 0.5 : 0) * this.thick;
  }

  valueAt(k, ix, iy) {
    const s = this.src[k];
    return s ? s.arr[s.off + iy * this.W + ix] : null;
  }

  apply() {
    const vis = this.op > 0.02;
    this.group.visible = vis;
    if (!vis) return;
    const col = EDGE.clone().lerp(HOT, this.hl);
    const b = 0.7 + 0.3 * this.hl;
    for (const p of this.planes) {
      p.mesh.material.opacity = this.op * this.base * (0.7 + 0.3 * this.hl);
      p.mesh.material.color.setScalar(b);
      p.edge.material.color.copy(col);
      p.edge.material.opacity = this.op;
    }
    if (this.box) { this.box.material.color.copy(col); this.box.material.opacity = this.op; }
  }
}

class Segs {
  constructor(cap, color, parent) {
    this.cap = cap; this.count = 0;
    this.pos = new Float32Array(cap * 6);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    this.mesh = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, depthWrite: false }));
    this.mesh.frustumCulled = false;
    parent.add(this.mesh);
  }

  begin() { this.count = 0; }

  add(ax, ay, az, bx, by, bz) {
    if (this.count >= this.cap) return;
    this.pos.set([ax, ay, az, bx, by, bz], this.count++ * 6);
  }

  arrow(ax, ay, az, bx, by, bz) {
    this.add(ax, ay, az, bx, by, bz);
    const dx = bx - ax, dz = bz - az, l = Math.hypot(dx, dz);
    if (l < 1e-6) return;
    const hl = Math.min(0.25, l * 0.35), ux = dx / l, uz = dz / l;
    for (const s of [-1, 1]) this.add(bx, by, bz, bx - hl * (ux + 0.5 * s * -uz), by, bz - hl * (uz + 0.5 * s * ux));
  }

  end(alpha) {
    this.mesh.geometry.attributes.position.needsUpdate = true;
    this.mesh.geometry.setDrawRange(0, this.count * 2);
    this.mesh.material.opacity = alpha;
    this.mesh.visible = alpha > 0.01 && this.count > 0;
  }
}

export class Stage {
  constructor(el, labelsEl) {
    this.el = el; this.labelsEl = labelsEl;
    this.renderer = new THREE.WebGLRenderer({ canvas: el, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setClearColor(BG);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 2000);
    this.camera.position.set(0, 40, 50);
    this.controls = new OrbitControls(this.camera, el);
    this.controls.enableDamping = false;
    this.controls.maxDistance = 600;
    this.controls.addEventListener("change", () => { this.dirty = true; });
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(10, 30, 20);
    this.scene.add(sun);
    this.blocks = new Map();
    this.curves = new Map();
    this.notes = new Map();
    this.pickables = [];
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.dirty = true;
    this.ray = new THREE.Raycaster();
    this.v = new THREE.Vector3();
    this.bodyHl = new Float32Array(MAXN);
    this.nBodies = 0;
    const bm = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshLambertMaterial(), MAXN);
    bm.setColorAt(0, new THREE.Color());
    bm.frustumCulled = false;
    bm.count = 0;
    this.bodies = bm;
    this.root.add(bm);
    this.ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 33 }, (_, i) => new THREE.Vector3(Math.cos(i / 16 * Math.PI), 0, Math.sin(i / 16 * Math.PI)))), new THREE.LineBasicMaterial({ color: 0x6fc3ff, transparent: true }));
    this.ring.visible = false;
    this.root.add(this.ring);
    this.segs = {};
    for (const [k, c, cap] of [["cic", 0x62d6a4, 16], ["pair", 0xff9a3c, 700], ["pairSel", 0xffd36e, 40], ["gather", 0x62d6a4, 300], ["pairAcc", 0xff9a3c, 300], ["vel", 0x6fc3ff, 300], ["move", 0xe4eaf5, 200]]) this.segs[k] = new Segs(cap, c, this.root);
    this.marks = [];
    for (let i = 0; i < POOL; i++) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0x6fc3ff, transparent: true, depthWrite: false, depthTest: false }));
      m.rotation.x = -Math.PI / 2;
      m.visible = false;
      m.renderOrder = 5;
      this.root.add(m);
      this.marks.push(m);
    }
    this.markList = [];
    this.ghost = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 8), new THREE.MeshBasicMaterial({ color: 0xe4eaf5, transparent: true, opacity: 0.35, depthWrite: false }), MAXN);
    this.ghost.frustumCulled = false;
    this.ghost.count = 0;
    this.root.add(this.ghost);
    this.resize();
    addEventListener("resize", () => this.resize());
  }

  resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    const wide = w > 760;
    this.camera.setViewOffset(w, h, wide ? 60 : 0, wide ? 0 : h * 0.16, w, h);
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  add(id, o) {
    const b = new Block(id, o);
    this.blocks.set(id, b);
    this.root.add(b.group);
    for (const p of b.planes) this.pickables.push(p.mesh);
    return b;
  }

  block(id) { return this.blocks.get(id); }

  // sim coordinates (x right, y up, arena centre 0) to world position on a block's top plane
  world(id, x, y, lift = 0, out = new THREE.Vector3()) {
    const b = this.blocks.get(id), g = b.group.position;
    return out.set(g.x + x / 64 * b.size, g.y + lift, g.z - y / 64 * b.sizeY);
  }

  addCurve(id, pts, color) {
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, depthWrite: false }));
    line.frustumCulled = false;
    this.root.add(line);
    this.curves.set(id, { line, n: pts.length });
    return line;
  }

  arc(id, a, b, lift, color, n = 40) {
    const c = new THREE.QuadraticBezierCurve3(a, new THREE.Vector3((a.x + b.x) / 2, Math.max(a.y, b.y) + lift, (a.z + b.z) / 2), b);
    return this.addCurve(id, c.getPoints(n - 1), color);
  }

  beginFrame() {
    for (const b of this.blocks.values()) { b.op = b.rest; b.hl = 0; }
    for (const c of this.curves.values()) c.frac = 0;
    this.bodyHl.fill(0);
    for (const s of Object.values(this.segs)) s.begin();
    this.segAlpha = {};
    this.markList.length = 0;
    this.ringAt = null;
    this.ghostList = null;
    for (const n of this.notes.values()) n.used = false;
  }

  setBodies(pos, n, spread) {
    this.nBodies = n;
    this.spread = spread;
    this.bodyPos = pos;
  }

  bodyWorld(i, out = new THREE.Vector3()) {
    return this.world("input", this.bodyPos[2 * i], this.bodyPos[2 * i + 1], 0.45, out);
  }

  note(key, text, p, alpha = 1, cls = "") {
    let n = this.notes.get(key);
    if (!n) {
      const el = document.createElement("div");
      el.className = "lab";
      this.labelsEl.appendChild(el);
      n = { el, text: "", cls: "", p: new THREE.Vector3() };
      this.notes.set(key, n);
    }
    if (n.text !== text) { n.el.innerHTML = text; n.text = text; }
    if (n.cls !== cls) { n.el.className = "lab " + cls; n.cls = cls; }
    n.p.copy(p);
    n.alpha = alpha;
    n.used = true;
  }

  endFrame() {
    for (const b of this.blocks.values()) b.apply();
    for (const [id, c] of this.curves) {
      c.line.visible = c.frac > 0.001 && (c.alpha || 0) > 0.01;
      c.line.geometry.setDrawRange(0, Math.max(2, Math.ceil(c.frac * c.n)));
      c.line.material.opacity = c.alpha || 0;
    }
    for (const [k, s] of Object.entries(this.segs)) s.end(this.segAlpha[k] || 0);
    this.updateBodies();
    this.marks.forEach((m, i) => {
      const e = this.markList[i];
      m.visible = !!e;
      if (!e) return;
      m.position.copy(e.p);
      m.scale.set(e.size, e.size, 1);
      m.material.opacity = e.alpha;
      m.material.color.set(e.color || 0x6fc3ff);
    });
    if (this.ringAt) { this.ring.visible = true; this.ring.position.copy(this.ringAt); this.ring.scale.setScalar(this.ringR); this.ring.material.opacity = this.ringA; } else this.ring.visible = false;
    const gl = this.ghostList;
    this.ghost.count = gl ? gl.length : 0;
    if (gl) {
      const m = new THREE.Matrix4();
      gl.forEach((p, i) => { m.compose(p, new THREE.Quaternion(), new THREE.Vector3(0.12, 0.12, 0.12)); this.ghost.setMatrixAt(i, m); });
      this.ghost.instanceMatrix.needsUpdate = true;
    }
    for (const n of this.notes.values()) if (!n.used) n.alpha = 0;
  }

  updateBodies() {
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), c = new THREE.Color();
    const n = this.nBodies, r = 0.11;
    s.set(r, r, r);
    this.bodies.count = n;
    for (let i = 0; i < n; i++) {
      this.bodyWorld(i, p);
      m.compose(p, q, s);
      this.bodies.setMatrixAt(i, m);
      this.bodies.setColorAt(i, c.copy(BODY).lerp(BODY_HOT, this.bodyHl[i]));
    }
    this.bodies.instanceMatrix.needsUpdate = true;
    if (this.bodies.instanceColor) this.bodies.instanceColor.needsUpdate = true;
  }

  render() {
    if (!this.dirty) return;
    this.dirty = false;
    this.renderer.render(this.scene, this.camera);
    const w = innerWidth, h = innerHeight, v = this.v;
    for (const n of this.notes.values()) {
      v.copy(n.p).project(this.camera);
      const on = n.alpha > 0.02 && v.z < 1 && Math.abs(v.x) < 1.15 && Math.abs(v.y) < 1.15;
      n.el.style.opacity = on ? n.alpha : 0;
      n.el.style.visibility = on ? "visible" : "hidden";
      if (on) n.el.style.transform = "translate(-50%,-100%) translate(" + ((v.x + 1) * w / 2).toFixed(1) + "px," + ((1 - v.y) * h / 2).toFixed(1) + "px)";
    }
  }

  setPose(target, pos) {
    this.controls.target.set(...target);
    this.camera.position.set(...pos);
    this.controls.update();
    this.dirty = true;
  }

  pick(cx, cy) {
    const r = this.el.getBoundingClientRect();
    this.ray.setFromCamera(new THREE.Vector2((cx - r.left) / r.width * 2 - 1, -((cy - r.top) / r.height * 2 - 1)), this.camera);
    const live = this.pickables.filter((m) => m.userData.block.op > 0.3 && m.userData.block.group.visible);
    const hits = this.ray.intersectObjects([...live, this.bodies], false);
    for (const h of hits) {
      if (h.object === this.bodies) return { type: "body", i: h.instanceId };
      const { block, k } = h.object.userData;
      const W = block.sub ? block.sub.w : block.W, H = block.sub ? block.sub.h : block.H;
      const ix = Math.min(W - 1, Math.floor(h.uv.x * W)), iy = Math.min(H - 1, Math.floor(h.uv.y * H));
      return { type: "cell", block, k, ix, iy, v: block.valueAt(k, ix, iy), p: h.point };
    }
    return null;
  }
}

export { fmt };
