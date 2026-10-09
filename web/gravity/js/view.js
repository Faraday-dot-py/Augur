import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CentralArch, fmt } from "./arch.js";

THREE.ColorManagement.enabled = false;

const BG = 0x06080c, FLOW = 500;
const RANK = { sec: 1, op: 2, col: 3, "col off": 3 };

// Three.js model view for the central-force page: the CentralArch diagram of one body, orbit camera, hover
// tooltips, HTML labels. Same set-up as web/token/js/scene.js, without the arena.
export class ModelView {
  constructor(host, labelHost, tip, onSelect) {
    this.host = host; this.labelHost = labelHost; this.tip = tip; this.onSelect = onSelect;
    const R = this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    R.setPixelRatio(Math.min(devicePixelRatio, 2));
    R.outputColorSpace = THREE.LinearSRGBColorSpace;
    R.setClearColor(BG);
    host.appendChild(R.domElement);
    const S = this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.5, 2000);
    const C = this.controls = new OrbitControls(this.camera, R.domElement);
    C.enableDamping = true; C.dampingFactor = 0.08; C.maxPolarAngle = 1.48; C.minDistance = 6; C.maxDistance = 400;
    C.addEventListener("start", () => { this.goal = null; });
    S.add(new THREE.HemisphereLight(0xe4ecff, 0x1a1f2a, 2.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.3);
    sun.position.set(60, 140, 30);
    S.add(sun);
    this.arch = new CentralArch(S);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * FLOW), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3 * FLOW), 3).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    this.flow = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.3, vertexColors: true, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false }));
    this.flow.frustumCulled = false;
    S.add(this.flow);
    this.ray = new THREE.Raycaster();
    this.v = new THREE.Vector3();
    this.pool = []; this.widths = new Map();
    this.goal = null; this.trace = null; this.sweep = performance.now(); this.hoverAt = null; this.down = null;
    this.W = this.H = 1; this.ins = [0, 0, 0, 0];
    const cv = R.domElement;
    cv.addEventListener("pointerdown", (e) => { this.down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
    cv.addEventListener("pointerup", (e) => {
      const d = this.down;
      this.down = null;
      if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 6 || performance.now() - d.t > 500) return;
      const hit = this.pick(e.clientX, e.clientY);
      if (hit && hit.cell.src >= 0) this.onSelect(hit.cell.src);
      if (e.pointerType !== "mouse") this.hoverAt = { x: e.clientX, y: e.clientY };
    });
    cv.addEventListener("pointermove", (e) => { this.hoverAt = { x: e.clientX, y: e.clientY }; if (e.buttons) this.tip.classList.remove("on"); });
    cv.addEventListener("pointerleave", () => { this.hoverAt = null; this.arch.hover = -1; this.tip.classList.remove("on"); });
  }

  set(tr) {
    this.trace = tr;
    this.arch.set(tr);
    const now = performance.now();
    if (now - this.sweep > 4000) this.sweep = now;
  }

  restartSweep() { this.sweep = performance.now(); }

  resize(w, h, ins) {
    this.W = w; this.H = h; this.ins = ins;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const cx = (ins[0] + w - ins[2]) / 2, cy = (ins[1] + h - ins[3]) / 2;
    this.camera.setViewOffset(w, h, w / 2 - cx, h / 2 - cy, w, h);
  }

  // frame the whole diagram inside the free area (insets = left, top, right, bottom px covered by UI)
  fit() {
    const w = 58, h = 64, cx = 28, cz = 28, el = innerWidth < innerHeight ? 1.25 : 1.0;
    const tgt = new THREE.Vector3(cx, 0, cz), dir = new THREE.Vector3(0, Math.sin(el), Math.cos(el));
    const cam = this.camera.clone();
    const [l, t, r, b] = this.ins, fw = this.W - l - r, fh = this.H - t - b;
    const pts = [];
    for (const y of [0, 3]) for (const sx of [-1, 1]) for (const sz of [-1, 1]) pts.push(new THREE.Vector3(cx + (sx * w) / 2, y, cz + (sz * h) / 2));
    let d = Math.max(w, h) * 2;
    const right = new THREE.Vector3(1, 0, 0), up = new THREE.Vector3().crossVectors(dir, right);
    for (let k = 0; k < 10; k++) {
      cam.position.copy(dir).multiplyScalar(d).add(tgt);
      cam.lookAt(tgt);
      cam.updateMatrixWorld();
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
      for (const p of pts) {
        const v = this.v.copy(p).project(cam);
        x0 = Math.min(x0, v.x); x1 = Math.max(x1, v.x); y0 = Math.min(y0, v.y); y1 = Math.max(y1, v.y);
      }
      const hh = d * Math.tan((cam.fov * Math.PI) / 360), hw = hh * cam.aspect;
      tgt.addScaledVector(right, ((x0 + x1) / 2) * hw * 0.8).addScaledVector(up, ((y0 + y1) / 2) * hh * 0.8);
      d *= Math.max(((x1 - x0) / 2) * this.W / fw, ((y1 - y0) / 2) * this.H / fh) / 0.94;
    }
    this.goal = { pos: cam.position.clone(), target: tgt };
  }

  pick(cx, cy) {
    const el = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera({ x: ((cx - el.left) / el.width) * 2 - 1, y: -((cy - el.top) / el.height) * 2 + 1 }, this.camera);
    const h = this.ray.intersectObject(this.arch.mesh)[0];
    return h ? { id: h.instanceId, cell: this.arch.cells[h.instanceId] } : null;
  }

  project(u, w) {
    const v = this.v.set(u, 0.1, w);
    v.project(this.camera);
    return v.z < 1 ? v : null;
  }

  hover() {
    this.arch.hover = -1;
    const hit = this.hoverAt && this.pick(this.hoverAt.x, this.hoverAt.y), tip = this.tip;
    this.renderer.domElement.style.cursor = hit && hit.cell.src >= 0 ? "pointer" : "";
    if (!hit) { tip.classList.remove("on"); return; }
    this.arch.hover = hit.id;
    tip.innerHTML = `<b>${hit.cell.val}</b><br><span>${hit.cell.name}</span>` + (hit.cell.src >= 0 ? `<br><span>click: follow body #${hit.cell.src}</span>` : "");
    tip.classList.add("on");
    const w = tip.offsetWidth, h = tip.offsetHeight, a = this.hoverAt;
    const x = Math.min(a.x + 14, innerWidth - w - 8), y = a.y + 16 + h > innerHeight - 8 ? a.y - h - 10 : a.y + 16;
    tip.style.transform = `translate(${x}px, ${y}px)`;
  }

  measure(text, cls) {
    const key = cls + "|" + text;
    if (!this.widths.has(key)) {
      const el = this.labelHost.appendChild(document.createElement("div"));
      el.className = cls;
      el.textContent = text;
      this.widths.set(key, [el.offsetWidth, el.offsetHeight]);
      el.remove();
    }
    return this.widths.get(key);
  }

  labels() {
    const a = this.project(0, 20), b = a && { x: a.x, y: a.y }, c = b && this.project(10, 20);
    const ppu = c ? Math.hypot((c.x - b.x) * this.W, (c.y - b.y) * this.H) / 20 : 0;
    const list = this.arch.labels.filter((l) => ppu >= (l.cls.startsWith("col") ? 6 : 3.5)).sort((p, q) => RANK[p.cls] - RANK[q.cls]);
    const placed = [];
    let n = 0;
    for (const l of list) {
      const p = this.project(l.u, l.w);
      if (!p || Math.abs(p.x) > 1.1 || Math.abs(p.y) > 1.1) continue;
      const [w, h] = this.measure(l.text, l.cls);
      const x = ((p.x + 1) / 2) * this.W - (l.left ? 0 : w / 2), y = ((1 - p.y) / 2) * this.H - h / 2;
      if (placed.some((r) => x < r[2] + 4 && x + w + 4 > r[0] && y < r[3] + 1 && y + h + 1 > r[1])) continue;
      placed.push([x, y, x + w, y + h]);
      const el = this.pool[n] || this.labelHost.appendChild(document.createElement("div"));
      this.pool[n++] = el;
      if (el.textContent !== l.text) el.textContent = l.text;
      if (el.className !== l.cls) el.className = l.cls;
      el.style.display = "";
      el.style.transform = `translate(${x}px, ${y}px)`;
    }
    for (let k = n; k < this.pool.length; k++) this.pool[k].style.display = "none";
  }

  frame(t) {
    if (this.goal) {
      this.camera.position.lerp(this.goal.pos, 0.09);
      this.controls.target.lerp(this.goal.target, 0.09);
      if (this.camera.position.distanceTo(this.goal.pos) < 0.2) this.goal = null;
    }
    this.controls.update();
    const u = (t - this.sweep) / 4000;
    this.arch.frame(u < 1 ? u * 64 - 4 : -1);
    const wp = this.arch.wirePos, nw = this.arch.wires.geometry.drawRange.count / 2;
    const P = this.flow.geometry.attributes.position, K = this.flow.geometry.attributes.color;
    const ph = (t * 0.00025) % 1;
    let n = 0;
    for (let q = 0; q < nw && n + 2 <= FLOW; q++) {
      for (let h = 0; h < 2; h++) {
        const s = (ph * (1 + (q % 3) * 0.3) + h * 0.5 + q * 0.137) % 1, b = 6 * q, bright = 0.5 + 0.5 * Math.sin(s * Math.PI);
        P.array.set([wp[b] + (wp[b + 3] - wp[b]) * s, 0.3, wp[b + 2] + (wp[b + 5] - wp[b + 2]) * s], 3 * n);
        K.array.set([0.3 * bright, 0.75 * bright, bright], 3 * n++);
      }
    }
    P.needsUpdate = K.needsUpdate = true;
    this.flow.geometry.setDrawRange(0, n);
    this.renderer.render(this.scene, this.camera);
    this.hover();
    this.labels();
  }
}

export { fmt };
