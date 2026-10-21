import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Block } from "../../scatter/js/stage.js";
import { lut, fmt } from "../../scatter/js/colors.js";

THREE.ColorManagement.enabled = false;

const FOV = 40, BG = 0x06080c, THICK = 2.2, GAP = 4, DOM = 26, TILE = 32, HALO = 16;
const SIM_C = [-44, 24], MOD_C = [-13, 24], ARR_C = [-28, -17], FAR_C0 = [24, 29], FAR_C1 = [70, 29], KBASE = [36.5, 29];
const LAYER_NAMES = ["potential (U-Net output)", "far term (coarse global, learned K)", "local kernel potential (learned K)", "density (CIC scatter)"];
const LAYER_KEYS = ["phi", "far", "phiK", "rho"];
const GOLD = 0xffd36e, CYAN = 0x6fe0ff, ORANGE = 0xff9a3c, VIOLET = 0xc79bff, BLUE = 0x6fc3ff, GREEN = 0x62d6a4;
const sizeOf = (H) => 0.8 * (12 - 1.5 * (7 - Math.log2(H)));
const hue = (k) => new THREE.Color().setHSL(((k * 137.508) % 360) / 360, 0.62, 0.64);
const V = (x, y, z = 0) => new THREE.Vector3(x, y, z);

function paintRaw(block, k, arr, aw, scale, med, kind) {
  const p = block.planes[k], d = p.data, W = block.W, H = block.H, L = lut(kind), sh = aw === W ? 0 : Math.round(Math.log2(W / aw)), inv = 1 / (scale || 1);
  for (let y = 0; y < H; y++) {
    const ro = (y >> sh) * aw;
    for (let x = 0; x < W; x++) {
      const t = (arr[ro + (x >> sh)] - med) * inv, q = 3 * (((t < -1 ? -1 : t > 1 ? 1 : t) * 127.5 + 127.5) | 0), i = 4 * (y * W + x);
      d[i] = L[q]; d[i + 1] = L[q + 1]; d[i + 2] = L[q + 2];
    }
  }
  p.map.needsUpdate = true;
  block.src[k] = { arr, off: 0, s: scale, aw, med };
}

function layerScale(list, size, signed) {
  let best = 0, top = 0;
  const meds = [];
  for (const a of list) {
    const st = Math.max(1, Math.floor(size / 600)), s = [];
    for (let i = 0; i < size; i += st) s.push(a[i]);
    s.sort((x, y) => x - y);
    const med = signed ? s[s.length >> 1] : 0;
    meds.push(med);
    const dev = s.map((v) => Math.abs(v - med)).sort((x, y) => x - y);
    best = Math.max(best, dev[Math.floor(dev.length * 0.985)]);
    top = Math.max(top, dev[dev.length - 1]);
  }
  return { scale: Math.max(best, 0.3 * top, 1e-12), meds };
}

class Segments {
  constructor(parent, color, cap, dashed = false) {
    this.cap = cap; this.n = 0;
    this.pos = new Float32Array(cap * 6);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setDrawRange(0, 0);
    this.mat = dashed ? new THREE.LineDashedMaterial({ color, transparent: true, depthWrite: false, dashSize: 0.55, gapSize: 0.45 }) : new THREE.LineBasicMaterial({ color, transparent: true, depthWrite: false });
    this.mesh = new THREE.LineSegments(this.geo, this.mat);
    this.mesh.frustumCulled = false;
    this.dashed = dashed;
    parent.add(this.mesh);
  }

  begin() { this.n = 0; }

  add(ax, ay, az, bx, by, bz) {
    if (this.n >= this.cap) return;
    this.pos.set([ax, ay, az, bx, by, bz], this.n++ * 6);
  }

  head(ax, ay, az, bx, by, bz, h = 0.5) {
    const dx = bx - ax, dy = by - ay, l = Math.hypot(dx, dy);
    if (l < 1e-6) return;
    const ux = dx / l, uy = dy / l;
    for (const s of [-1, 1]) this.add(bx, by, bz, bx - h * (ux + 0.5 * s * -uy), by - h * (uy + 0.5 * s * ux), bz);
  }

  arrow(ax, ay, az, bx, by, bz, h = 0.5) {
    this.add(ax, ay, az, bx, by, bz);
    this.head(ax, ay, az, bx, by, bz, h);
  }

  end(alpha) {
    this.geo.attributes.position.needsUpdate = true;
    this.geo.setDrawRange(0, this.n * 2);
    this.mat.opacity = alpha;
    this.mesh.visible = alpha > 0.01 && this.n > 0;
    if (this.dashed && this.n) this.mesh.computeLineDistances();
  }
}

export async function createView({ canvas, labelsEl, tip, handlers }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setClearColor(BG);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 2000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false;
  controls.maxDistance = 600;
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const sun = new THREE.DirectionalLight(0xffffff, 0.8);
  sun.position.set(10, 30, 40);
  scene.add(sun);

  let W = null, dirty = true, raf = 0, active = false, hover = null, pending = null, anim = 0;
  const notes = new Map();
  const state = { nt: 0, layer: -1, channel: 0, inspect: 0, flags: { ghosts: true, fx: true, far: true }, links: true, arrows: false, overlay: true, frame: null, sim: null, out: null, tick: -1 };

  const NARROW_KEEP = new Set(["simT", "modT", "tarr", "farT"]);

  function note(key, text, p, cls = "", alpha = 1) {
    if (innerWidth <= 760 && !NARROW_KEEP.has(key)) alpha = 0;
    let n = notes.get(key);
    if (!n) {
      const el = document.createElement("div");
      el.className = "lab";
      labelsEl.appendChild(el);
      n = { el, text: "", cls: "", p: V(0, 0, 0) };
      notes.set(key, n);
    }
    if (n.text !== text) { n.el.innerHTML = text; n.text = text; }
    if (n.cls !== cls) { n.el.className = "lab " + cls; n.cls = cls; }
    n.p.copy(p); n.alpha = alpha; n.used = true;
  }

  function resize() {
    const w = innerWidth, h = innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    dirty = true;
  }
  addEventListener("resize", () => { resize(); if (active && !W.userMoved) focus(innerWidth <= 760 ? "sim" : "overview"); });
  controls.addEventListener("change", () => { dirty = true; });
  controls.addEventListener("start", () => { if (W) W.userMoved = true; });

  function build(nt) {
    if (W) {
      scene.remove(W.root, W.wall);
      for (const b of W.blocks.values()) b.group.traverse((o) => { if (o.material) { o.material.map?.dispose(); o.material.dispose(); } if (o.geometry) o.geometry.dispose(); });
      W.wall.traverse((o) => { if (o.material) o.material.dispose(); if (o.geometry) o.geometry.dispose(); });
      for (const n of notes.values()) n.el.remove();
      notes.clear();
    }
    const L = nt * TILE, root = new THREE.Group(), wall = new THREE.Group();
    root.rotation.x = Math.PI / 2;
    scene.add(root, wall);
    const blocks = new Map();
    const add = (id, o) => {
      const b = new Block(id, { ...o, pos: [o.x, o.out || 0, -o.y] });
      blocks.set(id, b);
      root.add(b.group);
      return b;
    };
    W = { nt, L, T: nt * nt, root, wall, blocks, userMoved: false, hlTile: -1 };
    const T = W.T, pitch = nt === 4 ? 9.2 : 17, tsize = nt === 4 ? 6.6 : 12.5, gap = nt === 4 ? 1.7 : 2.4;
    W.pitch = pitch; W.tsize = tsize; W.gap = gap;
    W.tileXY = (k) => { const tx = Math.floor(k / nt), ty = k % nt; return [ARR_C[0] + (tx - (nt - 1) / 2) * pitch, ARR_C[1] + (ty - (nt - 1) / 2) * pitch]; };
    add("sim", { x: SIM_C[0], y: SIM_C[1], kind: "dens", H: 8, size: DOM, title: "exact sim" });
    add("mod", { x: MOD_C[0], y: MOD_C[1], kind: "pot", H: nt * 64, size: DOM, title: "model potential", opacity: 0.95 });
    for (let k = 0; k < T; k++) {
      const [bx, by] = W.tileXY(k);
      add("t" + k, { x: bx, y: by, kind: "pot", kinds: ["act", "far", "mom", "dens"], H: 128, n: 4, size: tsize, gap, opacity: 0.92, title: "tile " + Math.floor(k / nt) + "," + (k % nt) });
    }
    const Gp = nt * 16 + 16, cs = Gp * 0.15;
    add("crho", { x: FAR_C0[0], y: FAR_C0[1], kind: "dens", H: Gp, size: cs, title: "coarse density" });
    add("cphi", { x: FAR_C1[0], y: FAR_C1[1], kind: "pot", H: Gp, size: cs, title: "coarse potential" });
    W.Gp = Gp;
    let x = 8;
    const yEnc = -9, yDec = -33;
    const h0 = add("h0", { x: x + sizeOf(128) / 2, y: yEnc, out: 0, kind: "act", H: 128, C: 32, thick: THICK, size: sizeOf(128), title: "input conv" });
    x += sizeOf(128);
    W.unet = { h0, enc: [], dec: [], xs: [] };
    for (let l = 0; l <= 5; l++) {
      const H = 128 >> l, s = sizeOf(H);
      x += GAP + s / 2;
      W.unet.xs[l] = x;
      W.unet.enc.push(add("enc" + l, { x, y: yEnc, out: -3 * l, kind: "act", H, C: 32, thick: THICK, size: s, title: "encoder " + l }));
      x += s / 2;
    }
    for (let l = 4; l >= 0; l--) W.unet.dec[l] = add("dec" + l, { x: W.unet.xs[l], y: yDec, out: -3 * l, kind: "act", H: 128 >> l, C: 32, thick: THICK, size: sizeOf(128 >> l), title: "decoder " + l });

    const lineMat = (c, o = 1) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: o, depthWrite: false });
    const line = (pts, c, o = 1, loop = false) => {
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      const l = loop ? new THREE.LineLoop(g, lineMat(c, o)) : new THREE.Line(g, lineMat(c, o));
      l.frustumCulled = false;
      wall.add(l);
      return l;
    };
    for (const c of [SIM_C, MOD_C]) {
      const seam = new Segments(wall, 0x56627a, 2 * nt + 2);
      for (let i = 1; i < nt; i++) {
        const o = -DOM / 2 + DOM * i / nt;
        seam.add(c[0] + o, c[1] - DOM / 2, 0.06, c[0] + o, c[1] + DOM / 2, 0.06);
        seam.add(c[0] - DOM / 2, c[1] + o, 0.06, c[0] + DOM / 2, c[1] + o, 0.06);
      }
      seam.end(0.9);
    }
    W.sel = [0, 1].map(() => ({ core: line([V(0, 0), V(1, 0), V(1, 1), V(0, 1)], 0xffffff, 0.9, true), halo: line([V(0, 0), V(1, 0), V(1, 1), V(0, 1)], 0xffffff, 0.35, true) }));
    W.tileCore = Array.from({ length: T }, (_, k) => {
      const [bx, by] = W.tileXY(k), h = tsize / 4, z = 0.05;
      return line([V(bx - h, by - h, z), V(bx + h, by - h, z), V(bx + h, by + h, z), V(bx - h, by + h, z)], hue(k).getHex(), 0.95, true);
    });

    const sphere = new THREE.SphereGeometry(1, 12, 8);
    const mk = (color, n, opacity = 1) => {
      const m = new THREE.InstancedMesh(sphere, new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 }), n);
      m.count = 0; m.frustumCulled = false;
      wall.add(m);
      return m;
    };
    W.simB = mk(ORANGE, 700);
    W.modB = mk(BLUE, 700);
    W.simG = mk(ORANGE, 700, 0.55);
    const ring = new THREE.RingGeometry(0.34, 0.46, 20);
    W.rings = new THREE.InstancedMesh(ring, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide }), 6000);
    W.rings.setColorAt(0, new THREE.Color());
    W.rings.count = 0; W.rings.frustumCulled = false;
    wall.add(W.rings);
    W.links = new Segments(wall, 0xe4eaf5, 1400);
    W.accSim = new Segments(wall, ORANGE, 2800);
    W.accMod = new Segments(wall, BLUE, 2800);
    W.flowBody = new Segments(wall, GOLD, 600, true);
    W.flowBodyH = new Segments(wall, GOLD, 1200);
    W.flowField = new Segments(wall, CYAN, 400, true);
    W.flowFieldH = new Segments(wall, CYAN, 800);
    W.flowFar = new Segments(wall, ORANGE, 400, true);
    W.flowRho = new Segments(wall, GREEN, 400, true);
    W.skip = new Segments(wall, VIOLET, 200);
    W.inout = new Segments(wall, VIOLET, 200, true);
    const sk = W.skip;
    const quad = (a, b, lift, n = 24) => new THREE.QuadraticBezierCurve3(a, V((a.x + b.x) / 2, (a.y + b.y) / 2, Math.max(a.z, b.z) + lift), b).getPoints(n);
    for (let l = 0; l < 5; l++) {
      const pts = quad(V(W.unet.xs[l], yEnc - sizeOf(128 >> l) / 2, -3 * l), V(W.unet.xs[l], yDec + sizeOf(128 >> l) / 2, -3 * l), 5 + (5 - l) * 1.2);
      for (let i = 0; i < pts.length - 1; i++) sk.add(pts[i].x, pts[i].y, pts[i].z, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
    }
    {
      const pts = quad(V(W.unet.xs[5], yEnc - sizeOf(4) / 2, -15), V(W.unet.xs[4], yDec + sizeOf(8) / 2, -12), 2);
      for (let i = 0; i < pts.length - 1; i++) sk.add(pts[i].x, pts[i].y, pts[i].z, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
    }
    sk.end(0.7);
    W.quad = quad;

    buildChart();
    W.out = null;
    dirty = true;
  }

  let chart = null;
  function buildChart() {
    if (!chart) return;
    const { curves } = chart, g = W.wall;
    const w = 18, h = 7, rmax = curves.r[curves.r.length - 1], sc = (f) => Math.sign(f) * Math.max(0, Math.min(1, (Math.log10(Math.abs(f) + 1e-12) + 6) / 6.2)) * h;
    const pts = (a) => Array.from(curves.r, (r, i) => V(KBASE[0] + r / rmax * w, KBASE[1] + sc(a[i])));
    const mk = (p, c, o = 1) => {
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: o, depthWrite: false }));
      l.frustumCulled = false;
      g.add(l);
    };
    mk(pts(curves.shipped), 0x8a7fb0, 0.9);
    mk(pts(curves.exact), 0xe4eaf5, 0.45);
    mk(pts(curves.learned), BLUE, 1);
    mk([V(KBASE[0], KBASE[1] - h), V(KBASE[0], KBASE[1] + h), V(KBASE[0], KBASE[1]), V(KBASE[0] + w, KBASE[1])], 0x56627a);
    const x64 = KBASE[0] + 64 / rmax * w;
    mk([V(x64, KBASE[1] - h), V(x64, KBASE[1] + h)], 0xff6b6b, 0.6);
    const lane = new Segments(g, 0x56627a, 12);
    const cr = W.blocks.get("crho"), cp = W.blocks.get("cphi");
    lane.arrow(FAR_C0[0] + cr.size / 2 + 0.6, FAR_C0[1], 0.1, KBASE[0] - 1.2, FAR_C0[1], 0.1, 0.6);
    lane.arrow(KBASE[0] + w + 1.2, FAR_C0[1], 0.1, FAR_C1[0] - cp.size / 2 - 0.6, FAR_C1[1], 0.1, 0.6);
    lane.end(0.9);
    W.chart = { w, h, rmax, x64 };
  }

  function focus(name) {
    const nt = W.nt, aspect = innerWidth / innerHeight, t = Math.tan(FOV / 2 * Math.PI / 180);
    const R = {
      overview: [13, -1, 150, 84],
      sim: [-28.5, 24, 64, 32],
      tiles: [ARR_C[0], ARR_C[1], nt === 4 ? 42 : 40, nt === 4 ? 42 : 40],
      far: [48, 26, 80, 28],
      unet: [46, -21, 88, 42],
    }[name] || [13, -1, 150, 84];
    const wide = innerWidth > 760, bot = wide ? 100 : 64, padL = wide ? 310 : 0, hpx = innerHeight - bot, wpx = innerWidth - padL, k = wide ? 1.08 : 1.04;
    const d = Math.max(R[3] / 2 / t * innerHeight / hpx, R[2] / 2 / (t * aspect) * innerWidth / wpx) * k, upp = 2 * d * t / innerHeight, cy = R[1] - bot / 2 * upp;
    R[0] -= padL / 2 * upp;
    controls.target.set(R[0], cy, 0);
    camera.position.set(R[0] + 0.1 * d, cy + 0.05 * d, d);
    controls.update();
    W.userMoved = false;
    dirty = true;
  }

  // ---------------------------------------------------------------- per-frame content
  const tileOfPoint = (px, py, c) => {
    const nt = W.nt, L = W.L, u = (px - c[0] + DOM / 2) / DOM * L, v = (py - c[1] + DOM / 2) / DOM * L;
    const tx = Math.floor(u / TILE), ty = Math.floor(v / TILE);
    return tx < 0 || ty < 0 || tx >= nt || ty >= nt ? -1 : tx * nt + ty;
  };

  function wallOf(c, x, y) { return [c[0] + x / W.L * DOM, c[1] + y / W.L * DOM]; }

  function paintAll() {
    const fr = state.frame, out = state.out, nt = W.nt, T = W.T;
    if (!fr || !out) return;
    const sc = {};
    for (const name of LAYER_KEYS) sc[name] = layerScale(out.tiles.map((t) => t[name]), name === "far" ? 1024 : 16384, name !== "rho");
    for (let k = 0; k < T; k++) {
      const b = W.blocks.get("t" + k), o = out.tiles[k];
      paintRaw(b, 0, o.phi, 128, sc.phi.scale, sc.phi.meds[k], "pot");
      paintRaw(b, 1, o.far, 32, sc.far.scale, sc.far.meds[k], "far");
      paintRaw(b, 2, o.phiK, 128, sc.phiK.scale, sc.phiK.meds[k], "mom");
      paintRaw(b, 3, o.rho, 128, sc.rho.scale, 0, "dens");
    }
    const N = nt * 64, glob = new Float32Array(N * N);
    for (let k = 0; k < T; k++) {
      const tx = Math.floor(k / nt), ty = k % nt, phi = out.tiles[k].phi;
      for (let y = 0; y < 64; y++) glob.set(phi.subarray((32 + y) * 128 + 32, (32 + y) * 128 + 96), (ty * 64 + y) * N + tx * 64);
    }
    const gs = layerScale([glob], N * N, true);
    paintRaw(W.blocks.get("mod"), 0, glob, N, gs.scale, gs.meds[0], "pot");
    W.glob = glob;
    const G = W.Gp;
    if (fr.coarse) {
      const a = layerScale([out.coarse.rho], G * G, false), b = layerScale([out.coarse.phi], G * G, true);
      paintRaw(W.blocks.get("crho"), 0, out.coarse.rho, G, a.scale, 0, "dens");
      paintRaw(W.blocks.get("cphi"), 0, out.coarse.phi, G, b.scale, b.meds[0], "pot");
    } else {
      out.coarse.rho.fill(0); out.coarse.phi.fill(0);
      paintRaw(W.blocks.get("crho"), 0, out.coarse.rho, G, 1, 0, "dens");
      paintRaw(W.blocks.get("cphi"), 0, out.coarse.phi, G, 1, 0, "pot");
    }
    const u = W.unet;
    if (fr.unet) {
      u.h0.setTensor(out.unet.inp);
      out.unet.enc.forEach((a, l) => u.enc[l].setTensor(a));
      out.unet.dec.forEach((a, l) => u.dec[l].setTensor(a));
      u.tile = fr.unet.tile;
    }
    W.painted = true;
  }

  function bodyMatrix(m, i, x, y, r) {
    const mat = new THREE.Matrix4();
    mat.compose(V(x, y, 0.35), new THREE.Quaternion(), V(r, r, r));
    m.setMatrixAt(i, mat);
  }

  function compose() {
    const nt = W.nt, T = W.T, fr = state.frame, sim = state.sim, fl = state.flags, narrow = innerWidth <= 760;
    const blocks = W.blocks;
    for (const b of blocks.values()) { b.op = 1; b.hl = 0; }
    for (const k of [0, 1]) {
      const c = k ? MOD_C : SIM_C, s = W.sel[k], tk = state.inspect, tx = Math.floor(tk / nt), ty = tk % nt, h = DOM / nt, x0 = c[0] - DOM / 2 + tx * h, y0 = c[1] - DOM / 2 + ty * h, e = HALO / TILE * h;
      const setLoop = (l, ax, ay, bx, by) => {
        const p = l.geometry.attributes.position;
        p.setXYZ(0, ax, ay, 0.08); p.setXYZ(1, bx, ay, 0.08); p.setXYZ(2, bx, by, 0.08); p.setXYZ(3, ax, by, 0.08);
        p.needsUpdate = true;
      };
      s.core.material.color.copy(hue(tk));
      s.halo.material.color.copy(hue(tk));
      setLoop(s.core, x0, y0, x0 + h, y0 + h);
      setLoop(s.halo, x0 - e, y0 - e, x0 + h + e, y0 + h + e);
    }
    const lyr = state.layer;
    for (let k = 0; k < T; k++) {
      const b = blocks.get("t" + k);
      b.planes.forEach((p, i) => { p.mesh.visible = lyr < 0 || lyr === i; });
      if (!fl.far) b.planes[1].mesh.material.opacity = 0;
      if (k === state.inspect) b.hl = 0.9;
      if (hover && hover.tile === k) b.hl = Math.max(b.hl, 0.5);
      W.tileCore[k].visible = lyr < 0 || lyr > 0;
    }
    if (!fl.far) { blocks.get("crho").op = 0.15; blocks.get("cphi").op = 0.15; }
    blocks.get("mod").hl = hover && hover.block === "mod" ? 0.4 : 0;
    const U = W.unet;
    for (const b of [U.h0, ...U.enc, ...U.dec]) if (b) b.op = fr && fr.unet ? 1 : 0.3;
    for (const b of blocks.values()) b.apply();
    if (!fl.far) for (let k = 0; k < T; k++) blocks.get("t" + k).planes[1].mesh.material.opacity = 0.08;

    // bodies
    const n = fr ? fr.n : 0, ms = fr ? fr.pos : null, ss = sim ? sim.pos : null;
    const rb = nt === 4 ? 0.16 : 0.2;
    W.modB.count = W.simB.count = W.simG.count = ss && ms ? n : 0;
    W.links.begin(); W.accSim.begin(); W.accMod.begin();
    if (ss && ms) {
      for (let i = 0; i < n; i++) {
        const [mx, my] = wallOf(MOD_C, ms[2 * i], ms[2 * i + 1]), [sx, sy] = wallOf(SIM_C, ss[2 * i], ss[2 * i + 1]), [gx, gy] = wallOf(MOD_C, ss[2 * i], ss[2 * i + 1]);
        bodyMatrix(W.modB, i, mx, my, rb);
        bodyMatrix(W.simB, i, sx, sy, rb);
        bodyMatrix(W.simG, i, gx, gy, rb * 0.7);
        if (state.links && state.overlay) W.links.add(mx, my, 0.3, gx, gy, 0.3);
        if (state.arrows) {
          const sa = 2.2 * DOM / W.L, ma = fr.acc, ta = sim.acc;
          W.accMod.arrow(mx, my, 0.3, mx + ma[2 * i] * sa, my + ma[2 * i + 1] * sa, 0.3, 0.3);
          W.accSim.arrow(sx, sy, 0.3, sx + ta[2 * i] * sa, sy + ta[2 * i + 1] * sa, 0.3, 0.3);
        }
      }
      W.modB.instanceMatrix.needsUpdate = W.simB.instanceMatrix.needsUpdate = W.simG.instanceMatrix.needsUpdate = true;
      if (hover && hover.type === "body") {
        const i = hover.i, [mx, my] = wallOf(MOD_C, ms[2 * i], ms[2 * i + 1]), [sx, sy] = wallOf(SIM_C, ss[2 * i], ss[2 * i + 1]);
        W.links.add(mx - 0.7, my, 0.4, mx + 0.7, my, 0.4); W.links.add(mx, my - 0.7, 0.4, mx, my + 0.7, 0.4);
        W.links.add(sx - 0.7, sy, 0.4, sx + 0.7, sy, 0.4); W.links.add(sx, sy - 0.7, 0.4, sx, sy + 0.7, 0.4);
      }
    }
    W.simG.visible = state.overlay;
    W.links.end(0.5); W.accSim.end(0.9); W.accMod.end(0.9);

    // ghost rings: one ring per (body, receiving tile), coloured by the receiving tile
    let nr = 0;
    if (fr && ms && fl.ghosts) {
      const col = new THREE.Color();
      for (let k = 0; k < T && nr < 6000; k++) {
        const c = hue(k);
        for (let j = fr.ghOff[k]; j < fr.ghOff[k + 1] && nr < 6000; j++) {
          const i = fr.gh[j], [mx, my] = wallOf(MOD_C, ms[2 * i], ms[2 * i + 1]), mat = new THREE.Matrix4();
          const r = 0.55 + 0.12 * (k % 3);
          mat.compose(V(mx, my, 0.3), new THREE.Quaternion(), V(r, r, 1));
          W.rings.setMatrixAt(nr, mat);
          W.rings.setColorAt(nr, col.copy(c));
          nr++;
        }
      }
      W.rings.instanceMatrix.needsUpdate = true;
      if (W.rings.instanceColor) W.rings.instanceColor.needsUpdate = true;
    }
    W.rings.count = nr;

    // flows between tile slabs: ghost bodies (gold), halo potential (cyan); far term and density to the coarse grid (orange, green)
    W.flowBody.begin(); W.flowBodyH.begin(); W.flowField.begin(); W.flowFieldH.begin(); W.flowFar.begin(); W.flowRho.begin(); W.inout.begin();
    const counts = new Map();
    if (fr && fl.ghosts) {
      for (let k = 0; k < T; k++) for (let j = fr.ghOff[k]; j < fr.ghOff[k + 1]; j++) {
        const key = fr.tid[fr.gh[j]] * T + k;
        counts.set(key, (counts.get(key) || 0) + 1);
      }
    }
    const half = W.tsize / 2, off = 0.5;
    let cmax = 1;
    for (const c of counts.values()) cmax = Math.max(cmax, c);
    W.exch = counts;
    for (let a = 0; a < T; a++) {
      const [ax, ay] = W.tileXY(a);
      for (let b = 0; b < T; b++) {
        if (a === b) continue;
        const dx = Math.floor(b / nt) - Math.floor(a / nt), dy = (b % nt) - (a % nt);
        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) continue;
        const diag = dx !== 0 && dy !== 0;
        const sx = Math.sign(dx), sy = Math.sign(dy);
        const p0 = diag ? [ax + sx * half * 0.92, ay + sy * half * 0.92] : [ax + sx * (half + 0.15), ay + sy * (half + 0.15)];
        const p1 = diag ? [ax + sx * (W.pitch - half * 0.92), ay + sy * (W.pitch - half * 0.92)] : [ax + sx * (W.pitch - half - 0.15), ay + sy * (W.pitch - half - 0.15)];
        const perp = diag ? [0, 0] : [-sy * off * 0.5, sx * off * 0.5];
        const c = counts.get(a * T + b) || 0;
        if (c > 0) {
          const lines = 1 + Math.round(2 * c / cmax);
          for (let q = 0; q < lines; q++) {
            const o = (q - (lines - 1) / 2) * 0.35;
            const pp = diag ? [o, -o] : [-sy * (off * 0.35 + o), sx * (off * 0.35 + o)];
            W.flowBody.add(p0[0] + pp[0], p0[1] + pp[1], 0.4, p1[0] + pp[0], p1[1] + pp[1], 0.4);
            if (q === 0) W.flowBodyH.head(p0[0] + pp[0], p0[1] + pp[1], 0.4, p1[0] + pp[0], p1[1] + pp[1], 0.4, 0.45);
          }
        }
        if (fl.fx && !diag) {
          const pp = [sy * off * 0.5, -sx * off * 0.5];
          W.flowField.add(p0[0] + pp[0], p0[1] + pp[1], 0.4, p1[0] + pp[0], p1[1] + pp[1], 0.4);
          W.flowFieldH.head(p0[0] + pp[0], p0[1] + pp[1], 0.4, p1[0] + pp[0], p1[1] + pp[1], 0.4, 0.4);
        }
      }
    }
    if (fl.far) {
      const [cx, cy] = FAR_C1, [rx, ry] = FAR_C0;
      for (let k = 0; k < T; k++) {
        const [bx, by] = W.tileXY(k), z = -W.gap;
        const pts = W.quad(V(cx - blocks.get("cphi").size / 2, cy - blocks.get("cphi").size / 2, 0), V(bx + half * 0.8, by + half * 0.8, z), 10 + k % 4 * 1.5, 18);
        for (let i = 0; i < pts.length - 1; i++) W.flowFar.add(pts[i].x, pts[i].y, pts[i].z, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
        const q = W.quad(V(bx + half * 0.8, by + half, -3 * W.gap), V(rx - blocks.get("crho").size / 2, ry - blocks.get("crho").size / 2, 0), 6 + k % 4 * 1.2, 18);
        for (let i = 0; i < q.length - 1; i++) W.flowRho.add(q[i].x, q[i].y, q[i].z, q[i + 1].x, q[i + 1].y, q[i + 1].z);
      }
      W.flowFar.end(0.2); W.flowRho.end(0.2);
    } else { W.flowFar.end(0); W.flowRho.end(0); }
    {
      const [bx, by] = W.tileXY(state.inspect), h0 = W.unet.h0, g = W.gap;
      const a = W.quad(V(bx + half, by, -3 * g), V(h0.group.position.x - h0.size / 2, h0.group.position.z * -1, 0), 4, 22);
      for (let i = 0; i < a.length - 1; i++) W.inout.add(a[i].x, a[i].y, a[i].z, a[i + 1].x, a[i + 1].y, a[i + 1].z);
      const d0 = W.unet.dec[0], b2 = W.quad(V(d0.group.position.x, -d0.group.position.z - d0.size / 2, 0), V(bx + half * 0.3, by - half, 0), 3, 22);
      for (let i = 0; i < b2.length - 1; i++) W.inout.add(b2[i].x, b2[i].y, b2[i].z, b2[i + 1].x, b2[i + 1].y, b2[i + 1].z);
    }
    W.flowBody.end(fl.ghosts ? 0.95 : 0); W.flowBodyH.end(fl.ghosts ? 0.95 : 0); W.flowField.end(fl.fx ? 0.9 : 0); W.flowFieldH.end(fl.fx ? 0.9 : 0); W.inout.end(0.55);
    anim = (fl.ghosts && counts.size) || fl.fx ? 1 : 0;

    // labels
    const exact = "exact", learned = "learned";
    note("simT", "SIM &middot; exact, all pairs", V(SIM_C[0], SIM_C[1] + DOM / 2 + 0.6), exact);
    note("modT", `MODEL &middot; ${nt}x${nt} tiled`, V(MOD_C[0], MOD_C[1] + DOM / 2 + 0.6), learned);
    if (fr) {
      note("simK", "tick " + fr.tick, V(SIM_C[0], SIM_C[1] - DOM / 2 - 0.4), "");
      note("modK", "tick " + fr.tick + " &middot; outline: inspected tile", V(MOD_C[0], MOD_C[1] - DOM / 2 - 0.4), "");
    }
    for (let k = 0; k < T; k++) {
      const [bx, by] = W.tileXY(k), tx = Math.floor(k / nt), ty = k % nt;
      if (fr && nt === 2) note("tl" + k, `${tx},${ty} &middot; ${fr.ghOff[k + 1] - fr.ghOff[k]} ghosts in`, V(bx, by + W.tsize / 2 + 0.2), "");
      else if (!narrow) note("tl" + k, `${tx},${ty}`, V(bx, by + W.tsize / 2 + 0.1, 0), "");
      note("tm" + k, nt === 4 ? "" : "", V(bx, by, 0), "", 0);
    }
    {
      const [bx, by] = W.tileXY(0), h = W.tsize / 2;
      LAYER_NAMES.forEach((nm, i) => {
        const cls = i === 3 ? exact : learned;
        note("ly" + i, nm + (i === 3 ? " &middot; exact op" : ""), V(bx - h - 2.6, by - h + 0.2 + i * 1.7, -i * W.gap), cls + " left", 0.9);
      });
    }
    note("tarr", `${W.T} tiles &middot; core 32 + halo 16 &middot; one 128x128 U-Net, identical weights in every tile`, V(ARR_C[0], ARR_C[1] + (nt - 1) / 2 * W.pitch + W.tsize / 2 + 2.2), learned);
    note("crt", "coarse density, cell 2 (exact)", V(FAR_C0[0], FAR_C0[1] + blocks.get("crho").size / 2 + 0.5), exact, fl.far ? 1 : 0.4);
    note("cpt", "coarse potential (learned K)", V(FAR_C1[0], FAR_C1[1] + blocks.get("cphi").size / 2 + 0.5), learned, fl.far ? 1 : 0.4);
    note("kt", "far kernel force F(r), log scale (learned)", V(KBASE[0] + 9, KBASE[1] + 9.2), learned, fl.far ? 1 : 0.5);
    note("kr", "r = 100", V(KBASE[0] + 18, KBASE[1] - 0.2), "", 0.7);
    note("k0", "r = 0", V(KBASE[0] + 1.2, KBASE[1] - 0.2), "", 0.7);
    note("kp", "F = +1 (attractive)", V(KBASE[0] - 3.5, KBASE[1] + 7.6), "", 0.7);
    note("km", "F = -1 (repulsive)", V(KBASE[0] - 3.5, KBASE[1] - 6.4), "", 0.7);
    if (W.chart) note("k64", "r = 64", V(W.chart.x64, KBASE[1] - 7.6), "k2", 0.9);
    note("kl", "blue: learned R1b kernel", V(KBASE[0] + 9, KBASE[1] - 9.0), "learned", fl.far ? 1 : 0.5);
    note("ke", "white: exact softened law (reference, not used)", V(KBASE[0] + 9, KBASE[1] - 10.6), "", 0.85);
    note("ks", "violet: shipped kmlp, repulsive beyond r = 64", V(KBASE[0] + 9, KBASE[1] - 12.2), "k1", 0.9);
    note("farT", "far term: mass outside a tile's box, via the coarse grid (orange arcs)", V((FAR_C0[0] + FAR_C1[0]) / 2, FAR_C0[1] + 11.5), fl.far ? "" : "dim", 0.95);
    const unetTile = fr && fr.unet ? fr.unet.tile : state.inspect, ux = unetTile;
    note("ut", `U-Net of tile ${Math.floor(ux / nt)},${ux % nt} &middot; same weights in all ${T} tiles (click a tile) &middot; e = encoder, d = decoder`, V(46, 3.6), learned);
    for (const [id, b] of [["h0", U.h0], ...U.enc.map((b, l) => ["enc" + l, b]), ...U.dec.map((b, l) => ["dec" + l, b])]) {
      if (!b) continue;
      note("b" + id, id === "h0" ? "in conv &middot; 128&sup2;" : (id.startsWith("enc") ? "e" : "d") + id.slice(3) + " &middot; " + b.H + "&sup2;", V(b.group.position.x, -b.group.position.z + b.sizeY / 2 + 0.4, b.group.position.y), "", narrow ? 0 : 0.9);
    }
    note("inl", "in: density, momentum x2, kernel accel x2, previous potential", V(W.unet.h0.group.position.x + 11, -9 - 6.4), "", 0.8);
    for (const n of notes.values()) if (!n.used) n.alpha = 0;
    for (const n of notes.values()) n.used = false;
    dirty = true;
  }

  function render() {
    if (!dirty && !anim) return;
    const doAnim = anim && !dirty;
    dirty = false;
    if (anim) for (const s of [W.flowBody, W.flowField]) s.mat.dashOffset -= 0.04;
    renderer.render(scene, camera);
    const w = innerWidth, h = innerHeight, v = V(0, 0, 0);
    for (const n of notes.values()) {
      v.copy(n.p).project(camera);
      const on = n.alpha > 0.02 && v.z < 1 && Math.abs(v.x) < 1.15 && Math.abs(v.y) < 1.15;
      n.el.style.opacity = on ? n.alpha : 0;
      n.el.style.visibility = on ? "visible" : "hidden";
      if (on) n.el.style.transform = (n.cls.includes("left") ? "translate(-100%,-50%)" : "translate(-50%,-100%)") + " translate(" + ((v.x + 1) * w / 2).toFixed(1) + "px," + ((1 - v.y) * h / 2).toFixed(1) + "px)";
    }
    void doAnim;
  }

  function loop() {
    if (!active) return;
    raf = requestAnimationFrame(loop);
    if (W.needCompose) { W.needCompose = false; compose(); }
    render();
  }

  // ---------------------------------------------------------------- picking
  const ray = new THREE.Raycaster();
  function pick(cx, cy) {
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((cx - r.left) / r.width * 2 - 1, -((cy - r.top) / r.height * 2 - 1)), camera);
    const live = [];
    for (const b of W.blocks.values()) {
      if (b.op <= 0.3 || !b.group.visible) continue;
      b.group.updateWorldMatrix(true, true);
      for (const m of b.all && b.allPlanes ? b.allPlanes : b.planes.map((p) => p.mesh)) if (m.visible) live.push(m);
    }
    W.modB.boundingSphere = W.simB.boundingSphere = null;
    const hits = ray.intersectObjects([...live, W.modB, W.simB], false);
    for (const h of hits) {
      if (h.object === W.modB || h.object === W.simB) return { type: "body", i: h.instanceId, side: h.object === W.modB ? "model" : "sim" };
      const { block, k, chan } = h.object.userData, W2 = block.W, H2 = block.H;
      const ix = Math.min(W2 - 1, Math.floor(h.uv.x * W2)), iy = Math.min(H2 - 1, Math.floor(h.uv.y * H2));
      if (chan !== undefined) {
        let best = null;
        for (const g of hits) {
          if (g.object.userData.block !== block || g.object.userData.chan === undefined) continue;
          const jx = Math.min(W2 - 1, Math.floor(g.uv.x * W2)), jy = Math.min(H2 - 1, Math.floor(g.uv.y * H2)), v = block.valueAtChan(g.object.userData.chan, jx, jy);
          if (!best || Math.abs(v) > Math.abs(best.v)) best = { type: "cell", block: block.id, b: block, k: 0, ix: jx, iy: jy, v, chan: g.object.userData.chan };
        }
        return best;
      }
      const tile = /^t(\d+)$/.exec(block.id);
      return { type: "cell", block: block.id, b: block, k, ix, iy, v: block.valueAt(k, ix, iy), tile: tile ? +tile[1] : block.id === "mod" ? tileOfPoint(h.point.x, h.point.y, MOD_C) : block.id === "sim" ? tileOfPoint(h.point.x, h.point.y, SIM_C) : -1 };
    }
    return null;
  }

  function tipHtml(p) {
    const fr = state.frame, sim = state.sim, nt = W.nt;
    if (!p) return "";
    if (p.type === "body") {
      const i = p.i, mp = fr.pos, sp = sim.pos, e = Math.hypot(mp[2 * i] - sp[2 * i], mp[2 * i + 1] - sp[2 * i + 1]);
      const t = fr.tid[i];
      return `body ${i} &middot; owned by tile ${Math.floor(t / nt)},${t % nt}<br>model pos (${mp[2 * i].toFixed(2)}, ${mp[2 * i + 1].toFixed(2)}) vel (${fr.vel[2 * i].toFixed(2)}, ${fr.vel[2 * i + 1].toFixed(2)})<br>sim pos (${sp[2 * i].toFixed(2)}, ${sp[2 * i + 1].toFixed(2)})<br>position error ${e.toFixed(3)}<br>accel model (${fr.acc[2 * i].toFixed(3)}, ${fr.acc[2 * i + 1].toFixed(3)}) sim (${sim.acc[2 * i].toFixed(3)}, ${sim.acc[2 * i + 1].toFixed(3)})`;
    }
    const b = p.b, v = p.v === null || p.v === undefined ? "-" : fmt(p.v);
    const m = /^t(\d+)$/.exec(p.block);
    if (m) {
      const k = +m[1], tx = Math.floor(k / nt), ty = k % nt, W2 = b.W, cx = W.L * 0 + (-nt * TILE / 2 + (tx + 0.5) * TILE), cy = -nt * TILE / 2 + (ty + 0.5) * TILE;
      const sx = cx + (p.ix + 0.5) * 0.5 - 32, sy = cy + (p.iy + 0.5) * 0.5 - 32, core = Math.abs(sx - cx) < TILE / 2 && Math.abs(sy - cy) < TILE / 2;
      const layer = LAYER_NAMES[p.k];
      return `tile ${tx},${ty} &middot; ${layer}<br>${p.k === 1 ? "coarse cell " + (p.ix >> 2) + "," + (p.iy >> 2) + " of 32" : "cell " + p.ix + "," + p.iy + " of " + W2}, sim (${sx.toFixed(1)}, ${sy.toFixed(1)}), ${core ? "core" : "halo (belongs to a neighbour)"}<br>value ${v}`;
    }
    if (p.block === "mod") {
      const N = nt * 64, sx = (p.ix + 0.5) * 0.5 - W.L / 2, sy = (p.iy + 0.5) * 0.5 - W.L / 2, t = Math.floor((sx + W.L / 2) / TILE) * nt + Math.floor((sy + W.L / 2) / TILE);
      return `model potential (tile cores joined)<br>cell ${p.ix},${p.iy} of ${N}, sim (${sx.toFixed(1)}, ${sy.toFixed(1)}), tile ${Math.floor(t / nt)},${t % nt}<br>value ${v}`;
    }
    if (p.block === "sim") return "exact sim: no field, every pair is summed directly";
    if (p.block === "crho" || p.block === "cphi") return `${b.title} &middot; coarse cell ${p.ix},${p.iy} (2 units)<br>value ${v}`;
    const ch = p.chan !== undefined ? ` ch ${p.chan} (strongest on the ray)` : b.thick ? ` ch ${b.chan}` : "";
    return `${b.title}${ch}<br>cell ${p.ix}, ${p.iy} of ${b.W}<br>value ${v}`;
  }

  function showTip(p, x, y) {
    if (!p) { tip.hidden = true; return; }
    tip.hidden = false;
    tip.innerHTML = tipHtml(p);
    tip.style.left = Math.min(innerWidth - 300, x + 14) + "px";
    tip.style.top = Math.min(innerHeight - 100, y + 14) + "px";
  }

  const hkey = (q) => (q ? (q.type === "body" ? "b" + q.side + q.i : q.block + q.k + ":" + q.ix + "," + q.iy) : "");
  canvas.addEventListener("pointermove", (e) => {
    if (e.buttons || !W || !state.frame) return;
    if (!pending) requestAnimationFrame(() => {
      const [x, y] = pending; pending = null;
      const p = pick(x, y);
      if (hkey(p) !== hkey(hover)) { hover = p; W.needCompose = true; }
      showTip(p, x, y);
    });
    pending = [e.clientX, e.clientY];
  });
  canvas.addEventListener("pointerleave", () => { hover = null; tip.hidden = true; if (W) W.needCompose = true; });
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener("pointerup", (e) => {
    if (!down || !W || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const p = pick(e.clientX, e.clientY);
    if (e.pointerType !== "mouse") { hover = p; showTip(p, e.clientX, e.clientY); W.needCompose = true; }
    if (p && p.tile >= 0 && p.tile !== undefined) handlers.inspect(p.tile);
  });

  return {
    build(nt) { build(nt); state.nt = nt; if (state.frame) W.needCompose = true; if (chart && !W.chart) buildChart(); },
    setChart(curves) { chart = { curves }; if (W && !W.chart) buildChart(); },
    show(on) {
      active = on;
      canvas.hidden = labelsEl.hidden = !on;
      if (!on) { cancelAnimationFrame(raf); tip.hidden = true; return; }
      resize();
      focus(innerWidth <= 760 ? "sim" : "overview");
      W.needCompose = true;
      raf = requestAnimationFrame(loop);
    },
    setFrame(fr, sim, out) {
      state.frame = fr; state.sim = sim; state.out = out;
      paintAll();
      W.needCompose = true;
    },
    setSim(sim) { state.sim = sim; W.needCompose = true; },
    setFlags(f) { state.flags = { ...f }; W.needCompose = true; },
    pose(target, pos) { controls.target.set(...target); camera.position.set(...pos); controls.update(); W.userMoved = true; dirty = true; },
    setInspect(k) { state.inspect = k; W.needCompose = true; },
    setLayer(l) { state.layer = l; W.needCompose = true; },
    setChannel(c) {
      for (const b of [W.unet.h0, ...W.unet.enc, ...W.unet.dec]) if (b) b.setChannel(c);
      W.needCompose = true;
    },
    setOptions(o) { Object.assign(state, o); W.needCompose = true; },
    focus,
    get exch() { return W && W.exch; },
    get ctx() { return { W, state, camera, controls }; },
  };
}
