import * as THREE from "three";
import { sl, fmt } from "./colors.js";
import { bodyCorners } from "./cic.js";

export const LEVELS = 5;
export const sizeOf = (H) => 12 - 1.5 * (7 - Math.log2(H));
const THICK = 2.2, GAP = 4, ZD = 22;

export function buildLayout(stage) {
  const pos = {};
  const place = (id, o, x, y, z) => { pos[id] = new THREE.Vector3(x, y, z); return stage.add(id, { ...o, pos: [x, y, z] }); };

  place("input", { kind: "dens", kinds: ["dens", "mom", "mom", "far", "far", "pot"], H: 128, n: 6, size: 24, gap: 0.9, opacity: 0.85, title: "input stack" }, 0, 0, 0);
  place("kw", { kind: "wgt", H: 64, size: 5, title: "kmlp layer 2 weights" }, -24, 0, -30);
  place("kring", { kind: "far", H: 64, size: 8, title: "kernel K on the grid" }, -14, 0, -30);
  place("phik", { kind: "pot", H: 128, size: 12, title: "far-field potential" }, 2, 0, -30);
  place("ak", { kind: "far", H: 128, n: 2, size: 12, gap: 0.9, title: "far-field accel" }, 18, 0, -30);

  let x = 22;
  const h0 = place("h0", { kind: "act", H: 128, C: 32, thick: THICK, size: sizeOf(128), title: "input conv" }, x, 0, 0);
  x += sizeOf(128) / 2;
  const enc = [], dec = [], xs = [];
  for (let l = 0; l <= LEVELS; l++) {
    const H = 128 >> l, s = sizeOf(H);
    x += GAP + s / 2;
    xs[l] = x;
    place("enc" + l, { kind: "act", H, C: 32, thick: THICK, size: s, title: "encoder " + l }, x, -3 * l, 0);
    enc.push("enc" + l);
    x += s / 2;
  }
  for (let l = LEVELS - 1; l >= 0; l--) {
    const H = 128 >> l;
    place("dec" + l, { kind: "act", H, C: 32, thick: THICK, size: sizeOf(H), title: "decoder " + l }, xs[l], -3 * l, ZD);
    dec.push("dec" + l);
  }
  place("phi", { kind: "pot", H: 128, size: 12, title: "potential" }, 22, 0, ZD);
  place("grad", { kind: "far", H: 128, n: 2, size: 12, gap: 0.9, title: "-grad(phi)" }, 6, 0, ZD);
  place("agrid", { kind: "far", H: 128, n: 2, size: 12, gap: 0.9, title: "grid acceleration" }, -10, 0, ZD);
  place("wconv", { kind: "wgt", H: 96, W: 192, size: 9.6, sizeY: 4.8, rest: 0, title: "conv weights" }, 0, 6, 0);
  stage.block("wconv").sub = { w: 192, h: 96 };

  const top = (id, dy = 0) => pos[id].clone().add(new THREE.Vector3(0, dy, 0));
  const edge = (id, dz) => pos[id].clone().add(new THREE.Vector3(0, 0.2, dz));
  stage.arc("recur", top("phi", 0.3), new THREE.Vector3(0, -4.5, 12.5), 12, 0xe8c85a);
  stage.arc("gatherArc", top("agrid", 0.3), new THREE.Vector3(0, 0.4, 12.5), 6, 0x62d6a4);
  for (let l = 0; l < LEVELS; l++) stage.arc("skip" + l, top("enc" + l, THICK / 2), top("dec" + l, THICK / 2), 4 + (LEVELS - l) * 1.3, 0xc79bff);
  stage.arc("down", edge("enc" + LEVELS, 0), edge("dec" + (LEVELS - 1), 0), 2, 0xc79bff, 16);
  stage.arc("rhoK", top("input", 0.2).add(new THREE.Vector3(0, 0, -12)), top("phik", 0.2).add(new THREE.Vector3(-6, 0, 6)), 3, 0x62d6a4, 20);
  stage.arc("akIn3", top("ak", 0.2).add(new THREE.Vector3(-6, 0, 6)), new THREE.Vector3(6, -2.7, -12), 5, 0xff9a3c, 20);
  stage.arc("akIn4", top("ak", -0.7).add(new THREE.Vector3(-6, 0, 6)), new THREE.Vector3(9, -3.6, -12), 7, 0xff9a3c, 20);
  stage.arc("inH0", new THREE.Vector3(12, -5.4, 0), top("h0", -1).add(new THREE.Vector3(-6, 0, 0)), 3, 0xc79bff, 20);
  stage.arc("encPhi", top("dec0", 0).add(new THREE.Vector3(-6, 0, 0)), top("phi", 0).add(new THREE.Vector3(6, 0, 0)), 2, 0xc79bff, 20);
  stage.arc("phiGrad", top("phi", 0).add(new THREE.Vector3(-6, 0, 0)), top("grad", 0).add(new THREE.Vector3(6, 0, 0)), 1.5, 0xe8c85a, 20);
  stage.arc("gradAgrid", top("grad", 0).add(new THREE.Vector3(-6, 0, 0)), top("agrid", 0).add(new THREE.Vector3(6, 0, 0)), 1.5, 0xff9a3c, 20);
  stage.arc("akAgrid", top("ak", 0.2).add(new THREE.Vector3(6, 0, 6)), top("agrid", 0.4).add(new THREE.Vector3(0, 0, -6)), 12, 0xff9a3c, 30);

  const arcs = ["recur", "gatherArc", "down", "rhoK", "akIn3", "akIn4", "inH0", "encPhi", "phiGrad", "gradAgrid", "akAgrid", ...Array.from({ length: LEVELS }, (_, l) => "skip" + l)];
  return { pos, enc, dec, xs, arcs, THICK };
}

export function buildCharts(stage, curve, pair) {
  const base = new THREE.Vector3(-44, 0, -30);
  const W = 14, Hc = 5;
  let kmax = 0;
  for (let i = 0; i < curve.r.length; i++) kmax = Math.max(kmax, Math.abs(curve.raw[i]), Math.abs(curve.tap[i]));
  const sc = (k) => sl(k, kmax / 100) * Hc;
  const rmax = curve.r[curve.r.length - 1];
  const line = (a) => Array.from(curve.r, (r, i) => new THREE.Vector3(base.x + r / rmax * W, base.y + sc(a[i]), base.z));
  stage.addCurve("kraw", line(curve.raw), 0x8a7fb0);
  stage.addCurve("ktap", line(curve.tap), 0xff9a3c);
  stage.addCurve("kaxis", [new THREE.Vector3(base.x, base.y - Hc, base.z), new THREE.Vector3(base.x, base.y + Hc, base.z), new THREE.Vector3(base.x, base.y, base.z), new THREE.Vector3(base.x + W, base.y, base.z)], 0x56627a);
  const pb = new THREE.Vector3(-9, 1.5, 14.5), pw = 8, ph = 2.2;
  let pmax = 0;
  for (const v of pair.pair) pmax = Math.max(pmax, Math.abs(v));
  const pr = pair.r[pair.r.length - 1];
  stage.addCurve("pcurve", Array.from(pair.r, (r, i) => new THREE.Vector3(pb.x + r / pr * pw, pb.y + pair.pair[i] / (pmax || 1) * ph, pb.z)), 0xffd36e);
  stage.addCurve("paxis", [new THREE.Vector3(pb.x, pb.y - ph, pb.z), new THREE.Vector3(pb.x, pb.y + ph, pb.z), new THREE.Vector3(pb.x, pb.y, pb.z), new THREE.Vector3(pb.x + pw, pb.y, pb.z)], 0x56627a);
  return { kbase: base, kW: W, kH: Hc, kmax, rmax, pbase: pb, pw, ph, pmax, pr };
}

const GG = 128 * 128;

export function paintTrace(stage, tr) {
  const b = (id) => stage.block(id);
  for (let k = 0; k < 6; k++) b("input").paint(k, tr.input, k * GG);
  b("phik").paint(0, tr.phiK, 0, 0, true);
  b("ak").paint(0, tr.input, 3 * GG); b("ak").paint(1, tr.input, 4 * GG);
  b("h0").setTensor(tr.inp);
  tr.enc.forEach((a, l) => b("enc" + l).setTensor(a));
  tr.dec.forEach((a, l) => b("dec" + l).setTensor(a));
  b("phi").paint(0, tr.phi, 0, 0, true);
  b("grad").paint(0, tr.gradPhi[0]); b("grad").paint(1, tr.gradPhi[1]);
  b("agrid").paint(0, tr.aGrid[0]); b("agrid").paint(1, tr.aGrid[1]);
}

export function paintKernel(stage, c, w) {
  const kr = new Float32Array(64 * 64);
  let kmax = 0;
  for (let i = 0; i < c.r.length; i++) kmax = Math.max(kmax, Math.abs(c.tap[i]));
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const r = Math.hypot(x - 32, y - 32) * 0.5, f = Math.min(c.r.length - 1, Math.max(0, r / c.r[c.r.length - 1] * c.r.length - 1)), i0 = Math.floor(f);
      const v = c.tap[i0] + (c.tap[Math.min(i0 + 1, c.r.length - 1)] - c.tap[i0]) * (f - i0);
      kr[y * 64 + x] = sl(v, kmax / 100);
    }
  }
  stage.block("kring").paint(0, kr, 0, 1);
  stage.block("kw").paint(0, w["kmlp.2.weight"]);
  return kmax;
}

// lays one conv layer's kernels out as a (cout x 3) by (cin x 3) picture on the shared weights block
export function paintWeights(stage, w, name) {
  const b = stage.block("wconv");
  if (b.meta && b.meta.name === name) return b;
  const ww = w[name + ".weight"];
  const cout = name === "net.out" ? 1 : 32, cin = ww.length / cout / 9;
  const a = new Float32Array(192 * 96);
  for (let co = 0; co < cout; co++) for (let ci = 0; ci < cin; ci++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) a[(co * 3 + ky) * 192 + ci * 3 + kx] = ww[((co * cin + ci) * 3 + ky) * 3 + kx];
  b.paint(0, a);
  b.planes[0].map.repeat.set(cin * 3 / 192, cout * 3 / 96);
  b.sub = { w: cin * 3, h: cout * 3 };
  b.meta = { cin, cout, name };
  b.planes[0].mesh.scale.set(cin * 3 / 192, cout * 3 / 96, 1);
  return b;
}

export function hoverMarks(stage, p, tr, n) {
  if (!tr || !p) return;
  const mark = (block, ix, iy, alpha, color, lift) => {
    const h = 64 / block.W;
    stage.markList.push({ p: stage.world(block.id, (ix + 0.5) * h - 32, (iy + 0.5) * h - 32, lift), size: block.size / block.W * 1.1, alpha, color });
  };
  if (p.type === "body") {
    const i = p.i;
    stage.bodyHl[i] = 1;
    for (const [cx, cy] of bodyCorners(tr.pos[2 * i], tr.pos[2 * i + 1], 128, 64)) mark(stage.block("input"), cx, cy, 0.7, 0x6fc3ff, 0.04);
  } else if (p.type === "cell") {
    const { block, k, ix, iy } = p;
    const py = block.planes[k].mesh.position.y + 0.04;
    mark(block, ix, iy, 0.8, 0x6fc3ff, py);
    if (block.W === block.H && block.W <= 128 && !["kring", "kw", "wconv"].includes(block.id)) {
      const f = 128 / block.W;
      for (let i = 0; i < n; i++) {
        if (bodyCorners(tr.pos[2 * i], tr.pos[2 * i + 1], 128, 64).some(([cx, cy]) => Math.floor(cx / f) === ix && Math.floor(cy / f) === iy)) stage.bodyHl[i] = 1;
      }
    }
  }
}

export function tipText(p, tr, vel) {
  if (p.type === "body") {
    const i = p.i;
    return `body ${i}<br>pos (${tr.pos[2 * i].toFixed(2)}, ${tr.pos[2 * i + 1].toFixed(2)})<br>vel (${vel[2 * i].toFixed(2)}, ${vel[2 * i + 1].toFixed(2)})`;
  }
  const b = p.block;
  if (b.id === "wconv" && b.meta) return `${b.meta.name}<br>out ${Math.floor(p.iy / 3)}, in ${Math.floor(p.ix / 3)}, k(${p.iy % 3},${p.ix % 3}) = ${fmt(p.v)}`;
  const ch = b.thick ? ` ch ${b.chan}` : b.n > 1 ? ` #${p.k + 1}` : "";
  const x = b.W === b.H ? ` sim (${((p.ix + 0.5) * 64 / b.W - 32).toFixed(1)}, ${((p.iy + 0.5) * 64 / b.H - 32).toFixed(1)})` : "";
  return `${b.title}${ch}<br>cell ${p.ix}, ${p.iy} of ${b.W}${x}<br>value ${p.v === null ? "-" : fmt(p.v)}`;
}
