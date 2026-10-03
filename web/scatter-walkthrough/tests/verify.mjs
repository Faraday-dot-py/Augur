// Checks the trace hook of the scatter model: tracing must not change results, trace tensors must
// have the documented shapes and be self-consistent, and the traced potential must match the
// float64 PyTorch field stored in ../../scatter/tests/vectors.json.
import { readFileSync } from "node:fs";
import { ScatterNet, parseWeights, mulberry32, initBodies } from "../../scatter/js/scatter_model.js";
import { bodyCorners } from "../js/cic.js";

const sdir = new URL("../../scatter/", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(sdir + "weights.json"));
const bin = readFileSync(sdir + "weights.bin");
const weights = parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const vec = JSON.parse(readFileSync(sdir + "tests/vectors.json"));
const fail = (m) => { console.error("FAIL:", m); process.exit(1); };
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

const { pos: p0, vel: v0 } = initBodies(16, mulberry32(4738));
const n = 16, steps = 4;
const plain = new ScatterNet(weights), traced = new ScatterNet(weights);
const pa = Float64Array.from(p0), va = Float64Array.from(v0), pb = Float64Array.from(p0), vb = Float64Array.from(v0);
let tr;
for (let t = 0; t < steps; t++) {
  plain.step(pa, va, n);
  tr = {};
  traced.step(pb, vb, n, tr);
  if (!same(pa, pb) || !same(va, vb)) fail("traced step " + t + " differs from untraced");
  if (!same(plain.field, traced.field)) fail("potential differs at step " + t);
}
console.log("traced == untraced, bit-identical, " + steps + " steps, n=" + n);

const G = 128, GG = G * G;
const shape = (name, a, len) => { if (!a || a.length !== len) fail(name + " length " + (a && a.length) + " != " + len); };
shape("input", tr.input, 6 * GG);
shape("phiK", tr.phiK, GG);
shape("inp", tr.inp, 32 * GG);
shape("phi", tr.phi, GG);
shape("gather", tr.gather, 2 * n);
shape("pairAcc", tr.pairAcc, 2 * n);
shape("accel", tr.accel, 2 * n);
shape("pos", tr.pos, 2 * n);
if (tr.enc.length !== 6 || tr.dec.length !== 5) fail("unet level counts");
tr.enc.forEach((a, l) => shape("enc" + l, a, 32 * (G >> l) ** 2));
for (let l = 0; l < 5; l++) shape("dec" + l, tr.dec[l], 32 * (G >> l) ** 2);
tr.aGrid.forEach((a) => shape("aGrid", a, GG));
tr.gradPhi.forEach((a) => shape("gradPhi", a, GG));
console.log("trace shapes OK (input 6x128x128, enc 6 levels 128..4, dec 5 levels 128..8, phi, grads, per-body arrays)");

let d = 0;
for (let i = 0; i < GG; i++) d = Math.max(d, Math.abs(tr.aGrid[0][i] - tr.gradPhi[0][i] - tr.input[3 * GG + i]), Math.abs(tr.aGrid[1][i] - tr.gradPhi[1][i] - tr.input[4 * GG + i]));
if (!(d < 1e-6)) fail("aGrid != gradPhi + ak: " + d);
console.log("aGrid = -grad(phi) + ak, max diff", d.toExponential(2));

const ga = new Float64Array(2 * n);
for (let i = 0; i < n; i++) {
  for (const [cx, cy, w] of bodyCorners(tr.pos[2 * i], tr.pos[2 * i + 1], G, 64)) {
    ga[2 * i] += tr.aGrid[0][cy * G + cx] * w;
    ga[2 * i + 1] += tr.aGrid[1][cy * G + cx] * w;
  }
}
d = maxDiff(ga, tr.gather);
if (!(d < 1e-6)) fail("gather != bilinear sample of aGrid: " + d);
console.log("gather matches independent CIC sample of aGrid, max diff", d.toExponential(2));

let mx = 0, my = 0;
const acc = new Float64Array(2 * n);
for (let i = 0; i < n; i++) { acc[2 * i] = tr.gather[2 * i] + tr.pairAcc[2 * i]; acc[2 * i + 1] = tr.gather[2 * i + 1] + tr.pairAcc[2 * i + 1]; mx += acc[2 * i]; my += acc[2 * i + 1]; }
for (let i = 0; i < n; i++) { acc[2 * i] -= mx / n; acc[2 * i + 1] -= my / n; }
d = maxDiff(acc, tr.accel);
if (!(d < 1e-6)) fail("accel != gather + pair - mean: " + d);
console.log("accel = gather + pair - mean, max diff", d.toExponential(2));

let rho = 0;
for (let i = 0; i < GG; i++) rho += tr.input[i];
if (Math.abs(rho * 0.25 - n) > 1e-3 * n * 4) fail("rho mass " + rho * 0.25);
console.log("scattered mass sums to N:", (rho * 0.25).toFixed(4));

const s = vec.scenes.n8;
const net = new ScatterNet(weights);
const pos = Float64Array.from(s.pos[0].flat()), vel = Float64Array.from(s.vel[0].flat());
let t2;
for (let t = 0; t < vec.steps; t++) { t2 = {}; net.step(pos, vel, 8, t2); }
d = maxDiff(t2.phi, Float64Array.from(s.field0.flat()));
console.log("traced phi vs PyTorch field after " + vec.steps + " steps, max abs diff", d.toExponential(2));
if (!(d < 1e-3)) fail("phi vs torch");

const kc = net.kernelCurve(16, 160);
if (kc.r.length !== 160 || !kc.tap.every(Number.isFinite)) fail("kernel curve");
console.log("UNet intermediate levels vs PyTorch: not checked (vectors.json stores only the final potential)");
console.log("OK");
