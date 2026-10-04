// Checks the shipped precomputed runs against the page's own model and initial conditions.
import { readFileSync, statSync } from "node:fs";
import { initBodies, mulberry32 } from "../js/scatter_model.js";
import { loadNet, run, SEED, MAGIC, ckptId } from "../tools/precompute.mjs";

const dir = new URL("../precomputed/", import.meta.url).pathname;
const index = JSON.parse(readFileSync(dir + "index.json"));
const { manifest, net } = loadNet();
let fail = 0;
const check = (c, msg) => { console.log((c ? "ok   " : "FAIL ") + msg); if (!c) fail++; };

check(index.seed === SEED && index.ticks === 100 && index.id === ckptId(manifest), `index: seed ${index.seed}, ${index.ticks} ticks, weights id matches`);
const ns = Array.from({ length: 10 }, (_, i) => 10 * (i + 1));
check(JSON.stringify(Object.keys(index.runs).map(Number)) === JSON.stringify(ns), "index lists N = 10..100 step 10");

let bytes = 0;
for (const n of ns) {
  const buf = readFileSync(dir + index.runs[n]);
  bytes += buf.length;
  const h = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const okHead = buf.toString("ascii", 0, 4) === MAGIC && h.getUint32(8, true) === n && h.getUint32(12, true) === 100 && h.getUint32(16, true) === SEED && h.getUint32(24, true) === index.id;
  const data = new Float32Array(buf.buffer.slice(buf.byteOffset + 32, buf.byteOffset + buf.length));
  const { pos, vel } = initBodies(n, mulberry32(SEED));
  const t0 = new Float32Array(4 * n);
  t0.set(pos); t0.set(vel, 2 * n);
  let same = data.length === 101 * 4 * n;
  for (let i = 0; same && i < t0.length; i++) if (data[i] !== t0[i]) same = false;
  check(okHead && same && data.every(Number.isFinite), `N=${n}: header, size, tick 0 == initBodies(seed ${SEED}), finite`);
}
console.log(`total ${(bytes / 1024).toFixed(0)} KiB`);

const n = 10, K = 20;
const fresh = run(net, n, K);
const stored = new Float32Array(readFileSync(dir + index.runs[n]).buffer.slice(32 + readFileSync(dir + index.runs[n]).byteOffset));
let diff = 0;
for (let i = 0; i < fresh.length; i++) if (fresh[i] !== stored[i]) diff++;
check(diff === 0, `N=10, 20 ticks regenerate bit-exactly (${diff} differing floats of ${fresh.length})`);

const live = initBodies(n, mulberry32(SEED));
net.reset();
for (let t = 0; t < K; t++) net.step(live.pos, live.vel, n);
const exact = net.field.slice();
const f64 = (a, s, e) => Float64Array.from(a.subarray(s, e));
net.reset(); net.grow(n);
for (let k = K - 2; k <= K; k++) net.force(f64(stored, k * 4 * n, k * 4 * n + 2 * n), f64(stored, k * 4 * n + 2 * n, k * 4 * n + 4 * n), n, new Float64Array(2 * n));
let num = 0, den = 0;
for (let i = 0; i < exact.length; i++) { num += (exact[i] - net.field[i]) ** 2; den += exact[i] ** 2; }
console.log(`probe field (2 warm ticks) vs live field at tick ${K} (N=${n}): relative L2 error ${Math.sqrt(num / den).toExponential(2)}`);
if (fail) { console.error("FAIL"); process.exit(1); }
console.log("OK");
