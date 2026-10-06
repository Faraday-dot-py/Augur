// Checks the precomputed fixed-seed runs against the page's own model and initial conditions. The page only uses them as a reference: it recomputes every frame itself.
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

if (fail) { console.error("FAIL"); process.exit(1); }
console.log("OK");
