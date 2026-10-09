// Ghost truth (js/truth.js) vs scripts/gravity_sim.py: n=100, seed 4738, 1000 ticks.
// node tests/truth.mjs --init   writes truth_init.json (input of truth_ref.py)
// node tests/truth.mjs          checks conservation/boundedness and parity with truth_ref.json
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import * as truth from "../js/truth.js";
import { initBodies, mulberry32 } from "../js/scatter_model.js";

const dir = new URL(".", import.meta.url).pathname;
const N = 100, TICKS = 1000, DT = 0.1, SEED = 4738;
const { pos, vel } = initBodies(N, mulberry32(SEED));

if (process.argv[2] === "--init") {
  writeFileSync(dir + "truth_init.json", JSON.stringify({ n: N, seed: SEED, dt: DT, ticks: TICKS, pos: Array.from(pos), vel: Array.from(vel) }));
  process.exit(0);
}

const a = new Float64Array(2 * N);
truth.accel(pos, N, a);
const E = [truth.energy(pos, vel, N)], snaps = {};
let rmax = 0, vmax = 0;
const KEEP = new Set([10, 100, 1000]);
for (let t = 1; t <= TICKS; t++) {
  truth.step(pos, vel, N, DT, a);
  E.push(truth.energy(pos, vel, N));
  if (KEEP.has(t)) snaps[t] = Array.from(pos);
  for (let i = 0; i < N; i++) {
    rmax = Math.max(rmax, Math.hypot(pos[2 * i], pos[2 * i + 1]));
    vmax = Math.max(vmax, Math.hypot(vel[2 * i], vel[2 * i + 1]));
  }
}
const E0 = E[0];
const drift = Math.max(...E.map((e) => Math.abs(e - E0))) / Math.abs(E0);
let px = 0, py = 0;
for (let i = 0; i < N; i++) { px += vel[2 * i]; py += vel[2 * i + 1]; }
console.log(`E0 ${E0.toFixed(3)}  E1000 ${E[TICKS].toFixed(3)}  max |dE|/|E0| ${drift.toExponential(2)}  max r ${rmax.toFixed(1)}  max speed ${vmax.toFixed(2)}  |P| ${Math.hypot(px, py).toExponential(1)}`);
const fail = (m) => { console.error("FAIL: " + m); process.exit(1); };
if (!(E0 < 0)) fail("initial system not bound");
if (!(drift < 0.05)) fail("energy drift >= 5%");
if (!(E[TICKS] < 0)) fail("system unbound at end");

const refFile = dir + "truth_ref.json";
if (existsSync(refFile)) {
  const ref = JSON.parse(readFileSync(refFile));
  for (const t of [10, 100, 1000]) {
    let m = 0;
    for (let i = 0; i < 2 * N; i++) m = Math.max(m, Math.abs(snaps[t][i] - ref.pos[t][i]));
    const dE = Math.abs(E[t] - ref.E[t]);
    console.log(`vs gravity_sim.py @${t}: max pos diff ${m.toExponential(2)}  E diff ${dE.toExponential(2)}`);
    // positions at 1000 diverge from roundoff amplified by chaos; energy still matches
    if (t < 1000 && !(m < 1e-6)) fail(`pos parity @${t}`);
    if (!(dE < 0.1)) fail(`energy parity @${t}`);
  }
} else console.log("truth_ref.json missing: parity not checked");
console.log("truth OK");
