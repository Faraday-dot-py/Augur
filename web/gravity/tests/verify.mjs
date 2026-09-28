// Compares the JS CentralForceDynamics port against PyTorch rollouts dumped
// by scripts/export_web_testvectors_central.py.
import { readFileSync } from "node:fs";
import { CentralNet, parseWeights } from "../js/central_model.js";

const dir = new URL("..", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(dir + "weights.json"));
const bin = readFileSync(dir + "weights.bin");
const net = new CentralNet(parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)));
const vec = JSON.parse(readFileSync(dir + "tests/vectors.json"));

const flat = (rows) => Float64Array.from(rows.flat());
const maxDiff = (a, flat_) => {
  let m = 0;
  for (let i = 0; i < flat_.length; i++) m = Math.max(m, Math.abs(a[i] - flat_[i]));
  return m;
};

let worst1 = 0, worstN = 0;
for (const [name, s] of Object.entries(vec.scenes)) {
  const count = s.pos[0].length;
  let worst1Scene = 0;
  for (let t = 0; t < s.steps; t++) {
    const pos = flat(s.pos[t]), vel = flat(s.vel[t]);
    net.step(pos, vel, count);
    const d = Math.max(maxDiff(pos, flat(s.pos64[t])), maxDiff(vel, flat(s.vel64[t])));
    worst1Scene = Math.max(worst1Scene, d);
  }
  worst1 = Math.max(worst1, worst1Scene);

  // free rollout from step 0, compared against the float32 torch rollout
  const pos = flat(s.pos[0]), vel = flat(s.vel[0]);
  let dN = 0, d20 = 0;
  for (let t = 0; t < s.steps; t++) {
    net.step(pos, vel, count);
    const d = Math.max(maxDiff(pos, flat(s.pos[t + 1])), maxDiff(vel, flat(s.vel[t + 1])));
    if (t === 19) d20 = d;
    dN = d;
  }
  worstN = Math.max(worstN, dN);
  console.log(`${name}: bodies ${count}  1-step max|d| vs float64 torch ${worst1Scene.toExponential(2)}` +
    `  free-run max|d| vs float32 torch at step 20 ${d20.toExponential(2)}, at end (${s.steps}) ${dN.toExponential(2)}`);
}

console.log(`worst 1-step ${worst1.toExponential(2)}, worst free-run end ${worstN.toExponential(2)}`);
const GATE = 1e-4;
if (worst1 > GATE) {
  console.error(`FAIL: worst 1-step diff ${worst1.toExponential(2)} exceeds gate ${GATE}`);
  process.exit(1);
}
console.log(`PASS: 1-step diff within ${GATE}`);
