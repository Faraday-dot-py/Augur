// Compares the JS ScatterField port against float64 PyTorch rollouts dumped by
// scripts/export_web_testvectors_scatter.py.
import { readFileSync } from "node:fs";
import * as truth from "../js/truth.js";
import { ScatterNet, parseWeights } from "../js/scatter_model.js";

const dir = new URL("..", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(dir + "weights.json"));
const bin = readFileSync(dir + "weights.bin");
const weights = parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const vec = JSON.parse(readFileSync(dir + "tests/vectors.json"));
const flat = (rows) => Float64Array.from(rows.flat());
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

let t0 = performance.now();
const net = new ScatterNet(weights);
console.log(`kernel setup ${(performance.now() - t0).toFixed(0)} ms`);

let worst1 = 0;
for (const [name, s] of Object.entries(vec.scenes)) {
  net.reset();
  const n = s.pos[0].length;
  const pos = flat(s.pos[0]), vel = flat(s.vel[0]);
  const rep = [];
  t0 = performance.now();
  for (let t = 0; t < vec.steps; t++) {
    net.step(pos, vel, n);
    const d = Math.max(maxDiff(pos, flat(s.pos[t + 1])), maxDiff(vel, flat(s.vel[t + 1])));
    if (t === 0) worst1 = Math.max(worst1, d);
    if (t === 0 || t === 4 || t === 9 || t === vec.steps - 1) rep.push(`@${t + 1} ${d.toExponential(2)}`);
  }
  console.log(name, n, "bodies:", rep.join("  "), ` ${((performance.now() - t0) / vec.steps).toFixed(0)} ms/step`);
  if (name === "n8" && s.field0) {
    // potential after the last step is not stored; field0 is the field of step `steps`
    console.log("  field max abs diff", maxDiff(net.field, Float64Array.from(s.field0.flat())).toExponential(2));
  }
}
console.log("1-step worst", worst1.toExponential(2));
if (!(worst1 < 1e-4)) { console.error("FAIL: 1-step diff >= 1e-4"); process.exit(1); }
console.log("OK");

// ghost truth must attract: two bodies 3 apart, gravity_sim.accel gives +0.10663718 on body 0
{
  const a = new Float64Array(4);
  truth.accel(new Float64Array([0, 0, 3, 0]), 2, a);
  if (Math.abs(a[0] - 0.10663718) > 1e-7 || Math.abs(a[2] + 0.10663718) > 1e-7) { console.error("FAIL: truth.accel sign", a); process.exit(1); }
  console.log("truth attracts OK");
}
