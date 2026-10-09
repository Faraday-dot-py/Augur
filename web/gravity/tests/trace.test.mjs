// The per-body trace must reproduce what step() actually does: summed edge forces = accel, trace output = new state.
import { readFileSync } from "node:fs";
import { CentralNet, parseWeights, initBodies, mulberry32 } from "../js/central_model.js";

const dir = new URL("..", import.meta.url).pathname;
const bin = readFileSync(dir + "weights.bin");
const net = new CentralNet(parseWeights(JSON.parse(readFileSync(dir + "weights.json")), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)));
const n = 40, { pos, vel } = initBodies(n, mulberry32(4738));
const ref = { pos: Float64Array.from(pos), vel: Float64Array.from(vel) };
let worst = 0;
for (let t = 0; t < 5; t++) {
  const i = (7 * t) % n;
  const tr = net.step(pos, vel, n, i, n);
  net.step(ref.pos, ref.vel, n);
  if (tr.s1.n !== n - 1 || tr.s1.edges.length !== n - 1) { console.error("FAIL: neighbour count", tr.s1.n); process.exit(1); }
  const sum = [0, 0];
  for (const e of tr.s1.edges) { sum[0] += e.force[0]; sum[1] += e.force[1]; }
  worst = Math.max(worst, Math.abs(sum[0] - tr.a0[0]), Math.abs(sum[1] - tr.a0[1]),
    Math.abs(tr.s2.acc[0] - tr.a1[0]), Math.abs(tr.s2.acc[1] - tr.a1[1]),
    Math.abs(tr.newPos[0] - pos[2 * i]), Math.abs(tr.newPos[1] - pos[2 * i + 1]),
    Math.abs(tr.newVel[0] - vel[2 * i]), Math.abs(tr.newVel[1] - vel[2 * i + 1]),
    Math.abs(pos[2 * i] - ref.pos[2 * i]));
  const e = tr.s1.edges[0];
  const out = e.h2.reduce((a, h, k) => a + net.forceMlp.W2[k] * h, net.forceMlp.b2);
  worst = Math.max(worst, Math.abs(out - e.out), Math.abs(e.mag - e.out / (e.dist ** 2 + 1)));
}
if (worst > 1e-9) { console.error("FAIL: trace vs step", worst); process.exit(1); }
console.log("trace matches step OK", worst);
