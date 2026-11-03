// node scripts/notebook_assets/dump_trace.mjs [n] [warmup_ticks] [out.json]
// Runs the browser scatter-field model (seed 4738, initBodies) and dumps the traced tensors of one force pass.
import { writeFileSync } from "node:fs";
import { initBodies, mulberry32 } from "../../web/scatter/js/scatter_model.js";
import { loadNet, SEED } from "../../web/scatter/tools/precompute.mjs";

const n = +(process.argv[2] || 100);
const warm = +(process.argv[3] || 20);
const out = process.argv[4] || "results/notebook_assets/trace_n100_tick21.json";
const { manifest, net } = loadNet();
const { pos, vel } = initBodies(n, mulberry32(SEED));
net.reset();
const init = Array.from(pos.subarray(0, 2 * n));
for (let t = 0; t < warm; t++) net.step(pos, vel, n);
const trace = {};
net.step(pos, vel, n, trace);
const GG = net.G * net.G;
const f = (a) => Array.from(a);
writeFileSync(out, JSON.stringify({
  seed: SEED, n, tick: warm + 1, checkpoint: manifest.checkpoint, iters: manifest.iters, G: net.G, h: net.h,
  init_pos: init, pos: f(trace.pos), vel: f(trace.vel),
  density: f(trace.input.subarray(0, GG)), phiK: f(trace.phiK), phi: f(trace.phi),
  aGrid: [f(trace.aGrid[0]), f(trace.aGrid[1])],
  gather: f(trace.gather), pairAcc: f(trace.pairAcc), accel: f(trace.accel), momfix: trace.momfix,
}));
console.log("wrote", out, "G", net.G, "h", net.h, "ckpt", manifest.checkpoint, manifest.iters);
