// node tools/log_energy.mjs [out.json] [ticks] [n]
// Model and ghost-truth energy plus bodies outside the arena, per tick from tick 0, seed 4738.
import { writeFileSync } from "node:fs";
import { initBodies, mulberry32 } from "../js/scatter_model.js";
import { countOutside } from "../js/frames.js";
import * as truth from "../js/truth.js";
import { loadNet, SEED } from "./precompute.mjs";

const out = process.argv[2] || "energy_log.json";
const ticks = +(process.argv[3] || 1000);
const n = +(process.argv[4] || 100);
const { manifest, net } = loadNet();
const { extent, dt } = manifest.config;
const init = initBodies(n, mulberry32(SEED));
const pos = init.pos.slice(), vel = init.vel.slice();
const tpos = init.pos.slice(), tvel = init.vel.slice(), ta = new Float64Array(2 * n);
truth.accel(tpos, n, ta);
net.reset();
const log = { seed: SEED, n, dt, extent, checkpoint: manifest.checkpoint, iters: manifest.iters, E: [], truthE: [], outside: [], truthOutside: [], err: [] };
const rec = () => {
  log.E.push(truth.energy(pos, vel, n));
  log.truthE.push(truth.energy(tpos, tvel, n));
  log.outside.push(countOutside(pos, n, extent));
  log.truthOutside.push(countOutside(tpos, n, extent));
  let e = 0;
  for (let i = 0; i < n; i++) e += Math.hypot(pos[2 * i] - tpos[2 * i], pos[2 * i + 1] - tpos[2 * i + 1]);
  log.err.push(e / n);
};
rec();
for (let t = 1; t <= ticks; t++) {
  net.step(pos, vel, n);
  truth.step(tpos, tvel, n, dt, ta);
  rec();
  if (t % 50 === 0) {
    console.log(`t ${t}  E ${log.E[t].toFixed(2)}  truthE ${log.truthE[t].toFixed(2)}  outside ${log.outside[t]}  err ${log.err[t].toFixed(3)}`);
    writeFileSync(out, JSON.stringify(log));
  }
}
writeFileSync(out, JSON.stringify(log));
