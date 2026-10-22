// Checks the shipped ticks against the model and the exact sim, and the compute-more path against the continuous reference.
//  1. sim: shipped sim positions vs the exact sim recomputed from tick 0 (float32 storage only)
//  2. model: ticks 0..PARITY_TICKS recomputed from tick 0 vs the shipped ticks (float32 storage); PARITY_TICKS=100 for the whole file
//  3. compute more: restore from the shipped full-precision tick-100 state, step to tick 120, vs tests/ref_continue.json (the continuous run);
//     tick 100 -> 101 is printed on its own
//  4. field quantisation: shipped 32x32 8-bit fields vs the fresh full-resolution tensors (relative to the tile max)
//  5. resuming from a shipped (compressed) tick instead of the exact state: 1-step and 5-step difference, for information
// Usage: node precomputed.mjs   (PARITY_TICKS=n, default 5)
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { TiledDriver, makeScene } from "../js/tiled_model.js";
import { packFrame, dequantFrame, makeOut, framePhi, GG } from "../js/tframes.js";
import { decodeAsset, decodeState, DEFAULT } from "../js/precomputed.js";
import * as truth from "../../scatter/js/truth.js";
import { NodePool, weights } from "./nodepool.mjs";

const dir = new URL("..", import.meta.url).pathname;
const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const meta = JSON.parse(readFileSync(dir + `precomputed/${DEFAULT.name}.json`));
const binGz = readFileSync(dir + `precomputed/${DEFAULT.name}.bin.gz`), stGz = readFileSync(dir + "precomputed/state100_collapse.bin.gz");
const { frames, sims } = decodeAsset(meta, ab(gunzipSync(binGz)));
const st = decodeState(ab(gunzipSync(stGz)), meta.n, meta.T);
const ref = JSON.parse(readFileSync(dir + "tests/ref_continue.json"));
const PT = Math.min(+process.env.PARITY_TICKS || 5, meta.ticks), n = meta.n, last = meta.ticks;
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
let bad = 0;
const check = (c, m) => { console.log((c ? "ok   " : "FAIL ") + m); if (!c) bad++; };

console.log(`payload: ${(binGz.length / 1e6).toFixed(2)} MB ticks 0-${last} (gz), ${(stGz.length / 1e6).toFixed(2)} MB tick-${last} state (gz), total ${((binGz.length + stGz.length + JSON.stringify(meta).length) / 1e6).toFixed(2)} MB`);
check(frames.length === last + 1 && sims.length === last + 1 && frames.every((f, k) => f.tick === k), `${frames.length} shipped ticks 0-${last}`);

const sp = Float64Array.from(makeScene(meta.scene, meta.nt, n, meta.seed).pos), sv = Float64Array.from(makeScene(meta.scene, meta.nt, n, meta.seed).vel), sa = new Float64Array(2 * n);
truth.accel(sp, n, sa);
let worstSim = maxDiff(sp, sims[0].pos);
for (let k = 1; k <= last; k++) { truth.step(sp, sv, n, meta.dt, sa); worstSim = Math.max(worstSim, maxDiff(sp, sims[k].pos)); }
check(worstSim < 1e-5, `exact sim recomputed from tick 0 vs shipped, ticks 0-${last}: max |dpos| ${worstSim.toExponential(2)}`);
check(maxDiff(sp, st.spos) === 0, "shipped tick-100 sim state is the exact sim state");

const pool = new NodePool();
const flags = { ghosts: true, fx: true, far: true };
const drv = new TiledDriver(weights, pool, meta.nt);
drv.flags = { ...flags };
await drv.reset();
const { pos, vel } = makeScene(meta.scene, meta.nt, n, meta.seed);
let worstPos = 0, worstGh = 0;
const worstQ = {};
const t0 = performance.now(), out = makeOut(meta.T, meta.nt * 16 + 16);
for (let k = 0; k <= PT; k++) {
  const r = k === 0 ? await drv.seed(pos, vel, n, true) : await drv.step(pos, vel, n, true);
  worstPos = Math.max(worstPos, maxDiff(pos, frames[k].pos), maxDiff(r.acc, frames[k].acc) / 10);
  const [fr] = packFrame(r, drv, pos, vel, k, 0, flags);
  const gl = frames[k].gh.length === fr.gh.length && frames[k].gh.every((x, i) => x === fr.gh[i]);
  if (!gl) worstGh++;
  if (k === PT || k % 25 === 0) {
    dequantFrame(frames[k], out);
    for (let t = 0; t < meta.T; t++) {
      for (const name of ["phi", "rho", "phiK", "far"]) {
        const a = out.tiles[t][name], b = r.tiles[t][name];
        let mx = 0, e = 0;
        for (let j = 0; j < b.length; j++) { mx = Math.max(mx, Math.abs(b[j])); e = Math.max(e, Math.abs(a[j] - b[j])); }
        worstQ[name] = Math.max(worstQ[name] || 0, e / (mx || 1));
      }
    }
  }
}
console.log(`   ${PT + 1} ticks recomputed in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
check(worstPos < 1e-5, `model recomputed from tick 0 vs shipped, ticks 0-${PT}: max |dpos| (and |dacc|/10) ${worstPos.toExponential(2)}`);
check(worstGh === 0, "ghost lists identical");
const pc = (x) => (x * 100).toFixed(2) + "%";
check(worstQ.phi < 0.015 && worstQ.rho < 0.015 && worstQ.far < 0.015, `shipped 8-bit fields vs full precision, worst error as a share of the tile max: potential ${pc(worstQ.phi)}, density ${pc(worstQ.rho)}, far term ${pc(worstQ.far)}`);
console.log(`info: local-kernel potential (stored 32x32): worst error ${pc(worstQ.phiK)} of the tile max (pits at bodies are smoothed)`);

// compute more: the exact tick-100 state, a fresh pool and driver as the page makes them
const pool2 = new NodePool();
const drv2 = new TiledDriver(weights, pool2, meta.nt);
drv2.flags = { ...flags };
const p2 = Float64Array.from(st.pos), v2 = Float64Array.from(st.vel);
await drv2.restore(st.acc, st.phis);
let worst = 0, first = 0;
const t1 = performance.now();
for (let k = last + 1; k <= ref.from + ref.ticks.length - 1; k++) {
  await drv2.step(p2, v2, n);
  const i = k - ref.from, d = Math.max(maxDiff(p2, ref.pos[i]), maxDiff(v2, ref.vel[i]));
  if (k === last + 1) first = d;
  worst = Math.max(worst, d);
}
const ticksPerS = (ref.ticks.length - 2) / ((performance.now() - t1) / 1000);
check(first <= 1e-3, `tick ${last} -> ${last + 1} from the shipped state vs the continuous run: max diff ${first.toExponential(2)}`);
check(worst <= 1e-3, `ticks ${last + 1}-${ref.from + ref.ticks.length - 1} from the shipped state vs the continuous run: max diff ${worst.toExponential(2)} (tolerance 1e-3), ${ticksPerS.toFixed(2)} ticks/s in Node on ${pool2.size} threads`);

// resuming from a shipped tick (compressed potentials, float32 bodies)
const kk = 50, pool3 = new NodePool(), drv3 = new TiledDriver(weights, pool3, meta.nt);
drv3.flags = { ...flags };
const f = frames[kk], phis = Array.from({ length: meta.T }, (_, t) => framePhi(f, t, new Float32Array(GG)));
await drv3.restore(Float64Array.from(f.acc), phis);
const p3 = Float64Array.from(f.pos), v3 = Float64Array.from(f.vel), rep = [];
for (let s = 1; s <= 5; s++) { await drv3.step(p3, v3, n); rep.push(`@${s} ${maxDiff(p3, frames[kk + s].pos).toExponential(1)}`); }
console.log(`info: resuming from shipped tick ${kk} (compressed state), diff vs shipped ticks: ${rep.join("  ")}`);

pool.close(); pool2.close(); pool3.close();
process.exit(bad ? 1 : 0);
