// Builds the shipped ticks of the default scene: runs the JS tiled model (traced) and the exact sim for ticks 0..TICKS, writes
// precomputed/{tiled4_collapse300.json, tiled4_collapse300.bin.gz, state100.bin.gz} and tests/ref_continue.json (the continuous run's ticks
// TICKS..TICKS+EXTRA, the reference the compute-more path is checked against).
// Usage: node tools/precompute.mjs [TICKS=100] [EXTRA=20]
import { writeFileSync, mkdirSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { TiledDriver, TileNet, makeScene, kernelCurves } from "../js/tiled_model.js";
import { packFrame, toCompact } from "../js/tframes.js";
import { encodeAsset, encodeState, assetBytes, DEFAULT } from "../js/precomputed.js";
import * as truth from "../../scatter/js/truth.js";
import { NodePool, weights, manifest } from "../tests/nodepool.mjs";

const TICKS = +process.argv[2] || 100, EXTRA = +process.argv[3] || 20;
const dir = new URL("..", import.meta.url).pathname;
const { nt, n, scene, seed } = DEFAULT, T = nt * nt, flags = { ghosts: true, fx: true, far: true };
const { pos, vel } = makeScene(scene, nt, n, seed);
const spos = Float64Array.from(pos), svel = Float64Array.from(vel), sacc = new Float64Array(2 * n), dt = weights.config.dt;
truth.accel(spos, n, sacc);

const pool = new NodePool();
const drv = new TiledDriver(weights, pool, nt);
drv.flags = { ...flags };
await drv.reset();
const frames = [], sims = [], ref = [];
let state = null, t0 = performance.now();
for (let tick = 0; tick <= TICKS + EXTRA; tick++) {
  const ts = performance.now();
  const r = tick === 0 ? await drv.seed(pos, vel, n, true) : await drv.step(pos, vel, n, true);
  if (tick > 0) truth.step(spos, svel, n, dt, sacc);
  if (tick <= TICKS) {
    const [fr] = packFrame(r, drv, pos, vel, tick, performance.now() - ts, flags);
    frames.push(toCompact(fr));
    sims.push({ pos: Float32Array.from(spos), acc: Float32Array.from(sacc) });
  }
  if (tick === TICKS) {
    state = { pos: Float64Array.from(pos), vel: Float64Array.from(vel), acc: Float64Array.from(r.acc), spos: Float64Array.from(spos), svel: Float64Array.from(svel), phis: r.tiles.map((t) => Float32Array.from(t.phi)) };
  }
  if (tick >= TICKS) ref.push({ tick, pos: Array.from(pos), vel: Array.from(vel) });
  if (tick % 10 === 0) console.log(`tick ${tick} ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
pool.close();

const net = new TileNet(weights), cv = kernelCurves(net, 100, 200), arr = (a) => Array.from(a, (x) => +x.toPrecision(6));
const meta = { name: "tiled4_collapse300", nt, T, n, G: nt * 16 + 16, scene, seed, flags, dt, ticks: TICKS, checkpoint: weights.checkpoint, iters: weights.iters, kernel: manifest.kernel,
  config: weights.config, params: Object.values(weights.w).reduce((a, t) => a + t.length, 0), curves: { r: arr(cv.r), learned: arr(cv.learned), shipped: arr(cv.shipped), exact: arr(cv.exact) } };
const bin = gzipSync(Buffer.from(encodeAsset(meta, frames, sims)), { level: 9 });
const st = gzipSync(Buffer.from(encodeState(state)), { level: 9 });
mkdirSync(dir + "precomputed", { recursive: true });
writeFileSync(dir + "precomputed/tiled4_collapse300.json", JSON.stringify(meta));
writeFileSync(dir + "precomputed/tiled4_collapse300.bin.gz", bin);
writeFileSync(dir + "precomputed/state100.bin.gz", st);
writeFileSync(dir + "tests/ref_continue.json", JSON.stringify({ from: TICKS, n, ticks: ref.map((r) => r.tick), pos: ref.map((r) => r.pos), vel: ref.map((r) => r.vel) }));
console.log(`asset ${(bin.length / 1e6).toFixed(2)} MB gz (${(assetBytes(meta) / 1e6).toFixed(2)} MB raw), state ${(st.length / 1e6).toFixed(2)} MB gz, total ${((bin.length + st.length) / 1e6).toFixed(2)} MB; ${((performance.now() - t0) / 1000).toFixed(0)} s`);
