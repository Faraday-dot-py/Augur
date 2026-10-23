// WebGPU U-Net vs the JS CPU U-Net: output match on real tile inputs, a rollout through GpuPool vs the CPU rollout (and the PyTorch reference), and 16-tile batch timing vs CPU worker threads.
// Needs a WebGPU implementation in Node: npm i webgpu in any directory, then WEBGPU_MODULE=<dir>/node_modules/webgpu/index.js node webgpu_verify.mjs [scene]
// (STEPS=n sets the rollout length; software adapters such as llvmpipe check numerics only, their timings are not evidence about GPUs.)
import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { availableParallelism } from "node:os";
import { pathToFileURL } from "node:url";
import { TileNet, TiledDriver, LocalPool, parseWeights } from "../js/tiled_model.js";
import { GpuPool } from "../js/tiled_gpu.js";
import { UNetGPU } from "../js/unet_gpu.js";

const dir = new URL("..", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(dir + "weights.json"));
const bin = readFileSync(dir + "weights.bin");
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const weights = parseWeights(manifest, buf);
const vec = JSON.parse(readFileSync(dir + "tests/vectors.json"));
const STEPS = Math.min(+process.env.STEPS || vec.steps, vec.steps);
const name = process.argv[2] || "nt4_cluster";
const flat = (rows) => Float64Array.from(rows.flat());
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

if (!process.env.WEBGPU_MODULE) { console.log("UNTESTED: set WEBGPU_MODULE to the webgpu npm package entry (no WebGPU in plain Node)"); process.exit(0); }
const wg = await import(pathToFileURL(process.env.WEBGPU_MODULE).href);
Object.assign(globalThis, wg.globals);
const gpuApi = wg.create([]);
const s = vec.scenes[name], nt = s.nt, n = s.pos0.length, T = nt * nt, GG = 128 * 128;
const unet = await UNetGPU.tryCreate(manifest, buf, T, gpuApi);
if (!unet) { console.log("UNTESTED: no WebGPU adapter"); process.exit(0); }
console.log(`adapter: ${unet.adapter}${unet.software ? " (SOFTWARE: numerics only, timings are not GPU evidence)" : ""}; scene ${name}, ${T} tiles, ${STEPS} ticks`);

async function rollout(pool) {
  const drv = new TiledDriver(weights, pool, nt);
  await drv.reset();
  const pos = flat(s.pos0), vel = flat(s.vel0), outp = [], t0 = performance.now();
  for (let t = 0; t < STEPS; t++) { await drv.step(pos, vel, n); outp.push([Float64Array.from(pos), Float64Array.from(vel)]); }
  return { outp, ms: (performance.now() - t0) / STEPS };
}

const rec = { x: [], y: [] };
const orig = TileNet.prototype.unet;
TileNet.prototype.unet = function (x, tr) {
  const y = orig.call(this, x, tr);
  rec.x.push(Float32Array.from(x)); rec.y.push(y);
  return y;
};
const cpu = await rollout(new LocalPool(weights));
TileNet.prototype.unet = orig;
console.log(`CPU rollout (1 thread): ${(cpu.ms / 1000).toFixed(2)} s/tick, ${rec.x.length} U-Net calls recorded`);

let worstOut = 0, worstElem = 0;
for (const [label, first] of [["tick-0 force pass (zero recurrent field)", 0], [`final force pass (tick ${STEPS})`, rec.x.length - 2 * T]]) {
  const x = new Float32Array(T * 6 * GG);
  for (let k = 0; k < T; k++) x.set(rec.x[first + k], k * 6 * GG);
  const y = await unet.run(x);
  let dmax = 0, peak = 0, rel = 0;
  for (let k = 0; k < T; k++) for (let i = 0; i < GG; i++) { peak = Math.max(peak, Math.abs(rec.y[first + k][i])); }
  for (let k = 0; k < T; k++) for (let i = 0; i < GG; i++) {
    const r = rec.y[first + k][i], d = Math.abs(y[k * GG + i] - r);
    dmax = Math.max(dmax, d);
    if (Math.abs(r) > 1e-2 * peak) rel = Math.max(rel, d / Math.abs(r));
  }
  worstOut = Math.max(worstOut, dmax / peak); worstElem = Math.max(worstElem, rel);
  console.log(`  U-Net out ${label}: peak |phi| ${peak.toExponential(2)}, max abs diff ${dmax.toExponential(2)}, / peak ${(dmax / peak).toExponential(2)}, max elementwise rel diff (|phi|>1% peak) ${rel.toExponential(2)}`);
}

const gp = new GpuPool(weights, unet);
const gpu = await rollout(gp);
let line = [];
for (const t of [0, 4, 9, STEPS - 1]) {
  if (t >= STEPS) continue;
  const dc = Math.max(maxDiff(gpu.outp[t][0], cpu.outp[t][0]), maxDiff(gpu.outp[t][1], cpu.outp[t][1]));
  const dp = Math.max(maxDiff(gpu.outp[t][0], flat(s.tiled.pos[t + 1])), maxDiff(gpu.outp[t][1], flat(s.tiled.vel[t + 1])));
  const cp = Math.max(maxDiff(cpu.outp[t][0], flat(s.tiled.pos[t + 1])), maxDiff(cpu.outp[t][1], flat(s.tiled.vel[t + 1])));
  line.push(`@${t + 1} gpu-vs-cpu ${dc.toExponential(2)} gpu-vs-torch ${dp.toExponential(2)} (cpu-vs-torch ${cp.toExponential(2)})`);
}
console.log("  rollout (max abs over pos and vel): " + line.join("; "));
console.log(`  GpuPool rollout ${(gpu.ms / 1000).toFixed(2)} s/tick (includes CPU scatter/pair work and the recording pass)`);

const x = new Float32Array(T * 6 * GG);
for (let k = 0; k < T; k++) x.set(rec.x[rec.x.length - T + k], k * 6 * GG);
await unet.run(x);
const tg = await unet.timeCompute(7);
const tm = {};
await unet.run(x, tm);

const nw = Math.min(T, availableParallelism()), workers = Array.from({ length: nw }, () => new Worker(new URL("./unet_cpu_worker.mjs", import.meta.url), { workerData: { manifest, buf } }));
const share = (w) => Array.from({ length: Math.ceil(T / nw) }, (_, j) => w + j * nw).filter((k) => k < T).map((k) => rec.x[rec.x.length - T + k]);
const runCpu = () => new Promise((res) => {
  let left = nw; const t0 = performance.now();
  workers.forEach((w, i) => { w.once("message", () => { if (!--left) res(performance.now() - t0); }); w.postMessage({ xs: share(i) }); });
});
await runCpu();
const cs = []; for (let i = 0; i < 5; i++) cs.push(await runCpu());
cs.sort((a, b) => a - b);
workers.forEach((w) => w.terminate());
console.log(`timing, ${T} tiles: GPU compute ${tg.toFixed(1)} ms (median of 7; upload ${tm.upload.toFixed(1)}, readback ${tm.readback.toFixed(1)} ms) vs CPU ${nw} worker threads ${cs[2].toFixed(1)} ms (median of 5) -> speedup ${(cs[2] / tg).toFixed(2)}x${unet.software ? " (software adapter: meaningless for the 5x criterion)" : ""}`);
console.log(`summary: out/peak ${worstOut.toExponential(2)}, elementwise rel ${worstElem.toExponential(2)}`);
