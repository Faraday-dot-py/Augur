import { TileNet, parseWeights } from "./tiled_model.js";
import { UNetGPU } from "./unet_gpu.js";

const out = document.getElementById("out"), log = (s) => { out.textContent += s + "\n"; };
const T = +new URLSearchParams(location.search).get("tiles") || 16, GG = 128 * 128;
const med = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];

async function main() {
  const [manifest, buf] = await Promise.all([fetch("weights.json").then((r) => r.json()), fetch("weights.bin").then((r) => r.arrayBuffer())]);
  const weights = parseWeights(manifest, buf);
  log(`tiles ${T}, cores ${navigator.hardwareConcurrency}`);
  const gpu = await UNetGPU.tryCreate(manifest, buf, T);
  if (!gpu) { log("WebGPU unavailable (no navigator.gpu or no adapter): the page keeps using the CPU workers."); return; }
  log(`adapter: ${gpu.adapter}${gpu.software ? "  [SOFTWARE ADAPTER: speed numbers below are not GPU evidence]" : ""}`);

  const x = new Float32Array(T * 6 * GG);
  for (let t = 0; t < T; t++) for (let c = 0; c < 6; c++) for (let i = 0; i < GG; i++) {
    const px = i % 128, py = i >> 7;
    x[(t * 6 + c) * GG + i] = Math.exp(-(((px - 40 - 3 * t) ** 2 + (py - 64) ** 2) / 600)) * (c < 3 ? 1 : 0.1) * (1 + 0.3 * Math.sin(0.7 * px + c));
  }
  const y = await gpu.run(x);
  const net = new TileNet(weights), ref = net.unet(x.subarray(0, 6 * GG));
  let d = 0, p = 0;
  for (let i = 0; i < GG; i++) { d = Math.max(d, Math.abs(y[i] - ref[i])); p = Math.max(p, Math.abs(ref[i])); }
  log(`tile 0: GPU vs CPU max abs diff ${d.toExponential(2)} (peak |phi| ${p.toExponential(2)}, rel ${(d / p).toExponential(2)})   criterion <= 1e-4`);

  const gt = [], tm = {};
  const gc = await gpu.timeCompute(9);
  for (let i = 0; i < 5; i++) { await gpu.run(x, tm); gt.push(tm.upload + tm.compute + tm.readback); }
  log(`GPU ${T} tiles: compute ${gc.toFixed(1)} ms (median of 9); with upload and readback ${med(gt).toFixed(1)} ms`);

  const nw = Math.max(1, Math.min(16, (navigator.hardwareConcurrency || 4) - 2));
  const workers = await Promise.all(Array.from({ length: nw }, () => new Promise((res) => {
    const w = new Worker(new URL("./unet_bench_worker.js", import.meta.url), { type: "module" });
    w.onmessage = () => res(w);
    w.postMessage({ init: { manifest, buf } });
  })));
  const xs = Array.from({ length: T }, (_, t) => x.slice(t * 6 * GG, (t + 1) * 6 * GG));
  const runCpu = () => new Promise((res) => {
    let left = workers.length; const t0 = performance.now();
    workers.forEach((w, i) => { w.onmessage = () => { if (!--left) res(performance.now() - t0); }; w.postMessage({ xs: xs.filter((_, k) => k % nw === i) }); });
  });
  await runCpu();
  const ct = []; for (let i = 0; i < 5; i++) ct.push(await runCpu());
  workers.forEach((w) => w.terminate());
  log(`CPU ${nw} workers, same ${T} tiles: ${med(ct).toFixed(1)} ms (median of 5)`);
  const sp = med(ct) / gc;
  log(`speedup (CPU / GPU compute) ${sp.toFixed(2)}x   criterion >= 5x   ${gpu.software ? "UNTESTED (software adapter)" : sp >= 5 ? "PASS" : "FAIL"}`);
}
main().catch((e) => log("error: " + e.message));
