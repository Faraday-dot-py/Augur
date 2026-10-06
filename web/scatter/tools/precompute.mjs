// node tools/precompute.mjs [outDir] [ticks] [n ...]
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { ScatterNet, parseWeights, initBodies, mulberry32 } from "../js/scatter_model.js";

export const SEED = 4738;
export const MAGIC = "SCPC";

export function loadNet() {
  const dir = new URL("..", import.meta.url).pathname;
  const manifest = JSON.parse(readFileSync(dir + "weights.json"));
  const bin = readFileSync(dir + "weights.bin");
  return { manifest, net: new ScatterNet(parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength))) };
}

export function run(net, n, ticks) {
  const { pos, vel } = initBodies(n, mulberry32(SEED));
  net.reset();
  const data = new Float32Array((ticks + 1) * 4 * n);
  const put = (t) => { data.set(pos, t * 4 * n); data.set(vel, t * 4 * n + 2 * n); };
  put(0);
  for (let t = 1; t <= ticks; t++) { net.step(pos, vel, n); put(t); }
  return data;
}

export function encode(n, ticks, dt, ckpt, data) {
  const head = Buffer.alloc(32);
  head.write(MAGIC, 0, "ascii");
  head.writeUInt32LE(1, 4);
  head.writeUInt32LE(n, 8);
  head.writeUInt32LE(ticks, 12);
  head.writeUInt32LE(SEED, 16);
  head.writeFloatLE(dt, 20);
  head.writeUInt32LE(ckpt >>> 0, 24);
  return Buffer.concat([head, Buffer.from(data.buffer, data.byteOffset, data.byteLength)]);
}

export function ckptId(manifest) {
  let h = 2166136261;
  for (const c of String(manifest.checkpoint) + ":" + manifest.iters) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  return h >>> 0;
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const out = process.argv[2] || new URL("../precomputed/", import.meta.url).pathname;
  const ticks = +(process.argv[3] || 100);
  const ns = process.argv.length > 4 ? process.argv.slice(4).map(Number) : Array.from({ length: 10 }, (_, i) => 10 * (i + 1));
  mkdirSync(out, { recursive: true });
  const { manifest, net } = loadNet();
  const dt = manifest.config.dt, id = ckptId(manifest);
  const index = { seed: SEED, ticks, dt, checkpoint: manifest.checkpoint, iters: manifest.iters, id, runs: {} };
  for (const n of ns) {
    const t0 = performance.now();
    const file = `n${String(n).padStart(3, "0")}.bin`;
    writeFileSync(out + file, encode(n, ticks, dt, id, run(net, n, ticks)));
    index.runs[n] = file;
    console.log(file, `${((performance.now() - t0) / 1000).toFixed(1)} s`);
  }
  writeFileSync(out + "index.json", JSON.stringify(index, null, 1) + "\n");
}
