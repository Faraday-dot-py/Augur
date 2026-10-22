import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { availableParallelism } from "node:os";
import { parseWeights } from "../js/tiled_model.js";

const dir = new URL("..", import.meta.url).pathname;
export const manifest = JSON.parse(readFileSync(dir + "weights.json"));
const bin = readFileSync(dir + "weights.bin");
export const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
export const weights = parseWeights(manifest, buf);

export class NodePool {
  constructor(n = Math.min(16, availableParallelism())) {
    this.size = n; this.id = 0; this.pending = new Map();
    this.workers = Array.from({ length: n }, () => {
      const w = new Worker(new URL("./node_worker.mjs", import.meta.url), { workerData: { manifest, buf } });
      w.on("message", ({ id, res }) => { this.pending.get(id)(res); this.pending.delete(id); });
      return w;
    });
  }

  call(i, msg) {
    return new Promise((res) => { const id = ++this.id; this.pending.set(id, res); this.workers[i].postMessage({ id, msg }); });
  }

  close() { for (const w of this.workers) w.terminate(); }
}
