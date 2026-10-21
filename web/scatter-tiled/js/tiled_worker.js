import { TiledDriver, TileNet, LocalPool, parseWeights, kernelCurves } from "./tiled_model.js";
import { packFrame } from "./tframes.js";

class WorkerPool {
  constructor(size, manifest, buf) {
    this.size = size; this.id = 0; this.pending = new Map();
    this.workers = Array.from({ length: size }, () => {
      const w = new Worker(new URL("./tile_worker.js", import.meta.url), { type: "module" });
      w.onmessage = (e) => {
        const p = this.pending.get(e.data.id);
        this.pending.delete(e.data.id);
        if (e.data.error) p.rej(new Error(e.data.error)); else p.res(e.data.res);
      };
      return w;
    });
    this.ready = Promise.all(this.workers.map((w, i) => this.send(i, { init: { manifest, buf: buf.slice(0) } })));
  }

  send(i, body) { return new Promise((res, rej) => { const id = ++this.id; this.pending.set(id, { res, rej }); this.workers[i].postMessage({ id, ...body }); }); }

  call(i, msg) { return this.send(i, { msg }); }

  close() { for (const w of this.workers) w.terminate(); }
}

let manifest, buf, weights, pool, drv, pools = 0, wanted = 0;
let nt = 2, pos, vel, n = 0, tick = -1, epoch = 0, limit = -1, flags, inflight = Promise.resolve(), pumping = false, inspect = -1;

async function setup(newNt, flagsIn) {
  if (!drv || drv.nt !== newNt) {
    inspect = -1;
    const size = Math.max(1, Math.min(newNt * newNt, wanted));
    if (pool && pool.close) pool.close();
    pool = typeof Worker === "undefined" ? new LocalPool(weights) : new WorkerPool(size, manifest, buf);
    if (pool.ready) await pool.ready;
    drv = new TiledDriver(weights, pool, newNt);
    pools++;
  }
  drv.flags = { ...flagsIn };
  drv.inspect = inspect = Math.min(inspect < 0 ? (newNt * newNt) / 2 + newNt / 2 : inspect, newNt * newNt - 1);
  nt = newNt;
  flags = drv.flags;
}

async function produce(ep) {
  const t0 = performance.now();
  let r;
  if (tick < 0) { r = await drv.seed(pos, vel, n, true); tick = 0; } else { r = await drv.step(pos, vel, n, true); tick++; }
  const [fr, bufs] = packFrame(r, drv, pos, vel, tick, performance.now() - t0, flags);
  fr.epoch = ep;
  postMessage(fr, bufs);
}

function pump() {
  if (pumping) return;
  pumping = true;
  const ep = epoch;
  inflight = (async () => {
    try {
      while (ep === epoch && tick < limit) await produce(ep);
    } catch (err) {
      postMessage({ type: "error", message: err.message });
    }
    pumping = false;
  })();
}

async function start(m, restoreFrom) {
  epoch = m.epoch;
  await inflight;
  if (epoch !== m.epoch) return;
  nt = m.nt;
  await setup(m.nt, m.flags);
  limit = m.limit;
  if (restoreFrom) {
    pos = Float64Array.from(m.pos); vel = Float64Array.from(m.vel); n = pos.length / 2; tick = m.tick;
    await drv.restore(m.acc, m.phis);
  } else {
    pos = Float64Array.from(m.pos); vel = Float64Array.from(m.vel); n = pos.length / 2; tick = -1;
    await drv.reset();
  }
  pump();
}

async function init(m) {
  wanted = m.workers;
  [manifest, buf] = await Promise.all([fetch(new URL("../weights.json", import.meta.url)).then((r) => r.json()), fetch(new URL("../weights.bin", import.meta.url)).then((r) => r.arrayBuffer())]);
  weights = parseWeights(manifest, buf);
  const net = new TileNet(weights);
  postMessage({ type: "ready", config: weights.config, checkpoint: weights.checkpoint, iters: weights.iters, kernel: manifest.kernel, curves: kernelCurves(net, 100, 200),
    workers: typeof Worker === "undefined" ? 1 : wanted, params: Object.values(weights.w).reduce((a, t) => a + t.length, 0) });
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "init") init(m).catch((err) => postMessage({ type: "error", message: err.message }));
  else if (m.cmd === "reset") start(m, false).catch((err) => postMessage({ type: "error", message: err.message }));
  else if (m.cmd === "branch") start(m, true).catch((err) => postMessage({ type: "error", message: err.message }));
  else if (m.cmd === "want" && m.epoch === epoch) { limit = m.limit; if (drv && tick >= 0) pump(); }
  else if (m.cmd === "inspect" && drv) { drv.inspect = inspect = m.tile; }
};
