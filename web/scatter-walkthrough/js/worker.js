import { loadWeights, ScatterNet } from "../../scatter/js/scatter_model.js";
import { collect } from "../../scatter/js/trace.js";

let net, pos, vel, n = 0, tick = 0;

async function init() {
  const w = await loadWeights(new URL("../../scatter/weights", import.meta.url).href);
  net = new ScatterNet(w);
  postMessage({ type: "ready", config: w.config, checkpoint: w.checkpoint, iters: w.iters, curve: net.kernelCurve(24, 192), pair: net.kernelCurve(w.config.pp, 64) });
}

function run(advance) {
  const t0 = performance.now();
  const tr = {};
  const posBefore = pos.slice();
  net.grow(n);
  if (advance) {
    net.step(pos, vel, n, tr);
    tick++;
  } else {
    net.force(pos, vel, n, net.aTok, tr);
    net.aPrev = Float64Array.from(net.aTok.subarray(0, 2 * n));
    net.aPrevN = n;
  }
  tr.phi = tr.phi.slice();
  const bufs = [];
  collect(tr, bufs);
  const p = pos.slice(), v = vel.slice(), pb = posBefore;
  bufs.push(p.buffer, v.buffer, pb.buffer);
  postMessage({ type: "trace", tick, ms: performance.now() - t0, n, pos: p, vel: v, posBefore: pb, trace: tr }, bufs);
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "reset") {
    n = m.n; pos = Float64Array.from(m.pos); vel = Float64Array.from(m.vel); tick = 0;
    net.reset();
    run(false);
  } else if (m.cmd === "trace") {
    run(true);
  }
};

init().catch((err) => postMessage({ type: "error", message: err.message }));
