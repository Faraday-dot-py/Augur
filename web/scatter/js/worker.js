import { loadWeights, ScatterNet } from "./scatter_model.js";
import * as truth from "./truth.js";
import { quantise, countOutside } from "./frames.js";

let net, pos, vel, n = 0, tick = -1, epoch = 0, limit = -1, pumping = false;

async function init() {
  const w = await loadWeights(new URL("../weights", import.meta.url).href);
  net = new ScatterNet(w);
  postMessage({ type: "ready", config: w.config, checkpoint: w.checkpoint, iters: w.iters, curve: net.kernelCurve(24, 192), pair: net.kernelCurve(net.cfg.pp, 64) });
}

// tick 0 is the force pass that seeds the Verlet scheme; every later tick records its end-of-step pass
function produce() {
  const t0 = performance.now(), tr = {};
  if (tick < 0) {
    net.reset();
    net.grow(n);
    net.force(pos, vel, n, net.aTok, tr);
    net.aPrev = Float64Array.from(net.aTok.subarray(0, 2 * n));
    net.aPrevN = n;
    tick = 0;
  } else {
    net.step(pos, vel, n, tr);
    tick++;
  }
  const q = quantise(tr);
  const p = pos.slice(), v = vel.slice();
  const msg = { type: "frame", epoch, tick, n, ms: performance.now() - t0, pos: p, vel: v, energy: truth.energy(pos, vel, n), outside: countOutside(pos, n, net.cfg.extent), ...q };
  postMessage(msg, [p.buffer, v.buffer, q.q16.buffer, q.q8.buffer, q.M.buffer]);
}

function pump() {
  if (pumping) return;
  pumping = true;
  const go = () => {
    if (!net || !n || tick >= limit) { pumping = false; return; }
    produce();
    setTimeout(go, 0);
  };
  go();
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "reset") {
    n = m.n; pos = Float64Array.from(m.pos); vel = Float64Array.from(m.vel); tick = -1; epoch = m.epoch; limit = m.limit;
    pump();
  } else if (m.cmd === "want" && m.epoch === epoch) {
    limit = m.limit;
    pump();
  }
};

init().catch((err) => postMessage({ type: "error", message: err.message }));
