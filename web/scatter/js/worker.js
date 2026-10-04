import { loadWeights, ScatterNet } from "./scatter_model.js";
import * as truth from "./truth.js";
import { collect } from "./trace.js";

let net, pos, vel, n = 0, tick = 0, epoch = 0, running = false, busy = false, wantEnergy = false;

async function init() {
  const w = await loadWeights(new URL("../weights", import.meta.url).href);
  net = new ScatterNet(w);
  postMessage({ type: "ready", config: w.config, checkpoint: w.checkpoint, iters: w.iters });
}

function send(ms) {
  const p = pos.slice(), v = vel.slice();
  const msg = { type: "state", epoch, tick, ms, pos: p, vel: v, energy: wantEnergy ? truth.energy(pos, vel, n) : null };
  postMessage(msg, [p.buffer, v.buffer]);
}

function one() {
  const t0 = performance.now();
  net.step(pos, vel, n);
  tick++;
  send(performance.now() - t0);
}

function traceNow(curves, s) {
  const t0 = performance.now(), saved = net.field, tr = {};
  net.grow(s.n);
  if (s.warm) {
    net.reset();
    for (const [p, v] of s.warm) net.force(Float64Array.from(p), Float64Array.from(v), s.n, new Float64Array(2 * s.n));
  }
  net.force(s.pos, s.vel, s.n, new Float64Array(2 * s.n), tr);
  net.field = saved;
  const msg = { type: "trace", epoch: s.epoch, tick: s.tick, n: s.n, ms: performance.now() - t0, trace: tr };
  const bufs = [];
  collect(tr, bufs);
  if (curves) { msg.curve = net.kernelCurve(24, 192); msg.pair = net.kernelCurve(net.cfg.pp, 64); }
  postMessage(msg, bufs);
}

function loop() {
  if (!running || busy) return;
  busy = true;
  one();
  busy = false;
  setTimeout(loop, 0);
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "reset") {
    n = m.n; pos = Float64Array.from(m.pos); vel = Float64Array.from(m.vel); tick = 0; epoch = m.epoch; wantEnergy = m.energy;
    net.reset();
    send(0);
  } else if (m.cmd === "run") {
    running = m.run;
    if (running) loop();
  } else if (m.cmd === "step") {
    one();
  } else if (m.cmd === "energy") {
    wantEnergy = m.on;
    if (wantEnergy && net && n) postMessage({ type: "energy", epoch, energy: truth.energy(pos, vel, n) });
  } else if (m.cmd === "trace") {
    if (net && n) traceNow(m.curves, { pos, vel, n, tick, epoch });
  } else if (m.cmd === "tracetick") {
    if (net) traceNow(m.curves, { pos: Float64Array.from(m.pos), vel: Float64Array.from(m.vel), n: m.n, tick: m.tick, epoch: m.epoch, warm: m.warm });
  }
};

init().catch((err) => postMessage({ type: "error", message: err.message }));
