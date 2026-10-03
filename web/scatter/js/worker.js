import { loadWeights, ScatterNet } from "./scatter_model.js";

let net, pos, vel, n = 0, tick = 0, running = false, busy = false;

async function init() {
  const w = await loadWeights(new URL("../weights", import.meta.url).href);
  net = new ScatterNet(w);
  postMessage({ type: "ready", config: w.config, checkpoint: w.checkpoint, iters: w.iters });
}

function one() {
  const t0 = performance.now();
  net.step(pos, vel, n);
  tick++;
  postMessage({ type: "state", tick, ms: performance.now() - t0, pos: pos.slice(), vel: vel.slice(), field: net.field.slice() });
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
    n = m.n; pos = Float64Array.from(m.pos); vel = Float64Array.from(m.vel); tick = 0;
    net.reset();
    postMessage({ type: "state", tick, ms: 0, pos: pos.slice(), vel: vel.slice(), field: net.field.slice() });
  } else if (m.cmd === "run") {
    running = m.run;
    if (running) loop();
  } else if (m.cmd === "step") {
    one();
  }
};

init().catch((err) => postMessage({ type: "error", message: err.message }));
