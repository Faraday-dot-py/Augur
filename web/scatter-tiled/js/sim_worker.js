import * as truth from "../../scatter/js/truth.js";

let n = 0, dt = 0.1, pos, vel, acc, tick = 0, target = 0, epoch = 0, pumping = false;

function post() {
  const p = pos.slice(), v = vel.slice(), a = acc.slice();
  postMessage({ type: "sim", epoch, tick, pos: p, vel: v, acc: a, energy: truth.energy(pos, vel, n) }, [p.buffer, v.buffer, a.buffer]);
}

async function pump() {
  if (pumping) return;
  pumping = true;
  let slice = performance.now();
  const ep = epoch;
  while (ep === epoch && tick < target) {
    truth.step(pos, vel, n, dt, acc);
    tick++;
    post();
    if (performance.now() - slice > 10) {
      await new Promise((r) => setTimeout(r, 0));
      slice = performance.now();
    }
  }
  pumping = false;
  if (ep !== epoch) pump();
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "init") {
    n = m.n; dt = m.dt; epoch = m.epoch; tick = m.tick0 || 0; target = m.limit;
    pos = Float64Array.from(m.pos); vel = Float64Array.from(m.vel);
    acc = new Float64Array(2 * n);
    truth.accel(pos, n, acc);
    post();
    pump();
  } else if (m.cmd === "stop") {
    target = -1;
  } else if (m.cmd === "want" && m.epoch === epoch) {
    if (m.limit > target) target = m.limit;
    pump();
  }
};
