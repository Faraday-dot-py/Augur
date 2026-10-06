import * as truth from "./truth.js";

let n = 0, dt = 0.1, pos, vel, acc, tick = 0, target = 0, epoch = 0, pumping = false, ms = 0;

async function pump() {
  if (pumping) return;
  pumping = true;
  let slice = performance.now();
  while (tick < target) {
    const t0 = performance.now();
    truth.step(pos, vel, n, dt, acc);
    tick++;
    ms = performance.now() - t0;
    if (tick === target) {
      const p = pos.slice();
      postMessage({ type: "truth", epoch, tick, ms, pos: p, energy: truth.energy(pos, vel, n) }, [p.buffer]);
    }
    if (performance.now() - slice > 10) {
      await new Promise((r) => setTimeout(r, 0));
      slice = performance.now();
    }
  }
  pumping = false;
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "init") {
    n = m.n; dt = m.dt; epoch = m.epoch; tick = m.tick; target = m.tick;
    pos = Float64Array.from(m.pos); vel = Float64Array.from(m.vel);
    acc = new Float64Array(2 * n);
    truth.accel(pos, n, acc);
    const p = pos.slice();
    postMessage({ type: "truth", epoch, tick, ms: 0, pos: p, energy: truth.energy(pos, vel, n) }, [p.buffer]);
  } else if (m.cmd === "advance" && m.epoch === epoch) {
    if (m.tick > target) target = m.tick;
    pump();
  }
};
