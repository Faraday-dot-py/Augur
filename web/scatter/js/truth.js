// Ground-truth softened-gravity leapfrog, ported from scripts/gravity_sim.py
// (accel/energy/rollout), for the ghost overlay and energy comparison.
const EPS2 = 0.5 * 0.5;

export function accel(pos, count, out) {
  out.fill(0, 0, 2 * count);
  for (let i = 0; i < count; i++) {
    for (let j = 0; j < count; j++) {
      if (i === j) continue;
      const dx = pos[2 * i] - pos[2 * j], dy = pos[2 * i + 1] - pos[2 * j + 1];
      const r2 = dx * dx + dy * dy + EPS2;
      const inv = Math.pow(r2, -1.5);
      out[2 * i] += dx * inv; out[2 * i + 1] += dy * inv;
    }
  }
}

export function energy(pos, vel, count) {
  let ke = 0;
  for (let i = 0; i < 2 * count; i++) ke += vel[i] * vel[i];
  ke *= 0.5;
  let pe = 0;
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      const dx = pos[2 * i] - pos[2 * j], dy = pos[2 * i + 1] - pos[2 * j + 1];
      pe -= 1 / Math.sqrt(dx * dx + dy * dy + EPS2);
    }
  }
  return ke + pe;
}

// One dt step, 4 leapfrog substeps (matches gravity_sim.rollout defaults).
export function step(pos, vel, count, dt, a, substeps = 4) {
  const h = dt / substeps;
  for (let s = 0; s < substeps; s++) {
    for (let i = 0; i < 2 * count; i++) vel[i] += 0.5 * h * a[i];
    for (let i = 0; i < 2 * count; i++) pos[i] += h * vel[i];
    accel(pos, count, a);
    for (let i = 0; i < 2 * count; i++) vel[i] += 0.5 * h * a[i];
  }
}
