// From-scratch port of CentralForceDynamics (model/central_force.py): a
// single learned pairwise force f(log d) / (d^2 + 1), directed along the
// line between two bodies, summed over a neighbour-radius graph, integrated
// with velocity Verlet. No walls, no box, no gravity head -- unlike the
// token model this one never saw either during training, so none are
// ported (see docs/debugging/unified-dynamics-relativistic-speed-limit.md
// section 3 for why).

const W = 64;

export async function loadWeights(base = "weights") {
  const [manifest, buf] = await Promise.all([
    fetch(base + ".json").then((r) => r.json()),
    fetch(base + ".bin").then((r) => r.arrayBuffer()),
  ]);
  return parseWeights(manifest, buf);
}

export function parseWeights(manifest, buf) {
  const all = new Float32Array(buf);
  const w = {};
  for (const [name, t] of Object.entries(manifest.tensors)) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    w[name] = all.subarray(t.offset, t.offset + size);
  }
  return { w, config: manifest.config, checkpoint: manifest.checkpoint };
}

// Linear(1,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,1), same shape
// as the token model's RadialMlp; scalar in (log distance), scalar out.
class ForceMlp {
  constructor(w) {
    this.W0 = w["force.0.weight"]; this.b0 = w["force.0.bias"];
    this.W1 = w["force.2.weight"]; this.b1 = w["force.2.bias"];
    this.W2 = w["force.4.weight"]; this.b2 = w["force.4.bias"][0];
    this.h1 = new Float64Array(W);
    this.h2 = new Float64Array(W);
  }

  eval(x) {
    const { W0, b0, W1, b1, W2, h1, h2 } = this;
    for (let a = 0; a < W; a++) h1[a] = Math.tanh(W0[a] * x + b0[a]);
    let out = this.b2;
    for (let a = 0; a < W; a++) {
      let s = b1[a];
      const o = a * W;
      for (let b = 0; b < W; b++) s += W1[o + b] * h1[b];
      h2[a] = Math.tanh(s);
      out += W2[a] * h2[a];
    }
    return out;
  }
}

export class CentralNet {
  constructor({ w, config }) {
    this.cfg = config;
    this.forceMlp = new ForceMlp(w);
    this.cap = 0;
    this.grow(64);
  }

  grow(n) {
    if (n <= this.cap) return;
    this.cap = Math.max(n, this.cap * 2);
    this.a0 = new Float64Array(2 * this.cap);
    this.a1 = new Float64Array(2 * this.cap);
    this.pmid = new Float64Array(2 * this.cap);
  }

  // Undirected pairs within `radius`, each visited once, as
  // build_radius_graph (all-pairs, both directions) would produce, calling
  // visit(i, j, d, rx, ry) with rx, ry = pos[i] - pos[j]. Grid-bucketed
  // (string keys so the world can be unbounded, unlike the token model's
  // fixed 100x100 box), 3x3 cell lookup per body.
  forPairs(pos, count, radius, visit) {
    const heads = new Map();
    const cellX = new Int32Array(count), cellY = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      const x = pos[2 * i], y = pos[2 * i + 1];
      if (!(Number.isFinite(x) && Number.isFinite(y))) { cellX[i] = cellY[i] = 0x7fffffff; continue; }
      const cx = Math.floor(x / radius), cy = Math.floor(y / radius);
      cellX[i] = cx; cellY[i] = cy;
      const k = cx + "," + cy;
      const bucket = heads.get(k);
      if (bucket) bucket.push(i); else heads.set(k, [i]);
    }
    for (let i = 0; i < count; i++) {
      if (cellX[i] === 0x7fffffff) continue;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const bucket = heads.get((cellX[i] + dx) + "," + (cellY[i] + dy));
          if (!bucket) continue;
          for (const j of bucket) {
            if (j <= i) continue;
            const rx = pos[2 * i] - pos[2 * j], ry = pos[2 * i + 1] - pos[2 * j + 1];
            const d = Math.sqrt(rx * rx + ry * ry + 1e-12);
            if (d <= radius) visit(i, j, d, rx, ry);
          }
        }
      }
    }
  }

  // CentralForceDynamics.accel: acc[dst] += f(log d)/(d^2+1) * rel/d summed
  // over src within neighbor_radius, rel = pos[src] - pos[dst]. For an
  // unordered pair (i, j) with rx, ry = pos[i]-pos[j], the (src=i, dst=j)
  // edge adds to a[j] and the (src=j, dst=i) edge adds the negation to a[i].
  accel(p, count, a) {
    const { neighbor_radius: R } = this.cfg;
    a.fill(0, 0, 2 * count);
    this.forPairs(p, count, R, (i, j, d, rx, ry) => {
      const fp = this.forceMlp.eval(Math.log(d)) / (d * d + 1) / d;
      const fx = fp * rx, fy = fp * ry;
      a[2 * i] -= fx; a[2 * i + 1] -= fy;
      a[2 * j] += fx; a[2 * j + 1] += fy;
    });
  }

  // CentralForceDynamics.forward, velocity Verlet: writes pos/vel in place.
  step(pos, vel, count) {
    const { dt } = this.cfg;
    this.grow(count);
    const { a0, a1, pmid } = this;
    this.accel(pos, count, a0);
    for (let i = 0; i < 2 * count; i++) pmid[i] = pos[i] + vel[i] * dt + 0.5 * dt * dt * a0[i];
    this.accel(pmid, count, a1);
    for (let i = 0; i < 2 * count; i++) {
      const dp = 0.5 * dt * dt * a0[i];
      const dv = 0.5 * dt * (a0[i] + a1[i]);
      pos[i] += vel[i] * dt + dp;
      vel[i] += dv;
    }
  }

  // flat [i, j, i, j, ...] of pairs within `radius`, for drawing
  edgeList(pos, count, radius) {
    const out = [];
    this.forPairs(pos, count, radius, (i, j) => { out.push(i, j); });
    return out;
  }
}

// Zero-mean-momentum body spawn: constant density / virial-speed scaling
// relative to an 8-body baseline (scripts/gravity_sim.py init_bodies).
export function initBodies(n, rng, spread = 5.0, speed = 0.5) {
  const f = Math.pow(n / 8, 0.5), fv = Math.pow(n / 8, 0.25);
  const s = spread * f, v = speed * fv;
  const pos = new Float64Array(2 * n), vel = new Float64Array(2 * n);
  let mvx = 0, mvy = 0;
  for (let i = 0; i < n; i++) {
    pos[2 * i] = (rng() * 2 - 1) * s;
    pos[2 * i + 1] = (rng() * 2 - 1) * s;
    const vx = gauss(rng) * v, vy = gauss(rng) * v;
    vel[2 * i] = vx; vel[2 * i + 1] = vy;
    mvx += vx; mvy += vy;
  }
  mvx /= n; mvy /= n;
  for (let i = 0; i < n; i++) { vel[2 * i] -= mvx; vel[2 * i + 1] -= mvy; }
  return { pos, vel };
}

function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// small deterministic PRNG (mulberry32) so demo scenes are reproducible
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
