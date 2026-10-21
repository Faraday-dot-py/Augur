// Tiled scatter-field model (Exp E weights, scripts/scatter_tiling.py TiledScatter) with the R1b far kernel (scripts/scatter_far.py).
// nt x nt tiles of 32 sim units, each running one copy of the same 128x128 U-Net on its square plus a 16-unit halo (G = 128, cell 0.5).
// Mechanisms: ghost bodies (neighbours' bodies inside a tile's extended box feed its scatter and pair term; only own bodies keep their
// force), halo field exchange (the recurrent potential's halo cells are overwritten by the neighbours' interiors), and a coarse global
// far term (cell 2, learned kernel on all mass, minus each tile's own-box coarse potential). Momentum is re-centred globally.
import { ScatterNet, Mlp2, fft2, mulberry32 } from "../../scatter/js/scatter_model.js";
export { parseWeights, mulberry32 } from "../../scatter/js/scatter_model.js";

export const TILE = 32, HALO = 16, FAR_C = 4, GT = 128, HG = 32, GC = 64;
const CG = GT / FAR_C, CT = (TILE / 0.5) / FAR_C, CH = HALO / 0.5 / FAR_C;

const pow2 = (n) => { let p = 1; while (p < n) p <<= 1; return p; };
const twCache = new Map();
function twiddles(n) {
  let t = twCache.get(n);
  if (!t) {
    t = { cos: new Float64Array(n / 2), sin: new Float64Array(n / 2) };
    for (let k = 0; k < n / 2; k++) { t.cos[k] = Math.cos(2 * Math.PI * k / n); t.sin[k] = Math.sin(2 * Math.PI * k / n); }
    twCache.set(n, t);
  }
  return t;
}

function avgPool(a, n, c) {
  const m = n / c, out = new Float32Array(m * m), inv = 1 / (c * c);
  for (let y = 0; y < m; y++) {
    for (let x = 0; x < m; x++) {
      let s = 0;
      for (let dy = 0; dy < c; dy++) for (let dx = 0; dx < c; dx++) s += a[(y * c + dy) * n + x * c + dx];
      out[y * m + x] = s * inv;
    }
  }
  return out;
}

// torch F.interpolate(scale_factor=s, mode="bilinear", align_corners=False)
function upBilinear(a, n, s) {
  const N = n * s, out = new Float32Array(N * N);
  const i0 = new Int32Array(N), i1 = new Int32Array(N), w = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const src = Math.max((i + 0.5) / s - 0.5, 0), f = Math.floor(src);
    i0[i] = f; i1[i] = f + (f < n - 1 ? 1 : 0); w[i] = src - f;
  }
  for (let y = 0; y < N; y++) {
    const ra = i0[y] * n, rb = i1[y] * n, wy = w[y];
    for (let x = 0; x < N; x++) {
      const wx = w[x], a0 = a[ra + i0[x]], a1 = a[ra + i1[x]], b0 = a[rb + i0[x]], b1 = a[rb + i1[x]];
      out[y * N + x] = (a0 * (1 - wx) + a1 * wx) * (1 - wy) + (b0 * (1 - wx) + b1 * wx) * wy;
    }
  }
  return out;
}

export class TileNet extends ScatterNet {
  buildKernel() {
    this.kfar = new Mlp2(this.w, "kfar");
    this.kcache = new Map();
    const k = this.kernelFFT(this.n2, this.h);
    this.Kre = k.re; this.Kim = k.im;
  }

  // R1b potential kernel K(r) with the short-range split taper
  kfun(r) {
    const t = Math.min(r / this.cfg.pp, 1);
    return this.kfar.eval(r, Math.log(r + 0.5)) * (1 - (1 - t * t) ** 2);
  }

  kernelFFT(P, h) {
    const key = P + "_" + h;
    let k = this.kcache.get(key);
    if (k) return k;
    const half = P / 2, vals = new Float64Array((half + 1) * (half + 1));
    for (let i = 0; i <= half; i++) {
      for (let j = i; j <= half; j++) vals[i * (half + 1) + j] = vals[j * (half + 1) + i] = this.kfun(Math.sqrt((i * h) ** 2 + (j * h) ** 2));
    }
    const re = new Float64Array(P * P), im = new Float64Array(P * P);
    for (let i = 0; i < P; i++) {
      const di = Math.min(i, P - i);
      for (let j = 0; j < P; j++) re[i * P + j] = vals[di * (half + 1) + Math.min(j, P - j)];
    }
    fft2(re, im, P, false, twiddles(P));
    k = { re, im };
    this.kcache.set(key, k);
    return k;
  }

  // linear convolution of a Gv x Gv density with K on cell size h, times h^2 (zero-padded to a power of two >= 2 Gv)
  convPot(rho, Gv, h) {
    const P = pow2(2 * Gv), tw = twiddles(P), K = this.kernelFFT(P, h);
    const re = new Float64Array(P * P), im = new Float64Array(P * P);
    for (let y = 0; y < Gv; y++) for (let x = 0; x < Gv; x++) re[y * P + x] = rho[y * Gv + x];
    fft2(re, im, P, false, tw);
    for (let i = 0; i < P * P; i++) {
      const a = re[i], b = im[i];
      re[i] = a * K.re[i] - b * K.im[i];
      im[i] = a * K.im[i] + b * K.re[i];
    }
    fft2(re, im, P, true, tw);
    const out = new Float32Array(Gv * Gv), h2 = h * h;
    for (let y = 0; y < Gv; y++) for (let x = 0; x < Gv; x++) out[y * Gv + x] = re[y * P + x] * h2;
    return out;
  }

  // top-16 nearest neighbours within pp among all m packed bodies, queried for the first nOwn only
  pairOwn(P, m, nOwn, out) {
    const { cfg, ppmlp } = this;
    const pp = cfg.pp, heads = new Map(), next = new Int32Array(m).fill(-1);
    const cx = new Int32Array(m), cy = new Int32Array(m);
    for (let i = 0; i < m; i++) {
      cx[i] = Math.floor(P[2 * i] / pp); cy[i] = Math.floor(P[2 * i + 1] / pp);
      const key = cx[i] * 1048576 + cy[i];
      next[i] = heads.has(key) ? heads.get(key) : -1;
      heads.set(key, i);
    }
    const cand = [];
    for (let i = 0; i < nOwn; i++) {
      cand.length = 0;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          let j = heads.get((cx[i] + dx) * 1048576 + (cy[i] + dy));
          if (j === undefined) continue;
          for (; j >= 0; j = next[j]) {
            if (j === i) continue;
            const ddx = P[2 * j] - P[2 * i], ddy = P[2 * j + 1] - P[2 * i + 1];
            const r = Math.sqrt(ddx * ddx + ddy * ddy + 1e-8);
            if (r < pp) cand.push([r, ddx, ddy]);
          }
        }
      }
      if (cand.length > cfg.knn) { cand.sort((a, b) => a[0] - b[0]); cand.length = cfg.knn; }
      for (const [r, ddx, ddy] of cand) {
        const t = r / pp, win = (1 - t * t) ** 2, wgt = ppmlp.eval(r, Math.log(r + 0.05)) * win / r;
        out[2 * i] += wgt * ddx;
        out[2 * i + 1] += wgt * ddy;
      }
    }
  }
}

// the tiles held by one worker (or by the page itself): per-tile recurrent potential and the two force phases
export class TileSet {
  constructor(weights) {
    this.net = new TileNet(weights);
    this.cfg = weights.config;
    this.st = new Map();
    this.field = new Map();
  }

  handle(m) {
    switch (m.cmd) {
      case "reset": this.field.clear(); this.st.clear(); return [{}, []];
      case "scatter": return [this.scatter(m.tiles), []];
      case "finish": return this.finish(m);
      case "getField": return [{ f: this.field.get(m.k) }, []];
      case "setField": for (const t of m.tiles) this.field.set(t.k, t.f); return [{}, []];
      default: throw new Error("unknown cmd " + m.cmd);
    }
  }

  // CIC-scatter own and ghost bodies, convolve with the learned kernel (local part of the far field)
  scatter(tiles) {
    const net = this.net, GG = GT * GT, sc = net.cfg.in_scale / (net.h * net.h), res = [];
    for (const t of tiles) {
      const x = new Float32Array(6 * GG), { P, V, msk } = t, m = msk.length;
      const sp = new Float64Array(2 * m);
      let c = 0;
      const sv = new Float64Array(2 * m);
      for (let i = 0; i < m; i++) if (msk[i]) { sp[2 * c] = P[2 * i]; sp[2 * c + 1] = P[2 * i + 1]; sv[2 * c] = V[2 * i]; sv[2 * c + 1] = V[2 * i + 1]; c++; }
      net.corners(sp, c, (f, wt, i) => {
        x[f] += wt * sc;
        x[GG + f] += wt * sv[2 * i] * sc;
        x[2 * GG + f] += wt * sv[2 * i + 1] * sc;
      });
      const phiK = net.convPot(x.subarray(0, GG), GT, net.h);
      const rc = avgPool(x.subarray(0, GG), GT, FAR_C);
      this.st.set(t.k, { x, phiK, rc, P, nOwn: t.nOwn, msk });
      res.push({ k: t.k, rc });
    }
    return { tiles: res };
  }

  finish(m) {
    const net = this.net, GG = GT * GT, out = [], tr = [];
    for (const t of m.tiles) {
      const s = this.st.get(t.k), { x, P, nOwn, msk } = s;
      const kx = new Float32Array(GG), ky = new Float32Array(GG);
      net.negGrad(s.phiK, kx, ky);
      let far = null;
      if (t.win) {
        const own = net.convPot(s.rc, CG, net.h * FAR_C);
        far = new Float32Array(CG * CG);
        for (let i = 0; i < far.length; i++) far[i] = t.win[i] - own[i];
        const fine = upBilinear(far, CG, FAR_C), fx = new Float32Array(GG), fy = new Float32Array(GG);
        net.negGrad(fine, fx, fy);
        for (let i = 0; i < GG; i++) { kx[i] += fx[i]; ky[i] += fy[i]; }
      }
      x.set(kx, 3 * GG);
      x.set(ky, 4 * GG);
      x.set(this.field.get(t.k) || new Float32Array(GG), 5 * GG);
      const ut = m.unet === t.k ? { dec: [] } : null;
      const phi = net.unet(x, ut);
      this.field.set(t.k, phi);
      const gx = new Float32Array(GG), gy = new Float32Array(GG);
      net.negGrad(phi, gx, gy);
      for (let i = 0; i < GG; i++) { gx[i] += kx[i]; gy[i] += ky[i]; }
      const acc = new Float64Array(2 * nOwn);
      const sp = new Float64Array(2 * nOwn);
      let c = 0;
      const idx = [];
      for (let i = 0; i < nOwn; i++) if (msk[i]) { sp[2 * c] = P[2 * i]; sp[2 * c + 1] = P[2 * i + 1]; idx.push(i); c++; }
      net.corners(sp, c, (f, wt, i) => {
        acc[2 * idx[i]] += gx[f] * wt;
        acc[2 * idx[i] + 1] += gy[f] * wt;
      });
      net.pairOwn(P, msk.length, nOwn, acc);
      const phiInt = new Float32Array(GC * GC);
      for (let y = 0; y < GC; y++) phiInt.set(phi.subarray((HG + y) * GT + HG, (HG + y) * GT + HG + GC), y * GC);
      const r = { k: t.k, acc, phiInt };
      const bufs = [acc.buffer, phiInt.buffer];
      if (m.trace) {
        r.rho = x.slice(0, GG); r.phiK = s.phiK; r.far = far; r.phi = phi;
        bufs.push(r.rho.buffer);
        if (far) bufs.push(far.buffer);
        if (ut) { r.unet = { inp: ut.inp, enc: ut.enc, dec: ut.dec }; }
      }
      out.push(r);
      tr.push(...bufs);
    }
    return [{ tiles: out }, tr];
  }
}

export class LocalPool {
  constructor(weights) { this.set = new TileSet(weights); this.size = 1; }
  async call(i, m) { return this.set.handle(m)[0]; }
}

// geometry and the global parts (packing, far term, halo exchange, momentum re-centring); `pool.call(worker, msg)` runs tile work
export class TiledDriver {
  constructor(weights, pool, nt) {
    this.cfg = weights.config;
    this.net = new TileNet(weights);
    this.pool = pool;
    this.nt = nt; this.T = nt * nt;
    this.X0 = -nt * TILE / 2;
    this.flags = { ghosts: true, fx: true, far: true };
    this.dt = this.cfg.dt;
    this.aPrev = null;
    this.inspect = -1;
  }

  tileOf(k) { return [Math.floor(k / this.nt), k % this.nt]; }

  centre(k) { const [tx, ty] = this.tileOf(k); return [this.X0 + (tx + 0.5) * TILE, this.X0 + (ty + 0.5) * TILE]; }

  async reset() {
    this.aPrev = null;
    await Promise.all(Array.from({ length: this.pool.size }, (_, w) => this.pool.call(w, { cmd: "reset" })));
  }

  worker(k) { return k % this.pool.size; }

  pack(pos, n) {
    const { nt, T, X0, flags } = this, half = nt * TILE / 2 + 0.25, E = TILE + 2 * HALO;
    const own = Array.from({ length: T }, () => []), gh = Array.from({ length: T }, () => []);
    const tid = new Int32Array(n), inside = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const x = pos[2 * i], y = pos[2 * i + 1];
      const tx = Math.min(nt - 1, Math.max(0, Math.floor((x - X0) / TILE))), ty = Math.min(nt - 1, Math.max(0, Math.floor((y - X0) / TILE)));
      tid[i] = tx * nt + ty;
      own[tid[i]].push(i);
      inside[i] = Math.abs(x) < half && Math.abs(y) < half ? 1 : 0;
    }
    if (flags.ghosts) {
      for (let i = 0; i < n; i++) {
        if (!inside[i]) continue;
        const x = pos[2 * i], y = pos[2 * i + 1], tx = Math.floor(tid[i] / nt), ty = tid[i] % nt;
        for (let a = Math.max(0, tx - 1); a <= Math.min(nt - 1, tx + 1); a++) {
          for (let b = Math.max(0, ty - 1); b <= Math.min(nt - 1, ty + 1); b++) {
            if (a === tx && b === ty) continue;
            const lx = X0 + a * TILE - HALO, ly = X0 + b * TILE - HALO;
            if (x >= lx && x < lx + E && y >= ly && y < ly + E) gh[a * nt + b].push(i);
          }
        }
      }
    }
    const geo = [];
    for (let k = 0; k < T; k++) {
      const idx = Int32Array.from(own[k].concat(gh[k])), m = idx.length, [cx, cy] = this.centre(k);
      const P = new Float64Array(2 * m), V = new Float64Array(2 * m), msk = new Uint8Array(m);
      for (let j = 0; j < m; j++) {
        const i = idx[j];
        P[2 * j] = pos[2 * i] - cx; P[2 * j + 1] = pos[2 * i + 1] - cy;
        msk[j] = inside[i];
        if (this.vel) { V[2 * j] = this.vel[2 * i]; V[2 * j + 1] = this.vel[2 * i + 1]; }
      }
      geo.push({ k, idx, P, V, msk, nOwn: own[k].length, ghosts: gh[k] });
    }
    return { geo, tid };
  }

  // acceleration of every body; `trace` also returns the per-tile tensors (and the U-Net activations of the inspected tile)
  async force(pos, vel, n, trace = false) {
    const { T, nt, pool, flags } = this;
    this.vel = vel;
    const { geo, tid } = this.pack(pos, n);
    const groups = Array.from({ length: pool.size }, () => []);
    for (const g of geo) groups[this.worker(g.k)].push(g);
    const A = await Promise.all(groups.map((g, w) => (g.length ? pool.call(w, { cmd: "scatter", tiles: g.map((t) => ({ k: t.k, P: t.P, V: t.V, msk: t.msk, nOwn: t.nOwn })) }) : { tiles: [] })));
    const rc = new Array(T);
    for (const a of A) for (const t of a.tiles) rc[t.k] = t.rc;
    let wins = null;
    if (flags.far) {
      const Gp = nt * CT + 2 * CH, Rp = new Float32Array(Gp * Gp);
      for (let k = 0; k < T; k++) {
        const [tx, ty] = this.tileOf(k);
        for (let y = 0; y < CT; y++) {
          for (let x = 0; x < CT; x++) Rp[(CH + ty * CT + y) * Gp + CH + tx * CT + x] = rc[k][(CH + y) * CG + CH + x];
        }
      }
      const phig = this.net.convPot(Rp, Gp, this.net.h * FAR_C);
      wins = [];
      for (let k = 0; k < T; k++) {
        const [tx, ty] = this.tileOf(k), w = new Float32Array(CG * CG);
        for (let y = 0; y < CG; y++) for (let x = 0; x < CG; x++) w[y * CG + x] = phig[(ty * CT + y) * Gp + tx * CT + x];
        wins.push(w);
      }
      this.coarse = { rho: Rp, phi: phig, G: Gp };
    } else this.coarse = null;
    const B = await Promise.all(groups.map((g, w) => (g.length
      ? pool.call(w, { cmd: "finish", tiles: g.map((t) => ({ k: t.k, win: wins ? wins[t.k] : null })), trace, unet: trace ? this.inspect : -1 })
      : { tiles: [] })));
    const acc = new Float64Array(2 * n), res = new Array(T);
    for (const b of B) {
      for (const t of b.tiles) {
        res[t.k] = t;
        const g = geo[t.k];
        for (let j = 0; j < g.nOwn; j++) { acc[2 * g.idx[j]] = t.acc[2 * j]; acc[2 * g.idx[j] + 1] = t.acc[2 * j + 1]; }
      }
    }
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += acc[2 * i]; my += acc[2 * i + 1]; }
    mx /= n; my /= n;
    for (let i = 0; i < n; i++) { acc[2 * i] -= mx; acc[2 * i + 1] -= my; }
    if (flags.fx) await this.exchange(res);
    return { acc, geo, tid, tiles: res, momfix: [mx, my] };
  }

  // overwrite every tile's recurrent potential halo with the neighbours' interiors (zeros outside the domain)
  async exchange(res) {
    const { nt, T, pool } = this, N = nt * GC, p = N + 2 * HG, glob = new Float32Array(p * p);
    for (let k = 0; k < T; k++) {
      const [tx, ty] = this.tileOf(k);
      for (let y = 0; y < GC; y++) glob.set(res[k].phiInt.subarray(y * GC, (y + 1) * GC), (HG + ty * GC + y) * p + HG + tx * GC);
    }
    const fields = [];
    for (let k = 0; k < T; k++) {
      const [tx, ty] = this.tileOf(k), f = new Float32Array(GT * GT);
      for (let y = 0; y < GT; y++) f.set(glob.subarray((ty * GC + y) * p + tx * GC, (ty * GC + y) * p + tx * GC + GT), y * GT);
      fields.push({ k, f });
    }
    await this.setFields(fields);
  }

  async setFields(fields) {
    const groups = Array.from({ length: this.pool.size }, () => []);
    for (const f of fields) groups[this.worker(f.k)].push(f);
    await Promise.all(groups.map((g, w) => (g.length ? this.pool.call(w, { cmd: "setField", tiles: g }) : null)));
  }

  // seeds the Verlet scheme: one force pass at the initial state
  async seed(pos, vel, n, trace = false) {
    const r = await this.force(pos, vel, n, trace);
    this.aPrev = Float64Array.from(r.acc);
    this.n = n;
    return r;
  }

  // one velocity-Verlet step in place; the returned pass is the end-of-step force evaluation
  async step(pos, vel, n, trace = false) {
    if (!this.aPrev || this.n !== n) await this.seed(pos, vel, n);
    const { dt, aPrev } = this, vHalf = new Float64Array(2 * n);
    for (let i = 0; i < 2 * n; i++) {
      pos[i] += dt * vel[i] + 0.5 * dt * dt * aPrev[i];
      vHalf[i] = vel[i] + dt * aPrev[i];
    }
    const r = await this.force(pos, vHalf, n, trace);
    for (let i = 0; i < 2 * n; i++) {
      vel[i] += 0.5 * dt * (aPrev[i] + r.acc[i]);
      aPrev[i] = r.acc[i];
    }
    return r;
  }

  // resume from a displayed tick: per-tile potential windows (previous end-of-tick pass) and that tick's accelerations
  async restore(acc, phis) {
    this.aPrev = Float64Array.from(acc);
    this.n = acc.length / 2;
    const res = phis.map((phi) => {
      const phiInt = new Float32Array(GC * GC);
      for (let y = 0; y < GC; y++) phiInt.set(phi.subarray((HG + y) * GT + HG, (HG + y) * GT + HG + GC), y * GC);
      return { phiInt };
    });
    if (this.flags.fx) await this.exchange(res);
    else await this.setFields(phis.map((f, k) => ({ k, f: Float32Array.from(f) })));
  }
}

// R1b kernel force curve against the shipped kmlp and the exact softened law, for the kernel chart (force = dK/dr, inward)
export function kernelCurves(net, rmax, m) {
  const r = new Float32Array(m), learned = new Float32Array(m), shipped = new Float32Array(m), exact = new Float32Array(m);
  const old = (x) => {
    const t = Math.min(x / net.cfg.pp, 1);
    return net.kmlp.eval(x, Math.log(x + net.h)) * (1 - (1 - t * t) ** 2);
  };
  const d = 1e-3;
  for (let i = 0; i < m; i++) {
    const x = r[i] = rmax * (i + 1) / m;
    learned[i] = (net.kfun(x + d) - net.kfun(x - d)) / (2 * d);
    shipped[i] = (old(x + d) - old(x - d)) / (2 * d);
    exact[i] = x / (x * x + 0.25) ** 1.5;
  }
  return { r, learned, shipped, exact };
}

function gauss(rng) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

// Gaussian clusters as in the tiling evaluation (clipped to 0.45 of the domain, per-component speed vfac sqrt(n / 4 sigma), momentum removed)
export const SCENES = {
  collapse: { label: "Cold collapse", vfac: 0.3, clusters: 1 },
  warm: { label: "Warm cluster", vfac: 1.0, clusters: 1 },
  cross: { label: "Two clusters crossing", vfac: 0.3, clusters: 2 },
};

export function makeScene(kind, nt, n, seed = 4738) {
  const rng = mulberry32(seed), sc = SCENES[kind], L = nt * TILE, lim = 0.45 * L;
  const sigma = (nt === 2 ? 5 : 8) * Math.sqrt(n / (nt === 2 ? 100 : 300));
  const pos = new Float64Array(2 * n), vel = new Float64Array(2 * n);
  const per = sc.clusters === 2 ? Math.floor(n / 2) : n;
  for (let i = 0; i < n; i++) {
    const c = sc.clusters === 2 ? (i < per ? 0 : 1) : 0, s = sc.clusters === 2 ? sigma * 0.6 : sigma, m = sc.clusters === 2 ? (c ? n - per : per) : n;
    const cx = sc.clusters === 2 ? (c ? 0.2 : -0.2) * L : 0, cy = sc.clusters === 2 ? (c ? 0.05 : -0.05) * L : 0;
    const drift = sc.clusters === 2 ? (c ? -1 : 1) * 0.04 * L : 0;
    pos[2 * i] = Math.max(-lim, Math.min(lim, cx + gauss(rng) * s));
    pos[2 * i + 1] = Math.max(-lim, Math.min(lim, cy + gauss(rng) * s));
    const sd = sc.vfac * Math.sqrt(m / (4 * s));
    vel[2 * i] = gauss(rng) * sd + drift;
    vel[2 * i + 1] = gauss(rng) * sd + drift * 0.25;
  }
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += vel[2 * i]; my += vel[2 * i + 1]; }
  for (let i = 0; i < n; i++) { vel[2 * i] -= mx / n; vel[2 * i + 1] -= my / n; }
  return { pos, vel };
}
