// From-scratch port of ScatterField (scripts/scatter_field.py), variant ms_kp_pot_v_g128:
// tokens (pos, vel, unit mass) are CIC-scattered onto a 128x128 grid as [m, m*vx, m*vy];
// a learned kernel (FFT convolution of the density) gives a far-field acceleration; a
// 5-level UNet reads [scatter, kernel accel, previous potential] and outputs a potential
// whose negative gradient is bilinearly gathered at each token; a learned short-range
// pair term (|r| < 2, 16 nearest) is added, momentum is re-centred, and velocity Verlet
// integrates with one force evaluation per step. Positions are relative to the arena
// centre; tokens outside the grid get no grid force.

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
  return { w, config: manifest.config, checkpoint: manifest.checkpoint, iters: manifest.iters };
}

function erf(x) {
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.5 * x);
  const y = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return s * y;
}
const SQRT1_2 = Math.SQRT1_2;
const gelu = (x) => 0.5 * x * (1 + erf(x * SQRT1_2));

function geluInPlace(a) {
  for (let i = 0; i < a.length; i++) a[i] = gelu(a[i]);
}

// 3x3 same-padded conv, planes laid out [channel][y][x]; w is [cout][cin][3][3].
// Output channels are blocked by 4 (shared input loads) on a zero-padded copy of the input.
function conv3(x, cin, H, W, w, b, cout) {
  if (cout % 4) return conv3Plain(x, cin, H, W, w, b, cout);
  const Wp = W + 2, pp = (H + 2) * Wp;
  const xp = new Float32Array(cin * pp);
  for (let c = 0; c < cin; c++) {
    for (let y = 0; y < H; y++) xp.set(x.subarray(c * H * W + y * W, c * H * W + y * W + W), c * pp + (y + 1) * Wp + 1);
  }
  const out = new Float32Array(cout * H * W);
  const a0 = new Float32Array(W), a1 = new Float32Array(W), a2 = new Float32Array(W), a3 = new Float32Array(W);
  for (let cb = 0; cb < cout; cb += 4) {
    for (let y = 0; y < H; y++) {
      a0.fill(b[cb]); a1.fill(b[cb + 1]); a2.fill(b[cb + 2]); a3.fill(b[cb + 3]);
      for (let ci = 0; ci < cin; ci++) {
        for (let ky = 0; ky < 3; ky++) {
          const ib = ci * pp + (y + ky) * Wp;
          const o0 = (cb * cin + ci) * 9 + ky * 3, o1 = ((cb + 1) * cin + ci) * 9 + ky * 3;
          const o2 = ((cb + 2) * cin + ci) * 9 + ky * 3, o3 = ((cb + 3) * cin + ci) * 9 + ky * 3;
          const w00 = w[o0], w01 = w[o0 + 1], w02 = w[o0 + 2], w10 = w[o1], w11 = w[o1 + 1], w12 = w[o1 + 2];
          const w20 = w[o2], w21 = w[o2 + 1], w22 = w[o2 + 2], w30 = w[o3], w31 = w[o3 + 1], w32 = w[o3 + 2];
          let p0 = xp[ib], p1 = xp[ib + 1];
          for (let xx = 0; xx < W; xx++) {
            const p2 = xp[ib + xx + 2];
            a0[xx] += w00 * p0 + w01 * p1 + w02 * p2;
            a1[xx] += w10 * p0 + w11 * p1 + w12 * p2;
            a2[xx] += w20 * p0 + w21 * p1 + w22 * p2;
            a3[xx] += w30 * p0 + w31 * p1 + w32 * p2;
            p0 = p1; p1 = p2;
          }
        }
      }
      out.set(a0, cb * H * W + y * W); out.set(a1, (cb + 1) * H * W + y * W);
      out.set(a2, (cb + 2) * H * W + y * W); out.set(a3, (cb + 3) * H * W + y * W);
    }
  }
  return out;
}

function conv3Plain(x, cin, H, W, w, b, cout) {
  const out = new Float32Array(cout * H * W);
  const plane = H * W;
  for (let co = 0; co < cout; co++) {
    const o = co * plane;
    out.fill(b[co], o, o + plane);
    for (let ci = 0; ci < cin; ci++) {
      const xi = ci * plane;
      const wb = (co * cin + ci) * 9;
      for (let ky = 0; ky < 3; ky++) {
        const dy = ky - 1;
        const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
        for (let kx = 0; kx < 3; kx++) {
          const wv = w[wb + ky * 3 + kx];
          const dx = kx - 1;
          const x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
          for (let y = y0; y < y1; y++) {
            let io = o + y * W + x0, ii = xi + (y + dy) * W + x0 + dx;
            for (let xx = x0; xx < x1; xx++) out[io++] += wv * x[ii++];
          }
        }
      }
    }
  }
  return out;
}

function avgPool(x, C, H, W) {
  const h2 = H >> 1, w2 = W >> 1;
  const out = new Float32Array(C * h2 * w2);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < h2; y++) {
      for (let xx = 0; xx < w2; xx++) {
        const i = c * H * W + 2 * y * W + 2 * xx;
        out[c * h2 * w2 + y * w2 + xx] = 0.25 * (x[i] + x[i + 1] + x[i + W] + x[i + W + 1]);
      }
    }
  }
  return out;
}

// nearest x2 upsample of h (C planes) concatenated with skip (C planes): [up, skip]
function upCat(h, skip, C, H, W) {
  const H2 = H * 2, W2 = W * 2;
  const out = new Float32Array(2 * C * H2 * W2);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < H2; y++) {
      const src = c * H * W + (y >> 1) * W;
      const dst = c * H2 * W2 + y * W2;
      for (let xx = 0; xx < W2; xx++) out[dst + xx] = h[src + (xx >> 1)];
    }
  }
  out.set(skip, C * H2 * W2);
  return out;
}

// in-place radix-2 complex FFT of length n on interleaved-free re/im arrays with stride
function fft1(re, im, off, stride, n, inverse, tw) {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const a = off + i * stride, b = off + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t;
      t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  const sgn = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = tw.cos[k * step], wi = sgn * tw.sin[k * step];
        const a = off + (i + k) * stride, b = off + (i + k + half) * stride;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

function fft2(re, im, n, inverse, tw) {
  for (let r = 0; r < n; r++) fft1(re, im, r * n, 1, n, inverse, tw);
  for (let c = 0; c < n; c++) fft1(re, im, c, n, n, inverse, tw);
  if (inverse) {
    const s = 1 / (n * n);
    for (let i = 0; i < n * n; i++) { re[i] *= s; im[i] *= s; }
  }
}

class Mlp2 {
  // Linear(2,64) GELU Linear(64,64) GELU Linear(64,1)
  constructor(w, p) {
    this.W0 = w[p + ".0.weight"]; this.b0 = w[p + ".0.bias"];
    this.W1 = w[p + ".2.weight"]; this.b1 = w[p + ".2.bias"];
    this.W2 = w[p + ".4.weight"]; this.b2 = w[p + ".4.bias"][0];
    this.h1 = new Float64Array(64);
    this.h2 = new Float64Array(64);
  }

  eval(a, b) {
    const { W0, b0, W1, b1, W2, h1, h2 } = this;
    for (let i = 0; i < 64; i++) h1[i] = gelu(W0[2 * i] * a + W0[2 * i + 1] * b + b0[i]);
    let out = this.b2;
    for (let i = 0; i < 64; i++) {
      let s = b1[i];
      const o = i * 64;
      for (let j = 0; j < 64; j++) s += W1[o + j] * h1[j];
      out += W2[i] * gelu(s);
    }
    return out;
  }
}

export class ScatterNet {
  constructor({ w, config }) {
    this.cfg = config;
    this.w = w;
    const G = this.G = config.grid;
    this.h = config.extent / G;
    this.dt = config.dt;
    this.kmlp = new Mlp2(w, "kmlp");
    this.ppmlp = new Mlp2(w, "ppmlp");
    const n2 = 2 * G;
    this.n2 = n2;
    this.tw = { cos: new Float64Array(n2 / 2), sin: new Float64Array(n2 / 2) };
    for (let k = 0; k < n2 / 2; k++) {
      this.tw.cos[k] = Math.cos(2 * Math.PI * k / n2);
      this.tw.sin[k] = Math.sin(2 * Math.PI * k / n2);
    }
    this.buildKernel();
    this.field = new Float32Array(G * G);
    this.aPrev = null;
    this.rho = new Float32Array(G * G);
    this.cap = 0;
    this.grow(128);
  }

  grow(n) {
    if (n <= this.cap) return;
    this.cap = Math.max(n, this.cap * 2);
    this.aTok = new Float64Array(2 * this.cap);
    this.aPrev = null;
  }

  // K(r) from the kernel MLP with the short-range split taper, on the 2G x 2G periodic
  // distance grid used for linear convolution; stored as its 2D FFT.
  buildKernel() {
    const { G, n2, h } = this;
    const cfg = this.cfg;
    const re = new Float64Array(n2 * n2), im = new Float64Array(n2 * n2);
    const vals = new Float64Array((G + 1) * (G + 1));
    for (let i = 0; i <= G; i++) {
      for (let j = i; j <= G; j++) {
        const r = Math.sqrt((i * h) ** 2 + (j * h) ** 2);
        let k = this.kmlp.eval(r, Math.log(r + h));
        if (cfg.split) {
          const t = Math.min(r / cfg.pp, 1);
          k *= 1 - (1 - t * t) ** 2;
        }
        vals[i * (G + 1) + j] = vals[j * (G + 1) + i] = k;
      }
    }
    for (let i = 0; i < n2; i++) {
      const di = Math.min(i, n2 - i);
      for (let j = 0; j < n2; j++) re[i * n2 + j] = vals[di * (G + 1) + Math.min(j, n2 - j)];
    }
    fft2(re, im, n2, false, this.tw);
    this.Kre = re; this.Kim = im;
  }

  reset() {
    this.field.fill(0);
    this.aPrev = null;
  }

  // CIC corner iteration shared by scatter and gather: calls fn(flatIndex, weight, token)
  corners(pos, n, fn) {
    const { G, h, cfg } = this;
    const half = cfg.extent / 2;
    for (let i = 0; i < n; i++) {
      const ux = (pos[2 * i] + half) / h - 0.5, uy = (pos[2 * i + 1] + half) / h - 0.5;
      const ix = Math.floor(ux), iy = Math.floor(uy);
      const fx = ux - ix, fy = uy - iy;
      for (let oy = 0; oy < 2; oy++) {
        const yy = iy + oy;
        if (yy < 0 || yy >= G) continue;
        const wy = oy ? fy : 1 - fy;
        for (let ox = 0; ox < 2; ox++) {
          const xx = ix + ox;
          if (xx < 0 || xx >= G) continue;
          fn(yy * G + xx, (ox ? fx : 1 - fx) * wy, i);
        }
      }
    }
  }

  // grid net on [scatter(3), kernel accel(2), previous potential(1)] -> potential (G*G)
  unet(x) {
    const { w, G } = this;
    const L = this.cfg.levels;
    let h = conv3(x, 6, G, G, w["net.inp.weight"], w["net.inp.bias"], 32);
    geluInPlace(h);
    const block = (inp, cin, H, W, name) => {
      let a = conv3(inp, cin, H, W, w[name + ".0.weight"], w[name + ".0.bias"], 32);
      geluInPlace(a);
      a = conv3(a, 32, H, W, w[name + ".2.weight"], w[name + ".2.bias"], 32);
      geluInPlace(a);
      return a;
    };
    h = block(h, 32, G, G, "net.enc.0");
    const skips = [h];
    let H = G;
    for (let l = 1; l <= L; l++) {
      h = avgPool(h, 32, H, H);
      H >>= 1;
      h = block(h, 32, H, H, "net.enc." + l);
      skips.push(h);
    }
    for (let l = L - 1; l >= 0; l--) {
      h = upCat(h, skips[l], 32, H, H);
      H <<= 1;
      h = block(h, 64, H, H, "net.dec." + l);
    }
    return conv3(h, 32, G, G, w["net.out.weight"], w["net.out.bias"], 1);
  }

  // negative central-difference gradient with zero padding, into ax (x) and ay (y) arrays
  negGrad(phi, ax, ay) {
    const { G, h } = this;
    const inv = 1 / (2 * h);
    for (let y = 0; y < G; y++) {
      for (let x = 0; x < G; x++) {
        const l = x > 0 ? phi[y * G + x - 1] : 0, r = x < G - 1 ? phi[y * G + x + 1] : 0;
        const d = y > 0 ? phi[(y - 1) * G + x] : 0, u = y < G - 1 ? phi[(y + 1) * G + x] : 0;
        ax[y * G + x] = -(r - l) * inv;
        ay[y * G + x] = -(u - d) * inv;
      }
    }
  }

  // acceleration of every token; also leaves the new potential in this.field
  force(pos, vel, n, out) {
    const { G, h, cfg, n2 } = this;
    const GG = G * G;
    const x = new Float32Array(6 * GG);
    const sc = cfg.in_scale / (h * h);
    this.corners(pos, n, (f, wt, i) => {
      x[f] += wt * sc;
      x[GG + f] += wt * vel[2 * i] * sc;
      x[2 * GG + f] += wt * vel[2 * i + 1] * sc;
    });
    // kernel far field: phi = conv(rho, K) * h^2
    const re = new Float64Array(n2 * n2), im = new Float64Array(n2 * n2);
    for (let y = 0; y < G; y++) for (let xx = 0; xx < G; xx++) re[y * n2 + xx] = x[y * G + xx];
    fft2(re, im, n2, false, this.tw);
    for (let i = 0; i < n2 * n2; i++) {
      const a = re[i], b = im[i];
      re[i] = a * this.Kre[i] - b * this.Kim[i];
      im[i] = a * this.Kim[i] + b * this.Kre[i];
    }
    fft2(re, im, n2, true, this.tw);
    const phiK = new Float32Array(GG);
    for (let y = 0; y < G; y++) for (let xx = 0; xx < G; xx++) phiK[y * G + xx] = re[y * n2 + xx] * h * h;
    const kx = new Float32Array(GG), ky = new Float32Array(GG);
    this.negGrad(phiK, kx, ky);
    x.set(kx, 3 * GG);
    x.set(ky, 4 * GG);
    x.set(this.field, 5 * GG);
    const phi = this.unet(x);
    this.field = phi;
    const gx = new Float32Array(GG), gy = new Float32Array(GG);
    this.negGrad(phi, gx, gy);
    for (let i = 0; i < GG; i++) { gx[i] += kx[i]; gy[i] += ky[i]; }
    out.fill(0, 0, 2 * n);
    this.corners(pos, n, (f, wt, i) => {
      out[2 * i] += gx[f] * wt;
      out[2 * i + 1] += gy[f] * wt;
    });
    this.pairTerm(pos, n, out);
    if (cfg.momfix) {
      let mx = 0, my = 0;
      for (let i = 0; i < n; i++) { mx += out[2 * i]; my += out[2 * i + 1]; }
      mx /= n; my /= n;
      for (let i = 0; i < n; i++) { out[2 * i] -= mx; out[2 * i + 1] -= my; }
    }
  }

  // learned short-range pair force within cfg.pp over the (at most knn) nearest neighbours
  pairTerm(pos, n, out) {
    const { cfg, ppmlp } = this;
    const pp = cfg.pp, cell = pp;
    const heads = new Map();
    const next = new Int32Array(n).fill(-1);
    const cx = new Int32Array(n), cy = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      cx[i] = Math.floor(pos[2 * i] / cell); cy[i] = Math.floor(pos[2 * i + 1] / cell);
      const key = cx[i] * 1048576 + cy[i];
      next[i] = heads.has(key) ? heads.get(key) : -1;
      heads.set(key, i);
    }
    const cand = [];
    for (let i = 0; i < n; i++) {
      cand.length = 0;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          let j = heads.get((cx[i] + dx) * 1048576 + (cy[i] + dy));
          if (j === undefined) continue;
          for (; j >= 0; j = next[j]) {
            if (j === i) continue;
            const ddx = pos[2 * j] - pos[2 * i], ddy = pos[2 * j + 1] - pos[2 * i + 1];
            const r = Math.sqrt(ddx * ddx + ddy * ddy + 1e-8);
            if (r < pp) cand.push([r, ddx, ddy]);
          }
        }
      }
      if (cand.length > cfg.knn) { cand.sort((a, b) => a[0] - b[0]); cand.length = cfg.knn; }
      for (const [r, ddx, ddy] of cand) {
        const t = r / pp;
        const win = (1 - t * t) ** 2;
        const wgt = ppmlp.eval(r, Math.log(r + 0.05)) * win / r;
        out[2 * i] += wgt * ddx;
        out[2 * i + 1] += wgt * ddy;
      }
    }
  }

  // one velocity-Verlet step in place (one force evaluation, plus one on the first step)
  step(pos, vel, n) {
    this.grow(n);
    const { dt, aTok } = this;
    if (!this.aPrev || this.aPrevN !== n) {
      this.force(pos, vel, n, aTok);
      this.aPrev = Float64Array.from(aTok.subarray(0, 2 * n));
      this.aPrevN = n;
    }
    const aPrev = this.aPrev;
    const vHalf = new Float64Array(2 * n);
    for (let i = 0; i < 2 * n; i++) {
      pos[i] += dt * vel[i] + 0.5 * dt * dt * aPrev[i];
      vHalf[i] = vel[i] + dt * aPrev[i];
    }
    this.force(pos, vHalf, n, aTok);
    for (let i = 0; i < 2 * n; i++) {
      vel[i] += 0.5 * dt * (aPrev[i] + aTok[i]);
      aPrev[i] = aTok[i];
    }
  }
}

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

// scripts/gravity_sim.init_bodies(scale=True), relative to the arena centre
export function initBodies(n, rng, spread = 5, speed = 0.5) {
  const f = n / 8;
  spread *= Math.sqrt(f); speed *= Math.pow(f, 0.25);
  const pos = new Float64Array(2 * n), vel = new Float64Array(2 * n);
  for (let i = 0; i < 2 * n; i++) { pos[i] = (rng() * 2 - 1) * spread; vel[i] = gauss(rng) * speed; }
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += vel[2 * i]; my += vel[2 * i + 1]; }
  for (let i = 0; i < n; i++) { vel[2 * i] -= mx / n; vel[2 * i + 1] -= my / n; }
  return { pos, vel };
}
