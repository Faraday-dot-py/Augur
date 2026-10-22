// Frame packing for the tiled page: per-tile field tensors as sqrt-companded 16-bit with a per-tile max, and the inspected tile's U-Net
// activations as 8-bit with a per-channel max.
export const GG = 128 * 128, CC = 32 * 32;
export const LAYERS = [["rho", GG], ["phiK", GG], ["far", CC], ["phi", GG]];

const c16 = (u) => Math.round(((u < 0 ? -Math.sqrt(-u) : Math.sqrt(u)) + 1) * 32767.5);
const d16 = (q) => { const u = q / 32767.5 - 1; return u < 0 ? -u * u : u * u; };

export function pack16(list, size) {
  const q = new Uint16Array(list.length * size).fill(32768), m = new Float32Array(list.length);
  list.forEach((a, t) => {
    if (!a) return;
    let mx = 0;
    for (let j = 0; j < size; j++) { const v = Math.abs(a[j]); if (v > mx) mx = v; }
    m[t] = mx;
    const inv = mx > 0 ? 1 / mx : 0, o = t * size;
    for (let j = 0; j < size; j++) q[o + j] = c16(a[j] * inv);
  });
  return { q, m, size };
}

export function unpack16(p, t, out) {
  const { q, m, size } = p, o = t * size, mx = m[t];
  for (let j = 0; j < size; j++) out[j] = d16(q[o + j]) * mx;
  return out;
}

const uparts = (u) => [u.inp, ...u.enc, ...u.dec];

export function packUnet(u) {
  const ps = uparts(u);
  let n8 = 0;
  for (const a of ps) n8 += a.length;
  const q8 = new Uint8Array(n8), M = new Float32Array(ps.length * 32);
  let o = 0;
  ps.forEach((a, i) => {
    const n = a.length / 32;
    for (let ch = 0; ch < 32; ch++) {
      const off = ch * n;
      let mx = 0;
      for (let j = 0; j < n; j++) { const v = Math.abs(a[off + j]); if (v > mx) mx = v; }
      M[i * 32 + ch] = mx;
      const inv = mx > 0 ? 1 / mx : 0;
      for (let j = 0; j < n; j++) {
        const x = a[off + j] * inv;
        q8[o++] = Math.round(((x < 0 ? -Math.sqrt(-x) : Math.sqrt(x)) + 1) * 127.5);
      }
    }
  });
  return { q8, M };
}

export function makeUnetOut() {
  const f = (n) => new Float32Array(n);
  return { inp: f(32 * GG), enc: Array.from({ length: 6 }, (_, l) => f(32 * (GG >> (2 * l)))), dec: Array.from({ length: 5 }, (_, l) => f(32 * (GG >> (2 * l)))) };
}

export function unpackUnet(p, out) {
  let o = 0;
  uparts(out).forEach((a, i) => {
    const n = a.length / 32;
    for (let ch = 0; ch < 32; ch++) {
      const off = ch * n, mx = p.M[i * 32 + ch];
      for (let j = 0; j < n; j++) { const u = p.q8[o++] / 127.5 - 1; a[off + j] = (u < 0 ? -u * u : u * u) * mx; }
    }
  });
  return out;
}

export function makeOut(T, Gp) {
  const f = (n) => new Float32Array(n);
  return {
    tiles: Array.from({ length: T }, () => ({ rho: f(GG), phiK: f(GG), far: f(CC), phi: f(GG) })),
    coarse: { rho: f(Gp * Gp), phi: f(Gp * Gp), G: Gp },
    unet: makeUnetOut(),
  };
}

export function dequantFrame(fr, out) {
  if (fr.compact) return dequantCompact(fr, out);
  for (let t = 0; t < out.tiles.length; t++) for (const [name] of LAYERS) unpack16(fr.tiles[name], t, out.tiles[t][name]);
  if (fr.coarse) { unpack16(fr.coarse.rho, 0, out.coarse.rho); unpack16(fr.coarse.phi, 0, out.coarse.phi); } else { out.coarse.rho.fill(0); out.coarse.phi.fill(0); }
  if (fr.unet) unpackUnet(fr.unet, out.unet);
  return out;
}

// Compact frames (shipped ticks and aged-out computed ticks): sqrt-companded 8-bit with a per-tile max, no U-Net. The density and the potential
// (spiky: one pit per occupied cell) keep their 128x128 cells; the local-kernel potential is smooth and is area-averaged 4x4 to 32x32 (so is
// stored the far term, which is 32x32 already). Bodies, accelerations and ghost lists are kept as they are.
export const CR = 32, CQ = CR * CR;
export const COMPACT = { rho: GG, phiK: CQ, far: CQ, phi: GG };
export const q8c = (u) => Math.round(((u < 0 ? -Math.sqrt(-u) : Math.sqrt(u)) + 1) * 127.5);
export const d8c = (q) => { const u = q / 127.5 - 1; return u < 0 ? -u * u : u * u; };

function down4(a, out) {
  for (let y = 0; y < CR; y++) {
    for (let x = 0; x < CR; x++) {
      let s = 0;
      for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) s += a[(y * 4 + j) * 128 + x * 4 + i];
      out[y * CR + x] = s / 16;
    }
  }
}

export function pack8(list, size) {
  const q = new Uint8Array(list.length * size).fill(128), m = new Float32Array(list.length);
  list.forEach((a, t) => {
    let mx = 0;
    for (let j = 0; j < size; j++) { const v = Math.abs(a[j]); if (v > mx) mx = v; }
    m[t] = mx;
    const inv = mx > 0 ? 1 / mx : 0;
    for (let j = 0; j < size; j++) q[t * size + j] = q8c(a[j] * inv);
  });
  return { q, m, size };
}

export function unpack8(p, t, out) {
  const { q, m, size } = p, o = t * size, mx = m[t];
  for (let j = 0; j < size; j++) out[j] = d8c(q[o + j]) * mx;
  return out;
}

export function toCompact(fr) {
  const T = fr.nt * fr.nt, tiles = {}, tmp = new Float32Array(GG);
  for (const [name, size] of LAYERS) {
    const list = [];
    for (let t = 0; t < T; t++) {
      const lo = new Float32Array(COMPACT[name]);
      if (COMPACT[name] === size) unpack16(fr.tiles[name], t, lo); else { unpack16(fr.tiles[name], t, tmp); down4(tmp, lo); }
      list.push(lo);
    }
    tiles[name] = pack8(list, COMPACT[name]);
  }
  const cg = fr.coarse ? fr.coarse.G ** 2 : 0;
  const coarse = fr.coarse ? { G: fr.coarse.G, rho: pack8([unpack16(fr.coarse.rho, 0, new Float32Array(cg))], cg), phi: pack8([unpack16(fr.coarse.phi, 0, new Float32Array(cg))], cg) } : null;
  return { type: "frame", compact: true, tick: fr.tick, n: fr.n, ms: fr.ms, nt: fr.nt, flags: fr.flags, pos: fr.pos, vel: fr.vel, acc: fr.acc, tid: fr.tid, gh: fr.gh, ghOff: fr.ghOff,
    tiles, coarse, unet: null, momfix: fr.momfix, epoch: fr.epoch };
}

// 32x32 -> 128x128, bilinear, cell-centred
function up4(q, o, mx, out) {
  const src = new Float32Array(CQ);
  for (let j = 0; j < CQ; j++) src[j] = d8c(q[o + j]) * mx;
  for (let y = 0; y < 128; y++) {
    const fy = Math.min(CR - 1, Math.max(0, (y + 0.5) / 4 - 0.5)), y0 = Math.floor(fy), y1 = Math.min(CR - 1, y0 + 1), wy = fy - y0;
    for (let x = 0; x < 128; x++) {
      const fx = Math.min(CR - 1, Math.max(0, (x + 0.5) / 4 - 0.5)), x0 = Math.floor(fx), x1 = Math.min(CR - 1, x0 + 1), wx = fx - x0;
      out[y * 128 + x] = (src[y0 * CR + x0] * (1 - wx) + src[y0 * CR + x1] * wx) * (1 - wy) + (src[y1 * CR + x0] * (1 - wx) + src[y1 * CR + x1] * wx) * wy;
    }
  }
  return out;
}

export function dequantCompact(fr, out) {
  for (let t = 0; t < out.tiles.length; t++) {
    for (const [name, size] of LAYERS) {
      const p = fr.tiles[name];
      if (p.size === size) unpack8(p, t, out.tiles[t][name]); else up4(p.q, t * CQ, p.m[t], out.tiles[t][name]);
    }
  }
  if (fr.coarse) { unpack8(fr.coarse.rho, 0, out.coarse.rho); unpack8(fr.coarse.phi, 0, out.coarse.phi); } else { out.coarse.rho.fill(0); out.coarse.phi.fill(0); }
  return out;
}

// tile t's recurrent-potential window (128x128, float32) from a frame of either kind
export function framePhi(fr, t, out) {
  return fr.compact ? unpack8(fr.tiles.phi, t, out) : unpack16(fr.tiles.phi, t, out);
}

// everything the page holds for one tick, from a traced TiledDriver pass
export function packFrame(r, drv, pos, vel, tick, ms, flags) {
  const T = drv.T, n = pos.length / 2;
  const tiles = {};
  for (const [name, size] of LAYERS) tiles[name] = pack16(r.tiles.map((t) => t[name] || null), size);
  const co = drv.coarse;
  const coarse = co ? { rho: pack16([co.rho], co.G * co.G), phi: pack16([co.phi], co.G * co.G), G: co.G } : null;
  const ut = drv.inspect >= 0 ? r.tiles[drv.inspect].unet : null;
  const unet = ut ? { ...packUnet(ut), tile: drv.inspect } : null;
  const off = new Uint32Array(T + 1), gl = [];
  r.geo.forEach((g, k) => { for (let j = g.nOwn; j < g.idx.length; j++) gl.push(g.idx[j]); off[k + 1] = gl.length; });
  const fr = { type: "frame", tick, n, ms, nt: drv.nt, flags: { ...flags }, pos: pos.slice(), vel: vel.slice(), acc: Float64Array.from(r.acc), tid: Int16Array.from(r.tid),
    gh: Uint16Array.from(gl), ghOff: off, tiles, coarse, unet, momfix: r.momfix };
  const bufs = [fr.pos.buffer, fr.vel.buffer, fr.acc.buffer, fr.tid.buffer, fr.gh.buffer, off.buffer];
  for (const p of Object.values(tiles)) bufs.push(p.q.buffer, p.m.buffer);
  if (coarse) for (const p of [coarse.rho, coarse.phi]) bufs.push(p.q.buffer, p.m.buffer);
  if (unet) bufs.push(unet.q8.buffer, unet.M.buffer);
  return [fr, bufs];
}

export const frameBytes = (fr) => fr.pos.byteLength * 3 + Object.values(fr.tiles).reduce((a, p) => a + p.q.byteLength, 0)
  + (fr.coarse ? fr.coarse.rho.q.byteLength + fr.coarse.phi.q.byteLength : 0) + (fr.unet ? fr.unet.q8.byteLength : 0);
