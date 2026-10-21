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
  for (let t = 0; t < out.tiles.length; t++) for (const [name] of LAYERS) unpack16(fr.tiles[name], t, out.tiles[t][name]);
  if (fr.coarse) { unpack16(fr.coarse.rho, 0, out.coarse.rho); unpack16(fr.coarse.phi, 0, out.coarse.phi); } else { out.coarse.rho.fill(0); out.coarse.phi.fill(0); }
  if (fr.unet) unpackUnet(fr.unet, out.unet);
  return out;
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
