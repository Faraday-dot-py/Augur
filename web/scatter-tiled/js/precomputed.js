// The shipped ticks of the default scene (4x4 tiles, cold collapse, 300 bodies, seed 4738, all mechanisms on): compact frames of the model and the
// exact sim for ticks 0..last as one planar binary (gzip), plus the full-precision tick-`last` state the in-browser model resumes from.
import { COMPACT, GG } from "./tframes.js";

export const DEFAULT = { name: "tiled4_collapse300", nt: 4, scene: "collapse", n: 300, seed: 4738 };
const LAYER_NAMES = ["rho", "phiK", "far", "phi"];

// [name, array type, elements per tick]; the ghost lists are the one variable-length section (meta.ghStart)
function sections(meta) {
  const { n, T, G, ticks } = meta, K = ticks + 1;
  const list = [["pos", Float32Array, 2 * n], ["vel", Float32Array, 2 * n], ["acc", Float32Array, 2 * n], ["spos", Float32Array, 2 * n], ["sacc", Float32Array, 2 * n],
    ["momfix", Float32Array, 2], ["cmaxRho", Float32Array, 1], ["cmaxPhi", Float32Array, 1], ...LAYER_NAMES.map((l) => ["max_" + l, Float32Array, T]),
    ["ghOff", Uint16Array, T + 1], ["gh", Uint16Array, 0], ["tid", Int8Array, n], ["crho", Uint8Array, G * G], ["cphi", Uint8Array, G * G],
    ...LAYER_NAMES.map((l) => ["q_" + l, Uint8Array, T * COMPACT[l]])];
  let off = 0;
  return list.map(([name, Type, per]) => {
    const len = name === "gh" ? meta.ghStart[K] : per * K, s = { name, Type, off, len };
    off += Math.ceil(len * Type.BYTES_PER_ELEMENT / 4) * 4;
    s.end = off;
    return s;
  });
}

export const assetBytes = (meta) => sections(meta).at(-1).end;

// frames: compact model frames for ticks 0..last; sims: {pos, acc} of the exact sim for the same ticks
export function encodeAsset(meta, frames, sims) {
  const K = frames.length, ghStart = [0];
  for (const f of frames) ghStart.push(ghStart.at(-1) + f.gh.length);
  meta.ticks = K - 1; meta.ghStart = ghStart;
  const secs = sections(meta), buf = new ArrayBuffer(secs.at(-1).end), v = {};
  for (const s of secs) v[s.name] = new s.Type(buf, s.off, s.len);
  const T = meta.T, n = meta.n;
  frames.forEach((f, k) => {
    v.pos.set(f.pos, 2 * n * k); v.vel.set(f.vel, 2 * n * k); v.acc.set(f.acc, 2 * n * k);
    v.spos.set(sims[k].pos, 2 * n * k); v.sacc.set(sims[k].acc, 2 * n * k);
    v.momfix.set(f.momfix, 2 * k);
    v.cmaxRho[k] = f.coarse.rho.m[0]; v.cmaxPhi[k] = f.coarse.phi.m[0];
    v.crho.set(f.coarse.rho.q, meta.G * meta.G * k); v.cphi.set(f.coarse.phi.q, meta.G * meta.G * k);
    for (const l of LAYER_NAMES) { v["max_" + l].set(f.tiles[l].m, T * k); v["q_" + l].set(f.tiles[l].q, T * COMPACT[l] * k); }
    v.ghOff.set(f.ghOff, (T + 1) * k); v.gh.set(f.gh, ghStart[k]); v.tid.set(f.tid, n * k);
  });
  return buf;
}

export function decodeAsset(meta, buf) {
  const secs = sections(meta), v = {}, K = meta.ticks + 1, { n, T, G } = meta;
  for (const s of secs) v[s.name] = new s.Type(buf, s.off, s.len);
  const frames = [], sims = [];
  for (let k = 0; k < K; k++) {
    const sub = (a, per) => a.subarray(per * k, per * (k + 1)), tiles = {};
    for (const l of LAYER_NAMES) tiles[l] = { q: sub(v["q_" + l], T * COMPACT[l]), m: sub(v["max_" + l], T), size: COMPACT[l] };
    frames.push({ type: "frame", compact: true, shipped: true, tick: k, n, ms: 0, nt: meta.nt, flags: { ...meta.flags }, pos: sub(v.pos, 2 * n), vel: sub(v.vel, 2 * n), acc: sub(v.acc, 2 * n),
      tid: sub(v.tid, n), gh: v.gh.subarray(meta.ghStart[k], meta.ghStart[k + 1]), ghOff: sub(v.ghOff, T + 1), tiles,
      coarse: { G, rho: { q: sub(v.crho, G * G), m: v.cmaxRho.subarray(k, k + 1), size: G * G }, phi: { q: sub(v.cphi, G * G), m: v.cmaxPhi.subarray(k, k + 1), size: G * G } },
      unet: null, momfix: sub(v.momfix, 2) });
    sims.push({ type: "sim", tick: k, pos: sub(v.spos, 2 * n), acc: sub(v.sacc, 2 * n) });
  }
  return { frames, sims };
}

// full-precision state at the last shipped tick: float64 bodies of the model and the sim, float32 potential windows of every tile
export function encodeState(st) {
  const n2 = st.pos.length, T = st.phis.length, buf = new ArrayBuffer(5 * n2 * 8 + T * GG * 4);
  const f = new Float64Array(buf, 0, 5 * n2), p = new Float32Array(buf, 5 * n2 * 8, T * GG);
  [st.pos, st.vel, st.acc, st.spos, st.svel].forEach((a, i) => f.set(a, i * n2));
  st.phis.forEach((a, t) => p.set(a, t * GG));
  return buf;
}

export function decodeState(buf, n, T) {
  const n2 = 2 * n, f = new Float64Array(buf, 0, 5 * n2), p = new Float32Array(buf, 5 * n2 * 8, T * GG), g = (i) => f.slice(i * n2, (i + 1) * n2);
  return { pos: g(0), vel: g(1), acc: g(2), spos: g(3), svel: g(4), phis: Array.from({ length: T }, (_, t) => p.slice(t * GG, (t + 1) * GG)) };
}

// a server may or may not have decoded the .gz already: look at the magic bytes
export async function gunzip(res) {
  const raw = await res.arrayBuffer(), b = new Uint8Array(raw);
  if (b[0] !== 0x1f || b[1] !== 0x8b) return raw;
  return new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}
