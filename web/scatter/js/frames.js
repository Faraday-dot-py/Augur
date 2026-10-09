export const GG = 128 * 128;

const NSMALL = 7;
const CH = [6, 1, 1, 1, 1, 1, 1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32];

const parts = (t) => [t.input, t.phiK, t.phi, t.gradPhi[0], t.gradPhi[1], t.aGrid[0], t.aGrid[1], t.inp, ...t.enc, ...t.dec];

// sqrt-companded per-channel quantisation: 16 bit for the small field tensors, 8 bit for the UNet activations
export function quantise(tr) {
  const ps = parts(tr);
  let n16 = 0, n8 = 0, nch = 0;
  ps.forEach((a, i) => { if (i < NSMALL) n16 += a.length; else n8 += a.length; nch += CH[i]; });
  const q16 = new Uint16Array(n16), q8 = new Uint8Array(n8), M = new Float32Array(nch);
  let o16 = 0, o8 = 0, c = 0;
  ps.forEach((a, i) => {
    const C = CH[i], n = a.length / C;
    for (let ch = 0; ch < C; ch++, c++) {
      const off = ch * n;
      let m = 0;
      for (let j = 0; j < n; j++) { const v = Math.abs(a[off + j]); if (v > m) m = v; }
      M[c] = m;
      const inv = m > 0 ? 1 / m : 0;
      for (let j = 0; j < n; j++) {
        const u = a[off + j] * inv, s = (u < 0 ? -Math.sqrt(-u) : Math.sqrt(u)) + 1;
        if (i < NSMALL) q16[o16++] = Math.round(s * 32767.5); else q8[o8++] = Math.round(s * 127.5);
      }
    }
  });
  return { q16, q8, M };
}

export function makeOut() {
  const f = (n) => new Float32Array(n);
  return {
    input: f(6 * GG), phiK: f(GG), phi: f(GG), gradPhi: [f(GG), f(GG)], aGrid: [f(GG), f(GG)], inp: f(32 * GG),
    enc: Array.from({ length: 6 }, (_, l) => f(32 * (GG >> (2 * l)))), dec: Array.from({ length: 5 }, (_, l) => f(32 * (GG >> (2 * l)))),
    pos: null, vel: null, n: 0, tick: -1,
  };
}

export function dequant(fr, out) {
  const ps = parts(out);
  let o16 = 0, o8 = 0, c = 0;
  ps.forEach((a, i) => {
    const C = CH[i], n = a.length / C;
    for (let ch = 0; ch < C; ch++, c++) {
      const off = ch * n, m = fr.M[c];
      if (i < NSMALL) for (let j = 0; j < n; j++) { const u = fr.q16[o16++] / 32767.5 - 1; a[off + j] = (u < 0 ? -u * u : u * u) * m; }
      else for (let j = 0; j < n; j++) { const u = fr.q8[o8++] / 127.5 - 1; a[off + j] = (u < 0 ? -u * u : u * u) * m; }
    }
  });
  out.pos = fr.pos; out.vel = fr.vel; out.n = fr.n; out.tick = fr.tick;
  return out;
}

export const frameBytes = (fr) => fr.q16.byteLength + fr.q8.byteLength + fr.M.byteLength + fr.pos.byteLength + fr.vel.byteLength;

export function countOutside(pos, n, extent) {
  const h = extent / 2;
  let c = 0;
  for (let i = 0; i < n; i++) if (Math.abs(pos[2 * i]) > h || Math.abs(pos[2 * i + 1]) > h) c++;
  return c;
}
