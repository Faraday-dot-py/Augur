const BASE = [0.085, 0.1, 0.14];
const BLUE = [0.26, 0.58, 1.0], ORANGE = [1.0, 0.56, 0.16];

export const KINDS = {
  dens: { pos: [0.38, 0.84, 0.64], neg: BLUE },
  mom: { pos: [0.44, 0.76, 1.0], neg: ORANGE },
  far: { pos: ORANGE, neg: BLUE },
  act: { pos: [0.72, 0.58, 1.0], neg: [0.3, 0.8, 0.75] },
  pot: { pos: [0.96, 0.86, 0.4], neg: BLUE },
  wgt: { pos: [1.0, 0.5, 0.62], neg: [0.35, 0.6, 1.0] },
};

function divColor(t, k, out, o) {
  t = Math.max(-1, Math.min(1, t || 0));
  const a = Math.pow(Math.abs(t), 0.6);
  const e = t >= 0 ? k.pos : k.neg;
  const w = Math.max(0, a - 0.8) * 2;
  for (let q = 0; q < 3; q++) out[o + q] = 255 * Math.min(1, BASE[q] + (e[q] - BASE[q]) * a + w * (1 - e[q]));
}

const luts = {};
export function lut(kind) {
  if (luts[kind]) return luts[kind];
  const a = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) divColor(i / 127.5 - 1, KINDS[kind], a, 3 * i);
  return (luts[kind] = a);
}

export const css = (kind, sign = 1) => {
  const c = KINDS[kind][sign >= 0 ? "pos" : "neg"];
  return "rgb(" + c.map((v) => Math.round(v * 255)).join(",") + ")";
};

// signed log scale onto [-1, 1]: s maps to ~0.15, 100 s to 1
export const sl = (x, s = 1) => Math.sign(x) * Math.min(1, Math.log1p(Math.abs(x) / s) / Math.log1p(100));

export const fmt = (x) => {
  const a = Math.abs(x);
  return a !== 0 && (a >= 1e4 || a < 1e-3) ? x.toExponential(2) : x.toFixed(4);
};
