// Frame quantisation round trips and scene generation.
import { pack16, unpack16, packUnet, unpackUnet, makeUnetOut } from "../js/tframes.js";
import { makeScene, SCENES } from "../js/tiled_model.js";

let bad = 0;
const check = (c, m) => { console.log((c ? "ok   " : "FAIL ") + m); if (!c) bad++; };

const rng = (a) => () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296; };
const r = rng(4738);
const tiles = Array.from({ length: 3 }, (_, t) => Float32Array.from({ length: 16384 }, (_, i) => (r() - 0.5) * 2 * Math.pow(10, t - 1) * Math.exp(-(((i % 128) - 64) ** 2) / 900)));
const p = pack16(tiles, 16384);
let worst = 0;
tiles.forEach((a, t) => {
  const o = unpack16(p, t, new Float32Array(16384));
  let m = 0;
  for (let i = 0; i < a.length; i++) { m = Math.max(m, Math.abs(a[i])); worst = Math.max(worst, Math.abs(a[i] - o[i]) / p.m[t]); }
});
check(worst < 1e-4, `16-bit tile fields: worst error ${worst.toExponential(2)} of the tile's max`);

const u = makeUnetOut();
for (const a of [u.inp, ...u.enc, ...u.dec]) for (let i = 0; i < a.length; i++) a[i] = Math.sin(i * 0.37) * (1 + (i % 7));
const pu = packUnet(u), v = makeUnetOut();
unpackUnet(pu, v);
let w8 = 0;
[[u.inp, v.inp], ...u.enc.map((a, l) => [a, v.enc[l]]), ...u.dec.map((a, l) => [a, v.dec[l]])].forEach(([a, b], idx) => {
  const n = a.length / 32;
  for (let ch = 0; ch < 32; ch++) {
    let m = 0;
    for (let j = 0; j < n; j++) m = Math.max(m, Math.abs(a[ch * n + j]));
    for (let j = 0; j < n; j++) w8 = Math.max(w8, Math.abs(a[ch * n + j] - b[ch * n + j]) / m);
  }
});
check(w8 < 0.008, `8-bit U-Net activations: worst error ${(w8 * 100).toFixed(2)}% of the channel max`);

for (const nt of [2, 4]) {
  for (const kind of Object.keys(SCENES)) {
    const n = nt === 2 ? 100 : 300, a = makeScene(kind, nt, n), b = makeScene(kind, nt, n), c = makeScene(kind, nt, n, 1);
    let mx = 0, my = 0, ext = 0;
    for (let i = 0; i < n; i++) { mx += a.vel[2 * i]; my += a.vel[2 * i + 1]; ext = Math.max(ext, Math.abs(a.pos[2 * i]), Math.abs(a.pos[2 * i + 1])); }
    check(a.pos.every((x, i) => x === b.pos[i]) && !a.pos.every((x, i) => x === c.pos[i]), `${nt}x${nt} ${kind}: seed 4738 deterministic`);
    check(Math.abs(mx) < 1e-9 && Math.abs(my) < 1e-9 && ext <= 0.45 * nt * 32 + 1e-9, `${nt}x${nt} ${kind}: zero momentum, inside 0.45 of the domain (max |x| ${ext.toFixed(1)})`);
  }
}
if (bad) process.exit(1);
