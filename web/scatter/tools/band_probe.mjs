// node tools/band_probe.mjs [out.json]
// Field bands vs input shift: empty arena and one body at several offsets (8 cells = 4 units, half a 16-cell pooling block).
import { writeFileSync } from "node:fs";
import { loadNet } from "./precompute.mjs";

const out = process.argv[2] || "band_probe.json";
const { net } = loadNet();
const G = net.G, h = net.h;
const cell = (u) => Math.floor((u + net.cfg.extent / 2) / h);

function dips(prof, skip) {
  const res = [];
  for (let i = 12; i < G - 12; i++) {
    if (skip(i)) continue;
    let s = 0, c = 0;
    for (let k = i - 4; k <= i + 4; k++) if (!skip(k)) { s += prof[k]; c++; }
    res.push([i, prof[i] - s / c]);
  }
  res.sort((a, b) => a[1] - b[1]);
  const top = [];
  for (const [i, d] of res) if (top.every(([j]) => Math.abs(i - j) > 6)) top.push([i, +d.toFixed(5)]);
  return top.slice(0, 3);
}

function probe(name, bodies, ticks) {
  net.reset();
  const n = bodies.length / 2;
  net.grow(Math.max(n, 1));
  const pos = Float64Array.from(bodies), vel = new Float64Array(2 * n), a = new Float64Array(2 * Math.max(n, 1));
  for (let t = 0; t < ticks; t++) net.force(pos, vel, n, a);
  const f = net.field;
  const bx = n ? cell(pos[0]) : -100, by = n ? cell(pos[1]) : -100;
  const near = (i, b) => Math.abs(i - b) < 14;
  const rows = new Float64Array(G), cols = new Float64Array(G);
  let rc = 0, cc = 0;
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
    if (!near(x, bx)) rows[y] += f[y * G + x];
    if (!near(y, by)) cols[x] += f[y * G + x];
  }
  for (let i = 0; i < G; i++) { if (!near(i, bx)) cc++; if (!near(i, by)) rc++; }
  for (let i = 0; i < G; i++) { rows[i] /= cc; cols[i] /= rc; }
  const r = { name, body: n ? [bx, by] : null, rowDips: dips(rows, (i) => near(i, by)), colDips: dips(cols, (i) => near(i, bx)) };
  console.log(JSON.stringify(r));
  return r;
}

const res = [];
for (const ticks of [1, 3]) {
  res.push(probe(`empty t${ticks}`, [], ticks));
  const base = [-24.6, -12.6];
  for (const [dx, dy] of [[0, 0], [4, 0], [0, 4], [4, 4], [8, 8]]) {
    res.push(probe(`one body at (${base[0] + dx},${base[1] + dy}) t${ticks}`, [base[0] + dx, base[1] + dy], ticks));
  }
}
writeFileSync(out, JSON.stringify(res, null, 1));
