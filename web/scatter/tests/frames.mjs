import { loadNet } from "../tools/precompute.mjs";
import { initBodies, mulberry32 } from "../js/scatter_model.js";
import { quantise, makeOut, dequant, frameBytes } from "../js/frames.js";

const { net } = loadNet();
const n = 10, { pos, vel } = initBodies(n, mulberry32(4738));
net.reset();
for (let i = 0; i < 3; i++) net.step(pos, vel, n);
const tr = {};
net.step(pos, vel, n, tr);

const fr = { ...quantise(tr), pos: pos.slice(), vel: vel.slice(), n, tick: 4 };
const out = dequant(fr, makeOut());
const pairs = [["input", tr.input, out.input, 6], ["phiK", tr.phiK, out.phiK, 1], ["phi", tr.phi, out.phi, 1], ["inp", tr.inp, out.inp, 32],
  ["gradPhi0", tr.gradPhi[0], out.gradPhi[0], 1], ["aGrid1", tr.aGrid[1], out.aGrid[1], 1]];
tr.enc.forEach((a, l) => pairs.push(["enc" + l, a, out.enc[l], 32]));
tr.dec.forEach((a, l) => pairs.push(["dec" + l, a, out.dec[l], 32]));
let bad = 0;
for (const [name, a, b, C] of pairs) {
  const per = a.length / C;
  let worst = 0;
  for (let c = 0; c < C; c++) {
    let m = 0;
    for (let i = 0; i < per; i++) m = Math.max(m, Math.abs(a[c * per + i]));
    for (let i = 0; i < per; i++) worst = Math.max(worst, m ? Math.abs(a[c * per + i] - b[c * per + i]) / m : 0);
  }
  const lim = name.startsWith("enc") || name.startsWith("dec") || name === "inp" ? 0.008 : 0.0001;
  if (!(worst <= lim)) { bad++; console.log("FAIL", name, worst, lim); } else console.log("ok  ", name, "max err / channel max", worst.toExponential(2));
}
const mb = frameBytes(fr) / 1e6;
console.log("frame", mb.toFixed(2), "MB;", (mb * 130).toFixed(0), "MB for 100 ahead + 30 behind; float32 would be", (tr.input.length + tr.phiK.length + tr.inp.length + tr.phi.length + 4 * tr.phi.length + tr.enc.reduce((s, a) => s + a.length, 0) + tr.dec.reduce((s, a) => s + a.length, 0)) * 4 / 1e6, "MB");
if (out.pos !== fr.pos || out.tick !== 4) bad++;
if (bad) { console.log("FAILED"); process.exit(1); }
console.log("OK");
