// Compares the JS tiled scatter-field port with float64 PyTorch (scripts/export_web_testvectors_scatter_tiled.py): single-force accelerations for
// each mechanism setting, tile potentials, 20-step rollouts (tiled and whole-domain single grid), and JS tiled vs JS single grid.
// Usage: node verify.mjs [scene ...]   (STEPS=n limits the rollout length; tile work runs on worker threads)
import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { availableParallelism } from "node:os";
import { TileNet, TiledDriver, parseWeights } from "../js/tiled_model.js";

const dir = new URL("..", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(dir + "weights.json"));
const bin = readFileSync(dir + "weights.bin");
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const weights = parseWeights(manifest, buf);
const vec = JSON.parse(readFileSync(dir + "tests/vectors.json"));
const STEPS = Math.min(+process.env.STEPS || vec.steps, vec.steps);
const flat = (rows) => Float64Array.from(rows.flat());
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

class NodePool {
  constructor(n) {
    this.size = n; this.id = 0; this.pending = new Map();
    this.workers = Array.from({ length: n }, () => {
      const w = new Worker(new URL("./node_worker.mjs", import.meta.url), { workerData: { manifest, buf } });
      w.on("message", ({ id, res }) => { this.pending.get(id)(res); this.pending.delete(id); });
      return w;
    });
  }

  call(i, msg) {
    return new Promise((res) => { const id = ++this.id; this.pending.set(id, res); this.workers[i].postMessage({ id, msg }); });
  }

  close() { for (const w of this.workers) w.terminate(); }
}

const pool = new NodePool(Math.min(8, availableParallelism()));
const VARIANTS = { full: { ghosts: true, fx: true, far: true }, noghost: { ghosts: false, fx: true, far: true }, nofar: { ghosts: true, fx: true, far: false },
  nofx: { ghosts: true, fx: false, far: true }, naive: { ghosts: false, fx: false, far: false } };
let worst1 = 0, worstF = 0, worst20 = 0, fail = false;
const names = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(vec.scenes);

async function rollout(drv, s, ref) {
  await drv.reset();
  const n = s.pos0.length, pos = flat(s.pos0), vel = flat(s.vel0), rep = [], out = { pos: [], vel: [] };
  const t0 = performance.now();
  for (let t = 0; t < STEPS; t++) {
    await drv.step(pos, vel, n);
    const d = Math.max(maxDiff(pos, flat(ref.pos[t + 1])), maxDiff(vel, flat(ref.vel[t + 1])));
    if (t === 0 || t === 4 || t === 9 || t === STEPS - 1) rep.push(`@${t + 1} ${d.toExponential(2)}`);
    if (t === 0) worst1 = Math.max(worst1, d);
    if (t === STEPS - 1) worst20 = Math.max(worst20, d);
    out.pos.push(Float64Array.from(pos)); out.vel.push(Float64Array.from(vel));
  }
  console.log("   ", rep.join("  "), `${((performance.now() - t0) / STEPS / 1000).toFixed(2)} s/step`);
  return { out, pos, vel };
}

for (const name of names) {
  const s = vec.scenes[name], nt = s.nt, n = s.pos0.length;
  console.log(name, `nt ${nt}, ${n} bodies`);
  const drv = new TiledDriver(weights, pool, nt);
  const pos0 = flat(s.pos0), vel0 = flat(s.vel0);
  for (const [vn, fl] of Object.entries(VARIANTS)) {
    drv.flags = fl;
    await drv.reset();
    const r = await drv.force(pos0, vel0, n);
    const d = maxDiff(r.acc, flat(s.force[vn]));
    worstF = Math.max(worstF, d);
    let extra = "";
    if (vn === "full" && s.field_force) {
      const k = nt * nt / 2 + nt / 2, f = (await pool.call(drv.worker(k), { cmd: "getField", k })).f;
      extra = `  tile ${k} potential diff ${maxDiff(f, s.field_force).toExponential(2)} (peak ${Math.max(...s.field_force.map(Math.abs)).toExponential(1)})`;
    }
    if (vn === "full") {
      const st = s.ghost_stats, g = r.geo.reduce((a, t) => a + (t.idx.length - t.nOwn), 0) / r.geo.length;
      extra += `  ghosts/tile ${g.toFixed(2)} (torch ${st.ghosts_mean.toFixed(2)})`;
      if (Math.abs(g - st.ghosts_mean) > 1e-9) { fail = true; console.log("FAIL ghost count mismatch"); }
    }
    console.log(`    force ${vn.padEnd(8)} max |da| ${d.toExponential(2)} (peak |a| ${Math.max(...flat(s.force[vn]).map(Math.abs)).toFixed(3)})${extra}`);
  }
  drv.flags = VARIANTS.full;
  console.log("  tiled rollout vs PyTorch tiled");
  const tiled = await rollout(drv, s, s.tiled);
  if (s.tile_field) {
    const k = s.tile_k, f = (await pool.call(drv.worker(k), { cmd: "getField", k })).f;
    console.log(`    tile ${k} potential after step ${STEPS}: max diff ${maxDiff(f, s.tile_field[String(vec.steps)]).toExponential(2)}`);
  }
  if (s.naive) {
    drv.flags = VARIANTS.naive;
    console.log("  naive tiling (no ghosts, fx, far) vs PyTorch");
    await rollout(drv, s, s.naive);
    drv.flags = VARIANTS.full;
  }
  const G = nt * 64, net = new TileNet({ w: weights.w, config: { ...weights.config, grid: G, extent: nt * 32 } });
  const sp = flat(s.pos0), sv = flat(s.vel0);
  net.reset();
  let dts = 0, dsp = 0;
  const t0 = performance.now();
  for (let t = 0; t < STEPS; t++) {
    net.step(sp, sv, n);
    dts = maxDiff(sp, tiled.out.pos[t]);
    if (t === STEPS - 1) dsp = maxDiff(sp, flat(s.single.pos[t + 1]));
  }
  console.log(`  JS single grid G=${G}: vs PyTorch single @${STEPS} ${dsp.toExponential(2)}; JS tiled vs JS single @${STEPS} ${dts.toExponential(2)} (PyTorch tiled vs single ${maxDiff(flat(s.tiled.pos[STEPS]), flat(s.single.pos[STEPS])).toExponential(2)}) ${((performance.now() - t0) / STEPS / 1000).toFixed(2)} s/step`);
  worst20 = Math.max(worst20, dsp);
}
pool.close();
console.log(`worst: 1-step ${worst1.toExponential(2)}, step ${STEPS} ${worst20.toExponential(2)}, single force ${worstF.toExponential(2)}`);
if (!(worst1 < 1e-4) || !(worstF < 1e-3) || fail) { console.error("FAIL"); process.exit(1); }
console.log("OK");
