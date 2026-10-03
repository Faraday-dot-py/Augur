import { initBodies, mulberry32 } from "./scatter_model.js";
import * as truth from "./truth.js";
import { drawEnergy, drawSeries } from "./plots.js";

const stage = document.getElementById("stage");
const ctx = stage.getContext("2d");
const err = document.getElementById("err");
const heat = document.createElement("canvas");
const heatCtx = heat.getContext("2d");

const debug = document.getElementById("debug");
const state = {
  n: 40, pos: null, vel: null, field: null, G: 128, extent: 64, dt: 0.1,
  tPos: null, tVel: null, tAcc: null, tTick: 0,
  seed: 4738, tick: 0, running: true, showTruth: true, showField: true,
  ms: 0, ready: false, meta: null,
  err: [], modelE: [], truthE: [], histLen: 200,
};

const worker = new Worker("js/worker.js", { type: "module" });

function spawn(n) {
  const rng = mulberry32(state.seed + Math.floor(performance.now()));
  const { pos, vel } = initBodies(n, rng);
  state.n = n;
  state.tPos = Float64Array.from(pos); state.tVel = Float64Array.from(vel);
  state.tAcc = new Float64Array(2 * n);
  truth.accel(state.tPos, n, state.tAcc);
  state.tTick = 0; state.E0 = undefined; state.err = []; state.modelE = []; state.truthE = [];
  state.pos = pos; state.vel = vel; state.tick = 0;
  worker.postMessage({ cmd: "reset", n, pos, vel });
}

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "error") { err.textContent = "Failed to load weights: " + m.message; err.hidden = false; return; }
  if (m.type === "ready") {
    Object.assign(state, { G: m.config.grid, extent: m.config.extent, dt: m.config.dt, ready: true });
    state.meta = { checkpoint: m.checkpoint, iters: m.iters, config: m.config };
    document.getElementById("s-ckpt").textContent = `${m.checkpoint} (${m.iters} it)`;
    spawn(state.n);
    worker.postMessage({ cmd: "run", run: state.running });
    return;
  }
  if (m.pos.length !== 2 * state.n) return; // stale message from before a reset
  state.pos = m.pos; state.vel = m.vel; state.field = m.field; state.tick = m.tick; state.ms = m.ms;
  while (state.tTick < m.tick) {
    truth.step(state.tPos, state.tVel, state.n, state.dt, state.tAcc);
    state.tTick++;
  }
  if (m.tick === 0) return;
  let e2 = 0;
  for (let i = 0; i < state.n; i++) e2 += Math.hypot(state.pos[2 * i] - state.tPos[2 * i], state.pos[2 * i + 1] - state.tPos[2 * i + 1]);
  state.err.push(e2 / state.n);
  if (state.E0 === undefined) state.E0 = truth.energy(state.tPos, state.tVel, state.n);
  state.modelE.push((truth.energy(state.pos, state.vel, state.n) - state.E0) / Math.abs(state.E0));
  state.truthE.push((truth.energy(state.tPos, state.tVel, state.n) - state.E0) / Math.abs(state.E0));
  if (state.err.length > state.histLen) { state.err.shift(); state.modelE.shift(); state.truthE.shift(); }
};

function bounds() {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const scan = (p) => {
    for (let i = 0; i < state.n; i++) {
      const x = p[2 * i], y = p[2 * i + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  };
  scan(state.pos);
  if (state.showTruth) scan(state.tPos);
  if (!Number.isFinite(minX)) { minX = minY = -10; maxX = maxY = 10; }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const half = Math.min(state.extent / 2, Math.max(10, (maxX - minX) / 2, (maxY - minY) / 2) * 1.25);
  return { cx, cy, half };
}

function paintField() {
  const G = state.G;
  if (heat.width !== G) { heat.width = G; heat.height = G; }
  const f = state.field;
  const sorted = Float32Array.from(f).sort();
  const lo = sorted[Math.floor(0.01 * (f.length - 1))], hi = sorted[Math.floor(0.995 * (f.length - 1))];
  const img = heatCtx.createImageData(G, G);
  const span = hi - lo || 1;
  for (let i = 0; i < f.length; i++) {
    const v = Math.pow(Math.min(1, Math.max(0, (f[i] - lo) / span)), 0.6);
    img.data[4 * i] = 8 + 40 * v; img.data[4 * i + 1] = 14 + 90 * v; img.data[4 * i + 2] = 24 + 150 * v; img.data[4 * i + 3] = 255;
  }
  heatCtx.putImageData(img, 0, 0);
}

function render() {
  const w = stage.clientWidth, h = stage.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (stage.width !== w * dpr || stage.height !== h * dpr) { stage.width = w * dpr; stage.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#06080c"; ctx.fillRect(0, 0, w, h);
  if (!state.pos) return;

  const { cx, cy, half } = bounds();
  const scale = Math.min(w, h) / (2 * half);
  const sx = (x) => w / 2 + (x - cx) * scale;
  const sy = (y) => h / 2 + (y - cy) * scale;

  if (state.showField && state.field) {
    paintField();
    const x0 = sx(-state.extent / 2), y0 = sy(-state.extent / 2), side = state.extent * scale;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heat, x0, y0, side, side);
    ctx.strokeStyle = "#1c2535"; ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, side, side);
  }

  if (state.showTruth) {
    ctx.fillStyle = "#ff9a3c";
    ctx.globalAlpha = 0.5;
    for (let i = 0; i < state.n; i++) {
      ctx.beginPath(); ctx.arc(sx(state.tPos[2 * i]), sy(state.tPos[2 * i + 1]), 2.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = "#6fc3ff";
  for (let i = 0; i < state.n; i++) {
    ctx.beginPath(); ctx.arc(sx(state.pos[2 * i]), sy(state.pos[2 * i + 1]), 3.5, 0, Math.PI * 2); ctx.fill();
  }
}

function frame() {
  requestAnimationFrame(frame);
  render();
  document.getElementById("s-tick").textContent = state.tick;
  document.getElementById("s-bodies").textContent = state.n;
  document.getElementById("s-ms").textContent = state.ms ? state.ms.toFixed(0) + " ms" : "-";
  const e = state.err[state.err.length - 1];
  document.getElementById("s-err").textContent = e === undefined ? "-" : e.toFixed(3);
  if (!debug.hidden) {
    let out = 0, vmax = 0;
    for (let i = 0; i < state.n; i++) {
      if (Math.abs(state.pos[2 * i]) > state.extent / 2 || Math.abs(state.pos[2 * i + 1]) > state.extent / 2) out++;
      vmax = Math.max(vmax, Math.hypot(state.vel[2 * i], state.vel[2 * i + 1]));
    }
    document.getElementById("s-out").textContent = `${out} / ${state.n}`;
    document.getElementById("s-vmax").textContent = vmax.toFixed(2);
  }
  drawSeries(document.getElementById("c-err"), state.err, "#6fc3ff");
  drawEnergy(document.getElementById("c-energy"), state.modelE, state.truthE);
}
requestAnimationFrame(frame);

document.getElementById("b-pause").addEventListener("click", (e) => {
  state.running = !state.running;
  e.target.setAttribute("aria-pressed", String(!state.running));
  e.target.textContent = state.running ? "Pause" : "Resume";
  worker.postMessage({ cmd: "run", run: state.running });
});
document.getElementById("b-step").addEventListener("click", () => {
  if (state.running) document.getElementById("b-pause").click();
  worker.postMessage({ cmd: "step" });
});
document.getElementById("b-reset").addEventListener("click", () => spawn(state.n));
document.getElementById("b-truth").addEventListener("click", (e) => {
  state.showTruth = !state.showTruth;
  e.target.setAttribute("aria-pressed", String(state.showTruth));
});
document.getElementById("b-field").addEventListener("click", (e) => {
  state.showField = !state.showField;
  e.target.setAttribute("aria-pressed", String(state.showField));
});
document.getElementById("b-export").addEventListener("click", () => {
  const snap = {
    tick: state.tick, n: state.n, dt: state.dt, extent: state.extent, G: state.G, meta: state.meta, exportedAt: new Date().toISOString(),
    pos: Array.from(state.pos), vel: Array.from(state.vel), truthPos: Array.from(state.tPos), truthVel: Array.from(state.tVel),
    field: state.field ? Array.from(state.field) : null, err: state.err, modelE: state.modelE, truthE: state.truthE,
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(snap)], { type: "application/json" }));
  a.download = `scatter_state_tick${state.tick}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});
const rBodies = document.getElementById("r-bodies"), oBodies = document.getElementById("o-bodies");
rBodies.addEventListener("input", () => { oBodies.textContent = rBodies.value; });
rBodies.addEventListener("change", () => state.ready && spawn(parseInt(rBodies.value, 10)));

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === " ") { e.preventDefault(); document.getElementById("b-pause").click(); }
  else if (e.key === ".") document.getElementById("b-step").click();
  else if (e.key === "r") document.getElementById("b-reset").click();
  else if (e.key === "g") document.getElementById("b-truth").click();
  else if (e.key === "f") document.getElementById("b-field").click();
  else if (e.key === "d") debug.hidden = !debug.hidden;
});
