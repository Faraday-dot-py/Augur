import { initBodies, mulberry32 } from "./scatter_model.js";
import { drawEnergy, drawSeries } from "./plots.js";
import * as truth from "./truth.js";
import { createView } from "./model3d.js";
import { makeOut, dequant, frameBytes } from "./frames.js";

const err = document.getElementById("err");
const $ = (id) => document.getElementById(id);

const debug = $("debug");
const state = {
  n: 40, pos: null, vel: null, G: 128, extent: 64, dt: 0.1,
  tPos: null, tVel: null, tTick: 0, tMs: 0, truthOn: false, truthSince: null,
  mode: "fixed", seed: 4738, tick: -1, epoch: 0, running: true,
  ms: 0, ready: false, meta: null, E0: undefined, modelEnergy: 0,
  err: [], modelE: [], truthE: [], histLen: 200,
};
const RATE = 10, BUF_AHEAD = 100, BUF_BEHIND = 30;
const frames = new Map(), out = makeOut(), history = new Map();
let oldest = 0, newest = -1, want = -1, lastAdv = 0, stepPending = false, curves = null;
const arrivals = [], shownAt = [];
let plotsDirty = true, truthWorker = null, model = null;

const worker = new Worker("js/worker.js", { type: "module" });

function rate(list) {
  const now = performance.now();
  while (list.length && now - list[0] > 3000) list.shift();
  return list.length > 1 ? (list.length - 1) / ((list[list.length - 1] - list[0]) / 1000) : 0;
}

function updateStatus() {
  if (state.tick < 0) { $("mstat").textContent = state.ready ? "computing tick 0" : "loading weights"; return; }
  const ahead = newest - state.tick, starved = state.running && !frames.has(state.tick + 1);
  $("mstat").textContent = `tick ${state.tick} · buffer ${ahead}/${BUF_AHEAD} · ${rate(shownAt).toFixed(1)} ticks/s shown · ${rate(arrivals).toFixed(1)} frames/s computed${starved ? " · waiting for next frame" : ""}`;
}

function syncTruth() {
  if (model) model.setTruth(state.truthOn ? state.tPos : null);
}

function startRun(n) {
  state.n = n; state.epoch++;
  frames.clear(); history.clear();
  oldest = 0; newest = -1; want = BUF_AHEAD; stepPending = false;
  state.tick = -1; state.E0 = undefined; state.err = []; state.modelE = []; state.truthE = []; state.pos = state.vel = null;
  const { pos, vel } = initBodies(n, mulberry32(state.mode === "fixed" ? state.seed : state.seed + Math.floor(performance.now())));
  $("s-src").textContent = state.mode === "fixed" ? `fixed seed ${state.seed}` : "random start";
  worker.postMessage({ cmd: "reset", epoch: state.epoch, n, pos, vel, limit: want });
  if (state.truthOn) startTruth(pos, vel, 0, true);
  updateStatus();
}

function show(k) {
  const fr = frames.get(k);
  state.tick = k; state.pos = fr.pos; state.vel = fr.vel; state.ms = fr.ms; state.modelEnergy = fr.energy;
  if (state.E0 === undefined) state.E0 = fr.energy;
  state.modelE = [];
  for (let t = Math.max((state.truthSince || 0) + 1, k - state.histLen + 1); t <= k; t++) {
    const f = frames.get(t);
    if (f) state.modelE.push((f.energy - state.E0) / Math.abs(state.E0));
  }
  history.set(k, { pos: fr.pos });
  history.delete(k - 64);
  if (truthWorker) truthWorker.postMessage({ cmd: "advance", epoch: state.epoch, tick: k });
  dequant(fr, out);
  if (model) model.showFrame(out);
  for (; oldest < k - BUF_BEHIND; oldest++) frames.delete(oldest);
  if (k + BUF_AHEAD !== want) { want = k + BUF_AHEAD; worker.postMessage({ cmd: "want", epoch: state.epoch, limit: want }); }
  shownAt.push(performance.now());
  syncRange();
  plotsDirty = true;
  syncTruth();
  updateStatus();
}

function syncRange() {
  const r = $("r-tick");
  r.min = oldest; r.max = newest; r.value = state.tick;
  $("o-tick").textContent = state.tick;
}

function tryAdvance() {
  if (state.tick < 0) return;
  const next = state.tick + 1, now = performance.now();
  if (!frames.has(next)) return;
  if (stepPending) { stepPending = false; lastAdv = now; show(next); }
  else if (state.running && now - lastAdv >= 1000 / RATE) { lastAdv = now; show(next); }
}
setInterval(tryAdvance, 20);

function setRunning(on) {
  state.running = on;
  const b = $("b-pause");
  b.setAttribute("aria-pressed", String(!on));
  b.textContent = on ? "Pause" : "Resume";
  lastAdv = performance.now();
}

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  $("b-mode").setAttribute("aria-pressed", String(mode === "fixed"));
  $("b-mode").textContent = mode === "fixed" ? "Fixed seed" : "Random";
  rBodies.step = mode === "fixed" ? 10 : 1;
  if (mode === "fixed") {
    rBodies.value = Math.min(100, Math.max(10, Math.round(state.n / 10) * 10));
    oBodies.textContent = rBodies.value;
  }
  startRun(+rBodies.value);
}

function startTruth(pos, vel, tick, fresh) {
  stopTruth();
  state.truthOn = true; state.truthSince = tick; state.tTick = tick;
  state.tPos = Float64Array.from(pos); state.tVel = Float64Array.from(vel);
  state.err = []; state.truthE = []; state.modelE = []; state.E0 = fresh ? undefined : state.modelEnergy;
  truthWorker = new Worker("js/truth_worker.js", { type: "module" });
  truthWorker.onmessage = onTruth;
  truthWorker.postMessage({ cmd: "init", epoch: state.epoch, n: state.n, dt: state.dt, tick, pos, vel });
  syncTruthUI();
  syncTruth();
}

function stopTruth() {
  if (truthWorker) truthWorker.terminate();
  truthWorker = null;
  state.truthOn = false; state.truthSince = null; state.tPos = null; state.tVel = null; state.truthE = []; state.err = [];
  syncTruthUI();
  syncTruth();
  plotsDirty = true;
}

function syncTruthUI() {
  $("c-truth").checked = state.truthOn;
  $("c-err").hidden = $("h-err").hidden = $("k-truth").hidden = !state.truthOn;
  $("h-energy").textContent = state.truthOn ? "Energy drift (model vs truth)" : "Energy drift (model)";
  $("s-err").textContent = state.truthOn ? "-" : "off";
  $("s-tsince").textContent = state.truthOn ? `tick ${state.truthSince}` : "off";
  $("s-tlag").textContent = state.truthOn ? "0 ticks" : "off";
  $("s-tms").textContent = state.truthOn ? "-" : "off";
}

function onTruth(e) {
  const m = e.data;
  if (m.epoch !== state.epoch || !state.truthOn) return;
  state.tPos = m.pos; state.tTick = m.tick; state.tMs = m.ms;
  const h = history.get(m.tick);
  if (h && state.E0 !== undefined && m.tick > state.truthSince) {
    let e2 = 0;
    for (let i = 0; i < state.n; i++) e2 += Math.hypot(h.pos[2 * i] - m.pos[2 * i], h.pos[2 * i + 1] - m.pos[2 * i + 1]);
    state.err.push(e2 / state.n);
    state.truthE.push((m.energy - state.E0) / Math.abs(state.E0));
    if (state.err.length > state.histLen) { state.err.shift(); state.truthE.shift(); }
  }
  syncTruth();
  plotsDirty = true;
}

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "error") { err.textContent = "Failed to load weights: " + m.message; err.hidden = false; return; }
  if (m.type === "ready") {
    Object.assign(state, { G: m.config.grid, extent: m.config.extent, dt: m.config.dt, ready: true });
    state.meta = { checkpoint: m.checkpoint, iters: m.iters, config: m.config };
    $("s-ckpt").textContent = `${m.checkpoint} (${m.iters} it)`;
    curves = { curve: m.curve, pair: m.pair };
    if (model) model.setCurves(curves.curve, curves.pair);
    startRun(state.n);
    return;
  }
  if (m.type !== "frame" || m.epoch !== state.epoch) return;
  frames.set(m.tick, m);
  newest = m.tick;
  arrivals.push(performance.now());
  if (state.tick < 0) { lastAdv = performance.now(); show(0); } else syncRange();
  $("s-trace").textContent = `${newest - state.tick} / ${BUF_AHEAD}`;
  tryAdvance();
};

function frame() {
  requestAnimationFrame(frame);
  if (plotsDirty && !debug.hidden) {
    plotsDirty = false;
    if (state.truthOn) drawSeries($("c-err"), state.err, "#6fc3ff");
    drawEnergy($("c-energy"), state.modelE, state.truthE);
  }
}
requestAnimationFrame(frame);

function updateStats() {
  updateStatus();
  if (debug.hidden || !state.pos) return;
  $("s-tick").textContent = state.tick;
  $("s-bodies").textContent = state.n;
  $("s-ms").textContent = state.ms ? state.ms.toFixed(0) + " ms" : "-";
  $("s-trace").textContent = `${Math.max(0, newest - state.tick)} / ${BUF_AHEAD}`;
  if (state.truthOn) {
    const e = state.err[state.err.length - 1];
    $("s-err").textContent = e === undefined ? "-" : e.toFixed(3);
    $("s-tlag").textContent = `${Math.max(0, state.tick - state.tTick)} ticks`;
    $("s-tms").textContent = state.tMs ? state.tMs.toFixed(1) + " ms" : "-";
  }
  let out = 0, vmax = 0;
  for (let i = 0; i < state.n; i++) {
    if (Math.abs(state.pos[2 * i]) > state.extent / 2 || Math.abs(state.pos[2 * i + 1]) > state.extent / 2) out++;
    vmax = Math.max(vmax, Math.hypot(state.vel[2 * i], state.vel[2 * i + 1]));
  }
  $("s-out").textContent = `${out} / ${state.n}`;
  $("s-vmax").textContent = vmax.toFixed(2);
}
setInterval(updateStats, 250);

function toggleDebug() {
  debug.hidden = !debug.hidden;
  plotsDirty = true;
  updateStats();
}

$("b-pause").addEventListener("click", () => setRunning(!state.running));
$("b-step").addEventListener("click", () => {
  if (state.running) setRunning(false);
  stepPending = true;
  tryAdvance();
});
$("b-reset").addEventListener("click", () => startRun(state.n));
$("b-mode").addEventListener("click", (e) => { setMode(state.mode === "fixed" ? "random" : "fixed"); e.target.blur(); });
$("r-tick").addEventListener("input", (e) => {
  const k = +e.target.value;
  if (state.running) setRunning(false);
  if (frames.has(k)) show(k); else syncRange();
});
$("r-ch").addEventListener("input", (e) => {
  $("o-ch").textContent = +e.target.value < 0 ? "all" : e.target.value;
  if (model) model.setChannel(+e.target.value);
});
$("c-truth").addEventListener("change", (e) => {
  if (e.target.checked) startTruth(state.pos, state.vel, state.tick);
  else stopTruth();
  e.target.blur();
});
$("b-export").addEventListener("click", () => {
  const snap = {
    tick: state.tick, n: state.n, dt: state.dt, extent: state.extent, G: state.G, meta: state.meta, exportedAt: new Date().toISOString(),
    pos: Array.from(state.pos), vel: Array.from(state.vel),
    truthPos: state.truthOn ? Array.from(state.tPos) : null, truthVel: state.truthOn ? Array.from(state.tVel) : null,
    truthSinceTick: state.truthSince,
    err: state.truthOn ? state.err : null, modelE: state.modelE, truthE: state.truthOn ? state.truthE : null,
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(snap)], { type: "application/json" }));
  a.download = `scatter_state_tick${state.tick}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});
const rBodies = $("r-bodies"), oBodies = $("o-bodies");
rBodies.addEventListener("input", () => { oBodies.textContent = rBodies.value; });
rBodies.addEventListener("change", () => { if (state.ready) startRun(parseInt(rBodies.value, 10)); });

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
  if (e.key === "[" || e.key === "]") {
    const r = $("r-ch");
    r.value = Math.min(31, Math.max(-1, +r.value + (e.key === "]" ? 1 : -1)));
    r.dispatchEvent(new Event("input"));
  }
  else if (e.key === " ") { e.preventDefault(); $("b-pause").click(); }
  else if (e.key === ".") $("b-step").click();
  else if (e.key === "r") $("b-reset").click();
  else if (e.key === "d") toggleDebug();
});

const nav = document.querySelector("nav");
let idle = 0;
window.addEventListener("pointerdown", (e) => {
  if (e.pointerType === "mouse") return;
  nav.classList.add("active");
  clearTimeout(idle);
  idle = setTimeout(() => nav.classList.remove("active"), 3000);
});

async function main() {
  try {
    model = await createView({ canvas: $("stage3d"), labelsEl: $("labels"), tip: $("tip") });
  } catch (e) {
    console.error(e);
    err.textContent = "3D view unavailable: " + e.message; err.hidden = false;
    return;
  }
  $("mview").open = innerWidth > 760;
  model.show(true);
  if (curves) model.setCurves(curves.curve, curves.pair);
  if (state.tick >= 0) model.showFrame(out);
  syncTruth();
  window.__model = model;
  window.__frames = frames;
  window.__sim = () => {
    let bytes = 0;
    for (const f of frames.values()) bytes += frameBytes(f);
    return { tick: state.tick, ahead: newest - state.tick, frames: frames.size, bytes, shown: model.shown, shownRate: rate(shownAt), prodRate: rate(arrivals), epoch: state.epoch };
  };
}

main();
