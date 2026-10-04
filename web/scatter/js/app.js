import { initBodies, mulberry32 } from "./scatter_model.js";
import { drawEnergy, drawSeries } from "./plots.js";
import * as truth from "./truth.js";
import { createView } from "./model3d.js";

const err = document.getElementById("err");
const $ = (id) => document.getElementById(id);

const debug = $("debug");
const state = {
  n: 40, pos: null, vel: null, G: 128, extent: 64, dt: 0.1,
  tPos: null, tVel: null, tTick: 0, tMs: 0, truthOn: false, truthSince: null,
  mode: "pre", run: null, seed: 4738, tick: 0, epoch: 0, running: true,
  ms: 0, ready: false, meta: null, E0: undefined, modelEnergy: 0,
  err: [], modelE: [], truthE: [], histLen: 200,
};
const PRE_RATE = 10, WARM = 2, LIVE_TRACE_GAP = 600;
const runs = new Map();
let runIndex = null, playTimer = null;
const history = new Map();
let plotsDirty = true, truthWorker = null;
let model = null, traceBusy = false, traceKey = "", traceTick = -1, traceT = 0, curvesSent = false;

const worker = new Worker("js/worker.js", { type: "module" });

function syncBodies() {
  if (!model || !state.pos) return;
  model.setBodies(state.pos, state.vel, state.n);
  model.setTruth(state.truthOn ? state.tPos : null);
  updateStatus();
}

function updateStatus() {
  if (!state.pos) return;
  const lag = traceTick >= 0 && traceTick !== state.tick;
  $("mstat").textContent = traceTick < 0 ? "waiting for trace" : `bodies: tick ${state.tick} · slabs: tick ${traceTick}${traceBusy && lag ? " · updating" : ""}`;
}

function spawn(n) {
  const rng = mulberry32(state.seed + Math.floor(performance.now()));
  const { pos, vel } = initBodies(n, rng);
  state.n = n; state.epoch++;
  state.E0 = undefined; state.err = []; state.modelE = []; state.truthE = [];
  state.pos = pos; state.vel = vel; state.tick = 0;
  history.clear();
  worker.postMessage({ cmd: "reset", epoch: state.epoch, n, pos, vel, energy: !debug.hidden });
  if (state.truthOn) startTruth(pos, vel, 0, true);
  syncBodies();
  wantTrace();
}

async function loadRun(n) {
  if (runs.has(n)) return runs.get(n);
  if (!runIndex) runIndex = await (await fetch("precomputed/index.json")).json();
  const buf = await (await fetch("precomputed/" + runIndex.runs[n])).arrayBuffer();
  const head = new DataView(buf);
  if (new TextDecoder().decode(new Uint8Array(buf, 0, 4)) !== "SCPC" || head.getUint32(8, true) !== n) throw new Error("bad precomputed file for " + n);
  const run = { n, ticks: head.getUint32(12, true), data: new Float32Array(buf, 32), energy: null };
  run.energy = new Float64Array(run.ticks + 1).fill(NaN);
  runs.set(n, run);
  return run;
}

async function spawnPre(n) {
  state.n = n; state.epoch++; state.run = null;
  const epoch = state.epoch;
  stopPlay();
  let run;
  try { run = await loadRun(n); } catch (e) {
    err.textContent = "Failed to load precomputed run: " + e.message; err.hidden = false;
    return;
  }
  if (epoch !== state.epoch || state.mode !== "pre") return;
  err.hidden = true;
  $("s-src").textContent = `precomputed, seed ${state.seed}, ${run.ticks} ticks`;
  state.run = run;
  state.E0 = undefined; state.err = []; state.modelE = []; state.truthE = [];
  history.clear();
  $("r-tick").max = run.ticks;
  showTick(0);
  if (state.truthOn) startTruth(state.pos, state.vel, 0, true);
  if (state.running) startPlay();
}

function energyAt(k) {
  const run = state.run;
  if (Number.isNaN(run.energy[k])) {
    const o = k * 4 * run.n;
    run.energy[k] = truth.energy(run.data.subarray(o, o + 2 * run.n), run.data.subarray(o + 2 * run.n, o + 4 * run.n), run.n);
  }
  return run.energy[k];
}

function showTick(k) {
  const { n, data } = state.run;
  const o = k * 4 * n;
  state.tick = k; state.ms = 0;
  state.pos = Float64Array.from(data.subarray(o, o + 2 * n));
  state.vel = Float64Array.from(data.subarray(o + 2 * n, o + 4 * n));
  state.modelEnergy = energyAt(k);
  if (state.E0 === undefined) state.E0 = state.modelEnergy;
  state.modelE = [];
  for (let t = Math.max((state.truthSince || 0) + 1, k - state.histLen + 1); t <= k; t++) state.modelE.push((energyAt(t) - state.E0) / Math.abs(state.E0));
  history.set(k, { pos: state.pos });
  if (truthWorker) truthWorker.postMessage({ cmd: "advance", epoch: state.epoch, tick: k });
  $("r-tick").value = k; $("o-tick").textContent = k;
  plotsDirty = true;
  syncBodies();
  wantTrace();
}

function startPlay() {
  stopPlay();
  playTimer = setInterval(() => {
    if (state.tick < state.run.ticks) showTick(state.tick + 1);
    if (state.tick >= state.run.ticks) setRunning(false);
  }, 1000 / PRE_RATE);
}

function stopPlay() {
  clearInterval(playTimer);
  playTimer = null;
}

function setRunning(on) {
  state.running = on;
  const b = $("b-pause");
  b.setAttribute("aria-pressed", String(!on));
  b.textContent = on ? "Pause" : "Resume";
  if (state.mode === "live") { worker.postMessage({ cmd: "run", run: on }); return; }
  if (!on) stopPlay();
  else if (state.run && state.tick >= state.run.ticks) spawnPre(state.n);
  else if (state.run) startPlay();
}

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode; state.epoch++;
  stopPlay();
  document.body.classList.toggle("live", mode === "live");
  $("b-mode").setAttribute("aria-pressed", String(mode === "pre"));
  $("b-mode").textContent = mode === "pre" ? "Precomputed" : "Live";
  $("s-src").textContent = mode === "pre" ? `precomputed, seed ${state.seed}, ${runIndex ? runIndex.ticks : "?"} ticks` : "live, random start";
  rBodies.step = mode === "pre" ? 10 : 1;
  history.clear();
  if (mode === "pre") {
    worker.postMessage({ cmd: "run", run: false });
    rBodies.value = Math.min(100, Math.max(10, Math.round(state.n / 10) * 10));
    oBodies.textContent = rBodies.value;
    spawnPre(+rBodies.value);
  } else {
    state.run = null;
    spawn(state.n);
    worker.postMessage({ cmd: "run", run: state.running });
  }
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
  syncBodies();
}

function stopTruth() {
  if (truthWorker) truthWorker.terminate();
  truthWorker = null;
  state.truthOn = false; state.truthSince = null; state.tPos = null; state.tVel = null; state.truthE = []; state.err = [];
  syncTruthUI();
  syncBodies();
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
  syncBodies();
  plotsDirty = true;
}

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "error") { err.textContent = "Failed to load weights: " + m.message; err.hidden = false; return; }
  if (m.type === "trace") {
    traceBusy = false;
    traceT = performance.now();
    if (m.epoch === state.epoch && model) {
      traceKey = m.epoch + ":" + m.tick; traceTick = m.tick;
      if (m.curve) curvesSent = true;
      model.onTrace(m);
      $("s-trace").textContent = m.ms.toFixed(0) + " ms";
    }
    updateStatus();
    wantTrace();
    return;
  }
  if (m.type === "ready") {
    Object.assign(state, { G: m.config.grid, extent: m.config.extent, dt: m.config.dt, ready: true });
    state.meta = { checkpoint: m.checkpoint, iters: m.iters, config: m.config };
    $("s-ckpt").textContent = `${m.checkpoint} (${m.iters} it)`;
    if (state.mode === "live") {
      spawn(state.n);
      worker.postMessage({ cmd: "run", run: state.running });
    } else wantTrace();
    return;
  }
  if (m.epoch !== state.epoch) return;
  if (m.type === "energy") { state.modelEnergy = m.energy; if (state.E0 === undefined) state.E0 = m.energy; return; }
  state.pos = m.pos; state.vel = m.vel; state.tick = m.tick; state.ms = m.ms;
  if (m.energy !== null) { state.modelEnergy = m.energy; if (state.E0 === undefined) state.E0 = m.energy; }
  history.set(m.tick, { pos: m.pos });
  history.delete(m.tick - 64);
  if (truthWorker) truthWorker.postMessage({ cmd: "advance", epoch: state.epoch, tick: m.tick });
  if (m.energy !== null && m.tick > 0 && m.tick > (state.truthSince || 0)) {
    state.modelE.push((m.energy - state.E0) / Math.abs(state.E0));
    if (state.modelE.length > state.histLen) state.modelE.shift();
  }
  plotsDirty = true;
  syncBodies();
  wantTrace();
};

function wantTrace() {
  if (!model || traceBusy || !state.ready || !state.pos) return;
  if (traceKey === state.epoch + ":" + state.tick) return;
  if (state.mode === "live") {
    if (state.running && performance.now() - traceT < LIVE_TRACE_GAP) return;
    traceBusy = true;
    worker.postMessage({ cmd: "trace", curves: !curvesSent });
    return;
  }
  if (!state.run) return;
  const { n, data } = state.run, k = state.tick, warm = [];
  for (let t = Math.max(0, k - WARM); t < k; t++) warm.push([data.slice(t * 4 * n, t * 4 * n + 2 * n), data.slice(t * 4 * n + 2 * n, t * 4 * n + 4 * n)]);
  traceBusy = true;
  worker.postMessage({ cmd: "tracetick", epoch: state.epoch, tick: k, n, pos: state.pos.slice(), vel: state.vel.slice(), warm, curves: !curvesSent });
}
setInterval(wantTrace, 250);

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
  if (debug.hidden || !state.pos) return;
  $("s-tick").textContent = state.tick;
  $("s-bodies").textContent = state.n;
  $("s-ms").textContent = state.ms ? state.ms.toFixed(0) + " ms" : "-";
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
  worker.postMessage({ cmd: "energy", on: !debug.hidden });
  plotsDirty = true;
  updateStats();
}

$("b-pause").addEventListener("click", () => setRunning(!state.running));
$("b-step").addEventListener("click", () => {
  if (state.running) setRunning(false);
  if (state.mode === "live") worker.postMessage({ cmd: "step" });
  else if (state.run && state.tick < state.run.ticks) showTick(state.tick + 1);
});
$("b-reset").addEventListener("click", () => state.mode === "pre" ? spawnPre(state.n) : spawn(state.n));
$("b-mode").addEventListener("click", (e) => { setMode(state.mode === "pre" ? "live" : "pre"); e.target.blur(); });
$("r-tick").addEventListener("input", (e) => {
  if (state.mode !== "pre" || !state.run) return;
  if (state.running) setRunning(false);
  showTick(+e.target.value);
});
$("r-ch").addEventListener("input", (e) => {
  $("o-ch").textContent = e.target.value;
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
rBodies.addEventListener("change", () => {
  const v = parseInt(rBodies.value, 10);
  if (state.mode === "pre") spawnPre(v);
  else if (state.ready) spawn(v);
});

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
  if (e.key === "[" || e.key === "]") {
    const r = $("r-ch");
    r.value = Math.min(31, Math.max(0, +r.value + (e.key === "]" ? 1 : -1)));
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
  window.__model = model;
  syncBodies();
  wantTrace();
}

spawnPre(state.n);
main();
