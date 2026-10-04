import { initBodies, mulberry32 } from "./scatter_model.js";
import { drawEnergy, drawSeries } from "./plots.js";
import * as truth from "./truth.js";

const stage = document.getElementById("stage");
const ctx = stage.getContext("2d");
const err = document.getElementById("err");
const heat = document.createElement("canvas");
const heatCtx = heat.getContext("2d");
const $ = (id) => document.getElementById(id);

const debug = $("debug");
const state = {
  n: 40, pos: null, vel: null, field: null, G: 128, extent: 64, dt: 0.1,
  tPos: null, tVel: null, tTick: 0, tMs: 0, truthOn: false, truthSince: null,
  mode: "pre", run: null, seed: 4738, tick: 0, epoch: 0, running: true, showField: true,
  ms: 0, ready: false, meta: null, E0: undefined, modelEnergy: 0,
  err: [], modelE: [], truthE: [], histLen: 200,
};
const PRE_RATE = 10, WARM = 2;
const runs = new Map();
let runIndex = null, playTimer = null, probeBusy = false, probeKey = "", probedKey = "";
const cam = { auto: true, cx: 0, cy: 0, scale: 1 };
const history = new Map();
let dirty = true, plotsDirty = true, fieldDirty = false, truthWorker = null;
let model = null, modelOpen = false, modelLoading = false, traceBusy = false, traceKey = "", traceT = 0, curvesSent = false;

const worker = new Worker("js/worker.js", { type: "module" });

function touch() { dirty = true; }

function spawn(n) {
  const rng = mulberry32(state.seed + Math.floor(performance.now()));
  const { pos, vel } = initBodies(n, rng);
  state.n = n; state.epoch++;
  state.E0 = undefined; state.err = []; state.modelE = []; state.truthE = [];
  state.pos = pos; state.vel = vel; state.tick = 0;
  history.clear();
  worker.postMessage({ cmd: "reset", epoch: state.epoch, n, pos, vel, energy: !debug.hidden });
  if (state.truthOn) startTruth(pos, vel, 0, true);
  touch();
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
  state.n = n; state.epoch++;
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
  $("s-field").textContent = "approx (phi_prev warmed from the previous 2 ticks, ~5-9% off)";
  state.run = run;
  state.E0 = undefined; state.err = []; state.modelE = []; state.truthE = [];
  history.clear(); probeKey = "";
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
  touch(); plotsDirty = true;
  probe();
  wantTrace();
}

function probe() {
  if (state.mode !== "pre" || !state.run || !state.ready || probeBusy || !(state.showField || modelOpen)) return;
  const key = state.epoch + ":" + state.tick;
  if (probeKey === key) return;
  probeBusy = true; probeKey = key;
  const { n, data } = state.run, k = state.tick, warm = [];
  for (let t = Math.max(0, k - WARM); t < k; t++) warm.push([data.slice(t * 4 * n, t * 4 * n + 2 * n), data.slice(t * 4 * n + 2 * n, t * 4 * n + 4 * n)]);
  worker.postMessage({ cmd: "probe", epoch: state.epoch, tick: k, n, pos: state.pos.slice(), vel: state.vel.slice(), warm });
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
  $("s-field").textContent = mode === "pre" ? "approx (phi_prev warmed from the previous 2 ticks, ~5-9% off)" : "live";
  rBodies.step = mode === "pre" ? 10 : 1;
  history.clear(); probeKey = "";
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
}

function stopTruth() {
  if (truthWorker) truthWorker.terminate();
  truthWorker = null;
  state.truthOn = false; state.truthSince = null; state.tPos = null; state.tVel = null; state.truthE = []; state.err = [];
  syncTruthUI();
  touch(); plotsDirty = true;
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
  touch(); plotsDirty = true;
}

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "error") { err.textContent = "Failed to load weights: " + m.message; err.hidden = false; return; }
  if (m.type === "trace") {
    traceBusy = false;
    if (m.epoch !== state.epoch || !model) return;
    traceKey = m.epoch + ":" + m.tick; traceT = performance.now();
    if (m.curve) curvesSent = true;
    model.onTrace(m);
    return;
  }
  if (m.type === "ready") {
    Object.assign(state, { G: m.config.grid, extent: m.config.extent, dt: m.config.dt, ready: true });
    state.meta = { checkpoint: m.checkpoint, iters: m.iters, config: m.config };
    $("s-ckpt").textContent = `${m.checkpoint} (${m.iters} it)`;
    if (state.mode === "live") {
      spawn(state.n);
      worker.postMessage({ cmd: "run", run: state.running });
    } else probe();
    return;
  }
  if (m.type === "probed") {
    probeBusy = false;
    if (m.epoch === state.epoch) probedKey = m.epoch + ":" + m.tick;
    probe();
    wantTrace();
    return;
  }
  if (m.epoch !== state.epoch) return;
  if (m.type === "field") { state.field = m.field; fieldDirty = true; touch(); return; }
  if (m.type === "energy") { state.modelEnergy = m.energy; if (state.E0 === undefined) state.E0 = m.energy; return; }
  state.pos = m.pos; state.vel = m.vel; state.tick = m.tick; state.ms = m.ms;
  if (m.field) { state.field = m.field; fieldDirty = true; }
  if (m.energy !== null) { state.modelEnergy = m.energy; if (state.E0 === undefined) state.E0 = m.energy; }
  history.set(m.tick, { pos: m.pos });
  history.delete(m.tick - 64);
  if (truthWorker) truthWorker.postMessage({ cmd: "advance", epoch: state.epoch, tick: m.tick });
  if (m.energy !== null && m.tick > 0 && m.tick > (state.truthSince || 0)) {
    state.modelE.push((m.energy - state.E0) / Math.abs(state.E0));
    if (state.modelE.length > state.histLen) state.modelE.shift();
  }
  touch(); plotsDirty = true;
  wantTrace();
};

function wantTrace() {
  if (!modelOpen || traceBusy || !state.ready || !state.pos) return;
  if (state.mode === "pre" && probedKey !== state.epoch + ":" + state.tick) return;
  if (traceKey === state.epoch + ":" + state.tick) return;
  if (state.running && performance.now() - traceT < 1000) return;
  traceBusy = true;
  worker.postMessage({ cmd: "trace", curves: !curvesSent });
}
setInterval(wantTrace, 250);

async function toggleModel() {
  const on = !modelOpen;
  if (on && !model) {
    if (modelLoading) return;
    modelLoading = true;
    $("b-model").textContent = "Loading...";
    try {
      const { createView } = await import("./model3d.js");
      model = await createView({ canvas: $("stage3d"), labelsEl: $("labels"), tip: $("tip") });
      $("mview").open = innerWidth > 760;
      window.__model = model;
    } catch (e) {
      console.error(e);
      $("b-model").textContent = "Model view unavailable";
      $("b-model").disabled = true;
      return;
    } finally {
      modelLoading = false;
    }
    $("b-model").innerHTML = "Model<kbd>m</kbd>";
  }
  modelOpen = on;
  document.body.classList.toggle("model", on);
  $("b-model").setAttribute("aria-pressed", String(on));
  $("stage").hidden = on;
  $("mview").hidden = !on;
  model.show(on);
  worker.postMessage({ cmd: "field", on: on ? false : state.showField, quiet: state.mode === "pre" });
  probe();
  if (on) wantTrace(); else touch();
}

function bounds() {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const p = state.pos;
  for (let i = 0; i < state.n; i++) {
    const x = p[2 * i], y = p[2 * i + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX)) { minX = minY = -10; maxX = maxY = 10; }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const half = Math.min(state.extent / 2, Math.max(10, (maxX - minX) / 2, (maxY - minY) / 2) * 1.25);
  return { cx, cy, half };
}

function view(w, h) {
  const b = bounds();
  const fit = Math.min(w, h) / (2 * b.half);
  if (cam.auto) return { cx: b.cx, cy: b.cy, scale: fit, fit };
  return { cx: cam.cx, cy: cam.cy, scale: cam.scale, fit };
}

function zoomAt(px, py, k) {
  const w = stage.clientWidth, h = stage.clientHeight;
  const v = view(w, h);
  const s = Math.min(40 * v.fit, Math.max(0.25 * v.fit, v.scale * k));
  const wx = v.cx + (px - w / 2) / v.scale, wy = v.cy + (py - h / 2) / v.scale;
  cam.auto = false; cam.scale = s; cam.cx = wx - (px - w / 2) / s; cam.cy = wy - (py - h / 2) / s;
  syncFit(); touch();
}

function panBy(dx, dy) {
  const v = view(stage.clientWidth, stage.clientHeight);
  cam.auto = false; cam.scale = v.scale; cam.cx = v.cx - dx / v.scale; cam.cy = v.cy - dy / v.scale;
  syncFit(); touch();
}

function syncFit() { $("b-fit").setAttribute("aria-pressed", String(cam.auto)); }

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

function drawBodies(p, v, w, h, r, color, alpha) {
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  for (let i = 0; i < state.n; i++) {
    const x = w / 2 + (p[2 * i] - v.cx) * v.scale, y = h / 2 + (p[2 * i + 1] - v.cy) * v.scale;
    if (!(x >= -r && x <= w + r && y >= -r && y <= h + r)) continue;
    ctx.moveTo(x + r, y);
    ctx.arc(x, y, r, 0, Math.PI * 2);
  }
  ctx.fill();
  ctx.globalAlpha = 1;
}

function render() {
  dirty = false;
  const w = stage.clientWidth, h = stage.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (stage.width !== w * dpr || stage.height !== h * dpr) { stage.width = w * dpr; stage.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#06080c"; ctx.fillRect(0, 0, w, h);
  if (!state.pos) return;

  const v = view(w, h);
  if (state.showField && state.field) {
    if (fieldDirty) { paintField(); fieldDirty = false; }
    const x0 = w / 2 + (-state.extent / 2 - v.cx) * v.scale, y0 = h / 2 + (-state.extent / 2 - v.cy) * v.scale, side = state.extent * v.scale;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heat, x0, y0, side, side);
    ctx.strokeStyle = "#1c2535"; ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, side, side);
  }
  const r = Math.min(10, Math.max(3.5, 0.5 * v.scale));
  if (state.truthOn && state.tPos && state.tPos.length === 2 * state.n) drawBodies(state.tPos, v, w, h, r, "#ff9a3c", 0.5);
  drawBodies(state.pos, v, w, h, r, "#6fc3ff", 1);
}

function frame() {
  requestAnimationFrame(frame);
  if (dirty && !modelOpen) render();
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
  if (modelOpen) return;
  const v = view(stage.clientWidth, stage.clientHeight);
  $("s-zoom").textContent = `${(v.scale / v.fit).toFixed(2)}x ${cam.auto ? "auto" : "manual"}`;
  $("s-centre").textContent = `${v.cx.toFixed(1)}, ${v.cy.toFixed(1)}`;
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
$("b-model").addEventListener("click", (e) => { toggleModel(); e.target.blur(); });
$("r-ch").addEventListener("input", (e) => {
  $("o-ch").textContent = e.target.value;
  if (model) model.setChannel(+e.target.value);
});
$("b-fit").addEventListener("click", () => { cam.auto = true; syncFit(); touch(); });
$("b-field").addEventListener("click", (e) => {
  state.showField = !state.showField;
  e.target.setAttribute("aria-pressed", String(state.showField));
  worker.postMessage({ cmd: "field", on: state.showField, quiet: state.mode === "pre" });
  probe();
  touch();
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
    field: state.field ? Array.from(state.field) : null,
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

spawnPre(state.n);
const pointers = new Map();
stage.addEventListener("pointerdown", (e) => {
  stage.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
});
stage.addEventListener("pointermove", (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  if (pointers.size === 1) {
    panBy(e.clientX - p.x, e.clientY - p.y);
  } else if (pointers.size === 2) {
    const o = [...pointers.entries()].find(([id]) => id !== e.pointerId)[1];
    const d0 = Math.hypot(p.x - o.x, p.y - o.y), d1 = Math.hypot(e.clientX - o.x, e.clientY - o.y);
    const m0x = (p.x + o.x) / 2, m0y = (p.y + o.y) / 2, m1x = (e.clientX + o.x) / 2, m1y = (e.clientY + o.y) / 2;
    panBy(m1x - m0x, m1y - m0y);
    if (d0 > 0 && d1 > 0) zoomAt(m1x, m1y, d1 / d0);
  }
  p.x = e.clientX; p.y = e.clientY;
});
const endPointer = (e) => pointers.delete(e.pointerId);
stage.addEventListener("pointerup", endPointer);
stage.addEventListener("pointercancel", endPointer);
stage.addEventListener("wheel", (e) => {
  e.preventDefault();
  const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
  zoomAt(e.clientX, e.clientY, Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0015)));
}, { passive: false });
stage.addEventListener("dblclick", (e) => zoomAt(e.clientX, e.clientY, 2));
window.addEventListener("resize", touch);

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
  const cx = stage.clientWidth / 2, cy = stage.clientHeight / 2;
  if (e.key === "m") toggleModel();
  else if (modelOpen && (e.key === "[" || e.key === "]")) {
    const r = $("r-ch");
    r.value = Math.min(31, Math.max(0, +r.value + (e.key === "]" ? 1 : -1)));
    r.dispatchEvent(new Event("input"));
  }
  else if (modelOpen && (e.key === "f" || e.key === "0" || e.key === "+" || e.key === "=" || e.key === "-" || e.key.startsWith("Arrow"))) return;
  else if (e.key === " ") { e.preventDefault(); $("b-pause").click(); }
  else if (e.key === ".") $("b-step").click();
  else if (e.key === "r") $("b-reset").click();
  else if (e.key === "f") $("b-field").click();
  else if (e.key === "d") toggleDebug();
  else if (e.key === "0") $("b-fit").click();
  else if (e.key === "+" || e.key === "=") zoomAt(cx, cy, 1.25);
  else if (e.key === "-") zoomAt(cx, cy, 0.8);
  else if (e.key === "ArrowLeft") { e.preventDefault(); panBy(40, 0); }
  else if (e.key === "ArrowRight") { e.preventDefault(); panBy(-40, 0); }
  else if (e.key === "ArrowUp") { e.preventDefault(); panBy(0, 40); }
  else if (e.key === "ArrowDown") { e.preventDefault(); panBy(0, -40); }
});

const nav = document.querySelector("nav");
let idle = 0;
window.addEventListener("pointerdown", (e) => {
  if (e.pointerType === "mouse") return;
  nav.classList.add("active");
  clearTimeout(idle);
  idle = setTimeout(() => nav.classList.remove("active"), 3000);
});
