import { initBodies, mulberry32 } from "./scatter_model.js";
import { drawEnergy, drawSeries } from "./plots.js";

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
  seed: 4738, tick: 0, epoch: 0, running: true, showField: true,
  ms: 0, ready: false, meta: null, E0: undefined, modelEnergy: 0,
  err: [], modelE: [], truthE: [], histLen: 200,
};
const cam = { auto: true, cx: 0, cy: 0, scale: 1 };
const history = new Map();
let dirty = true, plotsDirty = true, fieldDirty = false, truthWorker = null;

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
  if (m.type === "ready") {
    Object.assign(state, { G: m.config.grid, extent: m.config.extent, dt: m.config.dt, ready: true });
    state.meta = { checkpoint: m.checkpoint, iters: m.iters, config: m.config };
    $("s-ckpt").textContent = `${m.checkpoint} (${m.iters} it)`;
    spawn(state.n);
    worker.postMessage({ cmd: "run", run: state.running });
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
};

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
  if (dirty) render();
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

$("b-pause").addEventListener("click", (e) => {
  state.running = !state.running;
  e.target.setAttribute("aria-pressed", String(!state.running));
  e.target.textContent = state.running ? "Pause" : "Resume";
  worker.postMessage({ cmd: "run", run: state.running });
});
$("b-step").addEventListener("click", () => {
  if (state.running) $("b-pause").click();
  worker.postMessage({ cmd: "step" });
});
$("b-reset").addEventListener("click", () => spawn(state.n));
$("b-fit").addEventListener("click", () => { cam.auto = true; syncFit(); touch(); });
$("b-field").addEventListener("click", (e) => {
  state.showField = !state.showField;
  e.target.setAttribute("aria-pressed", String(state.showField));
  worker.postMessage({ cmd: "field", on: state.showField });
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
rBodies.addEventListener("change", () => state.ready && spawn(parseInt(rBodies.value, 10)));

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
  if (e.key === " ") { e.preventDefault(); $("b-pause").click(); }
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
