import { makeScene, SCENES } from "./tiled_model.js";
import { makeOut, dequantFrame, framePhi, toCompact, frameBytes, GG } from "./tframes.js";
import { PRESETS, stateName, decodeAsset, decodeState, fetchProgress, gunzip } from "./precomputed.js";
import { drawSeries } from "../../scatter/js/plots.js";
import { createView } from "./view3d.js";

const $ = (id) => document.getElementById(id);
const err = $("err");
const params = new URLSearchParams(location.search);
const RATE = 10, BEHIND = 20;
const AHEAD = { 2: 50, 4: 24 };
const state = {
  nt: 4, scene: "collapse", n: 300, flags: { ghosts: true, fx: true, far: true }, running: true, tick: -1, runEpoch: 0, modelEpoch: 0, ready: false,
  seed: 4738, mode: "fixed", pos: null, vel: null, errs: [], workers: 1, dt: 0.1,
  pre: null, fromPre: false, preRun: false, live: true, cap: Infinity, computing: false, batchFrom: 0, moreOpen: false, produced: 0,
};
const frames = new Map(), sims = new Map(), errAt = new Map();
let oldest = 0, newest = -1, lastAdv = 0, stepPending = false, view = null, out = null, config = null, inspect = -1, curves = null;
const arrivals = [], shownAt = [];
let producer = null, simWorker = null, simState = 0, readyRes = null;
const readyP = new Promise((res) => { readyRes = res; });
const preLast = () => state.pre.meta.ticks;

const presetFor = () => { const p = PRESETS[state.scene]; return p && state.nt === p.nt && state.n === p.n && state.mode === "fixed" && Object.values(state.flags).every(Boolean) ? p : null; };
const preCache = new Map(), preFailed = new Set(), stateFiles = new Map();

function ensureProducer() {
  if (producer) return;
  producer = new Worker("js/tiled_worker.js", { type: "module" });
  producer.onmessage = onProducer;
  producer.postMessage({ cmd: "init", workers: Math.max(1, Math.min(16, +params.get("workers") || (navigator.hardwareConcurrency || 4) - 2)) });
}

function ensureSim() {
  if (simWorker) return;
  simWorker = new Worker("js/sim_worker.js", { type: "module" });
  simWorker.onmessage = onSim;
}

function loadState() {
  const m = state.pre.meta, p = PRESETS[m.scene];
  if (!stateFiles.has(m.name)) stateFiles.set(m.name, fetch("precomputed/" + stateName(p)).then(gunzip).then((b) => decodeState(b, m.n, m.T)));
  return stateFiles.get(m.name);
}

let loadT0 = 0;

// the file is planar (all ticks of one field together), so ticks only exist once it is all in: the tick count is the byte fraction of the 101 ticks
function setLoad(f, got, total) {
  const p = $("p-load"), t = $("t-load");
  if (f === null) { p.removeAttribute("value"); t.textContent = "loading shipped ticks"; return; }
  if (!loadT0) loadT0 = performance.now();
  const el = (performance.now() - loadT0) / 1000, K = 101;
  const eta = f >= 1 ? "unpacking" : el > 0.3 && f > 0.02 ? "~" + Math.max(1, Math.ceil(el * (1 - f) / f)) + " s left" : "estimating";
  p.value = f;
  t.textContent = `loading shipped ticks ${Math.round(100 * f)}% · ${(got / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB · ~${Math.floor(f * K)} of ${K} ticks · ${eta}`;
}

let loadingPre = false, loadSeq = 0;

async function loadPre(p) {
  loadingPre = true; loadT0 = 0;
  try {
    const meta = await fetch(`precomputed/${p.name}.json`).then((r) => r.json());
    const buf = await gunzip(await fetchProgress(`precomputed/${p.name}.bin.gz`, setLoad));
    preCache.set(p.name, { meta, ...decodeAsset(meta, buf) });
  } catch (e) {
    console.warn("no shipped ticks, computing live", e);
    preFailed.add(p.name);
  }
  loadingPre = false;
}

function rate(list) {
  const now = performance.now();
  while (list.length && now - list[0] > 4000) list.shift();
  return list.length > 1 ? (list.length - 1) / ((list[list.length - 1] - list[0]) / 1000) : 0;
}

const ahead = () => AHEAD[state.nt];

function status() {
  const el = $("mstat");
  $("loading").hidden = state.tick >= 0;
  if (state.tick < 0 && !state.fromPre && !loadingPre) { $("p-load").removeAttribute("value"); $("t-load").textContent = state.ready ? "computing tick 0" : "loading weights"; }
  if (state.tick < 0) { el.textContent = state.fromPre ? "loading shipped ticks" : state.ready ? "computing tick 0 (" + Math.min(state.nt * state.nt, state.wanted || 1) + " tile workers)" : "loading weights"; return; }
  if (state.fromPre && !producer) { el.textContent = `tick ${state.tick} · shipped ticks 0-${preLast()} played from the file, no model code running · U-Net activations only in computed ticks`; moreUi(); return; }
  moreUi();
  const tw = Math.min(state.nt * state.nt, state.wanted || 1), starved = state.running && !(frames.has(state.tick + 1) && sims.has(state.tick + 1));
  el.textContent = `tick ${state.tick} · buffer ${Math.max(0, newest - state.tick)}/${ahead()} · ${rate(shownAt).toFixed(1)} ticks/s shown · ${rate(arrivals).toFixed(2)} frames/s computed · ${tw} tile worker${tw > 1 ? "s" : ""}${starved ? " · waiting for next frame" : ""}`;
}

function startRun() {
  state.runEpoch++; state.modelEpoch++;
  frames.clear(); sims.clear(); errAt.clear();
  oldest = 0; newest = -1; state.tick = -1; stepPending = false; state.errs = [];
  state.cap = Infinity; state.computing = false; state.wantSent = undefined; state.simLimit = 0; state.moreShut = false; state.moreOpen = false;
  if (producer) producer.postMessage({ cmd: "stop" });
  if (simWorker) simWorker.postMessage({ cmd: "stop" });
  simState = 0;
  if (!view || view.nt !== state.nt) { view.build(state.nt); view.nt = state.nt; out = makeOut(state.nt * state.nt, state.nt * 16 + 16); }
  inspect = state.nt * state.nt / 2 + state.nt / 2;
  view.setInspect(inspect);
  view.setFlags(state.flags);
  const p = presetFor();
  if (p && params.get("live") === null && !preCache.has(p.name) && !preFailed.has(p.name)) {
    const seq = ++loadSeq;
    state.pre = null; state.fromPre = state.preRun = false;
    updateStatus();
    loadPre(p).then(() => { if (seq === loadSeq) startRun(); });
    return;
  }
  loadSeq++;
  state.pre = p ? preCache.get(p.name) || null : null;
  state.fromPre = state.preRun = !!state.pre;
  if (state.fromPre) { startPre(); return; }
  ensureProducer();
  if (!state.ready) { updateStatus(); return; }
  state.live = true;
  const seed = state.mode === "fixed" ? state.seed : (state.seed + Math.floor(performance.now())) >>> 0;
  state.runSeed = seed;
  const { pos, vel } = makeScene(state.scene, state.nt, state.n, seed);
  state.pos = pos;
  ensureSim();
  producer.postMessage({ cmd: "reset", epoch: state.modelEpoch, nt: state.nt, flags: state.flags, pos, vel, limit: ahead() });
  simWorker.postMessage({ cmd: "init", epoch: state.runEpoch, n: state.n, dt: state.dt, pos, vel, limit: ahead() + 20 });
  simState = 2;
  updateStatus();
}

// the shipped default scene: every tick comes from the file, no model or sim code runs until the user asks for more
function startPre() {
  const { meta, frames: pf, sims: ps } = state.pre;
  state.live = false;
  state.pos = pf[0].pos;
  pf.forEach((f, k) => { frames.set(k, f); sims.set(k, ps[k]); });
  newest = meta.ticks;
  state.wantSent = undefined;
  syncRange();
  lastAdv = performance.now();
  show(0);
}

function wantSim(limit) {
  state.simLimit = limit;
  if (simState === 2) { simWorker.postMessage({ cmd: "want", epoch: state.runEpoch, limit }); return; }
  if (simState === 1 || !state.preRun || limit <= preLast()) return;
  simState = 1;
  const ep = state.runEpoch;
  loadState().then((st) => {
    if (ep !== state.runEpoch) return;
    ensureSim();
    simWorker.postMessage({ cmd: "init", epoch: ep, n: state.n, dt: state.dt, pos: st.spos.slice(), vel: st.svel.slice(), tick0: preLast(), limit: state.simLimit });
    simState = 2;
  });
}

// pushes the model and sim limits for the tick on screen (or for the whole batch while computing more)
function kick() {
  if (!state.live) return;
  const k = Math.max(state.tick, 0), batch = state.cap !== Infinity, want = batch ? state.cap : k + ahead();
  wantSim(batch ? state.cap : want + 20);
  if (state.wantSent !== want) { state.wantSent = want; producer.postMessage({ cmd: "want", epoch: state.modelEpoch, limit: want }); }
}

async function computeMore() {
  if (!state.fromPre || state.computing) return;
  const count = +$("sel-more").value;
  state.batchFrom = newest;
  state.computing = true;
  state.cap = newest + count;
  if (state.live) { kick(); moreUi(); return; }
  moreUi();
  const ep = state.modelEpoch;
  const st = await loadState();
  ensureProducer();
  await readyP;
  if (ep !== state.modelEpoch || !state.computing) return;
  const last = preLast(), phis = st.phis.map((p) => p.slice());
  state.live = true;
  state.modelEpoch++;
  state.runEpoch++;
  state.wantSent = state.cap;
  producer.postMessage({ cmd: "branch", epoch: state.modelEpoch, nt: state.nt, flags: state.flags, tick: last, pos: st.pos.slice(), vel: st.vel.slice(), acc: st.acc.slice(), phis, limit: state.cap },
    phis.map((p) => p.buffer));
  simState = 0;
  wantSim(state.cap);
}

function stopMore() {
  state.computing = false;
  state.cap = newest;
  kick();
  moreUi();
}

function moreUi() {
  const box = $("more"), pre = state.fromPre, k = state.tick, last = state.pre ? preLast() : 0;
  const open = pre && (state.moreOpen || state.computing || (!state.moreShut && (k >= last || newest > last)));
  box.hidden = !open;
  box.style.setProperty("--navh", document.querySelector("nav").offsetHeight + 28 + "px");
  $("b-morenav").hidden = !pre;
  $("b-morenav").setAttribute("aria-pressed", String(open));
  if (!open) return;
  const done = Math.max(0, newest - state.batchFrom), total = Number.isFinite(state.cap) ? Math.max(1, state.cap - state.batchFrom) : 1;
  $("b-more").hidden = state.computing;
  $("b-more-stop").hidden = !state.computing;
  $("sel-more").disabled = state.computing;
  const pg = $("p-more");
  pg.hidden = !state.computing;
  pg.max = total; pg.value = done;
  if (state.computing) {
    $("m-title").textContent = `Computing in this browser: tick ${newest} of ${state.cap}`;
    $("m-stat").textContent = `${rate(arrivals).toFixed(2)} ticks/s on ${Math.min(state.nt * state.nt, state.wanted || 1)} tile workers`;
  } else if (newest > last) {
    $("m-title").textContent = `Computed through tick ${newest} (shipped through ${last})`;
    $("m-stat").textContent = "";
  } else {
    $("m-title").textContent = k >= last ? `Tick ${last} is the last shipped tick` : `Shipped ticks 0-${last}`;
    $("m-stat").textContent = "";
  }
}

function updateStatus() {
  const f = state.flags;
  for (const [id, key] of [["b-ghost", "ghosts"], ["b-fx", "fx"], ["b-far", "far"]]) $(id).setAttribute("aria-pressed", String(f[key]));
  status();
}

function seamStats(fr, sim) {
  const nt = state.nt, L = nt * 32, n = fr.n, e = { seam: [0, 0], inner: [0, 0], edge: [0, 0] };
  for (let i = 0; i < n; i++) {
    const x = sim.pos[2 * i], y = sim.pos[2 * i + 1], d = Math.hypot(fr.pos[2 * i] - x, fr.pos[2 * i + 1] - y);
    let ds = 1e9;
    for (let s = 1; s < nt; s++) { const b = -L / 2 + 32 * s; ds = Math.min(ds, Math.abs(x - b), Math.abs(y - b)); }
    const dout = L / 2 - Math.max(Math.abs(x), Math.abs(y)), c = dout < 8 ? e.edge : ds < 8 ? e.seam : e.inner;
    c[0] += d; c[1]++;
  }
  return e;
}

function show(k) {
  const fr = frames.get(k), sim = sims.get(k);
  state.tick = k;
  dequantFrame(fr, out);
  view.setFrame(fr, sim, out);
  let m = 0;
  for (let i = 0; i < fr.n; i++) m += Math.hypot(fr.pos[2 * i] - sim.pos[2 * i], fr.pos[2 * i + 1] - sim.pos[2 * i + 1]);
  m /= fr.n;
  errAt.set(k, m);
  state.errs = [];
  for (let t = Math.max(0, k - 199); t <= k; t++) if (errAt.has(t)) state.errs.push(errAt.get(t)); else state.errs.push(NaN);
  const e = seamStats(fr, sim), f = (c) => (c[1] ? (c[0] / c[1]).toFixed(3) : "-");
  $("s-tick").textContent = k;
  $("s-err").textContent = m.toFixed(3);
  $("s-seam").textContent = `${f(e.seam)} / ${f(e.inner)}`;
  $("s-seamn").textContent = `${e.seam[1]} / ${e.inner[1]}`;
  $("s-ms").textContent = fr.shipped ? "shipped" : fr.ms.toFixed(0) + " ms";
  if (!state.chTouched && fr.unet && state.autoCh !== state.runEpoch) {
    state.autoCh = state.runEpoch;
    const a = out.unet.dec[0], n = 128 * 128;
    let best = 0, bv = -1;
    for (let ch = 0; ch < 32; ch++) { let sum = 0; for (let j = 0; j < n; j += 7) sum += Math.abs(a[ch * n + j]); if (sum > bv) { bv = sum; best = ch; } }
    $("r-ch").value = best;
    $("r-ch").dispatchEvent(new Event("input"));
  }
  const fl = fr.flags;
  $("s-fl").textContent = `${fl.ghosts ? "ghosts" : "no ghosts"} · ${fl.fx ? "field exch" : "no field exch"} · ${fl.far ? "far" : "no far"}`;
  let gh = 0;
  for (let t = 0; t < fr.nt * fr.nt; t++) gh += fr.ghOff[t + 1] - fr.ghOff[t];
  $("s-gh").textContent = `${gh} (${(gh / (fr.nt * fr.nt)).toFixed(1)} per tile)`;
  history_trim(k);
  syncRange();
  plot();
  shownAt.push(performance.now());
  kick();
  status();
}

function history_trim(k) {
  for (let t = Math.max(0, k - BEHIND - 40); t < k - BEHIND; t++) {
    const f = frames.get(t);
    if (f && !f.compact) frames.set(t, toCompact(f));
  }
}

function plot() {
  const c = $("c-err");
  if (!$("stats").hidden) drawSeries(c, state.errs.filter(Number.isFinite), "#6fc3ff");
}

function syncRange() {
  const r = $("r-tick");
  r.min = oldest; r.max = newest; r.value = state.tick;
  $("o-tick").textContent = state.tick;
}

function tryAdvance() {
  if (state.tick < 0 || !view) return;
  const next = state.tick + 1, now = performance.now();
  if (!frames.has(next) || !sims.has(next)) return;
  if (stepPending) { stepPending = false; lastAdv = now; show(next); } else if (state.running && now - lastAdv >= 1000 / RATE) { lastAdv = now; show(next); }
}
setInterval(tryAdvance, 20);

function setRunning(on) {
  state.running = on;
  const b = $("b-pause");
  b.setAttribute("aria-pressed", String(!on));
  b.textContent = on ? "Pause" : "Resume";
  lastAdv = performance.now();
}

function branch() {
  const k = state.tick;
  if (k < 0) return;
  const fr = frames.get(k);
  state.modelEpoch++;
  for (const t of [...frames.keys()]) if (t > k) frames.delete(t);
  newest = k;
  state.fromPre = false; state.live = true; state.cap = Infinity; state.computing = false;
  const T = fr.nt * fr.nt, phis = [];
  for (let t = 0; t < T; t++) phis.push(framePhi(fr, t, new Float32Array(GG)));
  const pos = Float64Array.from(fr.pos), vel = Float64Array.from(fr.vel), acc = Float64Array.from(fr.acc), ep = state.modelEpoch;
  state.wantSent = k + ahead();
  ensureProducer();
  readyP.then(() => {
    if (ep !== state.modelEpoch) return;
    producer.postMessage({ cmd: "branch", epoch: ep, nt: state.nt, flags: state.flags, tick: k, pos, vel, acc, phis, limit: k + ahead() }, [pos.buffer, vel.buffer, acc.buffer, ...phis.map((p) => p.buffer)]);
    wantSim(k + ahead() + 20);
  });
  syncRange();
  moreUi();
}

function setFlag(key) {
  state.flags[key] = !state.flags[key];
  view.setFlags(state.flags);
  updateStatus();
  if (state.running) setRunning(false);
  branch();
}

function setNt(nt) {
  if (nt === state.nt) return;
  state.nt = nt;
  state.n = nt === 2 ? 100 : 300;
  const r = $("r-bodies");
  r.min = nt === 2 ? 20 : 60; r.max = nt === 2 ? 300 : 600; r.step = 10; r.value = state.n;
  $("o-bodies").textContent = state.n;
  for (const [id, v] of [["b-nt2", 2], ["b-nt4", 4]]) $(id).setAttribute("aria-pressed", String(v === nt));
  startRun();
}

function setInspect(k) {
  if (k === inspect) return;
  inspect = k;
  view.setInspect(k);
  if (!state.live) return;
  producer.postMessage({ cmd: "inspect", tile: k });
  if (state.tick >= 0) branch();
}

const mcapText = (params) => `${params.toLocaleString()} parameters shared by every tile (U-Net, short-range pair MLP) plus the 4,417-parameter far kernel. Each tile runs the same 128x128 grid network on its 32-unit core plus a 16-unit halo; tiles never see each other except through the exchanged ghost bodies, halo potential and the coarse far term.`;

function onProducer(e) {
  const m = e.data;
  if (m.type === "error") { err.textContent = "Model error: " + m.message; err.hidden = false; return; }
  if (m.type === "ready") {
    config = m.config; curves = m.curves; state.wanted = m.workers; state.dt = config.dt;
    $("s-ckpt").textContent = `${m.checkpoint} (${m.iters} it) + ${m.kernel}`;
    $("mcap").textContent = mcapText(m.params);
    state.ready = true;
    readyRes();
    if (view && !state.fromPre) { view.setChart(curves); startRun(); }
    return;
  }
  if (m.type !== "frame" || m.epoch !== state.modelEpoch) return;
  frames.set(m.tick, m);
  newest = Math.max(newest, m.tick);
  state.produced++;
  arrivals.push(performance.now());
  if (state.computing && newest >= state.cap) state.computing = false;
  if (state.tick < 0 && sims.has(m.tick)) { lastAdv = performance.now(); show(m.tick); } else syncRange();
  status();
}

function onSim(e) {
  const m = e.data;
  if (m.type !== "sim" || m.epoch !== state.runEpoch) return;
  sims.set(m.tick, m);
  if (state.tick < 0 && frames.has(m.tick)) { lastAdv = performance.now(); show(m.tick); }
}

$("b-pause").addEventListener("click", () => setRunning(!state.running));
$("b-step").addEventListener("click", () => { if (state.running) setRunning(false); stepPending = true; tryAdvance(); });
$("b-reset").addEventListener("click", () => startRun());
$("b-nt2").addEventListener("click", () => setNt(2));
$("b-nt4").addEventListener("click", () => setNt(4));
$("b-ghost").addEventListener("click", () => setFlag("ghosts"));
$("b-fx").addEventListener("click", () => setFlag("fx"));
$("b-far").addEventListener("click", () => setFlag("far"));
$("b-mode").addEventListener("click", (e) => {
  state.mode = state.mode === "fixed" ? "random" : "fixed";
  e.target.setAttribute("aria-pressed", String(state.mode === "fixed"));
  e.target.textContent = state.mode === "fixed" ? "Fixed seed" : "Random";
  e.target.blur();
  startRun();
});
$("sel-scene").addEventListener("change", (e) => { state.scene = e.target.value; e.target.blur(); startRun(); });
const rBodies = $("r-bodies");
rBodies.addEventListener("input", () => { $("o-bodies").textContent = rBodies.value; });
rBodies.addEventListener("change", () => { state.n = +rBodies.value; startRun(); });
$("r-tick").addEventListener("input", (e) => {
  const k = +e.target.value;
  if (state.running) setRunning(false);
  if (frames.has(k) && sims.has(k)) show(k); else syncRange();
});
$("r-ch").addEventListener("pointerdown", () => { state.chTouched = true; });
$("r-ch").addEventListener("input", (e) => {
  $("o-ch").textContent = +e.target.value < 0 ? "all" : e.target.value;
  if (view) view.setChannel(+e.target.value);
});
for (const b of document.querySelectorAll("[data-layer]")) b.addEventListener("click", () => {
  for (const o of document.querySelectorAll("[data-layer]")) o.setAttribute("aria-pressed", String(o === b));
  view.setLayer(+b.dataset.layer);
});
for (const b of document.querySelectorAll("[data-focus]")) b.addEventListener("click", () => view.focus(b.dataset.focus));
function toggleOpt(id, key) {
  const b = $(id), on = b.getAttribute("aria-pressed") !== "true";
  b.setAttribute("aria-pressed", String(on));
  view.setOptions({ [key]: on });
}
$("b-links").addEventListener("click", () => toggleOpt("b-links", "links"));
$("b-arrows").addEventListener("click", () => toggleOpt("b-arrows", "arrows"));
$("b-overlay").addEventListener("click", () => toggleOpt("b-overlay", "overlay"));
$("b-stats").addEventListener("click", () => { $("stats").hidden = !$("stats").hidden; $("b-stats").setAttribute("aria-pressed", String(!$("stats").hidden)); plot(); });

$("b-more").addEventListener("click", () => computeMore());
$("b-more-stop").addEventListener("click", () => stopMore());
$("b-morenav").addEventListener("click", () => { state.moreOpen = true; state.moreShut = false; moreUi(); $("b-more").focus(); });
$("b-more-x").addEventListener("click", () => { state.moreOpen = false; state.moreShut = true; moreUi(); });

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key;
  if (k === "[" || k === "]") { state.chTouched = true; const r = $("r-ch"); r.value = Math.min(31, Math.max(-1, +r.value + (k === "]" ? 1 : -1))); r.dispatchEvent(new Event("input")); }
  else if (k === " ") { e.preventDefault(); $("b-pause").click(); }
  else if (k === ".") $("b-step").click();
  else if (k === "r") $("b-reset").click();
  else if (k === "g") $("b-ghost").click();
  else if (k === "x") $("b-fx").click();
  else if (k === "f") $("b-far").click();
  else if (k === "e") $("b-links").click();
  else if (k === "a") $("b-arrows").click();
  else if (k === "o") $("b-overlay").click();
  else if (k === "d") $("b-stats").click();
  else if (k >= "1" && k <= "5") document.querySelectorAll("[data-layer]")[+k - 1].click();
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
    view = await createView({ canvas: $("stage3d"), labelsEl: $("labels"), tip: $("tip"), handlers: { inspect: setInspect } });
  } catch (e) {
    console.error(e);
    err.textContent = "3D view unavailable: " + e.message; err.hidden = false;
    return;
  }
  const nt = +params.get("nt") || 4;
  if (params.get("scene") && SCENES[params.get("scene")]) { state.scene = params.get("scene"); $("sel-scene").value = state.scene; }
  state.nt = 0;
  setNtInitial(nt);
  view.show(true);
  setInterval(status, 500);
  window.__view = view;
  window.__app = () => {
    let bytes = 0;
    for (const f of frames.values()) bytes += frameBytes(f);
    return { tick: state.tick, ahead: newest - state.tick, frames: frames.size, bytes, nt: state.nt, n: state.n, flags: { ...state.flags }, err: errAt.get(state.tick), shownRate: rate(shownAt), prodRate: rate(arrivals), workers: Math.min(state.nt * state.nt, state.wanted || 1), inspect,
      fromPre: state.fromPre, producerStarted: !!producer, simStarted: !!simWorker, produced: state.produced, newest, computing: state.computing, cap: state.cap };
  };
  window.__state = state;
  window.__frames = frames;
  const p0 = presetFor();
  if (p0 && params.get("live") === null) await loadPre(p0);
  state.pre = p0 ? preCache.get(p0.name) || null : null;
  if (state.pre) {
    const m = state.pre.meta;
    config = m.config; curves = m.curves; state.dt = m.dt;
    view.setChart(curves);
    $("s-ckpt").textContent = `${m.checkpoint} (${m.iters} it) + ${m.kernel}`;
    $("mcap").textContent = mcapText(m.params);
    startRun();
  } else {
    ensureProducer();
    if (state.ready) { view.setChart(curves); startRun(); }
  }
}

function setNtInitial(nt) {
  state.nt = nt;
  state.n = nt === 2 ? 100 : 300;
  const r = $("r-bodies");
  r.min = nt === 2 ? 20 : 60; r.max = nt === 2 ? 300 : 600; r.step = 10; r.value = state.n;
  $("o-bodies").textContent = state.n;
  for (const [id, v] of [["b-nt2", 2], ["b-nt4", 4]]) $(id).setAttribute("aria-pressed", String(v === nt));
  view.build(nt);
  view.nt = nt;
  out = makeOut(nt * nt, nt * 16 + 16);
}

main();
