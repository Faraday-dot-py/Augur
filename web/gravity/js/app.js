import { loadWeights, CentralNet, initBodies, mulberry32 } from "./central_model.js";
import * as truth from "./truth.js";
import { drawForceCurve, drawEnergy, drawHood } from "./plots.js";

const stage = document.getElementById("stage");
const ctx = stage.getContext("2d");
const err = document.getElementById("err");

let net, cfg;
try {
  const w = await loadWeights("weights");
  net = new CentralNet(w);
  cfg = w.config;
} catch (e) {
  err.textContent = "Failed to load weights: " + e.message;
  err.hidden = false;
  throw e;
}

const state = {
  n: 60, pos: null, vel: null,
  tPos: null, tVel: null, tAcc: null, // ground-truth ghost
  seed: 4738, tick: 0, running: true, showTruth: true, showEdges: true,
  ticksPerSec: 20, acc: 0, lastT: performance.now(),
  modelE: [], truthE: [], histLen: 240,
  sel: 0, trace: null, cam: null, model: false,
};
let mv = null;
const trueLaw = (d) => d / Math.pow(d * d + truth.EPS2, 1.5);

function setTrace(tr) {
  state.trace = tr;
  if (mv) mv.set(tr);
}

// trace the selected body's next step without advancing the sim
function refreshTrace() {
  const c = 2 * state.n, p = state.pos.slice(0, c), v = state.vel.slice(0, c);
  setTrace(net.step(state.pos, state.vel, state.n, state.sel));
  state.pos.set(p); state.vel.set(v);
}

function spawn(n) {
  const rng = mulberry32(state.seed + state.tick);
  const { pos, vel } = initBodies(n, rng);
  state.n = n; state.pos = pos; state.vel = vel;
  state.tPos = Float64Array.from(pos); state.tVel = Float64Array.from(vel);
  state.tAcc = new Float64Array(2 * n);
  truth.accel(state.tPos, n, state.tAcc);
  state.tick = 0; state.modelE = []; state.truthE = [];
  state.sel = Math.min(state.sel, n - 1);
  refreshTrace();
}
spawn(state.n);

function stepOnce() {
  setTrace(net.step(state.pos, state.vel, state.n, state.sel));
  if (state.showTruth) truth.step(state.tPos, state.tVel, state.n, cfg.dt, state.tAcc);
  state.tick++;
  state.modelE.push(0); // placeholder overwritten below (model has no closed-form PE; report KE only)
  let ke = 0;
  for (let i = 0; i < 2 * state.n; i++) ke += state.vel[i] * state.vel[i];
  state.modelE[state.modelE.length - 1] = 0.5 * ke;
  state.truthE.push(state.showTruth ? truth.energy(state.tPos, state.tVel, state.n) : NaN);
  if (state.modelE.length > state.histLen) { state.modelE.shift(); state.truthE.shift(); }
}

// camera: fit the bounding box of all bodies (model + ghost) with margin
function bounds() {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const scan = (p, n) => {
    for (let i = 0; i < n; i++) {
      const x = p[2 * i], y = p[2 * i + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  };
  scan(state.pos, state.n);
  if (state.showTruth) scan(state.tPos, state.n);
  if (!Number.isFinite(minX)) { minX = minY = -10; maxX = maxY = 10; }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const half = Math.max(10, (maxX - minX) / 2, (maxY - minY) / 2) * 1.25;
  return { cx, cy, half };
}

function render() {
  const w = stage.clientWidth, h = stage.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (stage.width !== w * dpr || stage.height !== h * dpr) { stage.width = w * dpr; stage.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#06080c"; ctx.fillRect(0, 0, w, h);

  const { cx, cy, half } = bounds();
  const scale = Math.min(w, h) / (2 * half);
  const sx = (x) => w / 2 + (x - cx) * scale;
  const sy = (y) => h / 2 + (y - cy) * scale;
  state.cam = { sx, sy };

  if (state.showEdges) {
    ctx.strokeStyle = "rgba(111,195,255,0.18)"; ctx.lineWidth = 1;
    const edges = net.edgeList(state.pos, state.n, cfg.neighbor_radius);
    ctx.beginPath();
    for (let e = 0; e < edges.length; e += 2) {
      const i = edges[e], j = edges[e + 1];
      ctx.moveTo(sx(state.pos[2 * i]), sy(state.pos[2 * i + 1]));
      ctx.lineTo(sx(state.pos[2 * j]), sy(state.pos[2 * j + 1]));
    }
    ctx.stroke();
  }

  if (state.showTruth) {
    ctx.fillStyle = "#ff9a3c";
    for (let i = 0; i < state.n; i++) {
      const x = sx(state.tPos[2 * i]), y = sy(state.tPos[2 * i + 1]);
      ctx.globalAlpha = 0.45;
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  ctx.fillStyle = "#6fc3ff";
  for (let i = 0; i < state.n; i++) {
    const x = sx(state.pos[2 * i]), y = sy(state.pos[2 * i + 1]);
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
  }

  const tr = state.trace;
  if (tr) {
    const px = sx(state.pos[2 * state.sel]), py = sy(state.pos[2 * state.sel + 1]);
    ctx.strokeStyle = "rgba(228,234,245,0.5)"; ctx.lineWidth = 1;
    ctx.beginPath();
    for (const e of tr.s1.edges) { ctx.moveTo(px, py); ctx.lineTo(sx(state.pos[2 * e.src]), sy(state.pos[2 * e.src + 1])); }
    ctx.stroke();
    ctx.strokeStyle = "#e4eaf5"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.stroke();
  }
}

let fpsT = performance.now(), fpsN = 0, fps = 0;
function frame(t) {
  requestAnimationFrame(frame);
  const dtReal = Math.min((t - state.lastT) / 1000, 0.25);
  state.lastT = t;
  if (state.running) {
    state.acc += dtReal * state.ticksPerSec;
    let steps = 0;
    while (state.acc >= 1 && steps < 8) { stepOnce(); state.acc -= 1; steps++; }
  }
  if (state.model && mv) mv.frame(t); else render();
  fpsN++;
  if (t - fpsT > 500) { fps = (fpsN * 1000) / (t - fpsT); fpsN = 0; fpsT = t; }

  document.getElementById("s-tick").textContent = state.tick;
  document.getElementById("s-bodies").textContent = state.n;
  document.getElementById("s-fps").textContent = fps.toFixed(0);
  const ke = state.modelE[state.modelE.length - 1] ?? 0;
  document.getElementById("s-ke").textContent = ke.toFixed(1);

  const tr = state.trace;
  const v2 = (a) => `(${fmt(a[0])}, ${fmt(a[1])})`;
  document.getElementById("sel-id").textContent = "#" + state.sel;
  document.getElementById("sel").innerHTML = tr
    ? `pos <b>${v2(tr.pos)}</b> vel <b>${v2(tr.vel)}</b><br>acc <b>${v2(tr.a0)}</b> from <b>${tr.s1.n}</b> neighbours`
    : "";
  drawHood(document.getElementById("c-hood"), tr);
  drawForceCurve(document.getElementById("c-force"), net.forceMlp, 0.05, cfg.neighbor_radius,
    tr ? tr.s1.edges.map((e) => e.dist) : [], trueLaw);
  drawEnergy(document.getElementById("c-energy"), state.modelE, state.truthE);
}
requestAnimationFrame(frame);

document.getElementById("b-pause").addEventListener("click", (e) => {
  state.running = !state.running;
  e.target.setAttribute("aria-pressed", String(!state.running));
  e.target.textContent = state.running ? "Pause" : "Resume";
});
document.getElementById("b-step").addEventListener("click", () => { state.running = false; stepOnce(); });
document.getElementById("b-reset").addEventListener("click", () => spawn(state.n));
document.getElementById("b-truth").addEventListener("click", (e) => {
  state.showTruth = !state.showTruth;
  e.target.setAttribute("aria-pressed", String(state.showTruth));
});
document.getElementById("b-edges").addEventListener("click", (e) => {
  state.showEdges = !state.showEdges;
  e.target.setAttribute("aria-pressed", String(state.showEdges));
});
const rBodies = document.getElementById("r-bodies"), oBodies = document.getElementById("o-bodies");
rBodies.addEventListener("input", () => { oBodies.textContent = rBodies.value; });
rBodies.addEventListener("change", () => spawn(parseInt(rBodies.value, 10)));
const rSpeed = document.getElementById("r-speed"), oSpeed = document.getElementById("o-speed");
rSpeed.addEventListener("input", () => {
  state.ticksPerSec = parseInt(rSpeed.value, 10);
  oSpeed.textContent = state.ticksPerSec;
});

const fmt = (x) => {
  const a = Math.abs(x);
  return a !== 0 && (a >= 1e4 || a < 1e-3) ? x.toExponential(2) : x.toFixed(3);
};

function select(i) {
  state.sel = i;
  refreshTrace();
  if (mv) mv.restartSweep();
}

stage.addEventListener("click", (e) => {
  if (!state.cam) return;
  const r = stage.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  let best = -1, bd = 18;
  for (let i = 0; i < state.n; i++) {
    const d = Math.hypot(state.cam.sx(state.pos[2 * i]) - x, state.cam.sy(state.pos[2 * i + 1]) - y);
    if (d < bd) { bd = d; best = i; }
  }
  if (best >= 0) select(best);
});
document.getElementById("b-pick").addEventListener("click", () => select(Math.floor(Math.random() * state.n)));

const archEl = document.getElementById("arch");
function insets() {
  const small = innerWidth <= 760, side = document.querySelector("aside").getBoundingClientRect();
  const top = small ? document.querySelector("header").getBoundingClientRect().bottom + 4 : 16;
  const bottom = innerHeight - document.querySelector("nav").getBoundingClientRect().top + 8;
  const left = small ? 0 : document.querySelector("header").getBoundingClientRect().right + 8;
  mv.resize(innerWidth, innerHeight, [left, top, small ? 0 : side.width + 24, bottom]);
  mv.fit();
}
async function setModel(on) {
  const btn = document.getElementById("b-model");
  if (on && !mv) {
    try {
      const { ModelView } = await import("./view.js");
      mv = new ModelView(archEl, document.getElementById("labels"), document.getElementById("tip"), select);
      mv.set(state.trace);
    } catch (e) {
      err.textContent = "3D view needs three.js from cdn.jsdelivr.net: " + e.message;
      err.hidden = false;
      setTimeout(() => { err.hidden = true; }, 4000);
      return;
    }
  }
  state.model = on;
  archEl.hidden = !on;
  document.body.classList.toggle("model", on);
  btn.setAttribute("aria-pressed", String(on));
  if (on) { insets(); mv.camera.position.copy(mv.goal.pos).multiplyScalar(1.4); mv.restartSweep(); }
  else document.getElementById("tip").classList.remove("on");
}
document.getElementById("b-model").addEventListener("click", () => setModel(!state.model));
addEventListener("resize", () => { if (state.model) insets(); });

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "m") document.getElementById("b-model").click();
  else if (e.key === "p") document.getElementById("b-pick").click();
  if (e.key === " ") { e.preventDefault(); document.getElementById("b-pause").click(); }
  else if (e.key === ".") document.getElementById("b-step").click();
  else if (e.key === "r") document.getElementById("b-reset").click();
  else if (e.key === "g") document.getElementById("b-truth").click();
  else if (e.key === "e") document.getElementById("b-edges").click();
});
