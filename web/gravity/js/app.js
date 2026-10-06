import { loadWeights, CentralNet, initBodies, mulberry32 } from "./central_model.js";
import * as truth from "./truth.js";
import { drawForceCurve, drawEnergy } from "./plots.js";

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
};

function spawn(n) {
  const rng = mulberry32(state.seed + state.tick);
  const { pos, vel } = initBodies(n, rng);
  state.n = n; state.pos = pos; state.vel = vel;
  state.tPos = Float64Array.from(pos); state.tVel = Float64Array.from(vel);
  state.tAcc = new Float64Array(2 * n);
  truth.accel(state.tPos, n, state.tAcc);
  state.tick = 0; state.modelE = []; state.truthE = [];
}
spawn(state.n);

function stepOnce() {
  net.step(state.pos, state.vel, state.n);
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
  render();
  fpsN++;
  if (t - fpsT > 500) { fps = (fpsN * 1000) / (t - fpsT); fpsN = 0; fpsT = t; }

  document.getElementById("s-tick").textContent = state.tick;
  document.getElementById("s-bodies").textContent = state.n;
  document.getElementById("s-fps").textContent = fps.toFixed(0);
  const ke = state.modelE[state.modelE.length - 1] ?? 0;
  document.getElementById("s-ke").textContent = ke.toFixed(1);

  drawForceCurve(document.getElementById("c-force"), net.forceMlp, 0.05, cfg.neighbor_radius,
    Array.from({ length: 0 }));
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

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === " ") { e.preventDefault(); document.getElementById("b-pause").click(); }
  else if (e.key === ".") document.getElementById("b-step").click();
  else if (e.key === "r") document.getElementById("b-reset").click();
  else if (e.key === "g") document.getElementById("b-truth").click();
  else if (e.key === "e") document.getElementById("b-edges").click();
});
