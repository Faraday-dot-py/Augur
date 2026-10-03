import { Stage } from "./stage.js";
import { buildLayout, buildCharts, paintTrace, paintKernel, paintWeights, hoverMarks, tipText } from "./layout.js";
import { Walk, PHASE_TITLES, PHASE_COUNT } from "./walk.js";
import { loadWeights, mulberry32, initBodies } from "../../scatter/js/scatter_model.js";

const $ = (id) => document.getElementById(id);
const err = (m) => { $("err").hidden = false; $("err").textContent = m; };
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const KEY = "scatterwalk.phase";

let stage, layout, charts, walk, worker, weights;
let S = { n: 0, trace: null, sel: 0, picked: false, tick: 0 };
let hoverRef = null, hoverPick = null, busy = true, lastKey = "", lastSays = "", seedN = 16;

const load = () => { try { return Math.min(PHASE_COUNT - 1, Math.max(0, parseInt(localStorage.getItem(KEY), 10) || 0)); } catch (e) { return 0; } };
const save = (i) => { try { localStorage.setItem(KEY, String(i)); } catch (e) { /* storage unavailable */ } };

function neighbours(tr, n, pp, knn) {
  const pairs = new Map();
  for (let i = 0; i < n; i++) {
    const c = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const r = Math.hypot(tr.pos[2 * j] - tr.pos[2 * i], tr.pos[2 * j + 1] - tr.pos[2 * i + 1]);
      if (r < pp) c.push([r, j]);
    }
    c.sort((a, b) => a[0] - b[0]);
    for (const [, j] of c.slice(0, knn)) pairs.set(Math.min(i, j) * 1000 + Math.max(i, j), [Math.min(i, j), Math.max(i, j)]);
  }
  return [...pairs.values()];
}

function onTrace(m) {
  const tr = m.trace;
  S = { ...S, n: m.n, trace: tr, pos: m.pos, vel: m.vel, tick: m.tick, ms: m.ms };
  let ma = 1e-12, mv = 1e-12;
  for (let i = 0; i < 2 * m.n; i++) { ma = Math.max(ma, Math.abs(tr.accel[i]), Math.abs(tr.gather[i]), Math.abs(tr.pairAcc[i])); mv = Math.max(mv, Math.abs(m.vel[i])); }
  S.accScale = 1.2 / ma;
  S.pairScale = 1.2 / Math.max(1e-12, ...tr.pairAcc.map(Math.abs));
  S.velScale = 1.2 / mv;
  S.nbrs = neighbours(tr, m.n, 2, 16);
  if (!S.picked || S.sel >= m.n) {
    const cnt = new Array(m.n).fill(0);
    for (const [i, j] of S.nbrs) { cnt[i]++; cnt[j]++; }
    S.sel = cnt.indexOf(Math.max(...cnt));
    S.picked = false;
  }
  stage.setBodies(tr.pos, m.n);
  paintTrace(stage, tr);
  walk.invalidate();
  busy = false;
  $("b-tick").disabled = false;
  $("status").textContent = "tick " + m.tick + " · " + m.ms.toFixed(0) + " ms";
  lastKey = "";
}

function onReady(m) {
  charts = buildCharts(stage, m.curve, m.pair);
  charts.maxK = paintKernel(stage, m.curve, weights.w);
  walk.env.charts = charts;
  reset(true);
}

function showWeights(name, anchor, alpha) {
  const b = paintWeights(stage, weights.w, name), { cin, cout } = b.meta;
  b.group.position.set(anchor.x, anchor.y + 2.4 + cout * 3 / 96 * 2.4, anchor.z);
  b.op = alpha;
  stage.note("wlab", name.replace("net.", "") + " weights: " + cout + " x " + cin + " kernels of 3 x 3", b.group.position.clone().add({ x: 0, y: 0.1 + 0, z: 0 }), alpha, "w");
  b.hl = 0.4;
}

function reset(first) {
  busy = true;
  $("b-tick").disabled = true;
  $("status").textContent = "computing";
  const { pos, vel } = initBodies(seedN, mulberry32(4738));
  S.picked = false;
  worker.postMessage({ cmd: "reset", n: seedN, pos, vel });
}

function tick() {
  if (busy) return;
  busy = true;
  $("b-tick").disabled = true;
  $("status").textContent = "computing";
  worker.postMessage({ cmd: "trace" });
}

function commentary(c) {
  const shown = c.says.filter((s) => s.start <= walk.time + 1e-6);
  const key = walk.phase + "|" + shown.length + "|" + S.sel + "|" + S.tick + "|" + (S.trace ? 1 : 0);
  if (key === lastSays) return;
  lastSays = key;
  $("text").innerHTML = shown.map((s) => "<p>" + s.html + "</p>").join("");
  const t = $("text");
  t.scrollTop = t.scrollHeight;
}

function extra() {
  hoverMarks(stage, hoverPick, S.trace, S.n);
}

function buildUi() {
  const ph = $("phases");
  PHASE_TITLES.forEach((t, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.setAttribute("role", "tab");
    b.innerHTML = `<i>${i + 1}</i><span>${t}</span>`;
    b.addEventListener("click", () => { walk.go(i, 0); walk.playing = !reduce; });
    ph.appendChild(b);
  });
  $("b-play").addEventListener("click", togglePlay);
  $("b-next").addEventListener("click", () => walk.next());
  $("b-prev").addEventListener("click", () => walk.prev());
  $("b-tick").addEventListener("click", () => { tick(); });
  $("b-reset").addEventListener("click", () => reset());
  $("r-time").addEventListener("input", (e) => { walk.playing = false; walk.time = e.target.value / 1000 * walk.length; });
  $("r-ch").addEventListener("input", (e) => {
    const c = +e.target.value;
    $("o-ch").textContent = c;
    for (const b of stage.blocks.values()) if (b.thick) b.setChannel(c);
    lastKey = "";
  });
  $("r-n").addEventListener("input", (e) => { $("o-n").textContent = e.target.value; });
  $("r-n").addEventListener("change", (e) => { seedN = +e.target.value; reset(); });
  $("text").addEventListener("pointerover", (e) => { const a = e.target.closest("a.ref"); hoverRef = a ? a.dataset.ref : null; lastKey = ""; });
  $("text").addEventListener("pointerleave", () => { hoverRef = null; lastKey = ""; });
  addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "range" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " " || e.key === "Spacebar") { if (e.target.tagName === "BUTTON") return; e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowRight") walk.next();
    else if (e.key === "ArrowLeft") walk.prev();
    else if (e.key === ".") tick();
    else if (e.key === "r") reset();
    else if (e.key === "0") lastKey = "", poseNow = true;
    else if (e.key === "[" || e.key === "]") {
      const r = $("r-ch");
      r.value = Math.min(31, Math.max(0, +r.value + (e.key === "]" ? 1 : -1)));
      r.dispatchEvent(new Event("input"));
    }
  });
}

let poseNow = false, lastPoseKey = "";
function togglePlay() {
  if (walk.playing) walk.playing = false;
  else if (walk.time >= walk.length - 1e-6) walk.next();
  else walk.playing = true;
}

function pointer() {
  const el = $("stage"), tip = $("tip");
  let down = null;
  el.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  el.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const p = S.trace && stage.pick(e.clientX, e.clientY);
    if (p && p.type === "body") { S.sel = p.i; S.picked = true; walk.invalidate(); poseNow = true; lastKey = ""; }
  });
  let pending = null;
  el.addEventListener("pointermove", (e) => {
    if (e.buttons || !S.trace) return;
    if (!pending) requestAnimationFrame(() => {
      const [x, y] = pending; pending = null;
      const p = stage.pick(x, y);
      const key = p ? (p.type === "body" ? "b" + p.i : p.block.id + p.k + ":" + p.ix + "," + p.iy) : "";
      const prev = hoverPick ? (hoverPick.type === "body" ? "b" + hoverPick.i : hoverPick.block.id + hoverPick.k + ":" + hoverPick.ix + "," + hoverPick.iy) : "";
      hoverPick = p;
      if (p) { tip.hidden = false; tip.innerHTML = tipText(p, S.trace, S.vel); tip.style.left = Math.min(innerWidth - 300, x + 14) + "px"; tip.style.top = Math.min(innerHeight - 80, y + 14) + "px"; } else tip.hidden = true;
      if (key !== prev) lastKey = "";
    });
    pending = [e.clientX, e.clientY];
  });
  el.addEventListener("pointerleave", () => { hoverPick = null; tip.hidden = true; lastKey = ""; });
}

function loop(t0) {
  let last = t0;
  const step = (now) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (walk && S.trace) {
      const moved = walk.update(dt);
      const key = walk.phase + "|" + walk.time.toFixed(3) + "|" + S.sel + "|" + S.tick + "|" + hoverRef + "|" + (hoverPick ? (hoverPick.type === "body" ? hoverPick.i : hoverPick.block.id + hoverPick.k + hoverPick.ix + "," + hoverPick.iy) : "");
      if (key !== lastKey || moved) {
        const poseKey = key.split("|").slice(0, 4).join("|");
        const timeChanged = poseKey !== lastPoseKey;
        lastPoseKey = poseKey;
        lastKey = key;
        const c = walk.frame(hoverRef, extra);
        if (timeChanged || poseNow) { const p = c.pose(); stage.setPose(p.target, p.pos); poseNow = false; }
        commentary(c);
        stage.dirty = true;
        const len = walk.length;
        $("r-time").value = Math.round(walk.time / len * 1000);
        $("b-play").textContent = walk.playing ? "Pause" : "Play";
        const k = document.createElement("kbd"); k.textContent = "space";
        $("b-play").appendChild(k);
        $("b-play").setAttribute("aria-pressed", walk.playing);
        $("ph-n").textContent = walk.phase + 1 + " / " + PHASE_COUNT;
        $("ph-t").textContent = PHASE_TITLES[walk.phase];
        [...$("phases").children].forEach((b, i) => b.setAttribute("aria-current", i === walk.phase));
        save(walk.phase);
      }
    }
    stage.render();
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

async function main() {
  try {
    stage = new Stage($("stage"), $("labels"));
    layout = buildLayout(stage);
    weights = await loadWeights(new URL("../../scatter/weights", import.meta.url).href);
    const env = { stage, layout, reduce, get S() { return S; }, charts: null, showWeights };
    walk = new Walk(env);
    buildUi();
    pointer();
    walk.go(load(), 0);
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") onReady(m);
      else if (m.type === "trace") { onTrace(m); if (m.tick === 0 && !reduce) walk.playing = true; }
      else if (m.type === "error") err("Worker error: " + m.message);
    };
    worker.onerror = (e) => err("Worker failed: " + e.message);
    window.__walk = { walk, stage, get S() { return S; } };
    loop(performance.now());
  } catch (e) {
    err("This page needs WebGL2 and module workers. " + e.message);
  }
}

main();
