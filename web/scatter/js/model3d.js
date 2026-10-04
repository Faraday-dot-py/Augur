import * as THREE from "three";
import { Stage } from "../../scatter-walkthrough/js/stage.js";
import { buildLayout, buildCharts, paintTrace, paintKernel, paintWeights, hoverMarks, tipText } from "../../scatter-walkthrough/js/layout.js";
import { css } from "../../scatter-walkthrough/js/colors.js";
import { loadWeights } from "./scatter_model.js";

const POSE = { target: [30, -3, 4], pos: [30, 100, 125] };
const KINDS = [["dens", "density"], ["mom", "momentum density"], ["far", "far-field force"], ["act", "UNet activations"], ["pot", "potential"], ["wgt", "learned weights"]];
const ARCS = [["#c79bff", "UNet flow, skips"], ["#e8c85a", "potential, recurrence"], ["#ff9a3c", "acceleration"], ["#62d6a4", "density, gather"]];
const WEIGHTS_AT = new THREE.Vector3(-28, 1, 0);
const $ = (id) => document.getElementById(id);

function weightsOf(id) {
  const m = /^(enc|dec)(\d)$/.exec(id);
  if (m) return `net.${m[1]}.${m[2]}.0`;
  return id === "h0" ? "net.inp" : id === "phi" ? "net.out" : null;
}

const NAMES = { input: "input stack, 6ch", kw: "kmlp W", kring: "K(r) grid", phik: "far potential", ak: "far accel", h0: "input conv", phi: "potential", grad: "-grad(phi)", agrid: "grid accel" };

function labelOf(b) {
  const m = /^(enc|dec)(\d)$/.exec(b.id);
  return m ? m[1] + " " + m[2] + " · " + b.H + "²" : NAMES[b.id] || b.title;
}

function describe(w) {
  let total = 0;
  for (const a of Object.values(w.w)) total += a.length;
  const c = w.config, ch = w.w["net.inp.weight"].length / (6 * 9);
  $("mleg").innerHTML = KINDS.map(([k, t]) => `<li><i style="background:linear-gradient(90deg,${css(k, -1)},${css(k, 1)})"></i>${t}</li>`).join("")
    + ARCS.map(([c2, t]) => `<li><i class="ln" style="background:${c2}"></i>${t}</li>`).join("");
  $("mcap").textContent = `${total.toLocaleString()} parameters. Input stack: 6 channels on a ${c.grid}x${c.grid} grid (${c.extent / c.grid} sim units per cell). UNet: ${ch} channels, ${c.grid}x${c.grid} down to ${c.grid >> c.levels}x${c.grid >> c.levels} and back with skip connections. Slabs are colored relative to their own range; hover for raw values, and over a UNet slab to see its conv weights.`;
}

export async function createView({ canvas, labelsEl, tip }) {
  const stage = new Stage(canvas, labelsEl);
  const layout = buildLayout(stage);
  const weights = await loadWeights(new URL("../weights", import.meta.url).href);
  describe(weights);

  function fitPose() {
    const t = POSE.target, portrait = innerWidth < innerHeight;
    if (portrait) stage.setPose([t[0] + 14, t[1], t[2]], [t[0] - 121, t[1] + 235, t[2] + 8]);
    else stage.setPose(t, POSE.pos);
  }
  fitPose();

  stage.ghost.material.color.set(0xff9a3c);
  stage.ghost.material.opacity = 0.6;
  let tr = null, charts = null, n = 0, hover = null, wname = "net.enc.0.0", ticked = -1, dirty = true, active = false, shown = false, raf = 0, pending = null;
  let cur = null, truthPos = null, wblock = null, wkey = "";

  function compose() {
    stage.beginFrame();
    for (const c of stage.curves.values()) { c.frac = 1; c.alpha = 0.8; }
    const narrow = innerWidth <= 760;
    if (cur) stage.setBodies(cur.pos, cur.n);
    if (cur && truthPos && truthPos.length === 2 * cur.n) stage.ghostList = Array.from({ length: cur.n }, (_, i) => stage.world("input", truthPos[2 * i], truthPos[2 * i + 1], 0.45));
    for (const b of stage.blocks.values()) {
      if (b.id === "wconv" || narrow) continue;
      const g = b.group.position;
      stage.note(b.id, labelOf(b), new THREE.Vector3(g.x, g.y + b.thick / 2 + 0.3, g.z + (b.id === "input" ? -1 : 1) * b.sizeY / 2), 0.95);
    }
    if (charts && !narrow) {
      const k = charts.kbase, p = charts.pbase;
      stage.note("kcap", "K(r), signed-log scale", new THREE.Vector3(k.x + 7, k.y + charts.kH + 1, k.z), 1);
      stage.note("kraw", "raw MLP", new THREE.Vector3(k.x + 12, k.y - charts.kH, k.z), 0.8, "k1");
      stage.note("ktap", "tapered (used)", new THREE.Vector3(k.x + 12, k.y - charts.kH - 1.2, k.z), 0.8, "k2");
      stage.note("pcap", "pair force vs r", new THREE.Vector3(p.x + 5, p.y + charts.ph + 1, p.z), 1);
    }
    if (tr) {
      if (wkey !== wname) { wblock = paintWeights(stage, weights.w, wname); wkey = wname; }
      const b = wblock, { cin, cout } = b.meta;
      b.group.position.set(WEIGHTS_AT.x, WEIGHTS_AT.y, WEIGHTS_AT.z);
      b.op = 1; b.hl = 0.4;
      if (!narrow) stage.note("wlab", "weights " + wname.replace("net.", "") + ": " + cout + "x" + cin + " 3x3", WEIGHTS_AT.clone().add(new THREE.Vector3(0, 0.1, 5)), 1, "w");
    }
    if (hover && hover.type === "cell") hover.block.hl = Math.max(hover.block.hl, 0.5);
    if (!cur || cur.n === n) hoverMarks(stage, hover, tr, n);
    stage.endFrame();
    stage.dirty = true;
  }

  function loop() {
    if (!active) return;
    if (dirty) { dirty = false; compose(); }
    stage.render();
    raf = requestAnimationFrame(loop);
  }

  function setHover(p) {
    const key = (q) => (q ? (q.type === "body" ? "b" + q.i : q.block.id + q.k + ":" + q.ix + "," + q.iy) : "");
    const changed = key(p) !== key(hover);
    hover = p;
    if (p && p.type === "cell") { const w = weightsOf(p.block.id); if (w && w !== wname) { wname = w; dirty = true; } }
    if (changed) dirty = true;
  }

  canvas.addEventListener("pointermove", (e) => {
    if (e.buttons || !tr) return;
    if (!pending) requestAnimationFrame(() => {
      const [x, y] = pending; pending = null;
      const p = stage.pick(x, y);
      setHover(p);
      if (p) { tip.hidden = false; tip.innerHTML = tipText(p, cur || tr, (cur || tr).vel); tip.style.left = Math.min(innerWidth - 300, x + 14) + "px"; tip.style.top = Math.min(innerHeight - 80, y + 14) + "px"; } else tip.hidden = true;
    });
    pending = [e.clientX, e.clientY];
  });
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener("pointerup", (e) => {
    if (e.pointerType === "mouse" || !tr || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const p = stage.pick(e.clientX, e.clientY);
    setHover(p);
    if (p) { tip.hidden = false; tip.innerHTML = tipText(p, cur || tr, (cur || tr).vel); tip.style.left = Math.min(innerWidth - 300, e.clientX + 14) + "px"; tip.style.top = Math.min(innerHeight - 80, e.clientY + 14) + "px"; } else tip.hidden = true;
  });
  canvas.addEventListener("pointerleave", () => { setHover(null); tip.hidden = true; });

  return {
    show(on) {
      active = on;
      canvas.hidden = labelsEl.hidden = !on;
      if (!on) { cancelAnimationFrame(raf); tip.hidden = true; return; }
      stage.resize();
      if (!shown) { fitPose(); shown = true; }
      dirty = true;
      raf = requestAnimationFrame(loop);
    },
    onTrace(m) {
      if (m.curve && !charts) {
        charts = buildCharts(stage, m.curve, m.pair);
        paintKernel(stage, m.curve, weights.w);
      }
      tr = m.trace; n = m.n; ticked = m.tick;
      if (!cur) stage.setBodies(tr.pos, n);
      paintTrace(stage, tr);
      dirty = true;
    },
    setBodies(pos, vel, count) {
      cur = { pos, vel, n: count };
      dirty = true;
    },
    setTruth(pos) {
      truthPos = pos;
      dirty = true;
    },
    setChannel(c) {
      for (const b of stage.blocks.values()) if (b.thick) b.setChannel(c);
      dirty = true;
    },
    tick: () => ticked,
    stage, layout,
  };
}
