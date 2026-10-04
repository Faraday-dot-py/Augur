import * as THREE from "three";
import { Stage } from "../../scatter-walkthrough/js/stage.js";
import { buildLayout, buildCharts, paintTrace, paintKernel, paintWeights, hoverMarks, tipText } from "../../scatter-walkthrough/js/layout.js";
import { css } from "../../scatter-walkthrough/js/colors.js";
import { loadWeights } from "./scatter_model.js";

const FOV = 40;
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
  $("mcap").textContent = `${total.toLocaleString()} parameters. Input stack: 6 channels on a ${c.grid}x${c.grid} grid (${c.extent / c.grid} sim units per cell). UNet: ${ch} channels, ${c.grid}x${c.grid} down to ${c.grid >> c.levels}x${c.grid >> c.levels} and back with skip connections. Slabs are colored relative to their own range; hover for values (stored 8-bit for UNet activations, 16-bit for fields, so tooltips show dequantised numbers), and over a UNet slab to see its conv weights. Channel "all" overlays every channel: opacity follows |value|, hue follows sign.`;
}

export async function createView({ canvas, labelsEl, tip }) {
  const stage = new Stage(canvas, labelsEl);
  const layout = buildLayout(stage);
  const weights = await loadWeights(new URL("../weights", import.meta.url).href);
  describe(weights);

  stage.narrowShift = 0;
  stage.setWall(true);
  stage.resize();

  // the layout is built on the floor and stood up by the stage, so floor z maps to wall -y
  function bounds() {
    let x0 = -46, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const b of stage.blocks.values()) {
      const g = b.group.position;
      x0 = Math.min(x0, g.x - b.size / 2); x1 = Math.max(x1, g.x + b.size / 2);
      y0 = Math.min(y0, -g.z - b.sizeY / 2); y1 = Math.max(y1, -g.z + b.sizeY / 2);
    }
    return { x0, x1, y0, y1 };
  }

  function fitPose() {
    const { x0, x1, y0, y1 } = bounds(), wide = innerWidth > 760, portrait = innerWidth < innerHeight;
    const top = wide ? $("mview").getBoundingClientRect().bottom + 10 : 0, bot = wide ? 80 : 0, hpx = innerHeight - top - bot, t = Math.tan(FOV / 2 * Math.PI / 180);
    const w = x1 - x0 + 6, h = y1 - y0 + 8, aspect = innerWidth / innerHeight;
    const d = Math.max(h / 2 / t * innerHeight / hpx, w / 2 / (t * aspect)) * 1.04;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2 + (top - bot) / 2 * 2 * d * t / innerHeight;
    stage.setPose([cx, cy, 0], [cx + (portrait ? 0 : 0.1 * d), cy + 0.05 * d, d]);
  }
  fitPose();
  $("mview").addEventListener("toggle", () => { if (shown) fitPose(); });

  stage.ghost.material.color.set(0xff9a3c);
  stage.ghost.material.opacity = 0.6;
  let tr = null, charts = null, n = 0, hover = null, wname = "net.enc.0.0", dirty = true, active = false, shown = false, raf = 0, pending = null;
  let truthPos = null, wblock = null, wkey = "";
  const seen = { bodies: -1, slabs: -1 };

  const flat = (p, dx, dz) => new THREE.Vector3(p.x + dx, p.y, p.z + dz);

  function compose() {
    stage.beginFrame();
    for (const c of stage.curves.values()) { c.frac = 1; c.alpha = 0.8; }
    const narrow = innerWidth <= 760;
    if (tr) stage.setBodies(tr.pos, tr.n);
    if (tr && truthPos && truthPos.length === 2 * tr.n) stage.ghostList = Array.from({ length: tr.n }, (_, i) => stage.world("input", truthPos[2 * i], truthPos[2 * i + 1], 0.45));
    for (const b of stage.blocks.values()) {
      if (b.id === "wconv" || narrow) continue;
      const g = b.group.position;
      stage.note(b.id, labelOf(b), new THREE.Vector3(g.x, g.y + b.thick / 2 + 0.3, g.z - b.sizeY / 2), 0.95);
    }
    if (charts && !narrow) {
      const k = charts.kbase, p = charts.pbase;
      stage.note("kcap", "K(r), signed-log scale", flat(k, 7, -charts.kH - 1), 1);
      stage.note("kraw", "raw MLP", flat(k, 12, charts.kH), 0.8, "k1");
      stage.note("ktap", "tapered (used)", flat(k, 12, charts.kH + 1.2), 0.8, "k2");
      stage.note("pcap", "pair force vs r", flat(p, 5, -charts.ph - 1), 1);
    }
    if (tr) {
      if (wkey !== wname) { wblock = paintWeights(stage, weights.w, wname); wkey = wname; }
      const b = wblock, { cin, cout } = b.meta;
      b.group.position.set(WEIGHTS_AT.x, WEIGHTS_AT.y, WEIGHTS_AT.z);
      b.op = 1; b.hl = 0.4;
      if (!narrow) stage.note("wlab", "weights " + wname.replace("net.", "") + ": " + cout + "x" + cin + " 3x3", WEIGHTS_AT.clone().add(new THREE.Vector3(0, 0.1, -b.sizeY / 2 - 0.4)), 1, "w");
    }
    if (hover && hover.type === "cell") hover.block.hl = Math.max(hover.block.hl, 0.5);
    hoverMarks(stage, hover, tr, n);
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
      if (p) { tip.hidden = false; tip.innerHTML = tipText(p, tr, tr.vel); tip.style.left = Math.min(innerWidth - 300, x + 14) + "px"; tip.style.top = Math.min(innerHeight - 80, y + 14) + "px"; } else tip.hidden = true;
    });
    pending = [e.clientX, e.clientY];
  });
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener("pointerup", (e) => {
    if (e.pointerType === "mouse" || !tr || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const p = stage.pick(e.clientX, e.clientY);
    setHover(p);
    if (p) { tip.hidden = false; tip.innerHTML = tipText(p, tr, tr.vel); tip.style.left = Math.min(innerWidth - 300, e.clientX + 14) + "px"; tip.style.top = Math.min(innerHeight - 80, e.clientY + 14) + "px"; } else tip.hidden = true;
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
    setCurves(curve, pair) {
      if (charts) return;
      charts = buildCharts(stage, curve, pair, true);
      paintKernel(stage, curve, weights.w);
      dirty = true;
    },
    showFrame(f) {
      tr = f; n = f.n;
      stage.setBodies(f.pos, n);
      seen.bodies = f.tick;
      paintTrace(stage, f);
      seen.slabs = f.tick;
      dirty = true;
    },
    setTruth(pos) {
      truthPos = pos;
      dirty = true;
    },
    setChannel(c) {
      for (const b of stage.blocks.values()) if (b.thick || b.n > 1) b.setChannel(c);
      dirty = true;
    },
    shown: seen,
    fit: fitPose,
    stage, layout,
  };
}
