import * as THREE from "three";
import { LEVELS } from "./layout.js";
import { bodyCorners } from "./cic.js";
import { fmt } from "./colors.js";

const ease = (t) => t * t * (3 - 2 * t);
const clamp = (t) => Math.min(1, Math.max(0, t));
const lerp3 = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const DEFAULT = { target: [30, -3, 4], pos: [30, 100, 125] };

const dim = (k, t) => `<span class="dim d-${k}">${t}</span>`;
const ref = (id, t) => `<a class="ref" data-ref="${id}">${t}</a>`;
const N_ = (n) => dim("N", `N = ${n}`), G_ = dim("G", "G = 128"), C_ = dim("C", "C = 32"), H_ = dim("h", "h = 0.5"), L_ = dim("L", "5 levels");

const GROUPS = (lay) => ({
  enc: lay.enc, dec: lay.dec, far: ["kring", "phik", "ak", "kw"], skips: [],
  stack: ["input"], pot: ["phi", "grad", "agrid"],
});

class Ctx {
  constructor(walk, i, time, dry) {
    Object.assign(this, { walk, env: walk.env, i, time, dry, maxEnd: 0, breaks: [], says: [], kfs: [], groups: GROUPS(walk.env.layout) });
    this.stage = walk.env.stage;
  }

  at(start, dur) {
    const end = start + dur;
    this.maxEnd = Math.max(this.maxEnd, end);
    return { start, dur, end, t: dur > 0 ? clamp((this.time - start) / dur) : this.time >= start ? 1 : 0, on: this.time >= start };
  }

  after(prev, dur, wait = 0) { return this.at(prev.end + wait, dur); }

  brk(ev) { this.breaks.push(ev.end); }

  pulse(ev) { return ev.on && ev.t < 1 ? Math.sin(Math.PI * ev.t) : 0; }

  say(ev, html) { this.says.push({ start: ev.start, html }); }

  cam(ev, target, pos) { this.kfs.push({ ev, target, pos }); }

  ids(id) { return this.groups[id] || [id]; }

  hl(id, v) { if (!this.dry) for (const b of this.ids(id)) { const k = this.stage.block(b); if (k) k.hl = Math.max(k.hl, v); } }

  op(id, v) { if (!this.dry) for (const b of this.ids(id)) { const k = this.stage.block(b); if (k) k.op = Math.min(k.op, v); } }

  only(ids, v = 0.12) {
    if (this.dry) return;
    const keep = new Set(ids.flatMap((i) => this.ids(i)));
    for (const [id, b] of this.stage.blocks) if (!keep.has(id)) b.op = Math.min(b.op, v);
  }

  arc(id, frac, alpha = 1) {
    if (this.dry) return;
    const c = this.stage.curves.get(id);
    if (c) { c.frac = Math.max(c.frac, frac); c.alpha = Math.max(c.alpha || 0, alpha); }
  }

  note(key, text, p, alpha = 1, cls = "") { if (!this.dry) this.stage.note(key, text, p, alpha, cls); }

  at3(id, dx = 0, dy = 0, dz = 0) { return this.env.layout.pos[id].clone().add(new THREE.Vector3(dx, dy, dz)); }

  seg(name, alpha) { if (!this.dry) this.stage.segAlpha[name] = Math.max(this.stage.segAlpha[name] || 0, alpha); }

  weights(name, anchor, alpha) {
    if (this.dry) return;
    this.env.showWeights(name, anchor, alpha);
  }

  pose() {
    let cur = this.walk.startPose(this.i);
    for (const k of this.kfs) {
      const e = this.env.reduce ? (k.ev.on ? 1 : 0) : ease(k.ev.t);
      if (!k.ev.on) break;
      const to = { target: k.target, pos: k.pos };
      cur = { target: lerp3(cur.target, to.target, e), pos: lerp3(cur.pos, to.pos, e) };
      if (k.ev.t < 1 && !this.env.reduce) break;
    }
    return cur;
  }

  finalPose() {
    let cur = this.walk.startPose(this.i);
    if (this.kfs.length) { const k = this.kfs[this.kfs.length - 1]; cur = { target: k.target, pos: k.pos }; }
    return cur;
  }
}

export class Walk {
  constructor(env) {
    this.env = env;
    this.phase = 0; this.time = 0; this.playing = false;
    this.cache = [];
  }

  invalidate() { this.cache = []; }

  info(i) {
    if (!this.cache[i]) {
      const c = new Ctx(this, i, Infinity, true);
      PHASES[i].run(c);
      this.cache[i] = { len: c.maxEnd + 0.6, breaks: [...new Set(c.breaks)].sort((a, b) => a - b), pose: c.finalPose() };
    }
    return this.cache[i];
  }

  startPose(i) { return i > 0 ? this.info(i - 1).pose : DEFAULT; }

  get length() { return this.info(this.phase).len; }

  update(dt) {
    if (!this.playing) return false;
    const info = this.info(this.phase), nt = this.time + dt;
    const b = info.breaks.find((x) => x > this.time + 1e-6 && x <= nt);
    if (b !== undefined) { this.time = b; this.playing = false; } else if (nt >= info.len) { this.time = info.len; this.playing = false; } else this.time = nt;
    return true;
  }

  next() {
    const info = this.info(this.phase);
    if (this.time >= info.len - 1e-6) {
      if (this.phase < PHASES.length - 1) this.go(this.phase + 1, 0);
      this.playing = true;
    } else this.playing = true;
  }

  prev() {
    const info = this.info(this.phase);
    const b = info.breaks.filter((x) => x < this.time - 0.05);
    this.playing = false;
    if (b.length) this.time = b[b.length - 1];
    else if (this.time > 0.05 || this.phase === 0) this.time = 0;
    else this.go(this.phase - 1, 0);
  }

  go(i, t = 0) { this.phase = i; this.time = t; this.playing = false; }

  frame(hover, extra) {
    const c = new Ctx(this, this.phase, this.time, false);
    this.env.stage.beginFrame();
    PHASES[this.phase].run(c);
    if (hover) { c.hl(hover, 1); if (hover === "bodies") for (let i = 0; i < this.env.stage.nBodies; i++) this.env.stage.bodyHl[i] = 1; }
    if (extra) extra();
    this.env.stage.endFrame();
    return c;
  }
}

function geom(env) {
  const S = env.S, tr = S.trace, i = S.sel;
  const px = tr.pos[2 * i], py = tr.pos[2 * i + 1];
  return { i, px, py, corners: bodyCorners(px, py, 128, 64), body: env.stage.bodyWorld(i) };
}

function cellMark(env, c, ix, iy, lift, alpha, color, block = "input", size = 1.3) {
  const h = 64 / 128;
  const p = env.stage.world(block, (ix + 0.5) * h - 32, (iy + 0.5) * h - 32, lift);
  const b = env.stage.block(block);
  env.stage.markList.push({ p, size: size * b.size / 128, alpha, color });
  return p;
}

function bodyArrows(c, name, arr, scale, alpha, only = -1) {
  if (c.dry) return;
  const st = c.stage, p = new THREE.Vector3();
  for (let i = 0; i < st.nBodies; i++) {
    if (only >= 0 && i !== only) continue;
    st.bodyWorld(i, p);
    st.segs[name].arrow(p.x, p.y, p.z, p.x + arr[2 * i] * scale, p.y, p.z - arr[2 * i + 1] * scale);
  }
  c.seg(name, alpha);
}

const levelName = (l) => (l < 0 ? "net.inp" : "net.enc." + l + ".0");

const PHASES = [
  {
    title: "Overview",
    run(c) {
      const e = c.env, n = e.S.n, lay = e.layout;
      const a = c.at(0, 0.9);
      c.cam(a, DEFAULT.target, DEFAULT.pos);
      c.say(c.at(0, 0), `One tick of the scatter-field model, drawn as the computation it is. ${N_(n)} bodies float above the arena. Each ${ref("input", "grid")} is an image of the arena (${G_} cells per side, each cell ${H_} sim units wide), so a field is drawn as a flat picture and a body as a sphere above it.`);
      const b = c.after(a, 1.1, 0.3);
      c.say(b, `Data flows in a U. Bodies are ${ref("bodies", "scattered")} onto the grid, a learned ${ref("far", "far-field kernel")} adds a global force estimate, a ${ref("enc", "UNet encoder")} shrinks the grid ${L_} down to 4 x 4, a ${ref("dec", "decoder")} grows it back with ${ref("skips", "skip connections")}, and the result is a ${ref("pot", "potential")} that every body reads back.`);
      const d = c.after(b, 1.2, 0.2);
      c.say(d, `Colours: green is density, blue/orange are signed values, purple are UNet activations (${C_} channels), pink are learned weights. Everything here is real data from one traced tick; nothing is mocked.`);
      const seq = [["bodies", 0], ["input", 1], ["far", 1], ["enc", 2], ["dec", 2], ["pot", 3]];
      const sw = c.after(d, 3.2, 0.2);
      seq.forEach(([id, k]) => {
        const t = (sw.t * 3.6) - k;
        if (id === "bodies") { if (!c.dry && t > 0 && t < 1) for (let q = 0; q < e.stage.nBodies; q++) e.stage.bodyHl[q] = 1; } else c.hl(id, t > 0 && t < 1 ? Math.sin(Math.PI * t) : 0);
      });
      const f = c.after(sw, 0.1, 0.2);
      c.say(f, `Press <kbd>Space</kbd> to play a step, <kbd>&rarr;</kbd> for the next one. Click a body to follow it. Drag to orbit, hover any cell for its value.`);
      c.brk(f);
    },
  },
  {
    title: "Scatter",
    run(c) {
      const e = c.env, g = geom(e), tr = e.S.trace, sc = 4;
      const a = c.at(0, 0.8);
      c.cam(a, [g.body.x, 0, g.body.z], [g.body.x + 0.5, 2.6, g.body.z + 2.6]);
      c.only(["input"]);
      c.hl("input", 0.5);
      c.say(c.at(0, 0), `Body ${g.i} sits at (${g.px.toFixed(2)}, ${g.py.toFixed(2)}) sim units. It does not touch one cell: cloud-in-cell splitting hands its mass to the <b>four nearest cell centres</b> in proportion to how close each is. The weights sum to 1.`);
      const b = c.after(a, 1.2, 0.2);
      for (const [k, [ix, iy, w]] of g.corners.entries()) {
        const p = cellMark(e, c, ix, iy, 0.03, 0.55 * b.t, 0x62d6a4, "input", 0.95);
        if (!c.dry) e.stage.segs.cic.add(g.body.x, g.body.y, g.body.z, p.x, p.y, p.z);
        c.note("w" + k, `w=${w.toFixed(2)}`, p.clone().add(new THREE.Vector3(0, 0.25, 0)), b.t, "w");
      }
      c.seg("cic", b.t);
      if (!c.dry) { e.stage.ringAt = g.body; e.stage.ringR = 0.32; e.stage.ringA = 1; }
      const w0 = g.corners.map((q) => q[2]);
      c.say(b, `Corner weights: ${w0.map((w) => w.toFixed(3)).join(", ")} (sum ${w0.reduce((s, v) => s + v, 0).toFixed(3)}). Each cell accumulates mass from every body that touches it, divided by the cell area ${dim("h", "h&sup2;")} to give a density, so the top plane is ${ref("input", "&rho;")}.`);
      const d = c.after(b, 1.0, 0.4);
      c.say(d, `The same weights also deposit momentum: the next two planes are &rho;v<sub>x</sub> and &rho;v<sub>y</sub>. The model is only ever shown these three channels, never individual bodies. (This port assumes every body has unit mass.)`);
      c.cam(d, [0, -2.2, 0], [0, 30, 16]);
      c.brk(d);
      const ex = c.after(d, 1.0, 0.1);
      c.hl("input", ex.on ? 0.4 : 0);
      c.say(ex, `Seen from above: the whole density plane for all ${N_(e.S.n)} bodies. Bodies outside the 64 x 64 arena would fall off the grid and receive no grid force at all.`);
    },
  },
  {
    title: "Far field",
    run(c) {
      const e = c.env, ch = e.charts, S = e.S;
      const a = c.at(0, 0.8);
      c.cam(a, [-14, 0, -28], [-14, 58, 0]);
      c.only(["input", "kw", "kring", "phik", "ak"], 0.2);
      c.say(c.at(0, 0), `Before the UNet sees anything, a learned <b>kernel</b> K(r) gives a global estimate. A small MLP (${ref("kw", "weights")}) maps distance r to a number; below r = 2 it is tapered to zero, because close pairs are handled separately later.`);
      const b = c.after(a, 1.8, 0.2);
      for (const id of ["kraw", "ktap", "kaxis"]) c.arc(id, id === "kaxis" ? 1 : b.t, 1);
      c.hl("kw", 0.6);
      c.hl("kring", b.on ? 1 : 0);
      c.note("kcap", "K(r), signed-log scale", new THREE.Vector3(ch.kbase.x + 7, ch.kbase.y + ch.kH + 1, ch.kbase.z), b.on ? 1 : 0);
      c.note("kraw", "raw MLP", new THREE.Vector3(ch.kbase.x + 12, ch.kbase.y - ch.kH, ch.kbase.z), b.on ? 0.8 : 0, "k1");
      c.note("ktap", "tapered (used)", new THREE.Vector3(ch.kbase.x + 12, ch.kbase.y - ch.kH - 1.2, ch.kbase.z), b.on ? 0.8 : 0, "k2");
      c.say(b, `The tapered curve K(r) is wrapped around the origin to make ${ref("kring", "this ring")} (every cell offset from the centre gets its own K). This is the learned object closest to interpretable: a force law as a function of distance.`);
      const d = c.after(b, 1.2, 0.5);
      c.arc("rhoK", d.t);
      c.hl("phik", d.on ? 0.8 : 0);
      c.hl("input", d.on ? 0.3 * Math.sin(Math.PI * d.t) : 0);
      c.say(d, `The density is convolved with K using an FFT (a linear convolution on a padded 256 x 256 grid), giving a ${ref("phik", "far-field potential")} &phi;<sub>K</sub>. A convolution mixes every cell with every other cell, so it is global, but it is linear and fixed: <b>not softmax attention</b>.`);
      const f = c.after(d, 1.2, 0.5);
      c.arc("akIn3", f.t); c.arc("akIn4", f.t);
      c.hl("ak", f.on ? 0.8 : 0);
      c.say(f, `Its negative gradient, a central difference over the grid, is the ${ref("ak", "kernel acceleration")} (a<sub>K,x</sub>, a<sub>K,y</sub>). These two planes are written into slots 4 and 5 of the ${ref("input", "input stack")}, together with the previous tick's potential in slot 6.`);
      c.brk(f);
    },
  },
  {
    title: "UNet",
    run(c) {
      const e = c.env, lay = e.layout, P = lay.pos;
      const a = c.at(0, 0.8);
      c.cam(a, [11, -1, 0], [11, 24, 26]);
      c.say(c.at(0, 0), `The ${ref("input", "six-channel input stack")} (&rho;, &rho;v<sub>x</sub>, &rho;v<sub>y</sub>, a<sub>K,x</sub>, a<sub>K,y</sub>, previous &phi;) goes into a UNet. First, a 3 x 3 convolution lifts 6 channels to ${C_}. Each slab below is one tensor: its footprint is the arena, its thickness the ${C_} channels. Use <kbd>[</kbd> <kbd>]</kbd> or the slider to scrub channels. Channels are not individually interpretable.`);
      let prev = a;
      const names = ["h0", ...lay.enc];
      let cur = -1;
      const evs = [];
      const wait = 1.0;
      names.forEach((id, k) => {
        const ev = c.after(prev, 0.8, k === 0 ? 0.6 : wait);
        evs.push(ev);
        const p = P[id];
        c.cam(ev, [p.x, p.y, p.z], [p.x + 4, p.y + 16, p.z + 22]);
        if (ev.on) cur = k;
        const H = k === 0 ? 128 : 128 >> (k - 1);
        c.say(ev, k === 0 ? `<b>Input conv</b>: 6 &rarr; ${C_} at ${G_}. Pink on top is its weight tensor: one 3 x 3 kernel per (output, input) channel pair, 32 x 6 of them.` : `<b>Encoder ${k - 1}</b>: two 3 x 3 convolutions with GELU, ${C_} channels at ${H} x ${H}.${k > 1 ? " Before it, 2 x 2 average pooling halved the resolution." : ""} The weight tensor shown is the first of its two convs.`);
        prev = ev;
      });
      c.brk(prev);
      const down = c.after(prev, 0.4, 0.2);
      c.say(down, `At the bottom the grid is just 4 x 4 cells, each seeing the whole arena. Now the decoder climbs back up.`);
      let q = down;
      const dec = [];
      lay.dec.forEach((id, k) => {
        const l = LEVELS - 1 - k;
        const ev = c.after(q, 0.8, wait);
        dec.push(ev);
        const p = P[id];
        c.cam(ev, [p.x, p.y, p.z - 6], [p.x + 4, p.y + 16, p.z + 16]);
        c.say(ev, k === 0 ? `<b>Decoder ${l}</b>: nearest-neighbour upsample, then <b>concatenate the skip</b> from ${ref("enc" + l, "encoder " + l)} (64 channels in), two convs back to ${C_}. Skip connections return fine detail that pooling threw away.` : `<b>Decoder ${l}</b>: upsample to ${128 >> l} x ${128 >> l}, concatenate ${ref("enc" + l, "encoder " + l)}'s skip.`);
        q = ev;
      });
      const out = c.after(q, 0.8, wait);
      c.cam(out, [lay.pos.phi.x, 0, lay.pos.phi.z], [lay.pos.phi.x + 4, 14, lay.pos.phi.z + 18]);
      c.say(out, `A last 1-channel conv gives the ${ref("phi", "potential &phi;")} on the full ${G_} grid.`);
      c.brk(out);
      const shown = cur;
      names.forEach((id, k) => { c.hl(id, k === shown ? 1 : k < shown ? 0.25 : 0); });
      lay.dec.forEach((id, k) => { const ev = dec[k]; c.hl(id, ev.on ? (k === dec.filter((x) => x.on).length - 1 ? 1 : 0.25) : 0); });
      lay.dec.forEach((id, k) => { const l = LEVELS - 1 - k; c.arc("skip" + l, dec[k].t); });
      c.arc("inH0", evs[0].t); c.arc("down", down.t); c.arc("encPhi", out.t);
      c.hl("phi", out.on ? 1 : 0);
      let wn = null, anchor = null;
      const nd = dec.filter((x) => x.on).length;
      if (out.on) { wn = "net.out"; anchor = c.at3("phi", 0, 6, 0); }
      else if (nd > 0) { const l = LEVELS - nd; wn = "net.dec." + l + ".0"; anchor = c.at3("dec" + l, 0, 7, 0); }
      else if (shown >= 0) { wn = levelName(shown - 1); anchor = c.at3(names[shown], 0, 7, 0); }
      if (wn) c.weights(wn, anchor, 1);
    },
  },
  {
    title: "Potential and gradient",
    run(c) {
      const e = c.env, P = e.layout.pos;
      const a = c.at(0, 0.8);
      c.cam(a, [6, 0, P.phi.z], [6, 40, P.phi.z + 28]);
      c.only(["phi", "grad", "agrid", "ak", "dec0"], 0.25);
      c.hl("phi", 0.7);
      c.say(c.at(0, 0), `The ${ref("phi", "potential &phi;")} is a single scalar per cell: a learned stand-in for gravitational potential energy. It is the recurrent state too: next tick it comes back in as input channel 6.`);
      const b = c.after(a, 1.0, 0.4);
      c.arc("phiGrad", b.t);
      c.hl("grad", b.on ? 0.9 : 0);
      c.say(b, `Force is the negative slope of the potential: a central difference over neighbouring cells, divided by 2${dim("h", "h")}, gives ${ref("grad", "&minus;&nabla;&phi;")} (x and y planes). Edges are zero-padded.`);
      const d = c.after(b, 1.0, 0.4);
      c.arc("gradAgrid", d.t); c.arc("akAgrid", d.t);
      c.hl("agrid", d.on ? 0.9 : 0);
      c.hl("ak", d.on ? 0.6 : 0);
      c.say(d, `The kernel acceleration from the far field is added back, giving the ${ref("agrid", "total grid acceleration")} <b>a<sub>grid</sub> = &minus;&nabla;&phi; + a<sub>K</sub></b>. The UNet therefore only has to learn a correction on top of the kernel.`);
      c.brk(d);
    },
  },
  {
    title: "Gather",
    run(c) {
      const e = c.env, g = geom(e), tr = e.S.trace, P = e.layout.pos;
      const a = c.at(0, 0.8);
      c.cam(a, [-5, 0, 10], [-5, 40, 44]);
      c.only(["agrid", "input"], 0.3);
      c.say(c.at(0, 0), `Each body now reads the field back with the <b>same four cells and weights</b> it scattered into (bilinear interpolation). Scatter and gather are transposes of each other.`);
      const b = c.after(a, 1.2, 0.2);
      c.arc("gatherArc", b.t);
      for (const [ix, iy, w] of g.corners) {
        cellMark(e, c, ix, iy, 0.03, 0.7 * b.t, 0x62d6a4, "agrid", 1.0);
        cellMark(e, c, ix, iy, 0.03, 0.4 * b.t, 0x62d6a4, "input", 0.95);
      }
      c.hl("agrid", b.on ? 0.8 : 0);
      if (!c.dry) { e.stage.ringAt = g.body; e.stage.ringR = 0.32; e.stage.ringA = 1; }
      const i = g.i, ax = tr.gather[2 * i], ay = tr.gather[2 * i + 1];
      c.say(b, `Body ${i}'s gathered grid acceleration: (${fmt(ax)}, ${fmt(ay)}). Arrows on the bodies are these accelerations, scaled to one common length so the largest is about one unit (not to physical scale).`);
      bodyArrows(c, "gather", tr.gather, e.S.accScale, b.t);
      c.brk(b);
      const d = c.after(b, 0.8, 0.3);
      c.cam(d, [g.body.x, 0, g.body.z], [g.body.x + 0.5, 12, g.body.z + 11]);
      c.say(d, `Zoomed in on the arena: every body now carries its own grid acceleration.`);
    },
  },
  {
    title: "Pair term",
    run(c) {
      const e = c.env, g = geom(e), tr = e.S.trace, S = e.S, ch = e.charts;
      const a = c.at(0, 0.8);
      c.cam(a, [g.body.x - 2, 0, g.body.z + 5], [g.body.x - 1.5, 16, g.body.z + 22]);
      c.only(["input"], 0.3);
      c.say(c.at(0, 0), `The grid is too coarse for close encounters, so a second learned term handles them: for each body, up to 16 neighbours closer than 2 units, found with a cell hash.`);
      const b = c.after(a, 1.0, 0.2);
      if (!c.dry) {
        const p = new THREE.Vector3(), q = new THREE.Vector3();
        for (const [i, j] of S.nbrs) {
          e.stage.bodyWorld(i, p); e.stage.bodyWorld(j, q);
          const seg = i === g.i || j === g.i ? e.stage.segs.pairSel : e.stage.segs.pair;
          seg.add(p.x, p.y, p.z, q.x, q.y, q.z);
        }
      }
      c.seg("pair", 0.6 * b.t); c.seg("pairSel", b.t);
      if (!c.dry) { e.stage.ringAt = g.body; e.stage.ringR = 0.32; e.stage.ringA = 1; }
      const mine = S.nbrs.filter(([i, j]) => i === g.i || j === g.i).length;
      c.say(b, `Lines connect bodies that are neighbours (${S.nbrs.length} pairs here; body ${g.i} has ${mine}). This is a sparse list of who influences whom, <b>not</b> attention: nothing is normalised by a softmax, and the neighbour set is purely geometric.`);
      const d = c.after(b, 1.2, 0.4);
      c.arc("pcurve", d.t); c.arc("paxis", 1);
      c.note("pcap", "pair force vs r", new THREE.Vector3(ch.pbase.x + 5, ch.pbase.y + ch.ph + 1, ch.pbase.z), d.on ? 1 : 0);
      c.say(d, `Each pair contributes a learned scalar of distance (small MLP) times a smooth window, along the line between the bodies. Orange arrows are the resulting per-body pair acceleration (scaled on their own, since they are much smaller than the grid term). The window kills the term at r = 2, matching where the kernel's taper turned on.`);
      bodyArrows(c, "pairAcc", tr.pairAcc, e.S.pairScale, d.t);
      c.brk(d);
      const f = c.after(d, 0.8, 0.2);
      c.say(f, `Finally the grid and pair accelerations are added and the mean is subtracted so total momentum cannot drift.`);
    },
  },
  {
    title: "Integrate",
    run(c) {
      const e = c.env, S = e.S, tr = S.trace, g = geom(e);
      const a = c.at(0, 0.8);
      c.cam(a, [g.body.x, 0, g.body.z], [g.body.x + 1.5, 14, g.body.z + 14]);
      c.only(["input"], 0.3);
      c.say(c.at(0, 0), `With the final acceleration a (arrows) and the current velocity v (blue), velocity-Verlet advances the bodies by dt = 0.1: <b>x &larr; x + dt&middot;v + &frac12;dt&sup2;&middot;a</b>.`);
      const b = c.after(a, 1.2, 0.2);
      bodyArrows(c, "gather", tr.accel, S.accScale, b.t);
      bodyArrows(c, "vel", S.vel, S.velScale, b.t);
      if (!c.dry) {
        const dt = 0.1, K = 25, gl = [], p = new THREE.Vector3(), q = new THREE.Vector3();
        for (let i = 0; i < S.n; i++) {
          const x = tr.pos[2 * i] + K * b.t * (dt * S.vel[2 * i] + 0.5 * dt * dt * tr.accel[2 * i]);
          const y = tr.pos[2 * i + 1] + K * b.t * (dt * S.vel[2 * i + 1] + 0.5 * dt * dt * tr.accel[2 * i + 1]);
          e.stage.world("input", x, y, 0.45, q);
          e.stage.bodyWorld(i, p);
          e.stage.segs.move.add(p.x, p.y, p.z, q.x, q.y, q.z);
          gl.push(q.clone());
        }
        e.stage.ghostList = gl;
      }
      c.seg("move", b.t);
      c.say(b, `Ghost spheres show where each body lands, with the displacement magnified 25x because one tick is tiny. The velocity then updates with the average of the old and new acceleration. Verlet needs the force twice per tick (before and after the move), so a full tick runs this whole pipeline <b>twice</b>; the page shows one pass.`);
      c.brk(b);
    },
  },
  {
    title: "Recurrence",
    run(c) {
      const e = c.env, P = e.layout.pos;
      const a = c.at(0, 0.8);
      c.cam(a, [11, -2, 11], [11, 40, 50]);
      c.say(c.at(0, 0), `The last piece: the potential this pass produced is fed back as input channel 6 of the next tick. It is the model's only memory, a recurrent state a bit like a residual stream carried through time.`);
      const b = c.after(a, 1.4, 0.2);
      c.arc("recur", b.t);
      c.hl("phi", b.on ? 0.9 : 0);
      c.hl("input", b.on ? 0.5 : 0);
      c.say(b, `Press <b>Next tick</b> (<kbd>.</kbd>) to run another traced tick and watch the walkthrough replay with new bodies, new fields and the fed-back potential.`);
      c.brk(b);
    },
  },
];

export const PHASE_TITLES = PHASES.map((p) => p.title);
export const PHASE_COUNT = PHASES.length;
