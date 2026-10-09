import { Arch, divColor, sl, fmt } from "../../token/js/arch.js";

export { divColor, fmt };
export const KEEP = 5;
const PITCH = 11;

// 3D diagram of one body's step through CentralForceDynamics (model/central_force.py): the pair-force MLP
// run once per neighbour, the sum into an acceleration, and the two force passes of velocity Verlet.
export class CentralArch extends Arch {
  plates() {
    this.plate(-1, -3.5, 9, 3.5);
    this.plate(-1, 4, PITCH * KEEP + 2, 32.5);
    this.plate(-1, 33.5, 24, 39);
    this.plate(-1, 40.5, 58, 52.5);
    this.plate(-1, 54, 22, 60);
  }

  set(tr) {
    const C = this.cells, L = this.labels, A = this.anchors;
    C.length = 0; L.length = 0; A.cols.length = 0;
    let nw = 0;
    const wire = (u0, w0, u1, w1) => {
      if (nw >= 400) return;
      this.wirePos.set([u0, 0.05, w0, u1, 0.05, w1], 6 * nw++);
    };
    const cell = (u, w, s, v, name, val, dim = false, src = -1) => C.push({ u, w, s, v, name, val, dim, par: null, src, h: s * (0.3 + 1.4 * Math.abs(v)) });
    const lab = (text, u, w, cls = "", left = false) => L.push({ text, u, w, cls, left });
    const vec = (u, w, s, gap, a, sc, name) => {
      cell(u, w, s, sc(a[0]), name + " x", fmt(a[0]));
      cell(u + gap, w, s, sc(a[1]), name + " y", fmt(a[1]));
    };
    const pos = (x) => sl(x, 1), vel = (x) => sl(x, 0.3), acc = (x) => sl(x, 0.01);
    if (!tr) { this.mesh.count = 0; this.wires.geometry.setDrawRange(0, 0); return; }
    const id = tr.index, dt = tr.dt;

    lab(`input: body #${id}`, -1, -2.4, "sec", true);
    vec(0, 1, 1.6, 2, tr.pos, pos, "position");
    vec(4, 1, 1.6, 2, tr.vel, vel, "velocity");

    const s1 = tr.s1;
    lab(`force MLP 1→64→64→1 (tanh), once per neighbour · ${s1.n} in range, ${Math.min(KEEP, s1.edges.length)} strongest drawn`, 12, -2.4, "sec", true);
    s1.edges.forEach((e, k) => {
      const u0 = k * PITCH + 1;
      const tag = `edge #${e.src} → #${id}`;
      lab(`#${e.src}  d ${e.dist.toFixed(2)}`, u0 + 4, 5.2, "col");
      A.cols.push({ src: e.src, u: u0 + 3.5, w: 8 });
      cell(u0 + 0.6, 8.5, 1.2, pos(e.rel[0]), `${tag} · offset x (pos #${e.src} - pos #${id})`, fmt(e.rel[0]), false, e.src);
      cell(u0 + 2.6, 8.5, 1.2, pos(e.rel[1]), `${tag} · offset y`, fmt(e.rel[1]), false, e.src);
      cell(u0 + 4.6, 8.5, 1.2, 1 - Math.min(1, e.dist / 30), `${tag} · distance d = |offset|`, fmt(e.dist), false, e.src);
      cell(u0 + 6.6, 8.5, 1.2, e.logd / 5, `${tag} · MLP input ln d`, fmt(e.logd), false, e.src);
      for (let r = 0; r < 8; r++) {
        for (let q = 0; q < 8; q++) {
          const a = r * 8 + q;
          cell(u0 + q, 11 + r, 0.8, e.h1[a], `${tag} · h1[${a}] = tanh(W0 ln d + b0)`, fmt(e.h1[a]), false, e.src);
          cell(u0 + q, 20 + r, 0.8, e.h2[a], `${tag} · h2[${a}] = tanh(W1 h1 + b1)`, fmt(e.h2[a]), false, e.src);
        }
      }
      cell(u0 + 0.5, 29.5, 1.1, e.out / 1.5, `${tag} · MLP out f`, fmt(e.out), false, e.src);
      cell(u0 + 2.1, 29.5, 1.1, e.scale, `${tag} · 1 / (d² + 1)`, fmt(e.scale), false, e.src);
      cell(u0 + 3.7, 29.5, 1.1, sl(e.mag, 1e-3), `${tag} · |force| = f / (d² + 1)`, fmt(e.mag), false, e.src);
      cell(u0 + 5.3, 29.5, 1.1, acc(e.force[0]), `${tag} · force on #${id} x = |force| · offset x / d`, fmt(e.force[0]), false, e.src);
      cell(u0 + 6.9, 29.5, 1.1, acc(e.force[1]), `${tag} · force on #${id} y`, fmt(e.force[1]), false, e.src);
      wire(4, 2.5, u0 + 3.5, 8);
      wire(u0 + 3.5, 9.5, u0 + 3.5, 11);
      wire(u0 + 3.5, 18.8, u0 + 3.5, 20);
      wire(u0 + 3.5, 27.8, u0 + 3.5, 29.5);
      wire(u0 + 3.5, 31, 11, 35.5);
    });
    if (s1.n > s1.edges.length) lab(`+${s1.n - s1.edges.length} weaker`, PITCH * KEEP - 3, 32.8, "col off");
    if (!s1.n) lab("no other bodies in range · force MLP idle", 20, 18, "sec");

    lab(`sum over all ${s1.n} neighbours = acceleration a0`, -1, 34, "sec", true);
    vec(4, 36, 1.4, 2.4, tr.a0, acc, "a0 (acceleration at current position)");

    lab(`velocity Verlet · dt = ${dt}`, -1, 41, "sec", true);
    const v0 = 2, v1 = 9, v2 = 16, v3 = 28, v4 = 46;
    vec(v0, 45, 1.3, 2, tr.a0, acc, "a0");
    vec(v1, 45, 1.3, 2, tr.dp, (x) => sl(x, 1e-4), "dp = ½ dt² a0");
    vec(v2, 45, 1.3, 2, tr.pmid, pos, "predicted position = pos + vel dt + dp (this is also the new position)");
    vec(v3, 45, 1.3, 2, tr.a1, acc, "a1 (same force law at the predicted positions)");
    vec(v4, 45, 1.3, 2, tr.dv, (x) => sl(x, 1e-3), "dv = ½ dt (a0 + a1)");
    lab("a0", v0 + 1, 42.8, "col"); lab("dp = ½dt²·a0", v1 + 1, 42.8, "col"); lab("pos + vel·dt + dp", v2 + 1, 42.8, "col");
    lab("a1", v3 + 1, 42.8, "col"); lab("dv = ½dt(a0 + a1)", v4 + 1, 42.8, "col");
    lab(`second pass at predicted positions · ${tr.s2.n} neighbours · strongest ${Math.min(KEEP, tr.s2.edges.length)} forces`, v1, 47.2, "sec", true);
    tr.s2.edges.forEach((e, k) => vec(v1 + 5.5 * k, 48.8, 1.1, 1.8, e.force, acc, `second pass edge #${e.src} → #${id} · force`));
    wire(4.4, 37.5, v0 + 1, 45);
    wire(v0 + 2.5, 45.6, v1, 45.6);
    wire(v1 + 2.5, 45.6, v2, 45.6);
    wire(v2 + 3.4, 45.6, v3, 45.6);
    wire(v3 + 3.4, 45.6, v4, 45.6);
    wire(v1 + 12, 48.8, v3 + 1, 46.6);

    lab(`output: body #${id}`, -1, 54.4, "sec", true);
    vec(0, 57, 1.6, 2, tr.newPos, pos, "new position = pos + vel dt + dp");
    vec(4, 57, 1.6, 2, tr.newVel, vel, "new velocity = vel + dv");
    vec(11, 57, 1.2, 2, tr.dp, (x) => sl(x, 1e-4), "dp");
    vec(16, 57, 1.2, 2, tr.dv, (x) => sl(x, 1e-3), "dv");
    wire(v4 + 1, 45.8, 10, 56.5);
    wire(v1 + 1, 45.8, 11.5, 56.5);
    A.output = [4, 57];
    this.wires.geometry.attributes.position.needsUpdate = true;
    this.wires.geometry.setDrawRange(0, 2 * nw);
    this.mesh.count = C.length;
  }
}
