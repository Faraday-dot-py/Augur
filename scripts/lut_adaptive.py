"""Opt-in LUT force tables and adaptive substeps for TokenFreeDynamics(conservative_contact=True),
and a LUT for CentralForceDynamics. Nothing here edits tracked model files: `install` sets an
instance-level `forward` on the dynamics module.

Variant spec strings (see `parse`): base | lut<N>[c] | skin<M> | lut<N>[c]+skin<M>
  lut<N>: N-interval table (linear interp), suffix c = cubic (Catmull-Rom)
  skin<M>: adaptive substeps; M=0 active = neighbour d<=2r / wall d<=r at step start;
           M>0 active if d <= 2r + M*(|vi|+|vj|)*dt (wall: r + M*|v|*dt + 0.5*g*dt^2)
"""
import math

import torch

from model.token_graph import build_radius_graph, build_radius_graph_cells


class Table:
    def __init__(self, fn, lo, hi, entries, cubic, log_x=False):
        self.lo, self.hi, self.n, self.cubic, self.log_x = lo, hi, entries, cubic, log_x
        self.h = (hi - lo) / entries
        grid = torch.linspace(lo, hi, entries + 1, dtype=torch.float64)
        self.grid = grid
        self.fn = fn
        self.y = None

    def to(self, device):
        with torch.no_grad():
            x = self.grid.to(device=device, dtype=torch.float32)
            if self.log_x:
                x = torch.exp(x)
            self.y = self.fn(x).float()
        return self

    def __call__(self, x):
        shape = x.shape
        x = x.reshape(-1)
        if self.log_x:
            x = torch.log(x.clamp(min=1e-30))
        t = (x - self.lo) / self.h
        i = t.floor().clamp(0, self.n - 1)
        f = (t - i).clamp(min=0.0)
        i = i.long()
        fc = f.clamp(0.0, 1.0)
        y0, y1 = self.y[i], self.y[i + 1]
        if self.cubic:
            ym, y2 = self.y[(i - 1).clamp(min=0)], self.y[(i + 2).clamp(max=self.n)]
            m0, m1 = 0.5 * (y1 - ym), 0.5 * (y2 - y0)
            f2, f3 = fc * fc, fc * fc * fc
            out = (2 * f3 - 3 * f2 + 1) * y0 + (f3 - 2 * f2 + fc) * m0 + (-2 * f3 + 3 * f2) * y1 + (f3 - f2) * m1
        else:
            out = y0 + (y1 - y0) * fc
        out = out + (f - fc) * (self.y[self.n] - self.y[self.n - 1])
        return out.reshape(shape)


def parse(spec):
    lut, cubic, mult = 0, False, None
    for part in spec.split("+"):
        if part.startswith("lut"):
            body = part[3:]
            cubic = body.endswith("c")
            lut = int(body.rstrip("c"))
        elif part.startswith("skin"):
            mult = float(part[4:])
    return lut, cubic, mult


class FastContact:
    def __init__(self, dyn, lut=0, cubic=False, mult=None):
        self.dyn, self.mult, self.lut = dyn, mult, lut
        self.scale = dyn.force_scale
        self.gvec = (dyn.gravity * 10.0).detach()
        self.gpad = 0.5 * float(self.gvec.norm()) * dyn.dt ** 2
        pair = lambda x: x * dyn.pair_force(x[:, None])[:, 0] * self.scale
        wall = lambda x: x * dyn.wall_force(x[:, None])[:, 0] * self.scale
        dev = self.gvec.device
        if lut:
            self.pair_t = Table(pair, 0.0, 1.0, lut, cubic).to(dev)
            self.wall_t = Table(wall, 0.0, 2.0, lut, cubic).to(dev)
            self.pair_g, self.wall_g = self.pair_t, self.wall_t
        else:
            self.pair_g = lambda pen: pen * dyn.pair_force(pen) * self.scale
            self.wall_g = lambda pen: pen * dyn.wall_force(pen.unsqueeze(-1)).squeeze(-1) * self.scale
        self.stats = {"active": [], "n": []}

    def graph(self, p, rad):
        return (build_radius_graph_cells if self.dyn.cell_graph else build_radius_graph)(p, rad)

    def accel(self, p):
        dyn = self.dyn
        n, r = dyn.n, dyn.radius
        x, y = p[:, 0], p[:, 1]
        d = torch.stack([x, (n - 1) - x, y, (n - 1) - y], dim=1)
        f = self.wall_g((r - d).clamp(min=0.0) / r)
        acc = torch.stack([f[:, 0] - f[:, 1], f[:, 2] - f[:, 3]], dim=1) + self.gvec
        edges = self.graph(p, 2 * r)
        if edges.shape[1] > 0:
            src, dst = edges[0], edges[1]
            rel = p[dst] - p[src]
            dist = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
            pen = (2 * r - dist).clamp(min=0.0) / (2 * r)
            acc = acc.index_add(0, dst, self.pair_g(pen) * rel / dist)
        return acc

    def verlet(self, p, v):
        h = self.dyn.dt / self.dyn.contact_substeps
        a = self.accel(p)
        for _ in range(self.dyn.contact_substeps):
            vh = v + 0.5 * h * a
            p = p + h * vh
            a = self.accel(p)
            v = vh + 0.5 * h * a
        return p, v

    def active_mask(self, pos, vel):
        dyn, m = self.dyn, self.mult
        n, r, dt = dyn.n, dyn.radius, dyn.dt
        speed = vel.norm(dim=1)
        x, y = pos[:, 0], pos[:, 1]
        wd = torch.minimum(torch.minimum(x, (n - 1) - x), torch.minimum(y, (n - 1) - y))
        active = wd <= r + (m * speed * dt + self.gpad if m > 0 else 0.0)
        if m == 0:
            edges = self.graph(pos, 2 * r)
            active[edges[0]] = True
            return active
        rad = 2 * r + 2 * m * float(speed.max()) * dt
        edges = self.graph(pos, rad)
        src, dst = edges[0], edges[1]
        d = (pos[dst] - pos[src]).norm(dim=-1)
        keep = d <= 2 * r + m * (speed[src] + speed[dst]) * dt
        active[src[keep]] = True
        return active

    def __call__(self, positions, velocities, hidden):
        dyn = self.dyn
        dt = dyn.dt
        if self.mult is None:
            p, v = self.verlet(positions, velocities)
        else:
            active = self.active_mask(positions, velocities)
            idx = active.nonzero().squeeze(1)
            self.stats["active"].append(int(idx.numel()))
            self.stats["n"].append(positions.shape[0])
            if idx.numel() == positions.shape[0]:
                p, v = self.verlet(positions, velocities)
            else:
                p = positions + velocities * dt + 0.5 * self.gvec * dt * dt
                v = velocities + self.gvec * dt
                if idx.numel() > 0:
                    pa, va = self.verlet(positions[idx], velocities[idx])
                    p = p.index_copy(0, idx, pa)
                    v = v.index_copy(0, idx, va)
        return p - (positions + velocities * dt), v - velocities, hidden


def install(model, spec):
    """Installs the variant on `model.dynamics` in place; returns the FastContact (or None for base)."""
    dyn = model.dynamics if hasattr(model, "dynamics") else model
    dyn.__dict__.pop("forward", None)
    if spec == "base":
        return None
    lut, cubic, mult = parse(spec)
    fc = FastContact(dyn, lut, cubic, mult)
    dyn.forward = fc
    return fc


def central_force_fn(dyn, lut=0, cubic=False, train_max=150.0, lo=1e-3, hi=8.0):
    base = lambda d: dyn.force(torch.log(d.clamp(max=train_max))) / (d ** 2 + 1.0)
    if not lut:
        return base
    dev = next(dyn.parameters()).device
    fn = lambda d: base(d.reshape(-1, 1)).reshape(-1)
    table = Table(fn, math.log(lo), math.log(hi), lut, cubic, log_x=True).to(dev)
    return lambda d: table(d)
