"""Opt-in tier-0 extensions for ScatterField (docs/theory/heterogeneous-tiered.md).

`attach_tier0(model, spec)` sets `model.t0`; with `model.t0 is None` ScatterField is bit-identical to before (hooks in ScatterField._force).
Every feature is a flag in the spec string (comma separated, `name` or `name=value`):

  moments=1|2   sub-cell moments about each cell centre: first (mean offset) and second (central) -> net input channels
  vdisp         velocity dispersion m*sigma^2 tensor per cell (3 ch)
  cross         position-velocity cross covariance per cell (4 ch)
  head=W        small per-particle MLP head (width W): gathered grid features + own velocity, velocity relative to the local
                mean flow, mass -> extra acceleration (adds to the mesh force before momfix, so total momentum is still removed)
  phid=D        per-particle hidden state (D floats) carried in the rollout state as (grid_field, ph); updated by the head
  phscatter     scatter m*ph as grid channels (needs phid)
  mat=C         C per-particle material channels (set_material); scattered as m*mat, also a head input
  ext=1|3|6     imposed external potential expansion as input channels: [phi], +[gx,gy], +[Hxx,Hxy,Hyy] (set_external)
  extapply      add the external acceleration -(g + H x) to the particle step (after momfix)
  bnd           boundary channels: occupancy, clipped SDF, type mask open/periodic/wall (set_boundary)
  halo=H        halo width in cells; the core tile is (grid - 2H) cells wide
  boundary=tile|open|periodic   tile (default): halo particles are neighbour copies; open: nothing outside the core scatters or
                feels force, halo field zeroed; periodic: wrap at core size L, halo = wrapped copies (needs halo >= net reach)
  taper=T       cosine taper of scattered mass to zero over the outer T cells of the grid (hole 3)
  vscale=S      velocity scale for head inputs
  extgain=S     gain on the ext input channels only (extapply keeps the physical amplitude)
  headmax=S     bound the head acceleration: S*tanh(a/S) (0 = unbounded)
  headgate      (contact models) gate the head output by the contact window set from outside (Tier0.gate)
  lora=R        rank-R low-rank adapter on every conv of the grid net (zero-init output, base weights keep their names)
  vclip=S       clamp the velocity inputs of the head to [-S, S]

New net input channels are appended after all existing ones, so `scatter_field.load_compat` zero-pads a plain checkpoint and
the model reproduces it at init (head last layer is zero-initialised).

Tier-0 interface for tier >= 1 (all work on a model with or without t0):
  moment_scatter(model, pos, vel, mass, mask) -> MomentGrid  exact raw extensive moments per cell (no net)
  tier0_forward(model, pos, vel, mass, mask, state)    -> Tier0Out  moments, mesh force grid, particle acceleration, state, aux
  model.gather(grid, pos)                                      bilinear gather matching the scatter kernel
"""
import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import scatter_field as sf

MOMENT_NAMES = ("m", "px", "py", "s1x", "s1y", "s2xx", "s2xy", "s2yy", "vvxx", "vvxy", "vvyy", "cxx", "cxy", "cyx", "cyy")
BND_CH = 5
BND_TYPES = ("open", "periodic", "wall")
BND_CLIP = 8.0


@dataclass
class Tier0Cfg:
    moments: int = 0
    vdisp: bool = False
    cross: bool = False
    head: int = 0
    phid: int = 0
    phscatter: bool = False
    mat: int = 0
    ext: int = 0
    extapply: bool = False
    bnd: bool = False
    halo: int = 0
    boundary: str = "tile"
    taper: int = 0
    vscale: float = 1.0
    extgain: float = 1.0
    headmax: float = 0.0
    vclip: float = 0.0
    headgate: bool = False
    lora: int = 0

    def n_extra(self):
        return (2 * (self.moments >= 1) + 3 * (self.moments >= 2) + 3 * self.vdisp + 4 * self.cross
                + self.phid * self.phscatter + self.mat + self.ext + BND_CH * self.bnd)

    def spec(self):
        out = []
        for f in fields(self):
            v, d = getattr(self, f.name), f.default
            if v != d:
                out.append(f.name if v is True else f"{f.name}={v}")
        return ",".join(out)


def parse_spec(spec):
    if isinstance(spec, Tier0Cfg):
        return spec
    kw = {}
    types = {f.name: f.type for f in fields(Tier0Cfg)}
    for t in [x.strip() for x in (spec or "").split(",") if x.strip()]:
        k, _, v = t.partition("=")
        assert k in types, f"unknown tier-0 flag {k}"
        if types[k] in (bool, "bool"):
            kw[k] = True if v == "" else v.lower() in ("1", "true")
        elif types[k] in (int, "int"):
            kw[k] = int(v)
        elif types[k] in (float, "float"):
            kw[k] = float(v)
        else:
            kw[k] = v
    cfg = Tier0Cfg(**kw)
    assert cfg.ext in (0, 1, 3, 6) and cfg.boundary in ("tile", "open", "periodic") and cfg.moments in (0, 1, 2)
    assert not cfg.phscatter or cfg.phid
    assert not cfg.phid or cfg.head
    assert cfg.boundary != "periodic" or cfg.halo >= 1
    return cfg


class MomentGrid(NamedTuple):
    """Raw extensive moments per cell, tensor (B, 15, G, G) in MOMENT_NAMES order, offsets d = (x - cell centre) / h in units of
    the cell width h of this grid. m = sum w*m, p = sum w*m*v, s1 = sum w*m*d, s2 = sum w*m*d d^T, vv = sum w*m*v v^T,
    c = sum w*m*d v^T (w = CIC weight). Sums, not densities: merge children into a parent by plain addition after shifting
    offsets to the parent centre and rescaling by h_child / h_parent (parallel axis); central moments:
    s2 - s1 s1^T / m, vv - p p^T / m, c - s1 p^T / m."""
    data: torch.Tensor
    names: tuple = MOMENT_NAMES

    def get(self, *names):
        return self.data[:, [self.names.index(n) for n in names]]


class Tier0Out(NamedTuple):
    moments: MomentGrid
    acc_grid: torch.Tensor
    field: torch.Tensor
    a_tok: torch.Tensor
    state: object
    aux: dict


class _Ctx:
    pass


def _wrap1(x, dim, hw, fold):
    G = x.shape[dim]
    n = G - 2 * hw
    lo, hi, mid = x.narrow(dim, 0, hw), x.narrow(dim, G - hw, hw), x.narrow(dim, hw, n)
    if fold:
        mid = torch.cat([mid.narrow(dim, 0, hw) + hi, mid.narrow(dim, hw, n - 2 * hw), mid.narrow(dim, n - hw, hw) + lo], dim)
    return torch.cat([mid.narrow(dim, n - hw, hw), mid, mid.narrow(dim, 0, hw)], dim)


def wrap_halo(x, hw, fold=False):
    """Periodic ghost cells on (B, C, G, G): fold=True adds halo spill-over into the opposite core edge first (scatter);
    then halo = wrapped copy of the core (also used to refresh recurrent hidden state)."""
    if hw <= 0:
        return x
    return _wrap1(_wrap1(x, 2, hw, fold), 3, hw, fold)


def zero_halo(x, hw):
    if hw <= 0:
        return x
    m = torch.zeros(x.shape[-2:], device=x.device, dtype=x.dtype)
    m[hw:-hw, hw:-hw] = 1.0
    return x * m


def _moment_vals(names, mm, vel, dx, dy, ph=None, mat=None):
    vx, vy = vel[..., 0], vel[..., 1]
    tab = {"m": mm, "px": mm * vx, "py": mm * vy, "s1x": mm * dx, "s1y": mm * dy, "s2xx": mm * dx * dx, "s2xy": mm * dx * dy,
           "s2yy": mm * dy * dy, "vvxx": mm * vx * vx, "vvxy": mm * vx * vy, "vvyy": mm * vy * vy, "cxx": mm * dx * vx,
           "cxy": mm * dx * vy, "cyx": mm * dy * vx, "cyy": mm * dy * vy}
    out = []
    for n in names:
        if n.startswith("ph"):
            out.append(mm * ph[..., int(n[2:])])
        elif n.startswith("mat"):
            out.append(mm * mat[..., int(n[3:])])
        else:
            out.append(tab[n])
    return torch.stack(out, -1)


def scatter_names(model, pos, vel, mass, mask, names, ph=None, mat=None, hw=0, periodic=False):
    """CIC scatter of the named extensive channels (same kernel as ScatterField.scatter, unscaled) -> (B, C, G, G)."""
    B, N = pos.shape[:2]
    G = model.grid
    mm = mass * mask
    _, w = model._corners(pos)
    out = torch.zeros(B * G * G, len(names), device=pos.device, dtype=pos.dtype)
    for (flat, wt), (ox, oy) in zip(model._corner_iter(pos), ((0, 0), (0, 1), (1, 0), (1, 1))):
        vals = _moment_vals(names, mm, vel, w[..., 0] - ox, w[..., 1] - oy, ph, mat)
        out = out.index_add(0, flat.reshape(-1), (vals * wt[..., None]).reshape(-1, len(names)))
    out = out.view(B, G, G, len(names)).permute(0, 3, 1, 2)
    return wrap_halo(out, hw, fold=True) if periodic else out


def _central(m, a, b, ab):
    return ab - a * b / m.clamp_min(1e-9)


def vdisp_channels(raw):
    m = raw["m"]
    return [_central(m, raw["px"], raw["px"], raw["vvxx"]), _central(m, raw["px"], raw["py"], raw["vvxy"]),
            _central(m, raw["py"], raw["py"], raw["vvyy"])]


def derived_inputs(raw, cfg):
    """raw: dict name -> (B,1,G,G) extensive channels. Returns list of net input channels (unscaled)."""
    m = raw["m"]
    ch = []
    if cfg.moments >= 1:
        ch += [raw["s1x"], raw["s1y"]]
    if cfg.moments >= 2:
        ch += [_central(m, raw["s1x"], raw["s1x"], raw["s2xx"]), _central(m, raw["s1x"], raw["s1y"], raw["s2xy"]),
               _central(m, raw["s1y"], raw["s1y"], raw["s2yy"])]
    if cfg.vdisp:
        ch += vdisp_channels(raw)
    if cfg.cross:
        ch += [_central(m, raw["s1x"], raw["px"], raw["cxx"]), _central(m, raw["s1x"], raw["py"], raw["cxy"]),
               _central(m, raw["s1y"], raw["px"], raw["cyx"]), _central(m, raw["s1y"], raw["py"], raw["cyy"])]
    return ch


def moment_scatter(model, pos, vel, mass, mask):
    """Exact raw extensive moments of all tokens on the model's grid (no network). Uses the model's boundary geometry when t0 is attached."""
    t0 = model.t0
    spos, smass = (pos, mass) if t0 is None else t0.geometry(model, pos, mass)
    hw, per = (0, False) if t0 is None else (t0.cfg.halo, t0.cfg.boundary == "periodic")
    return MomentGrid(scatter_names(model, spos, vel, smass, mask, MOMENT_NAMES, hw=hw, periodic=per))


class Tier0(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.record = False
        self.aux = {}
        self.material = None
        self.ids = None
        self.write = None
        self.n_global = 0
        self.pen_list = None
        self.pow_list = None
        self.gate = None
        self.ext_grid = None
        self.ext_poly = None
        self.ext_high = None
        self.bnd_grid = None
        self.edge = "open"
        assert not (cfg.boundary == "periodic" and (model.kernel or model.pp)), "periodic wrap does not cover kernel/pp terms"
        if cfg.head:
            d = 9 + cfg.phid + cfg.mat + 3 * cfg.vdisp
            self.headnet = nn.Sequential(nn.Linear(d, cfg.head), nn.GELU(), nn.Linear(cfg.head, cfg.head), nn.GELU(),
                                         nn.Linear(cfg.head, 2 + 2 * cfg.phid))
            nn.init.zeros_(self.headnet[-1].weight)
            nn.init.zeros_(self.headnet[-1].bias)

    def core(self, model):
        return (model.grid - 2 * self.cfg.halo) * model.h

    def geometry(self, model, pos, mass):
        c = self.cfg
        spos, smass = pos, mass
        L = self.core(model)
        if c.boundary == "periodic":
            spos = torch.remainder(pos + L / 2, L) - L / 2
        elif c.boundary == "open" and c.halo:
            smass = smass * (pos.abs().amax(-1) < L / 2).to(pos.dtype)
        if c.taper and c.boundary != "periodic":
            d = (model.extent / 2 - spos.abs()) / (c.taper * model.h)
            t = 0.5 * (1 - torch.cos(math.pi * d.clamp(0.0, 1.0)))
            smass = smass * t[..., 0] * t[..., 1]
        return spos, smass

    def set_material(self, material):
        self.material = material

    def set_ids(self, ids, n_global, write=None):
        """Key the per-particle hidden state by global particle id for tiled use: ids (B, N) long, -1 = padding. The state then carries a
        (B, n_global, phid) store; each window gathers its members' hidden and writes the update back (only where `write` (B, N) bool is
        true, e.g. the owner tile's particles; default all valid ids). None restores per-window state."""
        self.ids, self.n_global, self.write = ids, n_global, write

    def set_external(self, model, phi0=None, g=None, H=None, grid=None, high=None):
        """Imposed external potential expansion about the grid centre: phi = phi0 + g.x + x.H.x/2, H = (Hxx, Hxy, Hyy).
        `high=(Q, C)`: acceleration Taylor coefficients of order 2 (B,2,2,2) and 3 (B,2,2,2,2) added by extapply (not in the channels).
        Fills the ext input channels at cell centres; or pass `grid` (B, ext, G, G) directly (e.g. a parent's lagged expansion)."""
        c = self.cfg
        self.ext_high = high
        if grid is not None:
            self.ext_grid, self.ext_poly = grid, None
            return
        ref = next(t for t in (phi0, g, H) if t is not None)
        B, dev, dt = ref.shape[0], ref.device, ref.dtype
        phi0 = torch.zeros(B, device=dev, dtype=dt) if phi0 is None else phi0
        g = torch.zeros(B, 2, device=dev, dtype=dt) if g is None else g
        H = torch.zeros(B, 3, device=dev, dtype=dt) if H is None else H
        G = model.grid
        xs = (torch.arange(G, device=dev, dtype=dt) + 0.5) * model.h - model.extent / 2
        Y, X = torch.meshgrid(xs, xs, indexing="ij")
        s = lambda t: t[:, None, None]
        gx, gy = s(g[:, 0]) + s(H[:, 0]) * X + s(H[:, 1]) * Y, s(g[:, 1]) + s(H[:, 1]) * X + s(H[:, 2]) * Y
        phi = s(phi0) + s(g[:, 0]) * X + s(g[:, 1]) * Y + 0.5 * (s(H[:, 0]) * X * X + 2 * s(H[:, 1]) * X * Y + s(H[:, 2]) * Y * Y)
        ch = [phi, gx, gy] + [s(H[:, i]).expand(-1, G, G) for i in range(3)]
        self.ext_grid = torch.stack(ch[:c.ext], 1) if c.ext else None
        self.ext_poly = (g, H)

    def ext_acc(self, model, pos):
        if self.ext_poly is not None:
            g, H = self.ext_poly
            x, y = pos[..., 0], pos[..., 1]
            a = -torch.stack([g[:, None, 0] + H[:, None, 0] * x + H[:, None, 1] * y, g[:, None, 1] + H[:, None, 1] * x + H[:, None, 2] * y], -1)
            if self.ext_high is not None:
                Q, C = self.ext_high
                xj, xk, xl = pos[:, :, None, :, None, None], pos[:, :, None, None, :, None], pos[:, :, None, None, None, :]
                a = a + 0.5 * (Q[:, None] * pos[:, :, None, :, None] * pos[:, :, None, None, :]).sum((-1, -2))
                a = a + (C[:, None] * xj * xk * xl).sum((-1, -2, -3)) / 6
            return a
        assert self.ext_grid is not None and self.ext_grid.shape[1] >= 3
        return -model.gather(self.ext_grid[:, 1:3], pos)

    def set_boundary(self, model, sdf=None, edge=None):
        """sdf: optional (B,1,G,G) signed distance to obstacles (negative inside, sim units). edge: 'open'|'periodic'|'wall' (str or list of B)."""
        G, h, hw = model.grid, model.h, self.cfg.halo
        dev = model.net.inp.weight.device
        B = 1 if sdf is None else sdf.shape[0]
        edge = self.edge if edge is None else edge
        edges = [edge] * B if isinstance(edge, str) else list(edge)
        B = len(edges)
        idx = torch.arange(G, device=dev)
        dist = torch.minimum(idx, G - 1 - idx).float() - (hw - 0.5)
        dedge = torch.minimum(dist[:, None], dist[None, :]) * h
        sd = dedge[None, None].expand(B, 1, G, G)
        if sdf is not None:
            sd = torch.minimum(sd, sdf) if self.cfg.boundary != "periodic" else sdf
        occ = (sdf < 0).float() if sdf is not None else torch.zeros_like(sd)
        ring = (torch.minimum(dist[:, None], dist[None, :]) < 1.0).float()[None, None].expand(B, 1, G, G)
        typ = torch.zeros(B, 3, G, G, device=dev)
        for b, e in enumerate(edges):
            typ[b, BND_TYPES.index(e)] = ring[b, 0]
        typ[:, 2:3] = torch.maximum(typ[:, 2:3], occ)
        self.bnd_grid = torch.cat([occ, (sd / (BND_CLIP * h)).clamp(-1.0, 1.0), typ], 1)

    def _fit(self, g, B, n):
        if g is None:
            return None
        return g.expand(B, *g.shape[1:]) if g.shape[0] == 1 and B > 1 else g

    def begin(self, model, pos, vel, mass, mask, field):
        c = self.cfg
        ph = None
        if isinstance(field, tuple):
            field, ph = field
        B, N = pos.shape[:2]
        ctx = _Ctx()
        ctx.store = None
        if self.ids is not None and c.phid:
            if ph is None:
                ph = torch.zeros(B, self.n_global, c.phid, device=pos.device, dtype=pos.dtype)
            ctx.store = ph
            ph = ph.gather(1, self.ids.clamp_min(0)[..., None].expand(-1, -1, c.phid)) * (self.ids >= 0)[..., None].to(ph.dtype)
        elif ph is not None and ph.shape[1] != N:
            ph = None
        if c.phid and ph is None:
            ph = torch.zeros(B, N, c.phid, device=pos.device, dtype=pos.dtype)
        ctx.ph, ctx.extra, ctx.x0, ctx.vdisp = ph, None, None, None
        ctx.pos, ctx.smass = self.geometry(model, pos, mass)
        mat = None
        if c.mat:
            mat = self.material if self.material is not None else torch.zeros(B, N, c.mat, device=pos.device, dtype=pos.dtype)
        ctx.mat = mat
        need = c.moments or c.vdisp or c.cross or (c.phid and c.phscatter) or c.mat
        ex = []
        if need:
            names = ["m"]
            if c.moments or c.cross:
                names += ["s1x", "s1y"]
            if c.moments >= 2:
                names += ["s2xx", "s2xy", "s2yy"]
            if c.vdisp or c.cross:
                names += ["px", "py"]
            if c.vdisp:
                names += ["vvxx", "vvxy", "vvyy"]
            if c.cross:
                names += ["cxx", "cxy", "cyx", "cyy"]
            nd = len(names)
            if c.phid and c.phscatter:
                names += [f"ph{i}" for i in range(c.phid)]
            names += [f"mat{i}" for i in range(c.mat)]
            raw = scatter_names(model, ctx.pos, vel, ctx.smass, mask, names, ph, mat, c.halo, c.boundary == "periodic")
            sc = model.in_scale / model.h ** 2
            d = {n: raw[:, i:i + 1] for i, n in enumerate(names)}
            ex += [t * sc for t in derived_inputs(d, c)]
            if c.vdisp:
                ctx.vdisp = [t * sc for t in vdisp_channels(d)]
            ex += [raw[:, nd + i:nd + i + 1] * sc for i in range(len(names) - nd)]
        if c.ext:
            e = self.ext_grid
            ex.append(torch.zeros(B, c.ext, model.grid, model.grid, device=pos.device, dtype=pos.dtype) if e is None
                      else self._fit(e, B, c.ext).to(pos.dtype) * c.extgain)
        if c.bnd:
            if self.bnd_grid is None or self.bnd_grid.shape[-1] != model.grid:
                self.set_boundary(model, edge=c.boundary if c.boundary != "tile" else "open")
            ex.append(self._fit(self.bnd_grid, B, BND_CH).to(device=pos.device, dtype=pos.dtype))
        if ex:
            ctx.extra = torch.cat([t if t.shape[0] == B else t.expand(B, *t.shape[1:]) for t in ex], 1)
        return field, ctx

    def after_scatter(self, ctx, x):
        c = self.cfg
        if c.boundary == "periodic":
            x = wrap_halo(x, c.halo, fold=True)
        ctx.x0 = x
        return x

    def after_net(self, field):
        c = self.cfg
        if c.boundary == "periodic":
            return wrap_halo(field, c.halo)
        if c.boundary == "open" and c.halo:
            return zero_halo(field, c.halo)
        return field

    def head(self, model, ctx, a_tok, acc, vel, mass, mask):
        c = self.cfg
        ctx.mesh = a_tok
        ctx.a_head = None
        if not c.head:
            return a_tok
        x0 = ctx.x0
        rho = x0[:, :1]
        u = x0[:, 1:3] / rho.clamp_min(1e-9)
        grid = torch.cat([rho, u] + (ctx.vdisp if c.vdisp else []), 1)
        gg = model.gather(grid, ctx.pos)
        rho_g, ug = gg[..., 0:1], gg[..., 1:3]
        rel = (vel - ug) * c.vscale
        if c.vclip:
            vel, rel = vel.clamp(-c.vclip, c.vclip), rel.clamp(-c.vclip, c.vclip)
        f = [vel * c.vscale, rel, rel.norm(dim=-1, keepdim=True), mass[..., None], torch.log1p(rho_g.clamp_min(0.0)), a_tok]
        if c.phid:
            f.append(ctx.ph)
        if c.mat:
            f.append(ctx.mat)
        if c.vdisp:
            f.append(gg[..., 3:6])
        o = self.headnet(torch.cat(f, -1))
        ah = o[..., :2] if not c.headmax else c.headmax * torch.tanh(o[..., :2] / c.headmax)
        if self.gate is not None:
            ah = ah * self.gate[..., None]
        ctx.a_head = ah * mask[..., None]
        if self.pen_list is not None:
            self.pen_list.append((ctx.a_head ** 2).sum() / mask.sum().clamp_min(1.0))
        if self.pow_list is not None:
            self.pow_list.append(torch.relu((ctx.a_head * vel).sum(-1)).sum() / mask.sum().clamp_min(1.0))
        if c.phid:
            gate = torch.sigmoid(o[..., 2:2 + c.phid])
            ph = (1 - gate) * ctx.ph + gate * torch.tanh(o[..., 2 + c.phid:])
            ctx.ph = ph * mask[..., None]
        return a_tok + ctx.a_head

    def finish(self, model, ctx, a_out, field, pos, mask):
        c = self.cfg
        if c.boundary == "open" and c.halo:
            a_out = a_out * (pos.abs().amax(-1) < self.core(model) / 2).to(a_out.dtype)[..., None]
        if c.extapply and (self.ext_poly is not None or self.ext_grid is not None):
            a_out = a_out + self.ext_acc(model, pos) * mask[..., None]
        if self.record:
            self.aux = {"acc_mesh": ctx.mesh, "head_acc": ctx.a_head, "ph": ctx.ph, "x0": ctx.x0, "extra": ctx.extra,
                        "max_acc": (a_out ** 2).sum(-1).sqrt().amax().detach()}
        if c.phid and ctx.store is not None:
            ok = (self.ids >= 0) if self.write is None else ((self.ids >= 0) & self.write)
            store = ctx.store.clone()
            b = torch.arange(ctx.ph.shape[0], device=ctx.ph.device)[:, None].expand_as(self.ids)
            store[b[ok], self.ids[ok]] = ctx.ph[ok]
            return a_out, (field, store)
        return a_out, ((field, ctx.ph) if c.phid else field)

    def head_penalty(self, mask):
        """Mean squared head acceleration over live tokens (soft penalty for the residual channel); needs record=True."""
        a = self.aux["head_acc"]
        return (a ** 2).sum(-1).mul(mask).sum() / mask.sum().clamp_min(1.0)


def _rebuild_net(model, extra):
    old = model.net
    cin = old.inp.in_channels + extra
    width, cout = old.inp.out_channels, old.out.out_channels
    if isinstance(old, sf.LocalNet):
        net = sf.LocalNet(cin, cout, width, len(old.mid) + 1)
    else:
        net = sf.UNet(cin, cout, width, old.levels)
    nn.init.zeros_(net.out.weight)
    nn.init.zeros_(net.out.bias)
    return net.to(next(old.parameters()).device)


class LoRAConv2d(nn.Conv2d):
    def forward(self, x):
        return super().forward(x) + self.lora_b(self.lora_a(x))


def add_lora(net, rank):
    """In place: every Conv2d becomes LoRAConv2d (same weight/bias keys) with a parallel kxk conv to `rank` channels then a 1x1 back, zero-init."""
    for m in [m for m in net.modules() if type(m) is nn.Conv2d]:
        m.lora_a = nn.Conv2d(m.in_channels, rank, m.kernel_size, stride=m.stride, padding=m.padding).to(m.weight.device)
        m.lora_b = nn.Conv2d(rank, m.out_channels, 1).to(m.weight.device)
        nn.init.zeros_(m.lora_b.weight)
        nn.init.zeros_(m.lora_b.bias)
        m.__class__ = LoRAConv2d


def attach_tier0(model, spec=""):
    """Attach tier-0 extensions to a ScatterField (before load_compat of any checkpoint). Returns the model."""
    assert type(model) is sf.ScatterField or isinstance(model, sf.ScatterField)
    cfg = parse_spec(spec)
    extra = cfg.n_extra()
    if extra:
        model.net = _rebuild_net(model, extra)
    if cfg.lora:
        add_lora(model.net, cfg.lora)
    dev = next(model.parameters()).device
    model.t0 = Tier0(cfg, model).to(dev)
    model.t0_spec = cfg.spec()
    return model


def tier0_forward(model, pos, vel, mass, mask, state=None):
    """Tier-0 interface for upper tiers: one force evaluation returning the exact raw moments, the mesh output grid and the
    particle acceleration. state=None starts a fresh medium state."""
    added = model.t0 is None
    if added:
        attach_tier0(model, "")
    t0 = model.t0
    old = t0.record
    t0.record = True
    try:
        if state is None:
            state = model.init_field(pos.shape[0], pos.device, pos.dtype)
            state = state[0] if model.verlet else state
        a_tok, st = model.force(pos, vel, mass, mask, state)
        aux = dict(t0.aux)
        mesh = st[0] if isinstance(st, tuple) else st
        acc_grid = model.neg_grad(mesh[:, :1]) if model.potential else mesh[:, :2]
        out = Tier0Out(moment_scatter(model, pos, vel, mass, mask), acc_grid, mesh, a_tok, st, aux)
    finally:
        t0.record = old
        if added:
            model.t0 = None
    return out
