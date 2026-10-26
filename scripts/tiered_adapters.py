"""TierModel adapters (scripts/tiered_iface.py) for code that lives on other branches; imports are lazy so this module loads without them.

  TieredAdapter     agent C's model.tiered.TieredField (branch worktree-agent-a4059904ca1ecb03c): tier 0 + upper tiers, Verlet step.
  Tier0ExtAdapter   agent E's ScatterField with scripts/scatter_tier0.attach_tier0 (branch worktree-agent-a96914a7f2c210071): feeds the
                    external-field expansion and boundary channels from the harness ctx when the variant has them.
  TiledAdapter      scripts/scatter_tiling.TiledScatter (ghosts + field exchange + coarse far term; arm B of the hole-9 comparison).
  WideAdapter       Exp E weights on one dense wide grid (arm A of the hole-9 comparison).
Specs for scripts/tiered_eval.py --model: tiered:<nets.pt>[,<tier0.pt>]  t0ext:<ckpt>  tiled:<ckpt>[,nt=4,tile=32,halo=16,far=1]  wide:<ckpt>[,grid=256,extent=128]"""
import types

import torch

from scripts import tiered_iface as ti
from scripts import tiered_truth as tt


class TieredAdapter(ti.TierModel):
    tier = 1
    laws = frozenset({"grav"})
    caps = frozenset({"occ"})

    def __init__(self, field, name="tiered", dtype=torch.float32):
        self.f, self.name, self.dtype = field, name, dtype
        self.dt, self.reach = float(field.dt), float(field.n0 * field.h0)
        self.cell = float(field.h0)

    def init_state(self, B, device):
        return [self.f.init_state(B, device, self.dtype), None]

    def _c(self, *xs):
        return [x.to(self.dtype) for x in xs]

    @torch.no_grad()
    def force(self, pos, vel, mass, mask, state, ctx=None):
        p, v, m, k = self._c(pos, vel, mass, mask)
        a, st, _ = self.f.force(p, v, m, k, state[0])
        return a.to(pos.dtype), [st, state[1]]

    @torch.no_grad()
    def step(self, pos, vel, mass, mask, state, ctx=None):
        p, v, m, k = self._c(pos, vel, mass, mask)
        p2, v2, st, a = self.f.step_verlet(p, v, m, k, state[0], state[1])
        return p2.to(pos.dtype), v2.to(pos.dtype), [st, a]

    @torch.no_grad()
    def step_multirate(self, pos, vel, mass, mask, state, m_ratio):
        """RESPA macro step = m_ratio velocity-Verlet tier-0 steps with slow (upper tier) half-kicks of m_ratio*dt around them; advances m_ratio * dt.
        Own inner integrator: TieredField.step_respa's inner loop is not Verlet (v updated with a(t) only), which confounds multi-rate with the integrator."""
        p, v, m, k = self._c(pos, vel, mass, mask)
        f, dt = self.f, self.f.dt
        s0, hid, coefs = state[0]
        a_slow, (hid, coefs, _) = f.slow_force(p, v, m, k, (s0, hid, coefs))
        v = v + 0.5 * m_ratio * dt * a_slow
        a0 = state[1]
        if a0 is None:
            a0, s0 = f.tier0.force(p, v, m, k, s0)
            a0 = a0 * k[..., None]
        for _ in range(m_ratio):
            v = v + 0.5 * dt * a0
            p = p + dt * v
            a0, s0 = f.tier0.force(p, v, m, k, s0)
            a0 = a0 * k[..., None]
            v = v + 0.5 * dt * a0
        a_slow, (hid, coefs, _) = f.slow_force(p, v, m, k, (s0, hid, coefs))
        v = v + 0.5 * m_ratio * dt * a_slow
        return p.to(pos.dtype), v.to(pos.dtype), [(s0, hid, coefs), a0]

    def tier0_only(self):
        return Tier0Only(self.f.tier0, self.dtype, self.dt, self.reach, self.name + "/tier0")


class Tier0Only(ti.TierModel):
    laws = frozenset({"grav"})
    caps = frozenset()

    def __init__(self, t0, dtype, dt, reach, name):
        self.t0, self.dtype, self.dt, self.reach, self.name = t0, dtype, dt, reach, name

    def init_state(self, B, device):
        return self.t0.init_state(B, device, self.dtype)

    @torch.no_grad()
    def force(self, pos, vel, mass, mask, state, ctx=None):
        a, st = self.t0.force(pos.to(self.dtype), vel.to(self.dtype), mass.to(self.dtype), mask.to(self.dtype), state)
        return a.to(pos.dtype), st


def load_tiered(spec, device):
    """spec: '<nets.pt>[,<tier0.pt>]'; nets.pt is a checkpoint written by scripts/train_tiered.py (keys nets, args)."""
    from scripts import train_tiered as ttr

    parts = spec.split(",")
    st = torch.load(parts[0], map_location=device, weights_only=False)
    args = types.SimpleNamespace(**{**dict(order3=False, latent=0, err=False, learned_pool=False, residual=False, out_scale=1.0), **st["args"]})
    ttr.ORDER3[0] = bool(getattr(args, "order3", False))
    t0 = ttr.load_tier0(parts[1] if len(parts) > 1 else args.tier0, device)
    if len(parts) == 1 and "tier0" in st:
        t0.model.load_state_dict(st["tier0"])
    nets = ttr.make_nets(args.tiers, args, device)
    nets.load_state_dict(st["nets"])
    field = ttr.make_field(t0, nets).to(device).eval()
    return TieredAdapter(field, f"tiered{args.tiers}")


class Tier0ExtAdapter(ti.TierModel):
    laws = frozenset({"grav"})

    def __init__(self, model, name="t0ext", dtype=torch.float32):
        self.m, self.name, self.dtype = model, name, dtype
        self.cfg = model.t0.cfg
        self.dt, self.reach, self.cell = float(model.dt), float(model.extent), float(model.h)
        caps = set()
        if self.cfg.extapply:
            caps.add("ext")
        if self.cfg.bnd:
            caps.add("occ")
        if self.cfg.boundary == "periodic":
            caps.add("periodic")
        self.caps = frozenset(caps)

    def init_state(self, B, device):
        return self.m.init_field(B, device, self.dtype)

    def _feed(self, B, ctx):
        """External expansion (ExtField: a = a0 + J x, so g = -a0, H = -J as (Hxx,Hxy,Hyy)) and boundary channels from the harness ctx."""
        t0, cfg = self.m.t0, self.cfg
        ext = (ctx or {}).get("ext")
        if cfg.ext:
            if ext is None:
                z = torch.zeros(B, 2, device=next(self.m.parameters()).device, dtype=self.dtype)
                t0.set_external(self.m, g=z, H=torch.zeros(B, 3, device=z.device, dtype=self.dtype), phi0=z[:, 0])
            else:
                t = (ctx or {}).get("t", 0.0)
                s = (1 + ext.rate * t)[:, None]
                g = (-ext.a0 * s).to(self.dtype)
                J = ext.J * s[..., None]
                H = (-torch.stack([J[:, 0, 0], 0.5 * (J[:, 0, 1] + J[:, 1, 0]), J[:, 1, 1]], -1)).to(self.dtype)
                t0.set_external(self.m, phi0=torch.zeros(B, device=g.device, dtype=self.dtype), g=g, H=H)
        if cfg.bnd:
            bounds = (ctx or {}).get("bound") or [dict(kind="none")] * B
            edges = [b["kind"] if b["kind"] in ("periodic", "wall") else "open" for b in bounds]
            dev = next(self.m.parameters()).device
            sdf = None
            if any(b["kind"] == "wall" for b in bounds):
                G, h, E = self.m.grid, self.m.h, self.m.extent
                xs = (torch.arange(G, device=dev, dtype=torch.float32) + 0.5) * h - E / 2
                Y, X = torch.meshgrid(xs, xs, indexing="ij")
                sd = []
                for b in bounds:
                    if b["kind"] == "wall":
                        lo, hi = [float(x) for x in b["lo"]], [float(x) for x in b["hi"]]
                        sd.append(torch.minimum(torch.minimum(X - lo[0], hi[0] - X), torch.minimum(Y - lo[1], hi[1] - Y)))
                    else:
                        sd.append(torch.full_like(X, 1e3))
                sdf = torch.stack(sd)[:, None]
            t0.set_boundary(self.m, sdf=sdf, edge=edges)

    def _c(self, *xs):
        return [x.to(self.dtype) for x in xs]

    @torch.no_grad()
    def force(self, pos, vel, mass, mask, state, ctx=None):
        self._feed(pos.shape[0], ctx)
        p, v, m, k = self._c(pos, vel, mass, mask)
        f = state[0] if isinstance(state, tuple) else state
        store = (ctx or {}).get("ph_store")
        if store is not None and self.cfg.phid:
            self.m.t0.set_ids(ctx["ids"], ctx["n_global"], ctx.get("write"))
            try:
                a, (f2, store2) = self.m.force(p, v, m, k, (f, store))
            finally:
                self.m.t0.set_ids(None, 0)
            store.copy_(store2)
        else:
            a, f2 = self.m.force(p, v, m, k, f)
        return a.to(pos.dtype), (f2, state[1]) if isinstance(state, tuple) else f2

    @torch.no_grad()
    def step(self, pos, vel, mass, mask, state, ctx=None):
        from scripts import scatter_field as sf

        self._feed(pos.shape[0], ctx)
        p, v, m, k = self._c(pos, vel, mass, mask)
        p2, v2, _, st, _ = sf._step_m(self.m, p, v, m, k, state)
        return p2.to(pos.dtype), v2.to(pos.dtype), st


def load_t0ext(ckpt, device, dt=0.1):
    """ckpt written by scripts/train_tier0_ext.py (keys model, t0)."""
    from scripts import scatter_tier0 as t0m
    from scripts import train_scatter_field as tsf

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", dt).to(device)
    t0m.attach_tier0(model, ck["t0"])
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)
    return Tier0ExtAdapter(model, f"t0ext[{ck['t0']}]")


class WideAdapter(ti.ScatterAdapter):
    """Arm A: the Exp E weights evaluated on one dense wide grid (default 256 cells at h=0.5, extent 128)."""

    def __init__(self, model, name):
        super().__init__(model, name)
        self.tier = 0


def load_wide(spec, device, dt=0.1):
    from scripts import scatter_field as sf

    parts = spec.split(",")
    kw = dict(p.split("=") for p in parts[1:])
    grid, extent = int(kw.get("grid", 256)), float(kw.get("extent", 128.0))
    m = sf.ScatterField(grid=grid, extent=extent, net="unet", recurrent=True, hidden_ch=0, momfix=True, dt=dt, in_scale=1.0, potential=True,
                        kernel=True, pp=2.0, split=True, verlet=True).to(device)
    m.load_state_dict(torch.load(parts[0], map_location=device, weights_only=False)["model"])
    m.eval().requires_grad_(False)
    return WideAdapter(m, f"wide{grid}")


class TiledAdapter(ti.TierModel):
    """Arm B: scripts/scatter_tiling.TiledScatter over an nt x nt tile grid (one scene at a time; the tiler is unbatched)."""
    laws = frozenset({"grav"})
    caps = frozenset()
    tier = 0

    def __init__(self, tiler, name="tiledfar"):
        self.t, self.name = tiler, name
        self.dt, self.reach, self.cell = float(tiler.m.dt), float(tiler.nt * tiler.Lt), float(tiler.h)

    def init_state(self, B, device):
        return [(self.t.init_field(device), None) for _ in range(B)]

    @torch.no_grad()
    def force(self, pos, vel, mass, mask, state, ctx=None):
        out = torch.zeros_like(pos)
        for b in range(pos.shape[0]):
            k = mask[b] > 0
            a, f = self.t.force(pos[b, k].float(), vel[b, k].float(), mass[b, k].float(), state[b][0])
            out[b, k] = a.to(pos.dtype)
            state[b] = (f, state[b][1])
        return out, state

    @torch.no_grad()
    def step(self, pos, vel, mass, mask, state, ctx=None):
        P, V = pos.clone(), vel.clone()
        for b in range(pos.shape[0]):
            k = mask[b] > 0
            p, v, st = self.t.step(pos[b, k].float(), vel[b, k].float(), mass[b, k].float(), state[b])
            P[b, k], V[b, k] = p.to(pos.dtype), v.to(pos.dtype)
            state[b] = st
        return P, V, state


def load_tiled(spec, device, dt=0.1):
    from scripts import scatter_scenes as ss
    from scripts import scatter_tiling as stl

    parts = spec.split(",")
    kw = dict(p.split("=") for p in parts[1:])
    m = ss.build_model("E", dt, device, parts[0])
    m.opt = set(m.opt)
    t = stl.TiledScatter(m, int(kw.get("nt", 4)), float(kw.get("tile", 32)), float(kw.get("halo", 16)), ghosts=True, field_x=True,
                         far=bool(int(kw.get("far", 1))), far_c=int(kw.get("far_c", 4)))
    return TiledAdapter(t, f"tiled{kw.get('nt', 4)}x{kw.get('tile', 32)}{'+far' if int(kw.get('far', 1)) else ''}")
