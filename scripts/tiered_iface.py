"""Minimal model interface for the tiered-architecture eval harness, adapters for existing scatter-field checkpoints, a tiled
wrapper (tile + halo windows, optional sparse activation), and frozen-lower-tier hooks for tier>=1 data.

Interface (any object with these attributes/methods works; nothing in the model code is imported by the harness):
  name: str          dt: float (model step)        reach: float (window extent the model sees, sim units)
  tier: int          caps: set of {"ext", "occ", "periodic", "error", "multirate"} (inputs the model consumes itself)
  init_state(B, device) -> state
  force(pos, vel, mass, mask, state, ctx) -> (acc (B,N,2), state)      near-field acceleration (optional for step-only models)
  step(pos, vel, mass, mask, state, ctx) -> (pos, vel, state)          one model step of length dt
  [optional] error(pos, vel, mass, mask, state, ctx) -> (B,N) predicted per-particle force error (for the calibration group)
ctx is a dict: "ext" (tiered_truth.ExtField), "t" (float), "occ" (B,3,32,32), "bound" (list of dicts), "period" (B,2), "phys".
Positions are centred; the model window is [-reach/2, reach/2)^2. Tensors are fp64 on entry; adapters cast as needed."""
import math

import torch

from scripts import tiered_truth as tt


class TierModel:
    name, dt, reach, tier, caps = "base", 0.1, 64.0, 0, frozenset()
    laws = frozenset({"grav", "lj", "contact", "bonded"})

    def init_state(self, B, device):
        return None

    def force(self, pos, vel, mass, mask, state, ctx=None):
        raise NotImplementedError

    def step(self, pos, vel, mass, mask, state, ctx=None):
        a, state = self.force(pos, vel, mass, mask, state, ctx)
        dv = a * self.dt
        return pos + self.dt * (vel + 0.5 * dv), vel + dv, state


def advance(model, pos, vel, mass, mask, state, ctx):
    """One model step plus the harness-side external-field kick for models without the "ext" capability."""
    ctx = ctx or {}
    ext = ctx.get("ext")
    if ext is None or "ext" in model.caps:
        return model.step(pos, vel, mass, mask, state, ctx)
    t = ctx.get("t", 0.0)
    dt = model.dt
    a0 = ext.acc(pos, t) * mask[..., None]
    p, v, state = model.step(pos, vel, mass, mask, state, ctx)
    a1 = ext.acc(p, t + dt) * mask[..., None]
    return p + 0.5 * dt * dt * a0, v + 0.5 * dt * (a0 + a1), state


class ConstVel(TierModel):
    name = "constvel"

    def __init__(self, dt=0.1):
        self.dt, self.reach = dt, float("inf")

    def force(self, pos, vel, mass, mask, state, ctx=None):
        return torch.zeros_like(pos), state


class Oracle(TierModel):
    """Truth near-field law (ctx["phys"]) integrated with the harness truth integrator: harness self-check, err ~ 0."""
    name = "oracle"
    caps = frozenset({"ext", "occ", "periodic"})

    def __init__(self, dt=0.1, sub=4):
        self.dt, self.sub, self.reach = dt, sub, float("inf")

    def force(self, pos, vel, mass, mask, state, ctx=None):
        return ctx["phys"].acc(pos, vel, mass, mask), state

    def step(self, pos, vel, mass, mask, state, ctx=None):
        ext = ctx.get("ext")
        t = ctx.get("t", 0.0)
        sub = max(1, int(round(self.dt / ctx["h"]))) if ctx.get("h") else self.sub
        r = tt.rollout(ctx["phys"], pos, vel, mass, mask, 1, self.dt, sub, ext=ext, t0=t)
        return r["pos"][:, 1], r["vel"][:, 1], state


class ScatterAdapter(TierModel):
    """Existing scripts.scatter_field.ScatterField (fp32) behind the interface. No ext/occ/periodic inputs: the harness adds the
    external kick and the model sees only particles inside its window (tokens outside get zero force, as in the model)."""

    def __init__(self, model, name, dtype=torch.float32, warm=3):
        self.m, self.name, self.dtype, self.warm = model, name, dtype, warm
        self.laws = frozenset({"grav"})
        self.cell = float(model.h)
        self.dt, self.reach = float(model.dt), float(model.extent)
        self.caps = frozenset()

    def init_state(self, B, device):
        return self.m.init_field(B, device, self.dtype)

    def _c(self, *xs):
        return [x.to(self.dtype) for x in xs]

    @torch.no_grad()
    def force(self, pos, vel, mass, mask, state, ctx=None):
        """Static-configuration force. The model's input includes its previous output field, which is zero on the first call; for
        batch members whose incoming field is all zero the field is iterated `warm` extra times at fixed positions so force-level
        groups see the settled field the model has during a rollout."""
        p, v, m, k = self._c(pos, vel, mass, mask)
        f = state[0] if isinstance(state, tuple) else state
        a, f2 = self.m.force(p, v, m, k, f)
        if self.warm:
            fresh = (f.abs().amax((1, 2, 3)) == 0)
            if bool(fresh.any()):
                fw = f2
                for _ in range(self.warm):
                    aw, fw = self.m.force(p, v, m, k, fw)
                sel = fresh[:, None, None, None]
                a = torch.where(fresh[:, None, None], aw, a)
                f2 = torch.where(sel, fw, f2)
        a = a.to(pos.dtype)
        return (a, (f2, state[1])) if isinstance(state, tuple) else (a, f2)

    @torch.no_grad()
    def step(self, pos, vel, mass, mask, state, ctx=None):
        p, v, m, k = self._c(pos, vel, mass, mask)
        p2, v2, st, _ = self.m.step(p, v, m, k, state)
        return p2.to(pos.dtype), v2.to(pos.dtype), st


def load_scatter(ckpt, device, dt=0.1, name=None):
    from scripts import scatter_scenes as ss

    model = ss.build_model("E", dt, device, ckpt)
    return ScatterAdapter(model, name or "scatterE")


class Tiled(TierModel):
    """Applies a window model (reach = tile + 2*halo) over a domain [-span/2, span/2)^2 split into tiles; each particle takes the
    acceleration from the tile that owns it, halo copies are read-only. Per-tile recurrent field persists across steps. `sparse`
    activates only tiles whose window holds particles (missing tiles contribute nothing, as zero-mass halo)."""

    def __init__(self, base, span, tile=32.0, sparse=False, persist=True, refresh=False, batched=False, bucket=False):
        self.base, self.span, self.tile, self.sparse = base, float(span), float(tile), sparse
        self.persist, self.refresh, self.batched, self.bucket = persist, refresh, batched, bucket
        self.dt, self.reach, self.caps, self.laws = base.dt, float(span), base.caps, base.laws
        self.name = f"{base.name}+tiled{int(tile)}{'s' if sparse else ''}"
        n = max(1, int(round(span / tile)))
        self.nt = n
        c = (torch.arange(n, dtype=torch.float64) + 0.5) * tile - n * tile / 2
        self.cen = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
        self.last_active = 0.0
        self.last_tiles = n * n

    def init_state(self, B, device):
        return dict(fields=[None] * B, aprev=None, ften=[None] * B)

    def _owner(self, pos):
        n, T = self.nt, self.tile
        ij = ((pos + n * T / 2) / T).floor().clamp(0, n - 1).long()
        inside = ((pos >= -n * T / 2) & (pos < n * T / 2)).all(-1)
        return ij[..., 0] * n + ij[..., 1], inside

    def _force_one(self, b, pos, vel, mass, mask, state):
        """pos (N,2) for one scene -> acc (N,2); updates state["fields"][b]."""
        dev = pos.device
        cen = self.cen.to(dev)
        win = self.base.reach / 2
        own, inside = self._owner(pos)
        live = (mask > 0) & inside
        inwin = ((pos[None] - cen[:, None]).abs().amax(-1) < win) & live[None]
        cnt = inwin.sum(1)
        act = (cnt > 0) if self.sparse else torch.ones_like(cnt, dtype=torch.bool)
        ids = act.nonzero()[:, 0]
        self.last_active, self.last_tiles = float(len(ids)), len(cen)
        out = torch.zeros_like(pos)
        if len(ids) == 0:
            return out
        M = max(1, int(cnt[ids].max()))
        if self.bucket:
            M = min(pos.shape[0], 1 << (M - 1).bit_length())
            M = max(M, min(16, pos.shape[0]))
        order = torch.argsort((~inwin[ids]).to(torch.int8), dim=1, stable=True)[:, :M]
        ok = torch.gather(inwin[ids], 1, order)
        loc = pos[order] - cen[ids][:, None, :]
        proto0 = self.base.init_state(1, dev)
        f00 = proto0[0] if isinstance(proto0, tuple) else proto0
        if self.batched and not self.refresh and torch.is_tensor(f00):
            K = len(ids)
            if self.bucket:
                K = 1 << (K - 1).bit_length()
            ften = state["ften"][b]
            if ften is None:
                ften = state["ften"][b] = f00.new_zeros(len(cen), *f00.shape[1:])
            pad = K - len(ids)
            if pad:
                order = torch.cat([order, order.new_zeros(pad, M)])
                ok = torch.cat([ok, ok.new_zeros(pad, M)])
                ids_p = torch.cat([ids, ids.new_zeros(pad)])
            else:
                ids_p = ids
            loc = pos[order] - cen[ids_p][:, None, :]
            st = ften[ids_p]
            if pad:
                st = torch.cat([st[:len(ids)], torch.zeros_like(st[len(ids):])])
            a, st = self.base.force(loc, vel[order], mass[order], ok.to(pos.dtype), st, None)
            if self.persist:
                ften[ids] = st[:len(ids)]
            mine = ok & (own[order] == ids_p[:, None])
            mine[len(ids):] = False
            out.index_add_(0, order[mine], a[mine])
            return out
        fields = state["fields"][b]
        if fields is None:
            fields = state["fields"][b] = {}
        proto = self.base.init_state(1, dev)
        f0 = proto[0] if isinstance(proto, tuple) else proto
        if f0 is not None and not torch.is_tensor(f0):
            outs = []
            for j, i in enumerate(ids.tolist()):
                sti = fields.get(int(i)) or self.base.init_state(1, dev)
                aj, sti = self.base.force(loc[j:j + 1], vel[order[j]][None], mass[order[j]][None], ok[j:j + 1].to(pos.dtype), sti, None)
                fields[int(i)] = sti
                outs.append(aj)
            a = torch.cat(outs)
            mine = ok & (own[order] == ids[:, None])
            out.index_add_(0, order[mine], a[mine])
            if not self.persist:
                fields.clear()
            return out
        st = None if f0 is None else torch.cat([fields[int(i)] if int(i) in fields else torch.zeros_like(f0) for i in ids.tolist()])
        a, st = self.base.force(loc, vel[order], mass[order], ok.to(pos.dtype), st, None)
        if st is not None:
            for j, i in enumerate(ids.tolist()):
                fields[int(i)] = st[j:j + 1]
        mine = ok & (own[order] == ids[:, None])
        out.index_add_(0, order[mine], a[mine])
        if not self.persist:
            fields.clear()
        return out

    def _refresh(self, fields):
        """Overwrite every tile's halo cells of the recurrent field with the neighbouring tiles' interior cells (hole 7)."""
        if not fields:
            return
        n, G = self.nt, self.base.m.grid
        c = int(round(G * self.tile / self.base.reach))
        hw = (G - c) // 2
        ref = next(iter(fields.values()))
        glob = ref.new_zeros(1, ref.shape[1], n * c + 2 * hw, n * c + 2 * hw)
        for t, f in fields.items():
            ix, iy = divmod(t, n)
            glob[:, :, hw + iy * c:hw + (iy + 1) * c, hw + ix * c:hw + (ix + 1) * c] = f[:, :, hw:hw + c, hw:hw + c]
        for t in list(fields):
            ix, iy = divmod(t, n)
            new = glob[:, :, iy * c:iy * c + G, ix * c:ix * c + G].clone()
            new[:, :, hw:hw + c, hw:hw + c] = fields[t][:, :, hw:hw + c, hw:hw + c]
            fields[t] = new

    def force(self, pos, vel, mass, mask, state, ctx=None):
        B = pos.shape[0]
        if state is None or not isinstance(state, dict):
            state = self.init_state(B, pos.device)
        a = torch.stack([self._force_one(b, pos[b], vel[b], mass[b], mask[b], state) for b in range(B)])
        if self.refresh and self.persist:
            for b in range(B):
                if state["fields"][b]:
                    self._refresh(state["fields"][b])
        return a * mask[..., None], state

    def step(self, pos, vel, mass, mask, state, ctx=None):
        if state is None:
            state = self.init_state(pos.shape[0], pos.device)
        dt = self.dt
        if getattr(self.base, "m", None) is not None and getattr(self.base.m, "verlet", False):
            if state["aprev"] is None:
                state["aprev"] = self.force(pos, vel, mass, mask, state)[0]
            ap = state["aprev"]
            p2 = pos + dt * vel + 0.5 * dt * dt * ap
            an, _ = self.force(p2, vel + dt * ap, mass, mask, state)
            state["aprev"] = an
            return p2, vel + 0.5 * dt * (ap + an), state
        a, _ = self.force(pos, vel, mass, mask, state)
        dv = a * dt
        return pos + dt * (vel + 0.5 * dv), vel + dv, state


class Lower:
    """Frozen lower-tier hook for tier>=1 data: __call__(pos, vel, mass, mask) -> acc (B,N,2)."""
    name = "lower"

    def __call__(self, pos, vel, mass, mask):
        raise NotImplementedError


class ZeroLower(Lower):
    name = "zero"

    def __call__(self, pos, vel, mass, mask):
        return torch.zeros_like(pos)


class RangeLower(Lower):
    """Oracle near tier: exact softened-gravity pairs within rc only (target = far part beyond rc)."""

    def __init__(self, rc, eps=tt.EPS):
        self.rc, self.eps, self.name = rc, eps, f"range{rc:g}"

    def __call__(self, pos, vel, mass, mask):
        out = torch.zeros_like(pos)
        mm = mass * mask
        for i0, i1, d, r2, ok in tt._pair_iter(pos, mask):
            ok = ok & (r2 < self.rc ** 2)
            w = torch.where(ok, (r2 + self.eps ** 2) ** -1.5, torch.zeros_like(r2)) * mm[:, None, :]
            out[:, i0:i1] = (d * w[..., None]).sum(2)
        return out * mask[..., None]


class ModelLower(Lower):
    """A tier-0 model applied memoryless over the scene with tiles (fresh recurrent state every call)."""

    def __init__(self, model, span, sparse=True):
        self.model = model if isinstance(model, Tiled) else Tiled(model, span, sparse=sparse)
        self.name = f"model:{self.model.name}"

    def __call__(self, pos, vel, mass, mask):
        st = self.model.init_state(pos.shape[0], pos.device)
        return self.model.force(pos, vel, mass, mask, st)[0]
