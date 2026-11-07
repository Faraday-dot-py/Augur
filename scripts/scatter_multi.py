"""Mixed-interaction scatter field for the near-field retrain (docs/debugging/scatter-nearfield-retrain.md).

MultiScatterField = Exp E architecture (ms_kp_pot_v_g128: U-Net on a G=128 grid over extent 64, FFT kernel, pp=2 knn-16 pair term,
potential head, Verlet) with two species and a law descriptor:
  * bodies carry a species bit typ in {0, 1}; the mass density is scattered per species (channels m, m1);
  * the interaction is a 2x2 matrix of ordered-pair laws F_ab (receiver a, source b); theta (DC = 20 floats = 4 slots x 5, the
    scatter_far_general.theta_of encoding, zeros for an absent slot) is a constant input of the kernel MLP, the pair MLP and the U-Net;
  * kernel: K_ab(r) = MLP(r, log(r + .5), theta) -> 4 channels, one FFT convolution per ordered pair; pair term likewise (4 channels);
  * the U-Net outputs one potential per receiver species; a token reads the field of its own species;
  * no momentum fix (non-reciprocal laws do not conserve momentum).
With typ None (single species) the model is the same function with species 1 absent, so tiled / single-grid code can drive it.
Warm start from an Exp E checkpoint (`init_from_e`) reproduces E's gravity force at init for theta of any value.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import scatter_field as sf
from scripts.scatter_far_general import LAWS, theta_of

DC = 20
NR = 2048
GRID, EXTENT, DT, EPS = 128, 64.0, 0.1, 0.5
N_PAD = 300
CAP = 20.0
CIN = 3 + 1 + 4 + 2 + DC


class CondMLP(nn.Module):
    def __init__(self, dout, hidden=128):
        super().__init__()
        self.l1 = nn.Linear(2 + DC, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, dout)
        nn.init.zeros_(self.l3.weight)
        nn.init.zeros_(self.l3.bias)

    def forward(self, f, cond):
        c = cond.expand(f.shape[:-1] + (DC,))
        h = F.gelu(self.l1(torch.cat([f, c], -1)))
        return self.l3(F.gelu(self.l2(h)))

    def load_e(self, seq):
        """embed E's 2-64-64-1 MLP (Sequential keys 0, 2, 4) as output channel 0; extra units start isolated with zero output weight."""
        with torch.no_grad():
            H = self.l1.out_features
            self.l1.weight[:64, 2:] = 0.0
            self.l1.weight[:64, :2] = seq[0].weight
            self.l1.bias[:64] = seq[0].bias
            self.l2.weight[:64, 64:] = 0.0
            self.l2.weight[64:, :64] = 0.0
            self.l2.weight[:64, :64] = seq[2].weight
            self.l2.bias[:64] = seq[2].bias
            self.l3.weight.zero_()
            self.l3.bias.zero_()
            self.l3.weight[0, :64] = seq[4].weight[0]
            self.l3.bias[0] = seq[4].bias[0]


class PairNet(nn.Module):
    """pair MLP; forward(x) (x = (r, log(r + .05)) features) returns species-pair channel 00 for the single-species code paths."""

    def __init__(self):
        super().__init__()
        self.mlp = CondMLP(4)
        self.cond = None

    def cond_for(self, x):
        c = self.cond
        if c.shape[0] > 1:
            return c.view(c.shape[0], *([1] * (x.dim() - 2)), DC)
        return c[0]

    def forward(self, x):
        return self.mlp(x, self.cond_for(x))[..., :1]

    def forward_all(self, x):
        return self.mlp(x, self.cond_for(x))


class MultiScatterField(sf.ScatterField):
    def __init__(self, dt=DT, hidden=128):
        super().__init__(grid=GRID, extent=EXTENT, net="unet", recurrent=True, hidden_ch=0, dt=dt, momfix=False, in_scale=1.0,
                         potential=True, kernel=True, pp=2.0, split=True, verlet=True)
        self.cf = 2
        self.net = sf.UNet(CIN, 2, 32, 5)
        nn.init.zeros_(self.net.out.weight)
        nn.init.zeros_(self.net.out.bias)
        self.kmlp = CondMLP(4, hidden)
        self.ppmlp = PairNet()
        self.ppmlp.mlp = CondMLP(4, hidden)
        self.cond = torch.zeros(1, DC)
        self.typ = None

    def set_law(self, cond, typ=None):
        self.cond = cond
        self.typ = typ
        self.ppmlp.cond = cond

    def init_from_e(self, sd):
        """sd: state dict of an Exp E (ms_kp_pot_v_g128) ScatterField."""
        with torch.no_grad():
            src = sf.ScatterField(grid=GRID, extent=EXTENT, net="unet", recurrent=True, hidden_ch=0, dt=self.dt, momfix=True, in_scale=1.0,
                                  potential=True, kernel=True, pp=2.0, split=True, verlet=True)
            src.load_state_dict(sd)
            n_new, n_old = self.net.state_dict(), src.net.state_dict()
            for k, v in n_old.items():
                if k == "inp.weight":
                    w = torch.zeros_like(n_new[k])
                    w[:, 0:3] = v[:, 0:3]
                    w[:, 4:6] = v[:, 3:5]
                    w[:, 8:9] = v[:, 5:6]
                    n_new[k].copy_(w)
                elif k in ("out.weight", "out.bias"):
                    n_new[k].copy_(v.expand_as(n_new[k]) if v.shape[0] == 1 else v)
                else:
                    n_new[k].copy_(v)
            self.kmlp.load_e(src.kmlp)
            self.ppmlp.mlp.load_e(src.ppmlp)

    def _rgrid(self, ref):
        G = self.grid
        idx = torch.arange(2 * G, device=ref.device, dtype=ref.dtype)
        d = torch.minimum(idx, 2 * G - idx) * self.h
        return torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2), math.sqrt(2.0) * G * float(self.h) * (1 + 1e-6)

    def _ktab(self, rmax, dev, dtype):
        r = torch.linspace(0, rmax, NR, device=dev, dtype=dtype)
        f = torch.stack([r, torch.log(r + 0.5)], -1)
        c = self.cond.to(dev, dtype)
        out = self.kmlp(f[None].expand(c.shape[0], NR, 2), c[:, None, :])
        sp = 1 - (1 - (r / self.pp).clamp(max=1.0) ** 2) ** 2
        return (out * sp[None, :, None]).permute(0, 2, 1)

    def _kgrid(self, ref):
        r, rmax = self._rgrid(ref)
        tab = self._ktab(rmax, ref.device, ref.dtype)
        u = (r / rmax * (NR - 1)).clamp(0, NR - 1 - 1e-4)
        i0 = u.floor().long()
        w = u - i0
        return tab[..., i0] * (1 - w) + tab[..., i0 + 1] * w

    def _kernel_fft(self, rho):
        return torch.fft.rfft2(self._kgrid(rho)[0, 0])

    def kernel_acc_multi(self, rho0, rho1):
        G = self.grid
        Kf = torch.fft.rfft2(self._kgrid(rho0))
        r0 = torch.fft.rfft2(rho0, s=(2 * G, 2 * G))
        r1 = torch.fft.rfft2(rho1, s=(2 * G, 2 * G))
        phis = [torch.fft.irfft2(Kf[:, 2 * a:2 * a + 1] * r0 + Kf[:, 2 * a + 1:2 * a + 2] * r1, s=(2 * G, 2 * G))[..., :G, :G] * self.h ** 2
                for a in (0, 1)]
        return torch.cat([self.neg_grad(p) for p in phis], 1)

    def pp_acc(self, pos, mass, mask, knn=16, qmask=None):
        typ = self.typ
        if typ is None:
            return super().pp_acc(pos, mass, mask, knn, qmask)
        B, N = mask.shape
        k = min(knn, N - 1)
        d_all = pos[:, None, :, :] - pos[:, :, None, :]
        r_all = torch.sqrt((d_all ** 2).sum(-1) + 1e-8)
        bad = (1 - mask)[:, None, :] + torch.eye(N, device=pos.device, dtype=pos.dtype)[None]
        r_sel, idx = torch.topk(r_all + 1e6 * (bad > 0).to(pos.dtype), k, dim=-1, largest=False)
        d = torch.gather(d_all, 2, idx[..., None].expand(B, N, k, 2))
        mj = torch.gather((mass * mask)[:, None, :].expand(B, N, N), 2, idx)
        tj = torch.gather(typ[:, None, :].expand(B, N, N), 2, idx)
        ok = (r_sel < 1e5).to(pos.dtype) * mask[:, :, None]
        r = r_sel.clamp(max=1e5)
        gall = self.ppmlp.forward_all(torch.stack([r, torch.log(r + 0.05)], -1))
        sl = (2 * typ[:, :, None] + tj).long()
        g = torch.gather(gall, -1, sl[..., None])[..., 0]
        win = (1 - (r / self.pp).clamp(max=1.0) ** 2) ** 2
        w = g * win * mj * ok
        return (w[..., None] * d / r[..., None]).sum(2)

    def _force(self, pos, vel, mass, mask, field):
        typ = self.typ
        B = pos.shape[0]
        G = self.grid
        x = self.scatter(pos, vel, mass, mask)
        rho = x[:, :1]
        if typ is None:
            m1 = torch.zeros_like(rho)
            has1 = rho.new_zeros(B, 1, 1, 1)
        else:
            m1 = self.scatter(pos, vel, mass * typ, mask)[:, :1]
            has1 = (typ * mask).amax(1).view(-1, 1, 1, 1)
        ak = self.kernel_acc_multi(rho - m1, m1)
        ak = torch.cat([ak[:, :2], ak[:, 2:] * has1], 1)
        cmap = self.cond.to(pos.device, pos.dtype).view(-1, DC, 1, 1).expand(B, DC, G, G)
        field = self.net(torch.cat([x, m1, ak, field, cmap], 1))
        acc = torch.cat([self.neg_grad(field[:, :1]), self.neg_grad(field[:, 1:2])], 1) + ak
        a_tok = self.gather(acc[:, :2], pos)
        if typ is not None:
            a_tok = a_tok + typ[..., None] * (self.gather(acc[:, 2:], pos) - a_tok)
        a_tok = a_tok + self.pp_acc(pos, mass, mask)
        return a_tok * mask[..., None], field


def build(dt=DT):
    return MultiScatterField(dt)


# ---------------------------------------------------------------- laws and scenes

TRAIN_SHAPES = ["pow1", "pow2", "pow3", "yuk8", "yuk128", "lj"]
HELD_SHAPES = ["pow1.5", "yuk32"]
STRUCT_W = {"single": 2.0, "charge": 1.0, "predprey": 1.0}
PP_SHAPES = ("pow1", "pow2", "pow3", "pow1.5")


def allowed(shape):
    if shape == "lj":
        return ["single"]
    return ["single", "charge"] + (["predprey"] if shape in PP_SHAPES else [])


def slots_for(structure, shape, s):
    """4 ordered-pair slots 2a+b -> (shape, scale) or None. Charges: like repels, unlike attracts; predator(1)-prey(0): pred->prey repels,
    prey->pred attracts at half strength."""
    if structure == "single":
        return [(shape, s), None, None, None]
    if structure == "charge":
        return [(shape, -s), (shape, s), (shape, s), (shape, -s)]
    return [None, (shape, -s), (shape, 0.5 * s), None]


def cond_of(slots):
    out = []
    for sl in slots:
        out += [0.0] * 5 if sl is None else theta_of(LAWS[sl[0]], sl[1])
    return out


def params_of(slots):
    """(7, 4) fp64 rows a1 p1 a2 p2 il eps lj for the 4 slots."""
    out = np.zeros((7, 4))
    for j, sl in enumerate(slots):
        if sl is None:
            out[:, j] = (0.0, 2.0, 0.0, 2.0, 0.0, EPS, 0.0)
        else:
            a1, p1, a2, p2, il, eps, _ = LAWS[sl[0]]
            out[:, j] = (a1 * sl[1], p1, a2 * sl[1], p2, il, eps, 1.0 if sl[0] == "lj" else 0.0)
    return out


@torch.no_grad()
def accel(pos, pp, mk):
    d = pos[:, None, :, :] - pos[:, :, None, :]
    r = torch.sqrt((d ** 2).sum(-1))
    lj = pp[6] > 0.5
    rl = torch.where(lj, r.clamp_min(1.0), r)
    den = rl * rl + pp[5] ** 2
    coef = (pp[0] * den.pow(-(pp[1] + 1) / 2) + pp[2] * den.pow(-(pp[3] + 1) / 2)) * torch.exp(-rl * pp[4])
    Fm = coef * rl
    Fm = torch.where(lj, CAP * torch.tanh(Fm / CAP), Fm)
    ce = Fm / r.clamp_min(1e-9) * mk[:, None, :] * mk[:, :, None]
    return (ce[..., None] * d).sum(2)


@torch.no_grad()
def pair_tensors(params, typ):
    """params (B, 7, 4), typ (B, N) long -> (7, B, N, N) per-pair parameters, receiver i / source j slot 2 typ_i + typ_j."""
    B, N = typ.shape
    idx = (2 * typ[:, :, None] + typ[:, None, :]).view(B, 1, N * N).expand(B, 7, N * N)
    return torch.gather(params, 2, idx).view(B, 7, N, N).permute(1, 0, 2, 3).contiguous()


@torch.no_grad()
def simulate(pos, vel, typ, mask, params, steps, dt=DT, sub=4):
    pp = pair_tensors(params, typ.long())
    h = dt / sub
    ps, vs = [pos.clone()], [vel.clone()]
    a = accel(pos, pp, mask)
    for _ in range(steps):
        for _ in range(sub):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = accel(pos, pp, mask)
            vel = vel + 0.5 * h * a
        ps.append(pos.clone())
        vs.append(vel.clone())
    return torch.stack(ps, 1), torch.stack(vs, 1)


def init_scene(rng, kind):
    if kind == "uniform":
        n = int(rng.integers(10, 201))
        f = n / 8
        spread, speed = 5.0 * f ** 0.5, 0.5 * f ** 0.25
        pos = rng.uniform(-spread, spread, (n, 2))
        vel = rng.normal(0.0, speed, (n, 2))
    else:
        n = int(rng.integers(80, N_PAD + 1))
        sg = rng.uniform(5.0, 10.0)
        pos = np.clip(rng.normal(0.0, sg, (n, 2)), -28.0, 28.0)
        vel = rng.normal(0.0, rng.uniform(0.1, 0.6) * np.sqrt(n / (4 * sg)), (n, 2))
    vel -= vel.mean(0)
    return pos, vel


def gen_batch(rng, cases, device, steps):
    """cases: list of (shape, structure, amp or None). Returns dict of CPU/GPU tensors for B scenes (fp64 truth, stored fp32)."""
    B = len(cases)
    pos0 = np.zeros((B, N_PAD, 2))
    vel0 = np.zeros((B, N_PAD, 2))
    typ = np.zeros((B, N_PAD))
    mask = np.zeros((B, N_PAD))
    cond = np.zeros((B, DC))
    par = np.zeros((B, 7, 4))
    nn_ = np.zeros(B, dtype=np.int64)
    for b, (shape, structure, amp) in enumerate(cases):
        s = 10 ** rng.uniform(-1, 1) if amp is None else amp
        slots = slots_for(structure, shape, s)
        p, v = init_scene(rng, "uniform" if rng.random() < 0.5 else "cluster")
        n = p.shape[0]
        pos0[b, :n], vel0[b, :n], mask[b, :n], nn_[b] = p, v, 1.0, n
        if structure != "single":
            typ[b, :n] = (rng.random(n) < 0.5).astype(float)
        cond[b], par[b] = cond_of(slots), params_of(slots)
    t = lambda a: torch.tensor(a, device=device, dtype=torch.float64)
    P, V = simulate(t(pos0), t(vel0), t(typ), t(mask), t(par), steps)
    out = (P.abs().amax(-1) > 32.0) & (t(mask)[:, None, :] > 0)
    frac = out.any(1).double().sum(1) / t(mask).sum(1)
    ok = torch.isfinite(P).all(-1).all(-1).all(-1) & torch.isfinite(V).all(-1).all(-1).all(-1) & (frac <= 0.05)
    return {"P": P.float(), "V": V.float(), "typ": t(typ).float(), "mask": t(mask).float(), "cond": t(cond).float(), "n": nn_, "ok": ok,
            "params": t(par)}


def gen_cases(cases, seed, device, steps, chunk=64, tag=0):
    """Generate len(cases) accepted scenes (rejection on > 5% bodies leaving |x| > 32 or non-finite); cases are repeated until each is filled."""
    rng = np.random.default_rng([seed, tag])
    got = {k: [] for k in ("P", "V", "typ", "mask", "cond", "params")}
    gotn = []
    sid = []
    need = list(range(len(cases)))
    tries = 0
    while need and tries < 50:
        tries += 1
        for c0 in range(0, len(need), chunk):
            sub = need[c0:c0 + chunk]
            d = gen_batch(rng, [cases[i] for i in sub], device, steps)
            keep = d["ok"].cpu().numpy()
            for k in got:
                got[k].append(d[k][torch.tensor(keep, device=device)])
            gotn += list(d["n"][keep])
            sid += [sub[j] for j in range(len(sub)) if keep[j]]
        done = set(sid)
        need = [i for i in range(len(cases)) if i not in done]
    out = {k: torch.cat(v) for k, v in got.items()}
    out["n"] = torch.tensor(gotn)
    out["sid"] = torch.tensor(sid)
    return out


def exact_acc(pos, typ, mask, params):
    """pos (B,N,2) fp64; returns fp64 acceleration under the scene's law matrix."""
    if typ is None:
        typ = torch.zeros_like(mask)
    return accel(pos, pair_tensors(params, typ.long()), mask)
