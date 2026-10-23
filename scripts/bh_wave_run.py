"""BH-collapse wave-structure diagnosis: exact softened all-pairs truth vs plain particle-mesh (PM) vs the learned scatter-field model on the same IC.

Scene (same IC as scatter_bh.py / proposed_check.py): N unit-mass bodies, Gaussian sigma 8 clipped +-29, per-component speed std vfac*sqrt(N/(4 sigma)), c 10, dt .1, eps .5,
relativistic momentum state (v = p/sqrt(1+p^2/c^2)), kick-drift-kick. --rot adds solid-body rotation omega0 (rad per time unit) to the IC velocities.

Arms (comma list): truth4 (exact, fp64, 4 substeps/tick as gravity_sim.rollout_torch), truth1 (exact, 1 substep/tick = same step as the mesh arms), pm128 / pm256 / pm64
(own non-learned PM: CIC deposit, FFT convolution with the softened potential kernel on a 2x zero-padded grid, central-difference gradient, momfix, extent 64, 1 substep),
pm128s4 (pm128 with 4 substeps/tick), model_old / model_new (Exp E weights; new = ghost edge closure + R1b far kernel, mesh-only, pair term off).

Per tick diagnostics (about the centre of mass): radial counts (linear and core), mean v_r and v_t^2 per radial bin, azimuthal Fourier amplitudes m=1..6 in radial shells, angular momentum L,
sum |l|, rotation rates inside enclosed-mass radii, relativistic KE, virial sum p.v, softened potential W, 2-D density histograms (N <= 1e5), a fixed 20000-body position subsample.

Usage: PYTHONPATH=. python scripts/bh_wave_run.py --n 10000 --arms truth4,pm128,model_new --steps 150 --out results/bhwave/n1e4_s4738
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import proposed_check as pc
from scripts import scatter_bh as sb

EPS = 0.5
DEV = torch.device("cuda")
RLIN = np.arange(0, 48.01, 0.25)
RCORE = np.arange(0, 4.001, 0.02)
SHELLS = [0, 1, 2, 4, 8, 12, 16, 24, 32]
NM = 6
NSUB = 20000


def make_ic(n, seed, sigma, vfac, c, rot):
    rng = np.random.default_rng(seed)
    pos0 = np.clip(rng.normal(0.0, sigma, (n, 2)), -29.0, 29.0)
    vel0 = rng.normal(0.0, vfac * np.sqrt(n / (4 * sigma)), (n, 2))
    if rot:
        vel0 = vel0 + rot * np.stack([-pos0[:, 1], pos0[:, 0]], 1)
    vel0 -= vel0.mean(0)
    return pos0, vel0


def to_mom(vel, c):
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    return vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


@torch.no_grad()
def exact_acc(pos, elems=1.0e8):
    n = pos.shape[0]
    ch = max(1, int(elems / n))
    out = torch.empty_like(pos)
    for s in range(0, n, ch):
        e = min(n, s + ch)
        d = pos[None] - pos[s:e, None]
        inv = ((d ** 2).sum(-1) + EPS ** 2) ** -1.5
        idx = torch.arange(s, e, device=pos.device)
        inv[idx - s, idx] = 0.0
        out[s:e] = (d * inv[..., None]).sum(1)
    return out


@torch.no_grad()
def potential_energy(pos, elems=1.0e8):
    n = pos.shape[0]
    ch = max(1, int(elems / n))
    tot = torch.zeros((), dtype=torch.float64, device=pos.device)
    for s in range(0, n, ch):
        e = min(n, s + ch)
        d = pos[None] - pos[s:e, None]
        inv = ((d ** 2).sum(-1) + EPS ** 2) ** -0.5
        idx = torch.arange(s, e, device=pos.device)
        inv[idx - s, idx] = 0.0
        tot += inv.sum(dtype=torch.float64)
    return float(-0.5 * tot)


class Truth:
    def __init__(self, substeps, dt, c):
        self.sub, self.dt, self.c = substeps, dt, c
        self.dtype = torch.float64

    def init(self, pos, mom):
        self.pos, self.mom = pos, mom
        self.a = exact_acc(pos)

    def tick(self):
        h = self.dt / self.sub
        for _ in range(self.sub):
            self.mom = self.mom + 0.5 * h * self.a
            self.pos = self.pos + h * speed(self.mom, self.c)
            self.a = exact_acc(self.pos)
            self.mom = self.mom + 0.5 * h * self.a


class PM:
    def __init__(self, grid, extent, substeps, dt, c):
        self.G, self.ext, self.h, self.sub, self.dt, self.c = grid, extent, extent / grid, substeps, dt, c
        self.dtype = torch.float32
        idx = torch.arange(2 * grid, device=DEV, dtype=torch.float32)
        d = torch.minimum(idx, 2 * grid - idx) * self.h
        r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
        self.Kf = torch.fft.rfft2(-1.0 / torch.sqrt(r ** 2 + EPS ** 2))

    def corners(self, pos):
        u = (pos + self.ext / 2) / self.h - 0.5
        i0 = torch.floor(u)
        w = u - i0
        i0 = i0.long()
        for ox in (0, 1):
            for oy in (0, 1):
                ix, iy = i0[:, 0] + ox, i0[:, 1] + oy
                wt = (w[:, 0] if ox else 1 - w[:, 0]) * (w[:, 1] if oy else 1 - w[:, 1])
                ok = ((ix >= 0) & (ix < self.G) & (iy >= 0) & (iy < self.G)).float()
                yield iy.clamp(0, self.G - 1) * self.G + ix.clamp(0, self.G - 1), wt * ok

    def acc(self, pos):
        G = self.G
        rho = torch.zeros(G * G, device=DEV)
        for flat, w in self.corners(pos):
            rho.index_add_(0, flat, w / self.h ** 2)
        rho = rho.view(G, G)
        phi = torch.fft.irfft2(torch.fft.rfft2(rho, s=(2 * G, 2 * G)) * self.Kf, s=(2 * G, 2 * G))[:G, :G] * self.h ** 2
        p = torch.nn.functional.pad(phi[None, None], (1, 1, 1, 1))[0, 0]
        gx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.h)
        gy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.h)
        field = -torch.stack([gx, gy], -1).reshape(G * G, 2)
        a = 0
        for flat, w in self.corners(pos):
            a = a + field[flat] * w[:, None]
        return a - a.mean(0, keepdim=True)

    def init(self, pos, mom):
        self.pos, self.mom = pos, mom
        self.a = self.acc(pos)

    def tick(self):
        h = self.dt / self.sub
        for _ in range(self.sub):
            self.mom = self.mom + 0.5 * h * self.a
            self.pos = self.pos + h * speed(self.mom, self.c)
            self.a = self.acc(self.pos)
            self.mom = self.mom + 0.5 * h * self.a


class Model:
    def __init__(self, model, dt, c):
        self.m, self.dt, self.c = model, dt, c
        self.dtype = torch.float32

    def init(self, pos, mom):
        n = pos.shape[0]
        self.mass = torch.ones(1, n, device=DEV)
        self.field = self.m.init_field(1, DEV)[0]
        self.pos, self.mom = pos, mom
        self.a = self.force()

    def force(self):
        a, self.field = self.m.force(self.pos[None], speed(self.mom, self.c)[None], self.mass, self.mass, self.field)
        return a[0]

    def tick(self):
        self.mom = self.mom + 0.5 * self.dt * self.a
        self.pos = self.pos + self.dt * speed(self.mom, self.c)
        self.a = self.force()
        self.mom = self.mom + 0.5 * self.dt * self.a


@torch.no_grad()
def diagnose(pos, mom, c, sub, want_w, want_2d):
    pos, mom = pos.double(), mom.double()
    n = pos.shape[0]
    v = speed(mom, c)
    cen = pos.mean(0)
    x = pos - cen
    r = x.norm(dim=-1)
    out = {}
    out["centre"] = cen.cpu().numpy()
    bins = torch.tensor(RLIN, device=DEV)
    ib = torch.bucketize(r, bins) - 1
    ok = (ib >= 0) & (ib < len(RLIN) - 1)
    nb = len(RLIN) - 1
    cnt = torch.bincount(ib[ok], minlength=nb)
    out["rad_lin"] = cnt.cpu().numpy()
    rr = r.clamp_min(1e-9)
    vr = (v * x).sum(-1) / rr
    vt = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / rr
    out["vr_lin"] = torch.bincount(ib[ok], weights=vr[ok], minlength=nb).cpu().numpy()
    out["vt2_lin"] = torch.bincount(ib[ok], weights=(vt ** 2)[ok], minlength=nb).cpu().numpy()
    cb = torch.tensor(RCORE, device=DEV)
    ic = torch.bucketize(r, cb) - 1
    okc = (ic >= 0) & (ic < len(RCORE) - 1)
    out["rad_core"] = torch.bincount(ic[okc], minlength=len(RCORE) - 1).cpu().numpy()
    lz = x[:, 0] * mom[:, 1] - x[:, 1] * mom[:, 0]
    out["L"] = float(lz.sum())
    out["absl"] = float(lz.abs().sum())
    jz = x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]
    qs = torch.quantile(r[:: max(1, n // 1_000_000)], torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=DEV, dtype=torch.float64))
    out["rq"] = qs.cpu().numpy()
    om, oa, jm = [], [], []
    for q in qs:
        k = r < q
        r2 = (r[k] ** 2).sum()
        om.append(float(jz[k].sum() / r2))
        oa.append(float(jz[k].abs().sum() / r2))
        jm.append(float(jz[k].mean()))
    out["omega_net"], out["omega_abs"], out["jz_mean_in"] = np.array(om), np.array(oa), np.array(jm)
    out["p_median"], out["p_max"] = float(mom.norm(dim=-1).median()), float(mom.norm(dim=-1).max())
    out["frac_v_gt_0.9c"] = float((v.norm(dim=-1) > 0.9 * c).double().mean())
    gam = (1 - (v ** 2).sum(-1) / c ** 2).clamp(min=1e-12).rsqrt()
    out["KE"] = float((c ** 2 * (gam - 1)).sum())
    out["pv"] = float((mom * v).sum())
    out["frac_outside"] = float((pos.abs().max(-1).values > 32).double().mean())
    th = torch.atan2(x[:, 1], x[:, 0])
    am = np.zeros((len(SHELLS) - 1, NM))
    ns = np.zeros(len(SHELLS) - 1)
    for s in range(len(SHELLS) - 1):
        k = (r >= SHELLS[s]) & (r < SHELLS[s + 1])
        ns[s] = int(k.sum())
        if ns[s] > 0:
            for m in range(1, NM + 1):
                am[s, m - 1] = float(torch.complex(torch.cos(m * th[k]).sum(), torch.sin(m * th[k]).sum()).abs() / ns[s])
    out["azim"], out["azim_n"] = am, ns
    if want_w:
        out["W"] = potential_energy(pos)
    if want_2d:
        for name, hw, off in (("h2d32", 32.0, 0.0), ("h2d8", 8.0, 1.0)):
            q = pos - (cen if off else 0) + hw
            ij = (q * (128 / (2 * hw))).floor().long()
            okk = ((ij >= 0) & (ij < 128)).all(-1)
            flat = torch.where(okk, ij[:, 1] * 128 + ij[:, 0], torch.full_like(ij[:, 0], 128 * 128))
            out[name] = torch.bincount(flat, minlength=128 * 128 + 1)[:-1].view(128, 128).cpu().numpy().astype(np.int32)
    out["sub"] = pos[sub].float().cpu().numpy()
    return out


def stack(rows):
    res = {}
    for k in rows[0]:
        res[k] = np.stack([np.asarray(r[k]) for r in rows])
    return res


def run_arm(name, args, models, pos0, vel0):
    c, dt = args.c, args.dt
    n = pos0.shape[0]
    if name == "truth4":
        eng = Truth(4, dt, c)
    elif name == "truth1":
        eng = Truth(1, dt, c)
    elif name.startswith("pm"):
        g = int(name[2:5])
        eng = PM(g, 64.0, 4 if name.endswith("s4") else 1, dt, c)
    else:
        eng = Model(models[name], dt, c)
    dty = eng.dtype
    pos = torch.tensor(pos0, dtype=dty, device=DEV)
    vel = torch.tensor(vel0, dtype=dty, device=DEV)
    mom = to_mom(vel, c)
    g = torch.Generator(device=DEV)
    g.manual_seed(args.seed + 1)
    sub = torch.randperm(n, device=DEV, generator=g)[: min(n, NSUB)]
    eng.init(pos, mom)
    want_2d = n <= 1_000_000
    t0 = time.perf_counter()
    rows = []
    for t in range(args.steps + 1):
        if t:
            eng.tick()
        rows.append(diagnose(eng.pos, eng.mom, c, sub, want_w=(n <= 100_000 and t % args.wevery == 0), want_2d=want_2d))
        if t % 25 == 0:
            d = rows[-1]
            print(f"[{name} n={n}] tick {t} wall {time.perf_counter() - t0:.0f}s L {d['L']:.3e} |l| {d['absl']:.3e} KE/N {d['KE'] / n:.4g} pmax {d['p_max']:.3g} out {d['frac_outside']:.3f}", flush=True)
    res = {}
    for k in rows[0]:
        if k == "W":
            continue
        res[k] = np.stack([np.asarray(r[k]) for r in rows])
    if want_w_any(rows):
        res["W"] = np.array([r.get("W", np.nan) for r in rows])
    res["wall_s"] = np.array(time.perf_counter() - t0)
    return res


def want_w_any(rows):
    return any("W" in r for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--vfac", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--wevery", type=int, default=5)
    ap.add_argument("--ckpt", default="checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt")
    ap.add_argument("--ghost-ckpt", default="/home/adamwebb/edge_ck/ghost_lsq.pt")
    ap.add_argument("--kernel", default="file:results/far_kernels/r1b_mlp.pt+cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pos0, vel0 = make_ic(args.n, args.seed, args.sigma, args.vfac, args.c, args.rot)
    arms = args.arms.split(",")
    models = {}
    if "model_old" in arms:
        models["model_old"] = pc.build(args, DEV, "ms_kp_pot_v_g128", args.ckpt, "old")
    if "model_new" in arms:
        models["model_new"] = pc.build(args, DEV, "ms_kp_pot_v_g128_ghost", args.ghost_ckpt, args.kernel)
    meta = {"args": vars(args), "arms": {}}
    for name in arms:
        torch.cuda.empty_cache()
        res = run_arm(name, args, models, pos0, vel0)
        np.savez_compressed(f"{args.out}_{name}.npz", **res)
        meta["arms"][name] = {"wall_s": float(res["wall_s"])}
        json.dump(meta, open(f"{args.out}_meta.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
