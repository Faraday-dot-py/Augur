import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from model.central_force import CentralForceDynamics
from scripts import scatter_field as sf
from scripts import train_scatter_field as ts

EPS = 0.5
DT = 0.1
CFG = ts.EXPS["B"]
BINS = [0.0, 0.25, 0.5, 1.0, 2.0, 1e9]
BIN_NAMES = ["<.25", ".25-.5", ".5-1", "1-2", ">2"]


def exact_acc(pos, mass, mask):
    N = pos.shape[1]
    d = pos[:, None, :, :] - pos[:, :, None, :]
    r2 = (d ** 2).sum(-1) + EPS ** 2
    eye = torch.eye(N, device=pos.device, dtype=pos.dtype)
    w = r2 ** -1.5 * (mass * mask)[:, None, :] * mask[:, :, None] * (1 - eye)
    return (d * w[..., None]).sum(2)


class CFDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.dt = DT
        self.extent = 64.0
        self.force = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.force[4].weight)
        nn.init.zeros_(self.force[4].bias)

    def init_field(self, B, device, dtype=torch.float32):
        return None

    def accel(self, pos, mass, mask):
        N = pos.shape[1]
        d = pos[:, None, :, :] - pos[:, :, None, :]
        dist = torch.sqrt((d ** 2).sum(-1, keepdim=True) + 1e-12)
        f = self.force(torch.log(dist)) / (dist ** 2 + 1.0)
        eye = torch.eye(N, device=pos.device, dtype=pos.dtype)
        pm = (mask[:, :, None] * mask[:, None, :] * (1 - eye))[..., None]
        return (f * d / dist * pm).sum(2)

    def step(self, pos, vel, mass, mask, field):
        dt = self.dt
        a0 = self.accel(pos, mass, mask)
        pn = pos + vel * dt + 0.5 * dt * dt * a0
        a1 = self.accel(pn, mass, mask)
        dv = 0.5 * dt * (a0 + a1)
        return pn, vel + dv, None, dv


class ExactModel(nn.Module):
    def __init__(self, carry):
        super().__init__()
        self.carry, self.dt, self.extent = carry, DT, 64.0

    def init_field(self, B, device, dtype=torch.float32):
        return None

    def step(self, pos, vel, mass, mask, field):
        dt = self.dt
        a0 = field if (self.carry and field is not None) else exact_acc(pos, mass, mask)
        pn = pos + vel * dt + 0.5 * dt * dt * a0
        a1 = exact_acc(pn, mass, mask)
        dv = 0.5 * dt * (a0 + a1)
        return pn, vel + dv, (a1 if self.carry else None), dv


class SFx(sf.ScatterField):
    def __init__(self, kmode="learn", pmode="learn", selfsub=False, knn=16, **kw):
        super().__init__(**kw)
        self.kmode, self.pmode, self.selfsub, self.knn = kmode, pmode, selfsub, knn
        self._K = None

    def kernel_acc(self, rho):
        G = self.grid
        idx = torch.arange(2 * G, device=rho.device, dtype=rho.dtype)
        d = torch.minimum(idx, 2 * G - idx) * self.h
        r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
        s = 1 - (1 - (r / self.pp).clamp(max=1.0) ** 2) ** 2
        if self.kmode == "learn":
            K = self.kmlp(torch.stack([r, torch.log(r + self.h)], -1))[..., 0] * s
        else:
            K = -s / torch.sqrt(r ** 2 + EPS ** 2)
            if self.kmode == "res":
                K = K * (1 + self.kmlp(torch.stack([r, torch.log(r + self.h)], -1))[..., 0])
        self._K = K
        Kf = torch.fft.rfft2(K)
        rf = torch.fft.rfft2(rho, s=(2 * G, 2 * G))
        phi = torch.fft.irfft2(rf * Kf, s=(2 * G, 2 * G))[..., :G, :G] * self.h ** 2
        return self.neg_grad(phi)

    def self_acc(self, pos, mass, mask):
        G = self.grid
        i0, w = self._corners(pos)
        K = self._K
        ax = torch.zeros_like(pos[..., 0])
        ay = torch.zeros_like(pos[..., 0])
        wt = {}
        for ox in (0, 1):
            for oy in (0, 1):
                wt[(ox, oy)] = (w[..., 0] if ox else 1 - w[..., 0]) * (w[..., 1] if oy else 1 - w[..., 1])
        for px in (0, 1):
            for py in (0, 1):
                for ox in (0, 1):
                    for oy in (0, 1):
                        a, b = px - ox, py - oy
                        dkx = K[(a + 1) % (2 * G), b % (2 * G)] - K[(a - 1) % (2 * G), b % (2 * G)]
                        dky = K[a % (2 * G), (b + 1) % (2 * G)] - K[a % (2 * G), (b - 1) % (2 * G)]
                        ww = wt[(px, py)] * wt[(ox, oy)]
                        ax = ax - ww * dkx / (2 * self.h)
                        ay = ay - ww * dky / (2 * self.h)
        return torch.stack([ax, ay], -1) * (mass * mask)[..., None]

    def force(self, pos, vel, mass, mask, field):
        x = self.scatter(pos, vel, mass, mask)
        ak = self.kernel_acc(x[:, :1])
        x = torch.cat([x, ak], 1)
        if self.recurrent:
            x = torch.cat([x, field], 1)
        if self.nonet:
            field = x.new_zeros(x.shape[0], self.cf, self.grid, self.grid)
        else:
            field = self.net(x)
        acc = self.neg_grad(field[:, :1]) + ak
        a_tok = self.gather(acc, pos)
        if self.selfsub:
            a_tok = a_tok - self.self_acc(pos, mass, mask)
        a_tok = a_tok + self.pp_acc(pos, mass, mask, self.knn)
        mm = (mass * mask)[..., None]
        a_tok = a_tok - (mm * a_tok).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)
        return a_tok * mask[..., None], field

    def pp_acc(self, pos, mass, mask, knn=16):
        if self.pmode == "learn":
            return super().pp_acc(pos, mass, mask, knn)
        B, N = mask.shape
        k = min(knn, N - 1)
        d_all = pos[:, None, :, :] - pos[:, :, None, :]
        r_all = torch.sqrt((d_all ** 2).sum(-1) + 1e-8)
        bad = (1 - mask)[:, None, :] + torch.eye(N, device=pos.device, dtype=pos.dtype)[None]
        r_sel, idx = torch.topk(r_all + 1e6 * (bad > 0).to(pos.dtype), k, dim=-1, largest=False)
        d = torch.gather(d_all, 2, idx[..., None].expand(B, N, k, 2))
        mj = torch.gather((mass * mask)[:, None, :].expand(B, N, N), 2, idx)
        ok = (r_sel < 1e5).to(pos.dtype) * mask[:, :, None]
        r = r_sel.clamp(max=1e5)
        g = self.ppmlp(torch.stack([r, torch.log(r + 0.05)], -1))[..., 0]
        u = (r / self.pp).clamp(max=1.0) ** 2
        win = (1 - u) ** 2
        s = 1 - win
        q = torch.sqrt(r ** 2 + EPS ** 2)
        sp = 4 * (1 - u) * r / self.pp ** 2
        a_mesh = -sp / q + s * r / q ** 3
        a_exact = r / q ** 3 - a_mesh
        w = (a_exact + g * win) * mj * ok
        return (w[..., None] * d / r[..., None]).sum(2)


BASE = dict(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, momfix=True, dt=DT, in_scale=1.0,
            potential=True, kernel=True, pp=2.0, split=True, verlet=True)
SFVARS = {
    "sf_full": {},
    "sf_nonet": dict(nonet=True),
    "sf_norec": dict(recurrent=False),
    "sf_prior": dict(kmode="res", pmode="res"),
    "sf_prior_nonet": dict(kmode="res", pmode="res", nonet=True),
    "sf_exactk": dict(kmode="exact"),
    "sf_exactk_nonet": dict(kmode="exact", nonet=True),
    "sf_floor": dict(kmode="exact", pmode="res", nonet=True),
    "sf_floor_ss": dict(kmode="exact", pmode="res", nonet=True, selfsub=True),
    "sf_floor_ss_pp4": dict(kmode="exact", pmode="res", nonet=True, selfsub=True, pp=4.0),
    "sf_full_ss": dict(selfsub=True),
    "sf_prior_ss": dict(kmode="res", pmode="res", selfsub=True),
    "sf_prior_ss_nonet": dict(kmode="res", pmode="res", selfsub=True, nonet=True),
    "sf_prior_ss_pp4": dict(kmode="res", pmode="res", selfsub=True, pp=4.0),
    "sf_floor_pp6": dict(kmode="exact", pmode="res", nonet=True, pp=6.0, knn=32),
    "sf_floor_pp8": dict(kmode="exact", pmode="res", nonet=True, pp=8.0, knn=48),
    "sf_prior_pp6": dict(kmode="res", pmode="res", pp=6.0, knn=32),
    "sf_prior_pp8": dict(kmode="res", pmode="res", pp=8.0, knn=48),
    "sf_prior_pp8_nonet": dict(kmode="res", pmode="res", pp=8.0, knn=48, nonet=True),
}


def build(name):
    if name == "cf":
        return CFDense()
    if name == "exact_v2":
        return ExactModel(False)
    if name == "exact_v1":
        return ExactModel(True)
    kw = {**BASE, **SFVARS[name]}
    if name == "sf_full":
        return sf.ScatterField(**kw)
    return SFx(**kw)


def nn_dist(P, mask):
    d = torch.cdist(P, P)
    N = P.shape[-2]
    bad = (1 - mask)[..., None, :] + torch.eye(N, device=P.device, dtype=P.dtype)
    return (d + 1e9 * (bad > 0).to(P.dtype)).min(-1).values


def bin_stats(val, key, valid):
    out = {}
    tot = (val * valid).sum()
    for i, nm in enumerate(BIN_NAMES):
        sel = (key >= BINS[i]) & (key < BINS[i + 1]) & (valid > 0)
        n = int(sel.sum())
        out[nm] = {"n_frac": n / float(valid.sum()), "rms": float(val[sel].mean().sqrt()) if n else None,
                   "share": float(val[sel].sum() / tot) if n else 0.0}
    return out


def breakdown(Pm, Vm, P, V, mask):
    out = {}
    e2 = ((Pm - P) ** 2).sum(-1)
    valid = mask
    for T in (20, 100):
        mind = torch.stack([nn_dist(P[:, t], mask) for t in range(0, T + 1)]).min(0).values
        out[f"err2_at{T}_by_min_nn"] = bin_stats(e2[:, T], mind, valid)
    scene = (e2[:, 20] * mask).sum(1) / mask.sum(1)
    q = torch.quantile(scene.sqrt(), torch.tensor([0.1, 0.5, 0.9], device=scene.device, dtype=scene.dtype))
    srt = scene.sort(descending=True).values
    out["scene_err20_pctl_10_50_90"] = q.tolist()
    out["scene_top10pct_share_sq_err20"] = float(srt[: max(1, len(srt) // 10)].sum() / srt.sum())
    return out


@torch.no_grad()
def teacher_forced(model, P, V, M, mask, ts_=(0, 10, 20, 30, 40)):
    errs, truth, keys, vals, perr = [], [], [], [], []
    for t in ts_:
        pos, vel = P[:, t].float(), V[:, t].float()
        field = model.init_field(pos.shape[0], pos.device)
        p1, v1, _, _, dv = sf._step_m(model, pos, vel, M.float(), mask.float(), field)
        dvt = (V[:, t + 1] - V[:, t]).float()
        errs.append((((dv - dvt) / DT).double() ** 2).sum(-1))
        truth.append(((dvt / DT).double() ** 2).sum(-1))
        keys.append(nn_dist(P[:, t], mask))
        vals.append(mask)
    E, T, K, Vd = torch.stack(errs), torch.stack(truth), torch.stack(keys), torch.stack(vals)
    res = {"overall_rel_rms": float(((E * Vd).sum() / (T * Vd).sum()).sqrt()), "bins": {}}
    for i, nm in enumerate(BIN_NAMES):
        sel = (K >= BINS[i]) & (K < BINS[i + 1]) & (Vd > 0)
        if sel.sum():
            res["bins"][nm] = {"n_frac": float(sel.sum() / Vd.sum()), "abs_rms": float(E[sel].mean().sqrt()),
                               "rel_rms": float((E[sel].sum() / T[sel].sum()).sqrt()), "share": float(E[sel].sum() / (E * Vd).sum())}
    return res


def full_eval(model, sets, dev):
    res = {}
    for seed, (P, V, M, mask) in sets.items():
        Pm, Vm = sf.rollout(model, P[:, 0].float(), V[:, 0].float(), M.float(), mask.float(), P.shape[1] - 1)
        Pm, Vm = Pm.double(), Vm.double()
        m = sf.traj_metrics(Pm, Vm, P, V, M, mask, EPS)
        r = {"err": m["err"], "dE": m["energy_drift_rel_model"][100]}
        if seed == 9000:
            r["breakdown"] = breakdown(Pm, Vm, P, V, mask)
            r["teacher_forced"] = teacher_forced(model, P, V, M, mask)
        res[str(seed)] = r
    return res


def make_sets(dev):
    out = {}
    for seed in ts.EVAL_SEEDS:
        data = ts.make_data(CFG, 48, 100, seed, dev, DT, EPS)
        out[seed] = sf.to_tensors(data, dev)
    return out


def head(r):
    return [round(r["err"][i], 5) for i in (4, 9, 19, 49, 99)]


def stage_floors(args, dev, sets):
    res = {}
    cf = CFDense().to(dev)
    ref = CentralForceDynamics(dt=DT).to(dev)
    ref.load_state_dict({k.replace("force.", "force."): v for k, v in cf.state_dict().items()})
    with torch.no_grad():
        for p in list(cf.parameters()) + list(ref.parameters()):
            p.add_(0.1 * torch.randn_like(p))
        ref.load_state_dict(cf.state_dict())
        P, V, M, mask = sets[9000]
        n = int(mask[0].sum())
        p_rel = P[0, 0, :n].float()[None]
        a_dense = cf.accel(p_rel, M[:1, :n].float(), mask[:1, :n].float())[0]
        a_ref = ref.accel(p_rel[0] + sf.CENTER)
        res["cf_dense_vs_ref_max_rel_diff"] = float((a_dense - a_ref).abs().max() / a_ref.abs().max())
    print("cf dense vs CentralForceDynamics", res["cf_dense_vs_ref_max_rel_diff"], flush=True)
    for name in args.floors.split(","):
        m = build(name).to(dev)
        r = full_eval(m, sets, dev)
        res[name] = r
        print(name, "err@5/10/20/50/100", head(r["9000"]), "tf rel", r["9000"]["teacher_forced"]["overall_rel_rms"],
              {k: round(v["rel_rms"], 4) for k, v in r["9000"]["teacher_forced"]["bins"].items()}, flush=True)
    json.dump(res, open(f"{args.out}/floors.json", "w"))


def acc_of(model, pos, vel, mass, mask):
    if isinstance(model, CFDense):
        return model.accel(pos, mass, mask)
    f = model.init_field(pos.shape[0], pos.device)
    f = f[0] if model.verlet else f
    return model.force(pos, vel, mass, mask, f)[0]


@torch.no_grad()
def force_eval(model, P, V, M, mask):
    E, T, K, Vd = [], [], [], []
    for t in (0, 10, 20, 30):
        pos, vel = P[:, t].float(), V[:, t].float()
        a = acc_of(model, pos, vel, M.float(), mask.float())
        at = exact_acc(pos, M.float(), mask.float())
        E.append(((a - at).double() ** 2).sum(-1))
        T.append((at.double() ** 2).sum(-1))
        K.append(nn_dist(P[:, t], mask))
        Vd.append(mask)
    E, T, K, Vd = torch.stack(E), torch.stack(T), torch.stack(K), torch.stack(Vd)
    out = {"rel_rms": float(((E * Vd).sum() / (T * Vd).sum()).sqrt()), "bins": {}}
    for i, nm in enumerate(BIN_NAMES):
        sel = (K >= BINS[i]) & (K < BINS[i + 1]) & (Vd > 0)
        if sel.sum():
            out["bins"][nm] = {"abs_rms": float(E[sel].mean().sqrt()), "rel_rms": float((E[sel].sum() / T[sel].sum()).sqrt())}
    return out


def stage_e2(args, dev, sets):
    data = ts.make_data(CFG, 2000, 30, 4738, dev, DT, EPS)
    P, V, M, mask = [t.float() for t in sf.to_tensors(data, dev)]
    Pe, Ve, Me, ke = sets[9000]
    res = {}
    checkpoints = {0, 250, 500, 1000, 2000, args.e2_iters}
    for name in args.models.split(","):
        torch.manual_seed(4738)
        model = build(name).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        g = torch.Generator(device="cpu").manual_seed(4738)
        curve = {}
        t0 = time.time()
        for it in range(args.e2_iters + 1):
            if it in checkpoints:
                curve[it] = force_eval(model, Pe, Ve, Me, ke)
                print(name, it, "force rel", round(curve[it]["rel_rms"], 4), {k: round(v["rel_rms"], 4) for k, v in curve[it]["bins"].items()}, f"{time.time() - t0:.0f}s", flush=True)
            if it == args.e2_iters:
                break
            frac = it / args.e2_iters
            for pg in opt.param_groups:
                pg["lr"] = 1e-3 * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
            idx = torch.randint(P.shape[0], (16,), generator=g).to(dev)
            tt = torch.randint(0, 31, (16,), generator=g).to(dev)
            pos, vel, m, mk = P[idx, tt], V[idx, tt], M[idx], mask[idx]
            a = acc_of(model, pos, vel, m, mk)
            at = exact_acc(pos, m, mk)
            loss = (((a - at) ** 2) * mk[..., None]).sum() / (mk.sum() * 2)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(gn):
                opt.step()
        res[name] = {str(k): v for k, v in curve.items()}
        json.dump(res, open(f"{args.out}/e2.json", "w"))


def stage_e1(args, dev, sets):
    data = ts.make_data(CFG, 2000, 30, 4738, dev, DT, EPS)
    tensors = sf.to_tensors(data, dev)
    res = {}
    path = f"{args.out}/e1.json"
    if os.path.exists(path):
        res = json.load(open(path))
    jobs = []
    for spec in args.e1.split(";"):
        name, its = spec.split(":")
        for n in its.split(","):
            jobs.append((name, int(n)))
    for name, n in jobs:
        key = f"{name}@{n}"
        if key in res:
            continue
        torch.manual_seed(4738)
        model = build(name).to(dev)
        ck = f"{args.ckpt}/{name}_{n}.pt"
        if os.path.exists(ck):
            os.remove(ck)
        targs = SimpleNamespace(lr=1e-3, time_budget=0.0, iters=n, k_start=4, k_end=20, batch=16, steps=30, device=str(dev),
                                seed=4738, log_every=100)
        losses = []

        def log(s):
            if s.startswith("it ") and "loss" in s:
                parts = s.split()
                losses.append((int(parts[1]), float(parts[5])))
            else:
                print(s, flush=True)

        n_it, secs = ts.train(model, tensors, targs, ck, log)
        r = full_eval(model, sets, dev)
        r["train_iters"], r["train_seconds"], r["loss_curve"] = n_it, secs, losses
        r["force_eval"] = force_eval(model, *sets[9000])
        r["params"] = sum(p.numel() for p in model.parameters())
        res[key] = r
        print(key, "err@5/10/20/50/100", head(r["9000"]), "12000", head(r["12000"]), "force rel", round(r["force_eval"]["rel_rms"], 4),
              f"{secs:.0f}s", flush=True)
        json.dump(res, open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--models", default="cf,sf_full")
    ap.add_argument("--floors", default="exact_v2,sf_floor")
    ap.add_argument("--e2-iters", type=int, default=4000)
    ap.add_argument("--e1", default="")
    ap.add_argument("--out", default="results/gap")
    ap.add_argument("--ckpt", default="checkpoints/gap")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    dev = torch.device("cuda")
    sets = make_sets(dev)
    for st in args.stage.split(","):
        {"floors": stage_floors, "e2": stage_e2, "e1": stage_e1}[st](args, dev, sets)


if __name__ == "__main__":
    main()
