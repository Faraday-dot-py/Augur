"""Zero-shot dark-matter-like two-component probe for the scatter-field Exp E checkpoint (research only, no training).

Visible: N_V=100 unit-mass Plummer (a=3). Halo: total mass M_H=200 Plummer (a=6) either as live particles of mass m_h in
{1,3,10,30} (arms live_m1..live_m30), absent (novis_halo = visible only), or static: one external softened-point-mass
acceleration M_H r / (r^2 + a_h^2)^1.5 added to the model force and to the truth force (arm static). Metrics on VISIBLE
bodies: err@k (mean position error vs exact fp64 softened all-pairs, sim units), constant-velocity baseline, 1e-5 perturbed-truth
floor, rotation-curve error (v_c^2 = -r . a at t=0 on visible bodies, model vs truth, median relative error), r50 at end.
Usage: PYTHONPATH=. python scripts/scatter_dm_probe.py --ckpt CKPT --out results/scatter_dm_probe.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_ood as so

NV, MH, AV, AH, EPS, DT = 100, 200.0, 3.0, 6.0, 0.5, 0.1
STEPS = 50
KS = (5, 10, 20, 50)


def plum_pos(rng, n, a):
    u = rng.uniform(0.01, 0.99, n)
    r = np.minimum(a / np.sqrt(u ** (-2 / 3) - 1), 29.0)
    th = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def make(arm, seed):
    rng = np.random.default_rng(seed)
    pv = plum_pos(rng, NV, AV)
    mv = np.ones(NV)
    if arm in ("live_m1", "live_m3", "live_m10", "live_m30"):
        mh = float(arm[6:])
        nh = int(round(MH / mh))
        ph = plum_pos(rng, nh, AH)
        pos, mass = np.concatenate([pv, ph]), np.concatenate([mv, np.full(nh, mh)])
    else:
        pos, mass = pv, mv
    r = np.linalg.norm(pos, axis=1)
    order = np.argsort(r)
    menc = np.empty(len(r))
    menc[order] = np.cumsum(mass[order]) - 0.5 * mass[order]
    if arm == "static":
        menc = menc + MH * r ** 3 / (r ** 2 + AH ** 2) ** 1.5
    vc = np.sqrt(menc * r ** 2 / (r ** 2 + EPS ** 2) ** 1.5)
    th = np.arctan2(pos[:, 1], pos[:, 0])
    sgn = rng.choice([-1.0, 1.0], len(r))
    vel = (0.7 * sgn * vc)[:, None] * np.stack([-np.sin(th), np.cos(th)], 1)
    pos, vel = so.com_fix(pos, vel, mass)
    return pos, vel, mass


def ext_acc(p):
    r2 = (p ** 2).sum(-1, keepdim=True) + AH ** 2
    return -MH * p * r2 ** -1.5


def truth(pos0, vel0, mass, dev, static):
    h = DT / 4
    pos = torch.tensor(pos0, dtype=torch.float64, device=dev)
    vel = torch.tensor(vel0, dtype=torch.float64, device=dev)
    m = torch.tensor(mass, dtype=torch.float64, device=dev)

    def acc(p):
        d = p[None] - p[:, None]
        inv = ((d ** 2).sum(-1) + EPS ** 2) ** -1.5
        inv.fill_diagonal_(0.0)
        a = (d * inv[..., None] * m[None, :, None]).sum(1)
        return a + ext_acc(p) if static else a

    ps, a0 = [pos.clone()], acc(pos)
    a = a0
    for _ in range(STEPS):
        for _ in range(4):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = acc(pos)
            vel = vel + 0.5 * h * a
        ps.append(pos.clone())
    return torch.stack(ps), a0


def run(model, arm, seed, dev):
    static = arm == "static"
    pos0, vel0, mass = make(arm, seed)
    nv = NV
    Pt, a_true = truth(pos0, vel0, mass, dev, static)
    rng = np.random.default_rng(seed + 1)
    Pp, _ = truth(pos0 + 1e-5 * 5.0 * rng.normal(size=pos0.shape), vel0, mass, dev, static)
    orig = model.force
    if static:
        def force(pos, vel, m, mask, field):
            a, f = orig(pos, vel, m, mask, field)
            return a + ext_acc(pos) * mask[..., None], f
        model.force = force
    try:
        Pm, _, nonfinite = so.model_rollout(model, pos0, vel0, mass, STEPS, DT, None, dev)
        pos_t = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
        mt = torch.tensor(mass, dtype=torch.float32, device=dev)[None]
        mk = torch.ones_like(mt)
        with torch.no_grad():
            a_mod, _ = model.force(pos_t, torch.zeros_like(pos_t), mt, mk, model.init_field(1, dev, torch.float32)[0])
    finally:
        model.force = orig
    a_mod = a_mod[0].double()
    p0 = torch.tensor(pos0, dtype=torch.float64, device=dev)
    r = p0.norm(dim=-1)
    vc_m = (-(a_mod * p0).sum(-1)).clamp_min(1e-9).sqrt()
    vc_t = (-(a_true * p0).sum(-1)).clamp_min(1e-9).sqrt()
    vis = slice(0, nv)
    vc_err = float(((vc_m[vis] - vc_t[vis]).abs() / vc_t[vis]).median())
    rv = r[vis]
    terc = torch.quantile(rv, torch.tensor([1 / 3, 2 / 3], device=dev, dtype=torch.float64))
    vc_terc = []
    for lo, hi in ((0, terc[0]), (terc[0], terc[1]), (terc[1], 1e9)):
        s = (rv >= lo) & (rv < hi)
        vc_terc.append(float(((vc_m[vis][s] - vc_t[vis][s]).abs() / vc_t[vis][s]).median()))
    T = Pm.shape[0]
    out = {"arm": arm, "seed": seed, "n_total": len(mass), "nonfinite_tick": nonfinite, "vc_relerr_median": vc_err, "vc_relerr_terciles": vc_terc}
    for k in KS:
        if k >= T:
            continue
        out[f"err_vis@{k}"] = float((Pm[k, vis] - Pt[k, vis]).norm(dim=-1).mean())
        out[f"floor_vis@{k}"] = float((Pp[k, vis] - Pt[k, vis]).norm(dim=-1).mean())
        out[f"ballistic_vis@{k}"] = float((p0[vis] + k * DT * torch.tensor(vel0[vis], device=dev) - Pt[k, vis]).norm(dim=-1).mean())
        out[f"err_all@{k}"] = float((Pm[k] - Pt[k]).norm(dim=-1).mean())
    out["r50_vis_end_model"] = float(Pm[-1, vis].norm(dim=-1).median())
    out["r50_vis_end_truth"] = float(Pt[-1, vis].norm(dim=-1).median())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opt", default="")
    args = ap.parse_args()
    dev = "cuda"
    model = so.build_model(args, dev)
    arms = ["novis_halo", "live_m1", "live_m3", "live_m10", "live_m30", "static"]
    res = []
    for seed in (4738, 9000, 12000):
        for arm in arms:
            o = run(model, arm, seed, dev)
            res.append(o)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
