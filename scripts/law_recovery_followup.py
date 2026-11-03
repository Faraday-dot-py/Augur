"""Follow-up to law_recovery_probe.py (audit row 1): robustness of the pow1 exponent miss and the Yukawa range fit.
Per-placement radial accelerations are stored, then: (a) refit on three placement seeds, (b) denser distance grid, (c) fit ranges, (d) bootstrap over
placements per distance, (e) Yukawa range: free fit vs range fixed to truth, residual profile over lambda, and a fit of the exact truth curve.
The registered lines (p within 10%, eps within 20%) are applied unchanged. Usage: PYTHONPATH=. python scripts/law_recovery_followup.py --out results/law_recovery_followup.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_multi as sm
from scripts.law_recovery_probe import LAMS, TRUTH, _grid_fit, fit, load_multi

DISTS_BASE = np.geomspace(0.25, 14.0, 28)
DISTS_DENSE = np.geomspace(0.25, 14.0, 56)
RANGES = ((0.25, 14.0), (0.25, 8.0), (0.5, 14.0), (0.5, 8.0))
LAWS = ("pow1", "pow2", "pow3", "yuk8", "yuk32", "yuk128")


@torch.no_grad()
def raw_curves(model, dev, law, dists, n, warm, seed):
    g = torch.Generator().manual_seed(seed)
    slots = sm.slots_for("single", law, 1.0)
    cond = torch.tensor([sm.cond_of(slots)], dtype=torch.float32, device=dev).expand(n, -1).contiguous()
    par = torch.tensor(sm.params_of(slots), dtype=torch.float64, device=dev)[None].expand(n, -1, -1).contiguous()
    typ0 = torch.zeros(n, 2, device=dev)
    model.set_law(cond, typ0)
    full, truth = [], []
    for d in dists:
        c = (torch.rand(n, 1, 2, generator=g) - 0.5) * 4.0
        th = torch.rand(n, generator=g) * 2 * np.pi
        u = torch.stack([th.cos(), th.sin()], -1)[:, None]
        pos = torch.cat([c - 0.5 * d * u, c + 0.5 * d * u], 1).to(dev)
        vel = torch.zeros_like(pos)
        mass = torch.ones(n, 2, device=dev)
        mask = torch.ones(n, 2, device=dev)
        field = model.init_field(n, dev)[0]
        for _ in range(warm + 1):
            a, field = model.force(pos, vel, mass, mask, field)
        at = sm.exact_acc(pos.double(), typ0, mask.double(), par).float()
        uh = pos[:, 1] - pos[:, 0]
        uh = uh / uh.norm(dim=-1, keepdim=True)
        full.append((a[:, 0] * uh).sum(-1).cpu().numpy())
        truth.append((at[:, 0] * uh).sum(-1).cpu().numpy())
    return np.array(full), np.array(truth)


def fit_range(d, y, dmin, dmax, lam_free, lam_truth):
    m = (d >= dmin) & (d <= dmax)
    return fit(d[m], y[m], None if lam_free else np.inf)


def errs(f, law):
    pt, et, lt = TRUTH[law]
    if f is None:
        return None
    out = {"p": f["p"], "eps": f["eps"], "lam": f["lam"], "p_err": abs(f["p"] - pt) / pt, "eps_err": abs(f["eps"] - et) / et, "rms_log_resid": f["rms_log_resid"]}
    return out


def lam_profile(d, y):
    ly = np.log(np.clip(y, 1e-30, None))
    best = []
    for lam in LAMS:
        s, (p, eps, _, A) = _grid_fit(d, y, np.arange(0.4, 4.6, 0.04), np.geomspace(0.03, 3.0, 40), np.array([lam]))
        best.append({"lam": None if not np.isfinite(lam) else float(lam), "p": float(p), "eps": float(eps), "rms_log_resid": float(np.sqrt(s / len(d)))})
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--models", default="mixed_s4738:ckpt/main_s4738.pt,mixed_s4739:ckpt/main_s4739.pt")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {"n": a.n, "warm": a.warm, "boot": a.boot, "models": {}}
    rng = np.random.default_rng(4738)
    for spec in a.models.split(","):
        name, path = spec.split(":")
        model = load_multi(path, dev)
        res = out["models"][name] = {}
        for law in LAWS:
            lam_free = law.startswith("yuk")
            rec = {"placement_seeds": {}, "ranges": {}}
            for seed in (4738, 4739, 4740):
                y, yt = raw_curves(model, dev, law, DISTS_BASE, a.n, a.warm, seed)
                med = np.median(y, 1)
                rec["placement_seeds"][str(seed)] = errs(fit_range(DISTS_BASE, med, 0.25, 14.0, lam_free, None), law)
                if seed == 4738:
                    y0, yt0 = y, yt
                    for dmin, dmax in RANGES:
                        rec["ranges"][f"{dmin}-{dmax}"] = errs(fit_range(DISTS_BASE, med, dmin, dmax, lam_free, None), law)
            yd, ytd = raw_curves(model, dev, law, DISTS_DENSE, a.n, a.warm, 4738)
            rec["dense_grid_56"] = errs(fit_range(DISTS_DENSE, np.median(yd, 1), 0.25, 14.0, lam_free, None), law)
            boots = []
            for _ in range(a.boot if (law in ("pow1", "pow2", "yuk8", "yuk32", "yuk128")) else 0):
                idx = rng.integers(0, a.n, (len(DISTS_BASE), a.n))
                med = np.median(np.take_along_axis(y0, idx, 1), 1)
                e = errs(fit_range(DISTS_BASE, med, 0.25, 14.0, lam_free, None), law)
                if e:
                    boots.append((e["p_err"], e["eps_err"], e["lam"]))
            if boots:
                b = np.array([(x[0], x[1]) for x in boots])
                rec["bootstrap"] = {"n": len(boots), "p_err_mean": float(b[:, 0].mean()), "p_err_ci95": [float(np.percentile(b[:, 0], 2.5)), float(np.percentile(b[:, 0], 97.5))],
                                    "eps_err_mean": float(b[:, 1].mean()), "eps_err_ci95": [float(np.percentile(b[:, 1], 2.5)), float(np.percentile(b[:, 1], 97.5))],
                                    "lam_values": [x[2] for x in boots[:20]]}
            if lam_free:
                med = np.median(y0, 1)
                _, _, lt = TRUTH[law]
                rec["lam_fixed_to_truth"] = errs(fit(DISTS_BASE, med, lt), law)
                rec["free_fit"] = errs(fit(DISTS_BASE, med, None), law)
                rec["exact_truth_curve_free_fit"] = errs(fit(DISTS_BASE, np.median(yt0, 1), None), law)
                rec["lam_profile"] = lam_profile(DISTS_BASE, med)
            res[law] = rec
            e = rec["placement_seeds"]["4738"]
            print(name, law, "seed 4738 p_err %.3f eps_err %.3f" % (e["p_err"], e["eps_err"]), "| seeds p_err", [round(v["p_err"], 3) for v in rec["placement_seeds"].values()],
                  "| dense", round(rec["dense_grid_56"]["p_err"], 3), flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
