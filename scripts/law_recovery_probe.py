"""Interpretability rule recovery by full-model 2-body force probing (docs/debugging/top8-first-pass-2026-10-29.md row 1).
Two unit-mass bodies at separation d (random orientation and sub-cell offset), zero velocity, recurrent field warmed by `warm` fixed-position force
evaluations. Radial acceleration of body 0 (positive = attraction) is fitted to  F(d) = A d (d^2+eps^2)^(-(p+1)/2) exp(-d/lam)  by log-space grid search.
Checkpoints (Polaris ~/augur_nfr): kern/E_frozen.pt (Exp E gravity), ckpt/main_s4738.pt, ckpt/main_s4739.pt (mixed six-law MultiScatterField, law descriptor
input), ckpt/cmp_yuk32_single.pt, ckpt/cmp_pow1.5_single.pt (single-law comparators).
Usage: PYTHONPATH=. python scripts/law_recovery_probe.py --out results/law_recovery_probe.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_multi as sm
from scripts import scatter_scaling_ops as ops

DISTS = np.geomspace(0.25, 14.0, 28)
NEAR_MAX = 2.0
TRUTH = {"pow1": (1.0, 0.5, np.inf), "pow1.5": (1.5, 0.5, np.inf), "pow2": (2.0, 0.5, np.inf), "pow3": (3.0, 0.5, np.inf),
         "yuk8": (2.0, 0.5, 8.0), "yuk32": (2.0, 0.5, 32.0), "yuk128": (2.0, 0.5, 128.0)}
LAMS = np.concatenate([[np.inf], np.geomspace(2.0, 1024.0, 60)])


def form(d, p, eps, lam):
    return d * (d ** 2 + eps ** 2) ** (-(p + 1) / 2) * np.exp(-d / lam)


def _grid_fit(d, y, ps, epss, lams):
    ly = np.log(y)
    best = (np.inf, None)
    for p in ps:
        base = np.log(d)[None, None, :] - (p + 1) / 2 * np.log(d[None, None, :] ** 2 + epss[:, None, None] ** 2) - d[None, None, :] / lams[None, :, None]
        res = ly[None, None, :] - base
        la = res.mean(-1, keepdims=True)
        sse = ((res - la) ** 2).sum(-1)
        i = np.unravel_index(np.argmin(sse), sse.shape)
        if sse[i] < best[0]:
            best = (sse[i], (p, epss[i[0]], lams[i[1]], float(np.exp(la[i[0], i[1], 0]))))
    return best


def fit(d, y, lam_fixed=None):
    """returns dict p, eps, lam, A, rms_log_resid; lam_fixed None = free (power-law candidate included as inf), else fixed value."""
    ok = (y > 0) & np.isfinite(y)
    d, y = d[ok], y[ok]
    if len(d) < 6:
        return None
    lams = LAMS if lam_fixed is None else np.array([lam_fixed], dtype=float)
    s, (p, eps, lam, A) = _grid_fit(d, y, np.arange(0.4, 4.6, 0.02), np.geomspace(0.03, 3.0, 70), lams)
    lams2 = lams if lam_fixed is not None or not np.isfinite(lam) else lam * np.geomspace(0.7, 1.4, 25)
    s, (p, eps, lam, A) = _grid_fit(d, y, np.arange(p - 0.04, p + 0.0401, 0.002), eps * np.geomspace(0.8, 1.25, 40), lams2)
    return {"p": float(p), "eps": float(eps), "lam": float(lam) if np.isfinite(lam) else None, "A": A, "rms_log_resid": float(np.sqrt(s / len(d))), "n_points": int(len(d))}


def fits_for(d, y, law, lam_free):
    pt, et, lt = TRUTH[law]
    out = {}
    m_full = (d <= 14.0)
    m_near = (d <= NEAR_MAX)
    out["full"] = fit(d[m_full], y[m_full], None if lam_free else np.inf)
    out["near"] = fit(d[m_near], y[m_near], lt if np.isfinite(lt) else np.inf)
    for k, f in out.items():
        if f is not None:
            f["p_err"] = abs(f["p"] - pt) / pt
            f["eps_err"] = abs(f["eps"] - et) / et
            f["lam_err"] = (abs((f["lam"] or np.inf) - lt) / lt) if np.isfinite(lt) and f["lam"] else None
    return out


@torch.no_grad()
def curves(model, dev, shape, multi, n, warm, seed):
    g = torch.Generator().manual_seed(seed)
    if multi:
        slots = sm.slots_for("single", shape, 1.0)
        cond = torch.tensor([sm.cond_of(slots)], dtype=torch.float32, device=dev).expand(n, -1).contiguous()
        par = torch.tensor(sm.params_of(slots), dtype=torch.float64, device=dev)[None].expand(n, -1, -1).contiguous()
        typ0 = torch.zeros(n, 2, device=dev)
        model.set_law(cond, typ0)
    else:
        slots = sm.slots_for("single", "pow2", 1.0)
        par = torch.tensor(sm.params_of(slots), dtype=torch.float64, device=dev)[None].expand(n, -1, -1).contiguous()
        typ0 = torch.zeros(n, 2, device=dev)
    res = {"d": DISTS.tolist(), "full": [], "pair": [], "mesh": [], "truth": [], "full_mean": [], "full_std": []}
    for d in DISTS:
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
        ap = model.pp_acc(pos, mass, mask)
        at = sm.exact_acc(pos.double(), typ0, mask.double(), par).float()
        uh = (pos[:, 1] - pos[:, 0])
        uh = uh / uh.norm(dim=-1, keepdim=True)
        rad = lambda x: (x[:, 0] * uh).sum(-1)
        res["full"].append(float(rad(a).median()))
        res["full_mean"].append(float(rad(a).mean()))
        res["full_std"].append(float(rad(a).std()))
        res["pair"].append(float(rad(ap).median()))
        res["mesh"].append(float((rad(a) - rad(ap)).median()))
        res["truth"].append(float(rad(at).median()))
    return res


def load_multi(path, dev):
    m = sm.build().to(dev).eval()
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
    return m


def load_e(path, dev):
    m = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, in_scale=1.0, potential=True,
                        kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--models", default="E_frozen:kern/E_frozen.pt:E:pow2,mixed_s4738:ckpt/main_s4738.pt:M:pow1+pow1.5+pow2+pow3+yuk8+yuk32+yuk128+lj,"
                    "mixed_s4739:ckpt/main_s4739.pt:M:pow1+pow1.5+pow2+pow3+yuk8+yuk32+yuk128+lj,"
                    "yuk32_single:ckpt/cmp_yuk32_single.pt:M:yuk32,pow1.5_single:ckpt/cmp_pow1.5_single.pt:M:pow1.5")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {"seed": a.seed, "n": a.n, "warm": a.warm, "dists": DISTS.tolist(), "models": {}}
    d = DISTS
    for spec in a.models.split(","):
        name, path, kind, laws = spec.split(":")
        model = load_multi(path, dev) if kind == "M" else load_e(path, dev)
        out["models"][name] = {"ckpt": path}
        for law in laws.split("+"):
            cur = curves(model, dev, law, kind == "M", a.n, a.warm, a.seed)
            rec = {"curves": cur}
            if law != "lj":
                for key in ("full", "pair", "mesh", "truth"):
                    y = np.array(cur[key])
                    rec["fit_" + key] = fits_for(d, y, law, law.startswith("yuk"))
                rec["rel_err_vs_truth_median"] = float(np.median(np.abs(np.array(cur["full"]) - np.array(cur["truth"])) / np.abs(np.array(cur["truth"]))))
                f = rec["fit_full"]["full"]
                print(name, law, "full p %.3f eps %.3f lam %s" % (f["p"], f["eps"], f["lam"]) if f else "fit failed",
                      "| truth-fit p %.3f eps %.3f" % (rec["fit_truth"]["full"]["p"], rec["fit_truth"]["full"]["eps"]), flush=True)
            out["models"][name][law] = rec
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
