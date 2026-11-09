"""Feasibility test for deeper adaptive layers in the adaptive pair term (docs/debugging/pair-term-scaling-fallback-chase.md).
Builds a Plummer cloud, runs model.pp_acc under each opt variant (default, deeper levels, longer candidate lists, combinations), and records fallback bodies, per-level resolution,
steady time, the largest difference to the default output, the error against an independent brute-force exact top-knn pair term on sampled queries, and
(optionally) bit-identity of the default output against the git HEAD copy of scatter_field.py.
Usage: PYTHONPATH=. python scripts/pair_term_adaptive_depth_test.py --n 30000 --core-scale 0.025 --ckpt random --device cpu --out results/pair_term_adaptive_depth_test_cpu.json
"""
import argparse
import importlib.util
import json
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.pm_baseline_force_error import sample

VARIANTS = (("default", ()), ("adaptdeep", ("adaptdeep",)), ("adaptwide64", ("adaptwide64",)), ("adaptwide128", ("adaptwide128",)), ("adaptdeep-adaptwide64", ("adaptdeep", "adaptwide64")), ("adaptdeep-adaptwide128", ("adaptdeep", "adaptwide128")))


def make_model(mod, ckpt, dev):
    if ckpt == "random":
        torch.manual_seed(4738)
    model = mod.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, in_scale=1.0, potential=True,
                             kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    if ckpt == "random":
        torch.manual_seed(4738)
        torch.nn.init.normal_(model.ppmlp[-1].weight, std=0.3)
    else:
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    model.opt.add("adapt")
    return model


def brute_force(model, pos, mass, qi, chunk):
    p, m = pos[0], mass[0]
    n = p.shape[0]
    out = []
    for c0 in range(0, len(qi), chunk):
        q = qi[c0:c0 + chunk]
        d = p[None, :, :] - p[q][:, None]
        r_all = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        bad = torch.arange(n, device=p.device)[None] == q[:, None]
        r_sel, i = torch.topk(r_all + 1e6 * bad.to(p.dtype), model.knn, dim=-1, largest=False)
        ds = torch.gather(d, 1, i[..., None].expand(-1, -1, 2))
        out.append(model._pp_force(ds, r_sel, m[i]))
    return torch.cat(out)


def timed(fn, reps, dev):
    fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.time()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--core-scale", type=float, required=True)
    ap.add_argument("--ckpt", default="random")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--head-module", default=None)
    ap.add_argument("--variants", default=",".join(v for v, _ in VARIANTS))
    ap.add_argument("--untimed", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed)
    pos = torch.tensor(sample("plummer", a.n, rng, a.core_scale), dtype=torch.float32, device=dev)[None]
    mass = torch.ones(1, a.n, device=dev)
    mask = torch.ones(1, a.n, device=dev)
    qi = torch.tensor(np.sort(rng.choice(a.n, min(a.queries, a.n), replace=False)), device=dev)
    res = {"n": a.n, "core_scale": a.core_scale, "ckpt": a.ckpt, "device": a.device, "seed": a.seed, "variants": {}}
    model = make_model(sf, a.ckpt, dev)
    with torch.no_grad():
        ref = brute_force(model, pos, mass, qi, max(1, 2 ** 24 // a.n))
        rms = float(ref.norm(dim=-1).pow(2).mean().sqrt())
        res["ref_rms_acc"] = rms
        base = None
        for name, flags in VARIANTS:
            if name not in a.variants.split(","):
                continue
            model.opt = {"adapt", *flags}
            model.adapt_profile = False
            torch.cuda.synchronize() if dev.type == "cuda" else None
            t_first = time.time()
            out = model.pp_acc(pos, mass, mask, model.knn)
            torch.cuda.synchronize() if dev.type == "cuda" else None
            t_first = time.time() - t_first
            untimed = name in a.untimed.split(",")
            ts = [] if untimed else timed(lambda: model.pp_acc(pos, mass, mask, model.knn), a.reps, dev)
            model.adapt_profile = True
            model.adapt_keep_idx = True
            if not untimed:
                model.pp_acc(pos, mass, mask, model.knn)
            model.adapt_profile = False
            fb = model.adapt_stats.pop("fallback_idx", None)
            model.adapt_keep_idx = False
            st = json.loads(json.dumps(model.adapt_stats, default=str)) if not untimed else {"fallback": None}
            if base is None:
                base = out
            err = (out[0, qi] - ref).norm(dim=-1)
            res["variants"][name] = {"first_call_s": t_first, "steady_s": ts, "steady_mean_s": float(np.mean(ts)) if ts else None, "fallback": st["fallback"], "fallback_frac": st["fallback"] / a.n if ts else None,
                                     "levels": st.get("levels"), "over_cap": st.get("over_cap"), "level_s": st.get("level_s"), "fallback_s": st.get("fallback_s"),
                                     "max_abs_err_vs_bruteforce": float(err.max()), "rms_err_vs_bruteforce": float(err.pow(2).mean().sqrt()),
                                     "max_abs_diff_vs_default_all_bodies": float((out - base).abs().max()), "bit_equal_to_default": bool(torch.equal(out, base))}
            if fb is not None and 0 < len(fb) <= 5000 and a.n <= 200000:
                c = pos[0].median(0).values
                dd = torch.cdist(pos[0, fb], pos[0])
                res["variants"][name]["leftover_radius_from_median"] = (pos[0, fb] - c).norm(dim=-1).tolist()[:200]
                res["variants"][name]["leftover_r16"] = dd.kthvalue(model.knn + 1, dim=1).values.tolist()[:200]
            print(name, {k: res["variants"][name][k] for k in ("steady_mean_s", "fallback", "fallback_frac", "max_abs_err_vs_bruteforce", "max_abs_diff_vs_default_all_bodies")}, flush=True)
        if a.head_module:
            spec = importlib.util.spec_from_file_location("scatter_field_head", a.head_module)
            head = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(head)
            hm = make_model(head, a.ckpt, dev)
            hm.opt = {"adapt"}
            model.opt = {"adapt"}
            res["default_bit_identical_to_head"] = bool(torch.equal(hm.pp_acc(pos, mass, mask, hm.knn), model.pp_acc(pos, mass, mask, model.knn)))
            print("default bit identical to HEAD:", res["default_bit_identical_to_head"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
