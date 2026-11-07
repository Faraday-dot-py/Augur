"""Reciprocal neighbour list, eval-only step 0 (audit row 32, close-pair proposal R4; docs/debugging/director-feasibility-2026-11-07.md).
Frozen model kern/E_frozen.pt, dense pair path (N <= 2000). For compact-cluster, Plummer and two-clump scenes at N=1000 (mass 1, v 0, seed 4738) it measures
(a) the fraction of in-window pairs (r < pp window) held by only one body's k=16 list, (b) the change in squared force error of nn<1 bodies when the knn pair term is replaced by the
union-list pair term (pair kept if either body selects it, force on each end from the same per-source weight; no 2k cap), same frozen weights, same momfix,
(c) momfix mean-acceleration magnitude relative to rms |a|. Truth = fp64 softened point sum. Self-check: the single-sided path rebuilt here must equal model.pp_acc.
Usage: PYTHONPATH=. python scripts/reciprocal_neighbour_list_step0.py --ckpt kern/E_frozen.pt --out results/reciprocal_neighbour_list_step0.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts.pm_baseline_force_error import EPS, learned_acc, sample

N = 1000
K = 16
FAMILIES = (("compact_cluster", "plummer", 0.25), ("plummer", "plummer", 1.0), ("two_clump", "two_clump", 1.0))


def pair_terms(model, pos, mass, mask):
    B, n = mask.shape
    d_all = pos[:, None, :, :] - pos[:, :, None, :]
    r_all = torch.sqrt((d_all ** 2).sum(-1) + 1e-8)
    bad = (1 - mask)[:, None, :] + torch.eye(n, device=pos.device, dtype=pos.dtype)[None]
    _, idx = torch.topk(r_all + 1e6 * (bad > 0).to(pos.dtype), K, dim=-1, largest=False)
    S = torch.zeros(B, n, n, dtype=torch.bool, device=pos.device)
    S.scatter_(2, idx, True)
    S = S & (bad[:, :, :] == 0) & (mask[:, :, None] > 0)
    mj = (mass * mask)[:, None, :].expand(B, n, n)
    w = model._pair_w(r_all, mj, torch.ones_like(r_all))
    f = w[..., None] * d_all / r_all[..., None]
    win = r_all < model.ppe()
    return S, win, f


def momfix(model, a, mass, mask):
    mm = (mass * mask)[..., None]
    return a - (mm * a).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenes", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    if a.ckpt != "random":
        model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    assert model.pp_mode(N) == "dense" and model.knn == K, (model.pp_mode(N), model.knn)
    rng = np.random.default_rng(a.seed)
    res = {"seed": a.seed, "n": N, "k": K, "warm": a.warm, "families": {}}
    pp_orig = model.pp_acc
    for name, fam, scale in FAMILIES:
        pos = torch.tensor(np.stack([sample(fam, N, rng, scale) for _ in range(a.scenes)]), dtype=torch.float32, device=a.device)
        vel = torch.zeros_like(pos)
        mass = torch.ones(a.scenes, N, device=dev)
        mask = torch.ones(a.scenes, N, device=dev)
        at = torch.stack([ops.exact_acc(p.double(), EPS) for p in pos]).float()
        with torch.no_grad():
            S, win, f = pair_terms(model, pos, mass, mask)
            check = (f * S[..., None]).sum(2)
            ref = pp_orig(pos, mass, mask, model.knn)
            err = float((check - ref).abs().max() / ref.abs().max().clamp_min(1e-9))
            assert err < 1e-4, f"single-sided rebuild differs from model.pp_acc: {err}"
            U = S | S.transpose(1, 2)
            union_pp = (f * U[..., None]).sum(2)
            one_sided = (S ^ S.transpose(1, 2)) & win & (S | S.transpose(1, 2))
            in_win = (U & win)
            frac_one_sided = float(one_sided.sum() / in_win.sum().clamp_min(1))
            frac_missing = float(((win & ~U & ~torch.eye(N, dtype=torch.bool, device=dev)[None]).sum()) / (win.sum() - a.scenes * N).clamp_min(1))
            a_knn = learned_acc(model, pos, vel, mass, mask, a.warm)
            model.pp_acc = lambda p, m, k, knn=16, qmask=None, _u=union_pp: _u
            a_uni = learned_acc(model, pos, vel, mass, mask, a.warm)
            model.pp_acc = pp_orig
            model.momfix = False
            a_raw = learned_acc(model, pos, vel, mass, mask, a.warm)
            model.momfix = True
        d = pos[:, :, None, :] - pos[:, None, :, :]
        r = (d ** 2).sum(-1).sqrt() + torch.eye(N, device=dev)[None] * 1e9
        nn_close = r.min(-1).values < 1.0
        sse = lambda x: float((((x - at) ** 2).sum(-1) * nn_close).sum())
        per = []
        for b in range(a.scenes):
            m = nn_close[b]
            per.append(float((((a_uni[b] - at[b]) ** 2).sum(-1)[m]).sum() / (((a_knn[b] - at[b]) ** 2).sum(-1)[m]).sum().clamp_min(1e-12)) if int(m.sum()) else None)
        rms = (at.norm(dim=-1) ** 2).mean().sqrt()
        res["families"][name] = {"nn_lt_1_bodies": int(nn_close.sum()), "frac_pairs_one_sided": frac_one_sided, "frac_in_window_pairs_dropped_by_both": frac_missing,
                                 "sse_nn_lt_1_knn": sse(a_knn), "sse_nn_lt_1_union": sse(a_uni), "sse_ratio_union_over_knn_pooled": sse(a_uni) / sse(a_knn),
                                 "sse_ratio_per_scene": per, "scenes_union_better_by_2pct": sum(1 for x in per if x is not None and x <= 0.98),
                                 "scenes_with_nn_lt_1": sum(1 for x in per if x is not None),
                                 "momfix_mean_acc_over_rms": float((a_raw.mean(1).norm(dim=-1)).mean() / rms),
                                 "all_body_sse_ratio": float(((a_uni - at) ** 2).sum() / ((a_knn - at) ** 2).sum())}
        print(name, res["families"][name], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
