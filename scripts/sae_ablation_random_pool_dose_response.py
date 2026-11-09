"""Random-pool dose-response ablation for the mesh-net SAE (curiosity, no pass/kill). Like sae_ablation_dose_response.py, but each random draw takes, per top
feature (slot), one member sampled without replacement from the `pool_size` live non-top features nearest in log frequency to that slot's top feature (all
of them when fewer exist), with `draws` draws per k, so the random arm is a distribution. Every draw's members are saved. Per law and k the summary gives the
top-k median force error against the random draws (median, p5, p95, mean, max), the percentile rank of top-k among the draws (ties count half), and random
outlier draws and features.
Usage: PYTHONPATH=. python scripts/sae_ablation_random_pool_dose_response.py --ckpt ckpt/main_s4738.pt --sae results/sae_interp_lowdead_dec0_s4738_large_dictionaries_sae.pt --key mesh_256 --out results/x.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from scripts import scatter_multi as sm
from scripts.sae_ablation import encode, force_errors
from scripts.sae_ablation_dose_response import KS, coarse_mask
from scripts.sae_interp import LAWS, eta2, mesh_acts


def pooled_draw(rng, top, live_idx, freq, pool_size):
    top_set = set(top.tolist())
    pool = np.array([j for j in live_idx if j not in top_set])
    lf = np.log(freq[pool] + 1e-9)
    out = []
    for j in top.tolist():
        order = np.argsort(np.abs(lf - np.log(freq[j] + 1e-9)), kind="stable")
        cand = [int(pool[i]) for i in order[:pool_size] if int(pool[i]) not in out]
        out.append(int(rng.choice(cand)))
    return out


def summarise(top_err, rand_errs, members, recon_err):
    r = np.array(rand_errs)
    q1, q3 = np.percentile(r, [25, 75])
    fence = max(q3 + 3 * (q3 - q1), 1.2 * recon_err)
    outlier_draws = [d for d in range(len(r)) if r[d] > fence]
    by_feature = defaultdict(list)
    for d, m in enumerate(members):
        for j in m:
            by_feature[j].append(r[d])
    worst = sorted(((float(np.mean(v)), j, len(v)) for j, v in by_feature.items() if len(v) >= 3), reverse=True)[:5]
    return {"top": top_err, "recon": recon_err, "rand_median": float(np.median(r)), "rand_p5": float(np.percentile(r, 5)), "rand_p95": float(np.percentile(r, 95)),
            "rand_mean": float(r.mean()), "rand_max": float(r.max()), "rank_percentile": float(((r < top_err).sum() + 0.5 * (r == top_err).sum()) / len(r) * 100),
            "top_above_p95": bool(top_err > np.percentile(r, 95)), "outlier_fence": float(fence), "outlier_draws": outlier_draws,
            "worst_random_features": [{"feature": j, "mean_draw_error": m, "n_draws": n} for m, j, n in worst]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", default="dec0")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=50)
    ap.add_argument("--act-scenes", type=int, default=24)
    ap.add_argument("--heldout-rank", type=int, default=0)
    ap.add_argument("--laws", default=",".join(LAWS))
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = sm.build().to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    S = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(a.sae, map_location="cpu")[a.key].items()}
    X, lab, _, sid = mesh_acts(model, dev, a.act_scenes, 400, 3, a.seed, a.layer)
    if a.heldout_rank:
        grp = sid % a.act_scenes
        keep = grp > grp.max() - a.heldout_rank
        X, lab = X[keep], lab[keep]
    Z = encode(S, (X.to(dev) - S["mu"]) / S["sc"])
    live_t = (Z > 0).any(0)
    e_law = eta2(Z, lab.to(dev), len(LAWS))
    freq = (Z > 0).float().mean(0).cpu().numpy()
    live_idx = live_t.nonzero()[:, 0].cpu().numpy().tolist()
    order = torch.argsort(e_law * live_t, descending=True).cpu().numpy()
    ks = [k for k in KS if k <= len(live_idx) // 2]
    rng = np.random.default_rng(a.seed)
    arms = {"native": None, "sae_reconstruction": []}
    members = {}
    for k in ks:
        top = order[:k]
        arms[f"top{k}"] = top.tolist()
        members[k] = [pooled_draw(rng, top, live_idx, freq, a.pool_size) for _ in range(a.draws)]
        for d, m in enumerate(members[k]):
            arms[f"rand{k}_{d}"] = m
    state = {"idx": None}

    def hook(m, i, o):
        if state["idx"] is None:
            return o
        x = o[0]
        m_ = coarse_mask(model._rho_mask, x)
        z = encode(S, (x[:, m_].T - S["mu"]) / S["sc"])
        if state["idx"]:
            z[:, torch.tensor(state["idx"], device=dev)] = 0.0
        rec = (z @ S["W"] + S["bd"]) * S["sc"] + S["mu"]
        y = x.clone()
        y[:, m_] = rec.T
        return y[None]

    model.net.dec[int(a.layer[3:])].register_forward_hook(hook)
    out = {"ckpt": a.ckpt, "sae": a.sae, "key": a.key, "layer": a.layer, "seed": a.seed, "ks": ks, "draws": a.draws, "pool_size": a.pool_size, "n_live": len(live_idx),
           "act_scenes": a.act_scenes, "heldout_rank": a.heldout_rank, "n_scenes_force": a.scenes, "feature_freq": [round(float(f), 5) for f in freq],
           "pool_available_by_k": {str(k): len(live_idx) - k for k in ks}, "draw_members": {str(k): members[k] for k in ks},
           "top_features": {str(k): [{"feature": int(j), "eta2": float(e_law[j]), "freq": float(freq[j])} for j in order[:k]] for k in ks},
           "errors": {}, "summary": {}, "complete": False}
    for law in a.laws.split(","):
        out["errors"][law] = {}
        for name, idx in arms.items():
            state["idx"] = idx
            out["errors"][law][name] = force_errors(model, dev, law, a.scenes, a.seed + 1000, 3)["median"]
        E = out["errors"][law]
        out["summary"][law] = {str(k): summarise(E[f"top{k}"], [E[f"rand{k}_{d}"] for d in range(a.draws)], members[k], E["sae_reconstruction"]) for k in ks}
        print(law, "native", round(E["native"], 4), "recon", round(E["sae_reconstruction"], 4),
              {k: (round(out["summary"][law][str(k)]["top"], 3), round(out["summary"][law][str(k)]["rand_p95"], 3), round(out["summary"][law][str(k)]["rank_percentile"], 1)) for k in ks}, flush=True)
        json.dump(out, open(a.out, "w"))
    out["complete"] = True
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE", a.key)


if __name__ == "__main__":
    main()
