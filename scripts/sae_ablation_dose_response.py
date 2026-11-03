"""Dose-response causal check for the mesh-net SAE (curiosity, no pass/kill). Replaces the UNet.dec[0] output at mass-bearing cells by the SAE
reconstruction with k features zeroed, k in (1, 2, 5, 10, 20). Arms per k: top-k law-aligned features (highest ANOVA eta^2 among live), and `draws`
random sets of k live features, each member matched to one top feature by activation frequency (drawn from the 10 live non-top features nearest in
log frequency). Reports per-law force-error median/p90 against exact fp64 force.
Usage: PYTHONPATH=. python scripts/sae_ablation_dose_response.py --ckpt ckpt/main_s4738.pt --sae results/sae_interp_lowdead_dec0_s4738_sae.pt --key mesh_64 --out results/sae_ablation_dose_response_s4738_mesh_64.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_multi as sm
from scripts.sae_ablation import encode, force_errors
from scripts.sae_interp import LAWS, eta2, mesh_acts

KS = (1, 2, 5, 10, 20)


def coarse_mask(m_, x):
    s = m_.shape[0] // x.shape[-1]
    return m_ if s == 1 else m_.reshape(x.shape[-2], s, x.shape[-1], s).any(3).any(1)


def matched_draw(rng, top, live_idx, freq, taken_pool=10):
    pool = np.array([j for j in live_idx if j not in set(top.tolist())])
    lf = np.log(freq[pool] + 1e-9)
    out = []
    for j in top.tolist():
        order = np.argsort(np.abs(lf - np.log(freq[j] + 1e-9)))
        cand = [pool[i] for i in order if pool[i] not in out][:taken_pool]
        out.append(int(rng.choice(cand)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", default="dec0")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument("--act-scenes", type=int, default=12)
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
    for k in ks:
        top = order[:k]
        arms[f"top{k}"] = top.tolist()
        for d in range(a.draws):
            arms[f"rand{k}_{d}"] = matched_draw(rng, top, live_idx, freq, a.pool_size)
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
    out = {"ckpt": a.ckpt, "sae": a.sae, "key": a.key, "layer": a.layer, "ks": ks, "draws": a.draws, "n_live": len(live_idx), "pool_size_requested": a.pool_size, "heldout_rank": a.heldout_rank, "feature_freq": [round(float(f), 5) for f in freq], "n_scenes_force": a.scenes,
           "n_nontop_live_by_k": {str(k): len(live_idx) - k for k in ks}, "draw_members": {n: m for n, m in arms.items() if n.startswith("rand")},
           "top_features": {str(k): [{"feature": int(j), "eta2": float(e_law[j]), "freq": float(freq[j])} for j in order[:k]] for k in ks}, "errors": {}}
    for law in a.laws.split(","):
        out["errors"][law] = {}
        for name, idx in arms.items():
            state["idx"] = idx
            out["errors"][law][name] = force_errors(model, dev, law, a.scenes, a.seed + 1000, 3)
        print(law, {k: round(out["errors"][law][k]["median"], 4) for k in ["native", "sae_reconstruction"] + [f"top{k}" for k in ks]},
              {f"rand{k}_mean": round(float(np.mean([out["errors"][law][f"rand{k}_{d}"]["median"] for d in range(a.draws)])), 4) for k in ks}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
