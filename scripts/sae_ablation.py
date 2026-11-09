"""Causal check for the mesh-net SAE (curiosity, no pass/kill): replace the UNet.dec[0] output at mass-bearing cells by its SAE reconstruction with chosen
features zeroed, and measure the change in force error against the exact fp64 force, per law. Arms: native model, SAE reconstruction, SAE reconstruction
with the top law-aligned features ablated, SAE reconstruction with the same number of random live features ablated (control).
Usage: PYTHONPATH=. python scripts/sae_ablation.py --ckpt ckpt/main_s4738.pt --sae results/sae_interp_topk_dec0_s4738_sae.pt --key mesh_512 --out results/sae_ablation_s4738.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from scripts import scatter_multi as sm
from scripts.sae_interp import LAWS, eta2, mesh_acts


def encode(S, x):
    pre = (x - S["bd"]) @ S["We"] + S["be"]
    v, i = pre.topk(S["k"], dim=-1) if S["method"] == "topk" else (None, None)
    return torch.zeros_like(pre).scatter(-1, i, F.relu(v)) if S["method"] == "topk" else F.relu(pre)


@torch.no_grad()
def force_errors(model, dev, law, scenes, seed, warm, rho_thresh=0.05):
    rng = np.random.default_rng(seed)
    errs = []
    slots = sm.slots_for("single", law, 1.0)
    for s in range(scenes):
        p, v = sm.init_scene(rng, "uniform" if s % 2 == 0 else "cluster")
        n = p.shape[0]
        pad = lambda a: torch.tensor(np.concatenate([a, np.zeros((sm.N_PAD - n,) + a.shape[1:])]), dtype=torch.float32, device=dev)[None]
        pos, vel = pad(p), pad(v)
        mask = torch.zeros(1, sm.N_PAD, device=dev)
        mask[0, :n] = 1.0
        cond = torch.tensor([sm.cond_of(slots)], dtype=torch.float32, device=dev)
        typ = torch.zeros(1, sm.N_PAD, device=dev)
        model.set_law(cond, typ)
        model._rho_mask = (model.scatter(pos, vel, mask, mask)[0, 0] > rho_thresh)
        field = model.init_field(1, dev)[0]
        for _ in range(warm + 1):
            a, field = model.force(pos, vel, mask, mask, field)
        par = torch.tensor(sm.params_of(slots), dtype=torch.float64, device=dev)[None]
        at = sm.exact_acc(pos.double(), typ, mask.double(), par).float()
        rms = ((at.norm(dim=-1) ** 2 * mask).sum() / mask.sum()).sqrt()
        errs.append(((a - at).norm(dim=-1)[0, :n] / rms).cpu())
    e = torch.cat(errs)
    return {"median": float(e.median()), "p90": float(e.quantile(0.9)), "mean": float(e.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--key", default="mesh_512")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--n-ablate", type=int, default=5)
    ap.add_argument("--scenes", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = sm.build().to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    S = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(a.sae, map_location="cpu")[a.key].items()}
    X, lab, _, _ = mesh_acts(model, dev, 12, 400, 3, a.seed, "dec0")
    Xn = ((X.to(dev) - S["mu"].to(dev)) / S["sc"])
    Z = encode(S, Xn)
    e_law = eta2(Z, lab.to(dev), len(LAWS))
    live = (Z > 0).any(0)
    top = torch.argsort(e_law * live, descending=True)[: a.n_ablate]
    g = torch.Generator().manual_seed(a.seed)
    liv = live.nonzero()[:, 0].cpu()
    rand = liv[torch.randperm(len(liv), generator=g)[: a.n_ablate]].to(dev)
    top_info = [{"feature": int(j), "eta2": float(e_law[j]), "dominant_law": LAWS[int(torch.stack([Z[lab.to(dev) == i, j].mean() for i in range(len(LAWS))]).argmax())]} for j in top]
    arms = {"native": None, "sae_reconstruction": torch.tensor([], dtype=torch.long, device=dev), "ablate_top_law_features": top, "ablate_random_live_features": rand}
    mode = {"arm": None}

    def hook(m, i, o):
        if mode["arm"] is None:
            return o
        x = o[0]
        m_ = model._rho_mask
        cells = x[:, m_].T
        z = encode(S, (cells - S["mu"].to(dev)) / S["sc"])
        z[:, arms[mode["arm"]]] = 0.0
        rec = (z @ S["W"].to(dev) + S["bd"].to(dev)) * S["sc"] + S["mu"].to(dev)
        y = x.clone()
        y[:, m_] = rec.T
        return y[None]

    model.net.dec[0].register_forward_hook(hook)
    out = {"ckpt": a.ckpt, "sae": a.sae, "key": a.key, "top_law_features": top_info, "errors": {}}
    for law in LAWS:
        out["errors"][law] = {}
        for arm in arms:
            mode["arm"] = arm if arms[arm] is not None else None
            if arm == "sae_reconstruction":
                mode["arm"] = arm
            out["errors"][law][arm] = force_errors(model, dev, law, a.scenes, a.seed + 1000, 3)
        print(law, {k: round(v["median"], 4) for k, v in out["errors"][law].items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
