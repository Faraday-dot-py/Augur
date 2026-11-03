"""Activation patching for the mesh-net SAE (curiosity, no pass/kill). For a pair of laws (source -> target) and the same scene: record the SAE codes of
the decoder layer under the source law, then run the model under the target law with the codes of k chosen features at mass-bearing cells replaced by the
recorded source codes. Reports whether the predicted force moves toward the source law: shift fraction = <patched - target_native, source_native - target_native>
/ |source_native - target_native|^2 (0 = no movement, 1 = reaches the source-law prediction), and rms force error against each law's exact force.
Arms: top-k law-aligned features (highest ANOVA eta^2 among live) and `draws` random sets matched by activation frequency.
Usage: PYTHONPATH=. python scripts/sae_activation_patching.py --ckpt ckpt/main_s4739.pt --sae results/X_sae.pt --key mesh_128 --layer dec2 --out results/Y.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_multi as sm
from scripts.sae_ablation import encode
from scripts.sae_ablation_dose_response import coarse_mask, matched_draw
from scripts.sae_interp import LAWS, eta2, mesh_acts

PAIRS = [("pow1", "pow2"), ("pow2", "pow1"), ("pow2", "pow3"), ("pow1", "pow3"), ("yuk8", "yuk128"), ("yuk128", "yuk8"), ("pow2", "yuk32"), ("pow2", "lj")]


def make_scene(rng, s, dev):
    p, v = sm.init_scene(rng, "uniform" if s % 2 == 0 else "cluster")
    n = p.shape[0]
    pad = lambda a: torch.tensor(np.concatenate([a, np.zeros((sm.N_PAD - n,) + a.shape[1:])]), dtype=torch.float32, device=dev)[None]
    mask = torch.zeros(1, sm.N_PAD, device=dev)
    mask[0, :n] = 1.0
    return pad(p), pad(v), mask, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", default="dec0")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--scenes", type=int, default=16)
    ap.add_argument("--act-scenes", type=int, default=12)
    ap.add_argument("--heldout-rank", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=50)
    ap.add_argument("--ks", default="1,5,20")
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
    ks = [int(k) for k in a.ks.split(",") if int(k) <= len(live_idx) // 2]
    rng = np.random.default_rng(a.seed)
    arms = {}
    for k in ks:
        top = order[:k]
        arms[f"top{k}"] = top.tolist()
        for d in range(a.draws):
            arms[f"rand{k}_{d}"] = matched_draw(rng, top, live_idx, freq, a.pool_size)
    st = {"mode": None, "rec": [], "ptr": 0, "idx": None}

    def hook(m, i, o):
        if st["mode"] is None:
            return o
        x = o[0]
        m_ = coarse_mask(model._rho_mask, x)
        z = encode(S, (x[:, m_].T - S["mu"]) / S["sc"])
        if st["mode"] == "record":
            st["rec"].append(z.clone())
        else:
            if st["idx"]:
                ix = torch.tensor(st["idx"], device=dev)
                z[:, ix] = st["rec"][st["ptr"]][:, ix]
            st["ptr"] += 1
        rec = (z @ S["W"] + S["bd"]) * S["sc"] + S["mu"]
        y = x.clone()
        y[:, m_] = rec.T
        return y[None]

    model.net.dec[int(a.layer[3:])].register_forward_hook(hook)

    @torch.no_grad()
    def run(pos, vel, mask, law, mode, idx=None):
        slots = sm.slots_for("single", law, 1.0)
        typ = torch.zeros(1, sm.N_PAD, device=dev)
        model.set_law(torch.tensor([sm.cond_of(slots)], dtype=torch.float32, device=dev), typ)
        model._rho_mask = (model.scatter(pos, vel, mask, mask)[0, 0] > 0.05)
        field = model.init_field(1, dev)[0]
        st["mode"], st["idx"], st["ptr"] = mode, idx, 0
        if mode == "record":
            st["rec"] = []
        for _ in range(4):
            acc, field = model.force(pos, vel, mask, mask, field)
        par = torch.tensor(sm.params_of(slots), dtype=torch.float64, device=dev)[None]
        at = sm.exact_acc(pos.double(), typ, mask.double(), par).float()
        return acc, at

    out = {"ckpt": a.ckpt, "sae": a.sae, "key": a.key, "layer": a.layer, "ks": ks, "draws": a.draws, "n_live": len(live_idx), "pool_size_requested": a.pool_size,
           "heldout_rank": a.heldout_rank, "scenes": a.scenes, "top_features": {str(k): [{"feature": int(j), "eta2": float(e_law[j]), "freq": float(freq[j])} for j in order[:k]] for k in ks},
           "draw_members": {n: m for n, m in arms.items() if n.startswith("rand")}, "pairs": {}}
    for src, tgt in PAIRS:
        acc_sh = {n: [] for n in arms}
        err = {n: {"to_target": [], "to_source": []} for n in ["target_native", "source_native"] + list(arms)}
        scr = np.random.default_rng(a.seed + 2000)
        for s in range(a.scenes):
            pos, vel, mask, n = make_scene(scr, s, dev)
            a_src, t_src = run(pos, vel, mask, src, "record")
            a_tgt, t_tgt = run(pos, vel, mask, tgt, "patch", [])
            d = (a_src - a_tgt)[0, :n]
            dd = float((d ** 2).sum())
            rms = lambda t: float(((t ** 2).sum(-1)[0, :n]).mean().sqrt())
            e = lambda x, t: float(((x - t)[0, :n] ** 2).sum(-1).mean().sqrt()) / rms(t)
            err["target_native"]["to_target"].append(e(a_tgt, t_tgt)); err["target_native"]["to_source"].append(e(a_tgt, t_src))
            err["source_native"]["to_target"].append(e(a_src, t_tgt)); err["source_native"]["to_source"].append(e(a_src, t_src))
            for name, idx in arms.items():
                a_p, _ = run(pos, vel, mask, tgt, "patch", idx)
                acc_sh[name].append(float(((a_p - a_tgt)[0, :n] * d).sum()) / max(dd, 1e-20))
                err[name]["to_target"].append(e(a_p, t_tgt)); err[name]["to_source"].append(e(a_p, t_src))
        med = lambda v: float(np.median(v))
        res = {"shift_fraction_median": {n: med(v) for n, v in acc_sh.items()}, "error_median": {n: {k: med(v) for k, v in e_.items()} for n, e_ in err.items()}}
        out["pairs"][f"{src}->{tgt}"] = res
        sh = res["shift_fraction_median"]
        print(src, "->", tgt, {f"top{k}": round(sh[f"top{k}"], 3) for k in ks}, {f"rand{k}_mean": round(float(np.mean([sh[f"rand{k}_{d}"] for d in range(a.draws)])), 3) for k in ks},
              "tgt_native_err", round(res["error_median"]["target_native"]["to_target"], 3), "src_native_err_vs_tgt", round(res["error_median"]["source_native"]["to_target"], 3), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
