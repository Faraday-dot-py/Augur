"""Training-window vs eval-window body count N and local density (neighbours within R of each body inside the same window).
Training: harvested windows (frame 0 mask) and replayed isolated Exp E scenes (all bodies, first frame). Eval: the hetero_tier_checks phase-a
window cutouts (sparse/mid/dense), seeds given. Prints quantiles. Usage: PYTHONPATH=. python scripts/window_density_tab.py --data data3 --out tab.json"""
import argparse
import glob
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf
from scripts.hetero_tier_checks import DT, EPS, HALF, SCENES_A, SUB
from scripts.scatter_scaling import make_scene

R = 4.0
Q = (5, 25, 50, 75, 95)


def local_n(x):
    d = torch.cdist(x, x)
    return ((d < R).sum(-1) - 1).float()


def qs(v):
    v = np.asarray(v, dtype=float)
    return [round(float(np.percentile(v, q)), 2) for q in Q] if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pattern", default="hv3_s*.npz")
    ap.add_argument("--seeds", default="9000,12000")
    ap.add_argument("--n-iso", type=int, default=1500)
    ap.add_argument("--out", default="tab.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    Ns, Ls = [], []
    for f in sorted(glob.glob(f"{a.data}/{a.pattern}")):
        z = np.load(f)
        for pos, mk in zip(z["pos"][:, 0], z["mask_f"][:, 0]):
            x = torch.tensor(pos[mk > 0], device=dev)
            Ns.append(len(x))
            Ls.extend(local_n(x).cpu().tolist())
    res["harvest"] = dict(windows=len(Ns), N=qs(Ns), local=qs(Ls))
    data = tsf.make_data(tsf.EXPS["E"], a.n_iso, 40, 4738 + 77, dev, 0.1, 0.5)
    P, V, M, mask = [t.float() for t in sf.to_tensors(data, dev)]
    Ns, Ls = [], []
    for i in range(len(P)):
        x = P[i, 0][mask[i] > 0]
        Ns.append(len(x))
        Ls.extend(local_n(x).cpu().tolist())
    res["replay_expE"] = dict(scenes=len(Ns), N=qs(Ns), local=qs(Ls))
    for sname, (nb, sig, ext, centres) in SCENES_A.items():
        Ns, Ls = [], []
        for seed in [int(x) for x in a.seeds.split(",")]:
            p0, v0 = make_scene(nb, sig, 1.0, ext, seed, 0)
            Pt, _ = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), 20, DT, SUB, EPS)
            for c in centres:
                for t in (5, 10, 20):
                    rel = Pt[t] - torch.tensor(c, device=dev, dtype=torch.float64)
                    x = rel[(rel.abs() < HALF).all(-1)].float()
                    Ns.append(len(x))
                    Ls.extend(local_n(x).cpu().tolist())
        res["eval_" + sname] = dict(windows=len(Ns), N=qs(Ns), local=qs(Ls))
    for k, v in res.items():
        print("TAB", k, json.dumps(v), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
