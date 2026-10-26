"""CPU rollouts of scatter-gravity-clustered-baseline and tier0-matched-60k-iterations on multiclump scenes, saved for scripts/loss_error_analyze.py.
Run with PYTHONPATH at a tree that has scripts/tiered_eval.py (worktree hetero-tiered-integration):
  PYTHONPATH=<tree> python3 scripts/loss_error_roll.py --n 1000 --scenes 8 --seed 9100 --baseline E.pt --tier0 T.pt --out results/loss_analysis/roll_n1000.npz"""
import argparse
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--span", type=float, default=64.0)
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--ic", default="multiclump")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tier0", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(12)
    dev = torch.device("cpu")
    from scripts import tiered_adapters as ta
    from scripts import tiered_eval as te
    from scripts import tiered_iface as ti

    models = {"baseline": te.wrap(te.load_model(f"scatter:{a.baseline}", dev, 0.1), a.span)}
    t0 = ta.load_t0ext(a.tier0, dev, 0.1)
    models["tier0"] = te.wrap(t0, a.span)
    models["tier0_tiled"] = ti.Tiled(t0, a.span, tile=32.0, sparse=True)
    out = {}
    tp, tv, ms = [], [], []
    res = {k: ([], []) for k in models}
    for s in range(a.scenes):
        b = te.grav_batch(a.ic, a.n, a.span, a.seed + s, 1, dev)
        rt = te.truth_roll(b, 0.1, a.steps)
        tp.append(rt["pos"][0].numpy()), tv.append(rt["vel"][0].numpy()), ms.append(b["mass"][0].numpy())
        for k, m in models.items():
            t = time.time()
            rm = te.model_roll(m, b, a.steps)
            res[k][0].append(rm["pos"][0].numpy()), res[k][1].append(rm["vel"][0].numpy())
            e = float((rm["pos"] - rt["pos"]).norm(dim=-1)[:, 20].mean())
            print(f"scene {s} {k} err@20 {e:.4f} {time.time() - t:.0f}s", flush=True)
    out = dict(truth_pos=np.stack(tp), truth_vel=np.stack(tv), mass=np.stack(ms), n=a.n, span=a.span)
    for k, (p, v) in res.items():
        out[f"{k}_pos"], out[f"{k}_vel"] = np.stack(p), np.stack(v)
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
