"""Force error on rollout tracers by coordinate bin near the edge: model force vs exact force (from the model's evolved massive bodies) for tracers with max-norm
coordinate in 22-32, per tick; shows where in the rollout the in-grid edge force is still wrong after the kernel ghost fix."""
import argparse
import json

import numpy as np
import torch

from scripts import edge_probe as ep
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

BINS = [(22, 26), (26, 28), (28, 30), (30, 31), (31, 32)]


@torch.no_grad()
def run(model, dev, seed, n=20000, nt=2000, steps=100):
    p0 = ep.scene(n, seed)
    rng = np.random.default_rng(seed + 1)
    v0 = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
    v0 -= v0.mean(0)
    ang = rng.uniform(0, 2 * np.pi, nt)
    r0 = rng.uniform(22, 26, nt)
    tp = np.stack([r0 * np.cos(ang), r0 * np.sin(ang)], 1)
    tv = 0.95 * ep.C * np.stack([np.cos(ang), np.sin(ang)], 1)
    pos = torch.tensor(np.concatenate([p0, tp]), dtype=torch.float32, device=dev)[None]
    mom = ep.mom_of(torch.tensor(np.concatenate([v0, tv]), dtype=torch.float32, device=dev)[None])
    M = torch.tensor(np.concatenate([np.ones(n), np.zeros(nt)]), dtype=torch.float32, device=dev)[None]
    mk = torch.ones_like(M)
    f = model.init_field(1, dev)[0]
    acc = {b: {"n": 0, "err_rad": 0.0, "exact_rad": 0.0, "model_rad": 0.0, "max_model": 0.0, "max_exact": 0.0} for b in BINS}
    a, f = model.force(pos, ep.sb.speed(mom, ep.C), M, mk, f)
    for t in range(steps):
        tr = pos[0, n:]
        ae = ep.tracer_exact(pos, n, 0.0)
        c = tr.abs().amax(-1)
        rhat = tr / tr.norm(dim=-1, keepdim=True)
        am = a[0, n:]
        for b in BINS:
            sel = (c >= b[0]) & (c < b[1])
            if sel.any():
                rm = (am[sel] * rhat[sel]).sum(-1)
                re = (ae[sel] * rhat[sel]).sum(-1)
                d = acc[b]
                d["n"] += int(sel.sum())
                d["err_rad"] += float((rm - re).abs().sum())
                d["exact_rad"] += float(re.abs().sum())
                d["model_rad"] += float(rm.sum())
                d["max_model"] = max(d["max_model"], float(am[sel].norm(dim=-1).max()))
                d["max_exact"] = max(d["max_exact"], float(ae[sel].norm(dim=-1).max()))
        mom = mom + 0.5 * ep.DT * a
        pos = pos + ep.DT * ep.sb.speed(mom, ep.C)
        a, f = model.force(pos, ep.sb.speed(mom, ep.C), M, mk, f)
        mom = mom + 0.5 * ep.DT * a
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--configs", default="base=none")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    for spec in args.configs.split(","):
        name, opts = spec.split("=")
        m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128_ghost", 0.1).to(dev)
        sf.load_compat(m, torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
        m.eval()
        sf.apply_opts(m, opts.replace("+", ","))
        ops.chunked_pp(m)
        acc = run(m, dev, 9000)
        res[name] = {f"{b[0]}-{b[1]}": v for b, v in acc.items()}
        for b, v in acc.items():
            print(f"BAND {name} coord {b[0]}-{b[1]} n {v['n']} rel_err {v['err_rad'] / max(v['exact_rad'], 1e-9):.3f} mean_model_rad {v['model_rad'] / max(v['n'], 1):.2f} "
                  f"max|a| model {v['max_model']:.0f} exact {v['max_exact']:.0f}", flush=True)
    json.dump(res, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
