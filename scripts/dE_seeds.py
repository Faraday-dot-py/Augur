"""Many-seed paired energy-drift comparison: stock Exp E vs the learned ghost closure (and E-vs-E reruns for the noise floor).

Same scenes for every arm (nr100 and bh300 regimes of scripts/scatter_conservation.py, one scene per seed), 1000 ticks, per-scene |dE|/(K0+|U0|) at the
sampled ticks plus outside-grid fraction, bound fraction and virial. Arms are rolled sequentially in one process. `det` arms turn on
torch.use_deterministic_algorithms (index_add_ becomes sort-based), cudnn.deterministic and drop the cudnn.benchmark ("graph") opt; `nd` arms are the harness default stack.
Per-arm npz saved as soon as it finishes; an existing file is skipped (resume).

Usage: PYTHONPATH=. python scripts/dE_seeds.py --ckpt-e E.pt --ckpt-g ghost_lsq.pt --out results/dE_seeds [--n-scenes 128] [--arms E_nd1,G_nd1,...]
"""
import argparse
import os
import time

import numpy as np
import torch

from scripts import scatter_conservation as sc
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

VARIANTS = {"E": "ms_kp_pot_v_g128", "G": "ms_kp_pot_v_g128_ghost", "B": "ms_kp_pot_v_g128", "C": "ms_kp_pot_v_g128"}
TICKS = np.array([0, 10, 50, 100, 200, 300, 500, 700, 1000])
DT, EPS = 0.1, 0.5
FIELDS = ("E", "K", "U", "outgrid", "bound", "vir", "finite")
DEFAULT_ARMS = "E_det1,E_det2,G_det1,G_det2,E_nd1,E_nd2,E_nd3,G_nd1,G_nd2,G_nd3"


def scene_seeds(n):
    return [4738] + [20000 + i for i in range(1, n)]


def make_scenes(kind, n, seeds):
    ps, vs = [], []
    for s in seeds:
        p, v = sc.init_scene(kind, n, np.random.default_rng(s))
        ps.append(p)
        vs.append(v)
    return np.stack(ps), np.stack(vs)


class Rec(sc.Recorder):
    def __call__(self, *a, **k):
        det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            return super().__call__(*a, **k)
        finally:
            torch.use_deterministic_algorithms(det)


def build(arm, args, dev):
    det = "_det" in arm
    torch.use_deterministic_algorithms(det, warn_only=False)
    torch.backends.cudnn.deterministic = det
    model = tsf.build(tsf.EXPS["E"], VARIANTS[arm[0]], DT).to(dev)
    ck = {"E": args.ckpt_e, "G": args.ckpt_g, "B": args.ckpt_b, "C": args.ckpt_c}[arm[0]]
    sf.load_compat(model, torch.load(ck, map_location=dev, weights_only=False)["model"])
    model.eval()
    sf.apply_opts(model, "default,-graph" if det else "default")
    torch.backends.cudnn.benchmark = not det
    return model


def run_arm(arm, args, dev, seeds):
    model = build(arm, args, dev)
    out = {"ticks": TICKS, "seeds": np.array(seeds)}
    for name in args.regimes.split(","):
        kind, n, _ = sc.REGIMES[name]
        pos0, vel0 = make_scenes(kind, n, seeds)
        P0, V0 = torch.tensor(pos0, device=dev), torch.tensor(vel0, device=dev)
        rel = kind == "bh"
        r0 = (P0 - P0.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt()
        rec = Rec(rel, EPS, model.extent, r0, args.steps, len(seeds), dev, DT)
        t0 = time.time()
        done = sc.roll_model(model, P0, V0, rel, args.steps, rec, None, DT)
        torch.cuda.synchronize()
        for k in FIELDS:
            out[f"{name}_{k}"] = rec.rows[k][torch.tensor(TICKS, device=dev)].cpu().numpy()
        out[f"{name}_done"] = np.array(done)
        print(f"{arm} {name} B={len(seeds)} done={done} {time.time() - t0:.1f}s", flush=True)
        del rec
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-e", required=True)
    ap.add_argument("--ckpt-g", default="")
    ap.add_argument("--ckpt-b", default="")
    ap.add_argument("--ckpt-c", default="")
    ap.add_argument("--out", default="results/dE_seeds")
    ap.add_argument("--n-scenes", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--regimes", default="nr100,bh300")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    seeds = scene_seeds(args.n_scenes)
    print(torch.__version__, torch.cuda.get_device_name(dev), flush=True)
    for arm in args.arms.split(","):
        f = f"{args.out}/{arm}.npz"
        if os.path.exists(f):
            print(f"skip {arm} (exists)", flush=True)
            continue
        np.savez_compressed(f, **run_arm(arm, args, dev, seeds))
        print(f"saved {f}", flush=True)


if __name__ == "__main__":
    main()
