"""Check that pruning ghosts out of the pair term is exact in a collapsed fp32 state, and measure the run-to-run noise floor
(index_add_ nondeterminism) it must be compared with. Usage: PYTHONPATH=. python scripts/scatter_prune_check.py --ckpt ... --out results/scatter_prune_check.json"""
import argparse
import json

import torch

from scripts import scatter_field as sf
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling_cores import CoreTiled, uniform_cores
from scripts.scatter_tiling_eval import load


def stat(a, b, ref):
    d = (a - b).norm(dim=-1)
    n = ref.norm(dim=-1)
    return {"max_abs": float(d.max()), "p999": float(d.quantile(0.999)), "n_gt_1e-3_rel": int((d > 1e-3 * n.clamp_min(1e-3)).sum()),
            "max_force": float(n.max()), "rel_sum": float(d.sum() / n.sum())}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = load(a.ckpt, dev)
    sf.apply_opt(model, "adapt")
    res = {}
    for n, tick in ((20000, 10), (60000, 10), (60000, 40)):
        p0, v0 = make_scene(n, 16.0, 0.3, 128.0, 4738, 0)
        pos = torch.tensor(p0, dtype=torch.float32, device=dev)[None]
        vel = torch.tensor(v0, dtype=torch.float32, device=dev)[None]
        mass = torch.ones(n, device=dev)
        msk = torch.ones(1, n, device=dev)
        model.grid, model.extent, model.h = 256, 128.0, 0.5
        st = (model.init_field(1, dev)[0], None)
        for _ in range(tick):
            pos, vel, st, _ = model.step(pos, vel, mass[None], msk, st)
        p, v = pos[0], vel[0]
        out = {}
        forces = {}
        for name, prune in (("a", False), ("b", False), ("c", True), ("d", True)):
            T = CoreTiled(model, uniform_cores(), far=True, prune_pair=prune)
            forces[name] = T.force(p, v, mass, T.slice_fields(st[0][0]))[0]
        out["noise_unpruned_vs_unpruned"] = stat(forces["a"], forces["b"], forces["a"])
        out["noise_pruned_vs_pruned"] = stat(forces["c"], forces["d"], forces["c"])
        out["pruned_vs_unpruned"] = stat(forces["c"], forces["a"], forces["a"])
        res[f"N{n}_t{tick}"] = out
        print(f"N{n} t{tick}", out, flush=True)
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
