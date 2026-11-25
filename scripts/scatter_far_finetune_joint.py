"""R5 augmentation: fine-tune ONLY the kernel (4,417 params, U-Net / pair term frozen) end-to-end through the model rollout on wide
Gaussian scenes (single G=512 grid, which the tiled+far scheme reproduces to .004), loss = relative acceleration error vs exact fp64
along the model's own rollout (BPTT over --unroll ticks with the recurrent field). Training scene seeds >= 30000 (never 4738/9000/12000).
--anchor W adds W * (K - K_start)^2 on r <= 28 (keeps the interior kernel near the starting one).
Usage: PYTHONPATH=. python scripts/scatter_far_finetune_joint.py --ckpt kern/E_frozen.pt --init kern/r1_mlp.pt --out kern/j1.pt [--anchor 100]
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import scatter_far as sfar
from scripts import scatter_scaling_ops as ops
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling_eval import EPS, load

L, N = 256.0, 4800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=4738)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    m = load(a.ckpt, dev)
    for p in m.parameters():
        p.requires_grad_(False)
    ref = sfar.load_kernel(a.init, dev)
    mod = sfar.load_kernel(a.init, dev)
    for p in mod.parameters():
        p.requires_grad_(True)
    mod.train()
    sfar.install(m, mod)
    ops.retarget(m, round(L / 0.5), L)
    opt = torch.optim.Adam(mod.parameters(), lr=a.lr)
    rk = torch.rand(4096, device=dev) * 28.0
    log = []
    t0 = time.time()
    for it in range(a.iters + 1):
        lr = a.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * it / a.iters)))
        for g in opt.param_groups:
            g["lr"] = lr
        seed = 30000 + int(rng.integers(1_000_000))
        sigma = float(rng.uniform(24, 40))
        p0, v0 = make_scene(N, sigma, float(rng.uniform(0.7, 1.3)), L, seed, 0)
        pos = torch.tensor(p0, dtype=torch.float32, device=dev)[None]
        vel = torch.tensor(v0, dtype=torch.float32, device=dev)[None]
        mass = torch.ones(1, N, device=dev)
        msk = torch.ones(1, N, device=dev)
        st = (m.init_field(1, dev)[0], None)
        loss = 0.0
        for k in range(a.unroll):
            pos, vel, st, _ = m.step(pos, vel, mass, msk, st)
            a_m = st[1][0]
            a_x = ops.exact_acc(pos[0].detach().double(), EPS).float()
            loss = loss + ((a_m - a_x).norm(dim=-1).sum() / a_x.norm(dim=-1).sum())
        loss = loss / a.unroll
        if a.anchor > 0:
            with torch.no_grad():
                kr = ref(rk)
            loss = loss + a.anchor * ((mod(rk) - kr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0:
            print(f"it {it} loss {float(loss):.4f} sigma {sigma:.1f} t {time.time() - t0:.0f}s", flush=True)
            log.append((it, float(loss)))
        if it % 100 == 0:
            torch.save({"arch": "mlp", "kw": {"pp": m.pp, "split": m.split}, "state": {k: v.detach().clone() for k, v in mod.state_dict().items()},
                        "meta": {"args": vars(a), "it": it}}, a.out)
    json.dump({"args": vars(a), "log": log}, open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
