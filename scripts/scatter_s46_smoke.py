"""Equivalence/backward checks for the opt-in scatter-field inputs (cu, epsin, norm). Run on Polaris:
PYTHONPATH=. python scripts/scatter_s46_smoke.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
At the trained operating point (h 0.5, eps 0.5, unit masses, modest density) each variant warm-started from E must reproduce E's force."""
import argparse

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

BASE = "ms_kp_pot_v_g128"


def load(ckpt, variant, dev):
    m = tsf.build(tsf.EXPS["E"], variant, 0.1).to(dev)
    sf.load_compat(m, torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    return m.eval()


def scene(n, dev, seed=4738, mass=None):
    rng = np.random.default_rng(seed)
    pos = torch.tensor(np.clip(rng.normal(0, 8.0, (1, n, 2)), -29, 29), dtype=torch.float32, device=dev)
    vel = torch.tensor(rng.normal(0, 0.5, (1, n, 2)), dtype=torch.float32, device=dev)
    m = torch.ones(1, n, device=dev) if mass is None else mass
    return pos, vel, m, torch.ones(1, n, device=dev)


def force(model, args):
    pos, vel, m, mk = args
    f = model.init_field(1, pos.device)[0]
    with torch.no_grad():
        return model.force(pos, vel, m, mk, f)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    base = load(args.ckpt, BASE, dev)
    sc = scene(200, dev)
    ref = force(base, sc)
    print("base |a| mean", float(ref.norm(dim=-1).mean()), flush=True)
    for v in ("cu", "eps", "mnorm", "dnorm", "cu_eps_mnorm"):
        m = load(args.ckpt, f"{BASE}_{v}", dev)
        a = force(m, sc)
        d = float((a - ref).abs().max())
        print(f"{v}: max |dA| vs base at operating point {d:.2e} (rel {d / float(ref.abs().max()):.2e})", flush=True)
        assert d < 1e-3 * float(ref.abs().max()) + 1e-5, v
    m = load(args.ckpt, f"{BASE}_mnorm", dev)
    heavy = scene(200, dev, mass=torch.full((1, 200), 3.0, device=dev))
    a3, a1 = force(m, heavy), force(base, scene(200, dev))
    print("mnorm equal-mass 3: |a3|/|a1| mean", float(a3.norm(dim=-1).mean() / a1.norm(dim=-1).mean()), "(3 = linear in mass)", flush=True)
    m = load(args.ckpt, f"{BASE}_cu", dev)
    for G in (64, 128, 256):
        ops.retarget(m, G, 64.0)
        a = force(m, sc)
        print(f"cu G{G}: |a| mean {float(a.norm(dim=-1).mean()):.4f} rel diff to base {float((a - ref).norm() / ref.norm()):.3f}", flush=True)
    m = load(args.ckpt, f"{BASE}_eps", dev)
    for e in (0.1, 0.5, 2.0):
        m.eps_cur = e
        a = force(m, sc)
        print(f"eps {e}: |a| mean {float(a.norm(dim=-1).mean()):.4f}", flush=True)
    for v in ("cu", "eps", "mnorm", "dnorm", "cu_eps_mnorm"):
        m = load(args.ckpt, f"{BASE}_{v}", dev).train()
        pos, vel, mm, mk = scene(64, dev)
        pos = pos.clone().requires_grad_(True)
        field = m.init_field(1, dev)
        p, vv, _, field, _ = sf._step_m(m, pos, vel, mm, mk, field)
        p, vv, _, field, _ = sf._step_m(m, p, vv, mm, mk, field)
        (p.sum() + vv.sum()).backward()
        g = sum(float(x.grad.abs().sum()) for x in m.parameters() if x.grad is not None)
        print(f"{v}: backward ok, grad sum {g:.3e}", flush=True)
    m = load(args.ckpt, f"{BASE}_cu", dev)
    for flag in ("cell", "adapt"):
        sf.apply_opt(m, flag)
        big = scene(1500, dev)
        a = force(m, big)
        m.opt.discard(flag)
        b = force(m, big)
        print(f"cu pair-term {flag} vs dense max diff {float((a - b).abs().max()):.2e}", flush=True)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
