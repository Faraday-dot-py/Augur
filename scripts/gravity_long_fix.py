import argparse
import time

import numpy as np
import torch

from model.token_free import TokenFreeDynamics
from scripts import gravity_sim as gs
from scripts.gravity_long import energy_t


def run(args, fix):
    p0, v0 = gs.init_bodies(args.bodies, np.random.default_rng(args.seed), scale=True)
    dev = torch.device(args.device)
    dyn = TokenFreeDynamics(n=1000, neighbor_radius=100.0, pair_impulse=True).to(dev)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    mp, mv = torch.tensor(p0, dtype=torch.float32, device=dev), torch.tensor(v0, dtype=torch.float32, device=dev)
    hidden = torch.zeros(args.bodies, dyn.hidden_dim, device=dev)
    e0 = energy_t(mp.double(), mv.double(), args.eps)
    out = [p0.copy()]
    energies = []
    for step in range(1, args.steps + 1):
        with torch.no_grad():
            dp, dv, hidden = dyn(mp, mv, hidden)
            if fix in ("mom", "mom_energy"):
                dp = dp - dp.mean(0, keepdim=True)
                dv = dv - dv.mean(0, keepdim=True)
            mp = mp + mv * args.dt + dp
            mv = mv + dv
            if fix == "mom_energy" and step % args.every == 0:
                e = energy_t(mp.double(), mv.double(), args.eps)
                ke = 0.5 * float((mv.double() ** 2).sum())
                target_ke = ke - (e - e0)
                if target_ke > 0 and ke > 0:
                    mv = mv * float((target_ke / ke) ** 0.5)
        if step % args.every == 0:
            out.append(mp.cpu().numpy().astype(np.float64))
            energies.append(energy_t(mp.double(), mv.double(), args.eps))
    return np.stack(out), np.array(energies)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_dynamics_v2.pt")
    ap.add_argument("--bodies", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fix", default="mom")
    ap.add_argument("--cache", default="results/gravity_long.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    truth = np.load(args.cache)["truth"]
    t0 = time.time()
    model, en = run(args, args.fix)
    n = len(model)
    gap = np.linalg.norm(model - truth[:n], axis=2).mean(1)
    com = model.mean(1)
    r = np.array([np.median(np.linalg.norm(m - np.median(m, 0), axis=1)) for m in model])
    tr = np.array([np.median(np.linalg.norm(m - np.median(m, 0), axis=1)) for m in truth[:n]])
    print(f"fix={args.fix} time {time.time()-t0:.0f}s")
    for i in [1, 5, 10, 25, 50, 100, 250, 500]:
        if i < n:
            print(f"step {i*args.every}: gap {gap[i]:.1f} COMdrift {np.linalg.norm(com[i]-com[0]):.1f} medR model {r[i]:.1f} truth {tr[i]:.1f} E {en[i-1]:.0f}")
    if args.out:
        np.savez_compressed(args.out, model=model, energy_m=en, every=args.every)
