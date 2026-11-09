"""Rollout dump (Polaris GPU) and grid render (local) for MultiScatterField checkpoints.
Dump:   PYTHONPATH=. python scripts/render_multi_rollout_grid.py dump --ckpt s4738=a.pt --ckpt s4739=b.pt --e-ckpt kern/E_frozen.pt --out traj.npz
Render: python scripts/render_multi_rollout_grid.py render --npz traj.npz --outdir videos
Per (model, case): truth/model positions of the first NB bodies of the first NS scenes (eval seed 9000, same scenes as scatter_multi_eval), per-tick mean
position error for all scenes, and a bad flag (non-finite or |x| > 1e3) per scene."""
import argparse

import numpy as np

CASES = ["grav", "pow1", "HELD_pow1.5", "HELD_yuk32", "HELD_pp_pow1.5"]
NS, NB, SEED, STEPS = 4, 6, 9000, 20


def dump(a):
    import torch

    from scripts import scatter_field as sf
    from scripts import scatter_multi as sm
    from scripts.scatter_multi_eval import CASES as EC

    dev = torch.device("cuda")
    models = {}
    for spec in a.ckpt:
        nm, path = spec.split("=")
        m = sm.build().to(dev).eval()
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
        models[nm] = (m, False)
    if a.e_ckpt:
        e = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=sm.DT, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
        e.load_state_dict(torch.load(a.e_ckpt, map_location=dev, weights_only=False)["model"])
        models["E"] = (e, True)
    out = {}
    for c in CASES:
        shape, st = EC[c]
        d = sm.gen_cases([(shape, st, 1.0)] * a.n, SEED, dev, STEPS, tag=2)
        P, V, mask, typ, cond = d["P"], d["V"], d["mask"], d["typ"], d["cond"]
        S, N = P.shape[0], P.shape[2]
        mass = torch.ones(S, N, device=dev)
        idx = np.stack([np.nonzero(mask[s].cpu().numpy() > 0)[0][:NB] for s in range(NS)])
        out[f"{c}/truth"] = np.stack([P[s][:, idx[s]].cpu().numpy() for s in range(NS)])
        for nm, (m, is_e) in models.items():
            if is_e and c != "grav":
                continue
            with torch.no_grad():
                if not is_e:
                    m.set_law(cond, typ)
                p, v = P[:, 0], V[:, 0]
                field = m.init_field(S, dev)
                traj, errs = [p.clone()], []
                for t in range(1, STEPS + 1):
                    p, v, field, _ = m.step(p, v, mass, mask, field)
                    traj.append(p.clone())
                    errs.append((((p - P[:, t]).norm(dim=-1) * mask).sum(1) / mask.sum(1)).cpu().numpy())
            T = torch.stack(traj, 1)
            bad = (~torch.isfinite(T).all(-1).all(-1).all(-1) | (torch.nan_to_num(T, nan=1e9).abs().amax((1, 2, 3)) > 1e3)).cpu().numpy()
            out[f"{c}/{nm}/model"] = np.stack([T[s][:, idx[s]].cpu().numpy() for s in range(NS)])
            out[f"{c}/{nm}/err"] = np.stack(errs, 1)
            out[f"{c}/{nm}/bad"] = bad
            print(c, nm, "bad", int(bad.sum()), "/", S, "err20", float(np.nanmean(out[f"{c}/{nm}/err"][:, -1])), flush=True)
    np.savez_compressed(a.out, **out)


def render(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(a.npz)
    names = sorted({k.split("/")[1] for k in d.files if k.endswith("/model")})
    for nm in [n for n in names if n != "E"]:
        cols = [(c, nm) for c in CASES if f"{c}/{nm}/model" in d.files]
        if "grav/E/model" in d.files:
            cols.insert(1, ("grav", "E"))
        fig, ax = plt.subplots(NS + 1, len(cols), figsize=(3.6 * len(cols), 3.4 * (NS + 1)))
        for j, (c, mn) in enumerate(cols):
            T, M = d[f"{c}/truth"], d[f"{c}/{mn}/model"]
            bad, err = d[f"{c}/{mn}/bad"], d[f"{c}/{mn}/err"]
            for s in range(NS):
                a_ = ax[s, j]
                for b in range(T.shape[2]):
                    col = f"C{b}"
                    a_.plot(T[s, :, b, 0], T[s, :, b, 1], col, lw=3, alpha=0.3)
                    with np.errstate(all="ignore"):
                        a_.plot(M[s, :, b, 0], M[s, :, b, 1], col + "--", lw=1)
                    a_.plot(*T[s, 0, b], "ko", ms=3)
                a_.set_aspect("equal")
                a_.set_title(f"{c} [{mn}] scene {s}" + ("  BAD" if bad[s] else ""), color="r" if bad[s] else "k", fontsize=8)
                if bad[s]:
                    for sp in a_.spines.values():
                        sp.set_color("r")
                        sp.set_linewidth(2)
            a_ = ax[NS, j]
            for s in range(err.shape[0]):
                a_.semilogy(np.arange(1, err.shape[1] + 1), err[s] + 1e-9, color="r" if bad[s] else "C0", lw=0.6, alpha=0.5)
            a_.semilogy(np.arange(1, err.shape[1] + 1), np.nanmean(err, 0) + 1e-9, "k", lw=2)
            a_.set_title(f"mean pos err vs tick (all {err.shape[0]} scenes; bad={int(bad.sum())}; black=mean)", fontsize=7)
            a_.set_xlabel("tick")
        plt.tight_layout()
        plt.savefig(f"{a.outdir}/multi_rollout_grid_{nm}.png", dpi=80)
        print(f"{a.outdir}/multi_rollout_grid_{nm}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("dump")
    p.add_argument("--ckpt", action="append", default=[])
    p.add_argument("--e-ckpt", default="")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--out", required=True)
    p = sp.add_parser("render")
    p.add_argument("--npz", required=True)
    p.add_argument("--outdir", default="videos")
    a = ap.parse_args()
    dump(a) if a.cmd == "dump" else render(a)
