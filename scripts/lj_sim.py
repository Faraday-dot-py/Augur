import argparse
import json
import math

import numpy as np
import torch

RC = 2.5
SQ3 = math.sqrt(3.0)


def _sf_consts(rc):
    u = 4 * (rc ** -12 - rc ** -6)
    f = 24 * (2 * rc ** -13 - rc ** -7)
    return u, f


def force(pos, box, rc=RC, chunk=1024):
    """Shifted-force Lennard-Jones (eps=sigma=m=1), periodic minimum image, all pairs.
    Returns (acc, pe, virial) with pe total potential and virial sum_{i<j} r.f."""
    n = pos.shape[0]
    u_rc, f_rc = _sf_consts(rc)
    acc = torch.zeros_like(pos)
    pe = pos.new_zeros(())
    vir = pos.new_zeros(())
    idx = torch.arange(n, device=pos.device)
    for s in range(0, n, chunk):
        rel = pos[s:s + chunk, None, :] - pos[None, :, :]
        rel = rel - box * torch.round(rel / box)
        r2 = (rel ** 2).sum(-1)
        rows = idx[s:s + chunk]
        r2[torch.arange(len(rows)), rows] = 1.0
        mask = r2 < rc * rc
        mask[torch.arange(len(rows)), rows] = False
        r = torch.sqrt(r2)
        ir6 = r2 ** -3
        fmag = 24 * (2 * ir6 * ir6 - ir6) / r - f_rc
        fmag = torch.where(mask, fmag, torch.zeros_like(fmag))
        acc[s:s + chunk] = (fmag / r)[..., None].mul(rel).sum(1)
        u = 4 * (ir6 * ir6 - ir6) - u_rc + f_rc * (r - rc)
        pe = pe + 0.5 * torch.where(mask, u, torch.zeros_like(u)).sum()
        vir = vir + 0.5 * (fmag * r).sum()
    return acc, pe, vir


def lattice(nx, ny, rho, jitter, rng):
    a = math.sqrt(2.0 / (SQ3 * rho))
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    x = (ix + 0.5 * (iy % 2)) * a
    y = iy * a * SQ3 / 2
    pos = np.stack([x.ravel(), y.ravel()], 1)
    pos = pos + rng.normal(0.0, jitter * a, pos.shape)
    box = np.array([nx * a, ny * a * SQ3 / 2])
    return pos % box, box


def maxwell(n, temp, rng):
    v = rng.normal(0.0, 1.0, (n, 2))
    v -= v.mean(0)
    return v * math.sqrt(temp * 2 * n / (v ** 2).sum())


SCENES = {
    "gas": dict(rho=0.05, T=1.0),
    "liquid": dict(rho=0.70, T=0.60),
    "solid": dict(rho=0.95, T=0.30),
    "coexist": dict(rho=0.35, T=0.40, jitter=0.05),
    "melt": dict(rho=0.90, T=0.10, ramp=(0.10, 1.20)),
    "quench": dict(rho=0.80, T=1.20, ramp=(1.20, 0.15)),
}


def run(pos, vel, box, steps, rec, dt, device, ramp=None, tau=0.5, checkpoint=None, force_fn=force):
    """Velocity Verlet. Berendsen thermostat toward a linear T ramp when `ramp` is given, else NVE."""
    n = pos.shape[0]
    box_t = torch.tensor(box, device=device, dtype=torch.float64)
    p = torch.tensor(pos, device=device, dtype=torch.float64)
    v = torch.tensor(vel, device=device, dtype=torch.float64)
    area = float(box[0] * box[1])
    a, pe, vir = force_fn(p, box_t)
    frames_p, frames_v, stats = [], [], []
    for step in range(steps + 1):
        if step % rec == 0:
            ke = 0.5 * (v ** 2).sum()
            frames_p.append(p.cpu().numpy().astype(np.float32))
            frames_v.append(v.cpu().numpy().astype(np.float32))
            stats.append([step, float(ke) / n, float(pe) / n, (float(ke) + float(pe)) / n,
                          (float(ke) + 0.5 * float(vir)) / area])
            if checkpoint and step % (rec * 50) == 0:
                np.savez(checkpoint, pos=p.cpu().numpy(), vel=v.cpu().numpy(), box=box, step=step)
        if step == steps:
            break
        v = v + 0.5 * dt * a
        p = (p + dt * v) % box_t
        a, pe, vir = force_fn(p, box_t)
        v = v + 0.5 * dt * a
        if ramp is not None:
            target = ramp[0] + (ramp[1] - ramp[0]) * step / steps
            temp = float((v ** 2).sum()) / (2 * n)
            v = v * math.sqrt(1 + dt / tau * (target / temp - 1))
    return np.stack(frames_p), np.stack(frames_v), np.array(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--ny", type=int, default=56)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--rec", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/lj")
    args = ap.parse_args()
    for name in args.scenes.split(","):
        sc = SCENES[name]
        rng = np.random.default_rng(args.seed)
        pos, box = lattice(args.nx, args.ny, sc["rho"], sc.get("jitter", 0.02), rng)
        vel = maxwell(len(pos), sc["T"], rng)
        fp, fv, st = run(pos, vel, box, args.steps, args.rec, args.dt, args.device, ramp=sc.get("ramp"))
        np.savez_compressed(f"{args.out}_{name}.npz", pos=fp, vel=fv, box=box, stats=st, dt=args.dt, rho=sc["rho"])
        print(name, "N", len(pos), "box", box, "E/N first,last", st[0, 3], st[-1, 3], "T last", st[-1, 1], "P last", st[-1, 4], flush=True)


if __name__ == "__main__":
    main()
