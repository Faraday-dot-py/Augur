"""Tier-0 training scenes with a random smooth imposed external field (hole 3 / 4 in docs/theory/heterogeneous-tiered.md).

Truth = softened 2D-gravity N-body (G=1, eps=0.5, dt=0.1, 4 substeps, same as gravity_sim) plus the external acceleration
-(g + H x) (potential phi = g.x + x.H.x/2 about the window centre). A tier-0 model with ext channels + extapply that learns only the
near-field force reproduces these trajectories, i.e. target = truth - external.
"""
import numpy as np
import torch

from scripts import gravity_sim as gs

KINDS = ("scale", "cluster", "filled")


def ext_params(num, rng, p_zero=0.2, sigma=(0.003, 0.05)):
    s = np.exp(rng.uniform(np.log(sigma[0]), np.log(sigma[1]), num))
    g = rng.normal(size=(num, 2)) * s[:, None]
    H = rng.normal(size=(num, 3)) * (s / 16.0)[:, None]
    zero = rng.random(num) < p_zero
    g[zero], H[zero] = 0.0, 0.0
    return np.zeros(num), g, H


def init_scene(kind, rng):
    if kind == "scale":
        n = int(rng.integers(10, 201))
        p, v = gs.init_bodies(n, rng, scale=True)
        return p - gs.CENTER, v
    if kind == "cluster":
        n = int(rng.integers(80, 301))
        sg = rng.uniform(5.0, 10.0)
        p = np.clip(rng.normal(0.0, sg, (n, 2)), -28.0, 28.0)
        v = rng.normal(0.0, rng.uniform(0.1, 0.6) * np.sqrt(n / (4 * sg)), (n, 2))
        return p, v - v.mean(0)
    n = int(rng.integers(100, 301))
    p = rng.uniform(-31.0, 31.0, (n, 2))
    v = rng.normal(0.0, 0.5 * (n / 8) ** 0.25, (n, 2))
    return p, v - v.mean(0)


@torch.no_grad()
def rollout_ext(pos, vel, steps, device, g, H, dt=0.1, substeps=4, eps=0.5):
    h = dt / substeps
    pos, vel = torch.tensor(pos, device=device), torch.tensor(vel, device=device)
    g, H = torch.tensor(g, device=device), torch.tensor(H, device=device)

    def acc(p):
        d = p[None, :, :] - p[:, None, :]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
        inv.fill_diagonal_(0.0)
        a = (d * inv[..., None]).sum(1)
        x, y = p[:, 0], p[:, 1]
        return a - torch.stack([g[0] + H[0] * x + H[1] * y, g[1] + H[1] * x + H[2] * y], -1)

    ps, vs = [pos.cpu().numpy() + gs.CENTER], [vel.cpu().numpy()]
    a = acc(pos)
    for _ in range(steps):
        for _ in range(substeps):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = acc(pos)
            vel = vel + 0.5 * h * a
        ps.append(pos.cpu().numpy() + gs.CENTER)
        vs.append(vel.cpu().numpy())
    return np.stack(ps), np.stack(vs)


def make_scenes(num, steps, seed, device, mix=(0.35, 0.35, 0.30), p_zero=0.2):
    """Returns (data, ext) with data = list of (P, V) absolute coords (T,N,2) and ext = (phi0 (S,), g (S,2), H (S,3))."""
    rng = np.random.default_rng(seed)
    phi0, g, H = ext_params(num, rng, p_zero)
    kinds = rng.choice(len(KINDS), num, p=np.array(mix) / sum(mix))
    data = []
    for i in range(num):
        p, v = init_scene(KINDS[kinds[i]], rng)
        data.append(rollout_ext(p, v, steps, device, g[i], H[i]))
    return data, (phi0, g, H), kinds
