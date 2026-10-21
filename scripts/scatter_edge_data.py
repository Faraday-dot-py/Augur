"""Open-domain training scenes for the scatter-field grid-edge fix: clusters that cross the grid box [-32, 32]^2 in both directions (exit and enter), with exact
softened all-pairs float64 truth over ALL bodies (infinite domain), so force labels include the mass outside the box.
Same output format as scatter_field.make_cluster_dataset: list of (P, V) (T, N, 2) in absolute coordinates (CENTER offset), unit masses."""
import numpy as np

from scripts import gravity_sim as gs
from scripts import scatter_field as sf

C = sf.CENTER


def _cluster(rng, n, centre, sigma, vfac):
    pos = rng.normal(0.0, sigma, (n, 2)) + centre
    vel = rng.normal(0.0, vfac * np.sqrt(n / (4 * sigma)), (n, 2))
    return pos, vel


def open_scene(rng):
    r = rng.random()
    if r < 0.35:
        n = int(rng.integers(80, 301))
        sg = rng.uniform(5.0, 10.0)
        c = rng.uniform(-12.0, 12.0, 2)
        pos, vel = _cluster(rng, n, c, sg, rng.uniform(0.1, 0.6))
        vel = vel + rng.uniform(0.5, 3.0) * (pos - c) / sg
    elif r < 0.70:
        n = int(rng.integers(80, 301))
        sg = rng.uniform(4.0, 9.0)
        a = rng.uniform(0, 2 * np.pi)
        c = rng.uniform(20.0, 34.0) * np.array([np.cos(a), np.sin(a)])
        pos, vel = _cluster(rng, n, c, sg, rng.uniform(0.1, 0.5))
        b = rng.uniform(0, 2 * np.pi)
        vel = vel + rng.uniform(0.0, 4.0) * np.array([np.cos(b), np.sin(b)])
    else:
        n1 = int(rng.integers(60, 201))
        sg = rng.uniform(4.0, 8.0)
        a = rng.uniform(0, 2 * np.pi)
        c = rng.uniform(38.0, 50.0) * np.array([np.cos(a), np.sin(a)])
        p1, v1 = _cluster(rng, n1, c, sg, rng.uniform(0.1, 0.4))
        v1 = v1 - rng.uniform(2.0, 6.0) * np.array([np.cos(a), np.sin(a)])
        n2 = int(rng.integers(40, 151))
        p2, v2 = _cluster(rng, n2, np.zeros(2), rng.uniform(4.0, 8.0), rng.uniform(0.1, 0.4))
        pos, vel = np.concatenate([p1, p2]), np.concatenate([v1, v2])
    vel = vel - vel.mean(0)
    return pos + C, vel


def make_open_dataset(num, steps, seed, device, dt=0.1, eps=0.5):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        pos, vel = open_scene(rng)
        P, V = gs.rollout_torch(pos, vel, steps, device, dt=dt, eps=eps)
        out.append((P, V))
    return out
