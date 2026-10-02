"""Error floor of one-force-eval-per-step integrators with the EXACT force, vs the 4-substep Verlet truth.
Schemes at dt: kick_drift (what ScatterField.step does: x += dt*v + .5dt^2 a(x); v += dt*a(x)),
verlet2 (CentralForceDynamics: 2 force evals), leapfrog1 (staggered, 1 eval/step, second order)."""
import json

import numpy as np

from scripts import gravity_sim as gs

EPS, DT = 0.5, 0.1


def kick_drift(p, v, steps):
    out = [p]
    for _ in range(steps):
        a = gs.accel(p, EPS)
        p, v = p + DT * v + 0.5 * DT ** 2 * a, v + DT * a
        out.append(p)
    return np.stack(out)


def verlet2(p, v, steps):
    out = [p]
    for _ in range(steps):
        a0 = gs.accel(p, EPS)
        p = p + DT * v + 0.5 * DT ** 2 * a0
        v = v + 0.5 * DT * (a0 + gs.accel(p, EPS))
        out.append(p)
    return np.stack(out)


def leapfrog1(p, v, steps):
    out = [p]
    w = v - 0.5 * DT * gs.accel(p, EPS)
    for _ in range(steps):
        a = gs.accel(p, EPS)
        w = w + DT * a
        p = p + DT * w
        out.append(p)
    return np.stack(out)


def main():
    res = {}
    for seed in (9000, 12000):
        data = gs.make_dataset(48, (2, 2), 100, seed, device=None, dt=DT, eps=EPS)
        for name, fn in (("kick_drift", kick_drift), ("verlet2", verlet2), ("leapfrog1", leapfrog1)):
            e = np.zeros(100)
            for P, V in data:
                pr = fn(P[0].copy(), V[0].copy(), 100)
                e += np.linalg.norm(pr - P, axis=-1).mean(-1)[1:]
            e /= len(data)
            res[f"{name}_{seed}"] = e.tolist()
            print(name, seed, "err@5/10/20/50/100", [round(e[i], 5) for i in (4, 9, 19, 49, 99)], flush=True)
    json.dump(res, open("results/integrator_floor.json", "w"))


if __name__ == "__main__":
    main()
