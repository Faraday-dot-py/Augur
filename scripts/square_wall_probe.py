import json
import sys

import numpy as np
import torch

from scripts import scatter_bh as sb
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

dev = torch.device("cuda")
ckpt = sys.argv[1]
c, dt, eps = 10.0, 0.1, 0.5
model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", dt).to(dev)
model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
model.eval()
print("grid", model.grid, "extent", model.extent, "h", model.h, "momfix", model.momfix, "pp", model.pp, "potential", model.potential, "kernel", model.kernel, "far", model.far, flush=True)
ops.chunked_pp(model)
pp_fn = model.pp_acc


def scene(n, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "clipgauss":
        p = np.clip(rng.normal(0.0, 8.0, (n, 2)), -29.0, 29.0)
    elif kind == "truncgauss":
        p = rng.normal(0.0, 8.0, (4 * n, 2))
        p = p[(np.abs(p) < 28.0).all(1)][:n]
    else:
        r = 20.0 * np.sqrt(rng.uniform(size=n)); a = rng.uniform(0, 2 * np.pi, n)
        p = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    return p


@torch.no_grad()
def static_probe(kind, n):
    p0 = scene(n, kind, 4738)
    xs = np.arange(16.0, 46.01, 0.5)
    tests = []
    for ang in (0.0, np.pi / 2, np.pi / 4):
        tests.append(np.stack([xs * np.cos(ang), xs * np.sin(ang)], 1))
    tp = np.concatenate(tests)
    pos = np.concatenate([p0, tp])
    mass = np.concatenate([np.ones(n), np.zeros(len(tp))])
    P = torch.tensor(pos, dtype=torch.float32, device=dev)[None]
    M = torch.tensor(mass, dtype=torch.float32, device=dev)[None]
    mk = torch.ones_like(M)
    vel = torch.zeros_like(P)
    res = {"kind": kind, "n": n, "xs": xs.tolist()}
    ex = ops.exact_acc(torch.tensor(pos, dtype=torch.float64, device=dev), eps, torch.tensor(mass, dtype=torch.float64, device=dev)).cpu().numpy()[n:]
    res["exact"] = ex.tolist()
    for name, ppf in (("full", pp_fn), ("mesh", lambda pos, mass, mask, knn=16: torch.zeros_like(pos))):
        model.pp_acc = ppf
        a, _ = model.force(P, vel, M, mk, model.init_field(1, dev)[0])
        res[name] = a[0, n:].cpu().numpy().tolist()
    model.pp_acc = pp_fn
    return res


@torch.no_grad()
def tracer_rollout(n, nt, steps, seed):
    p0 = scene(n, "clipgauss", seed)
    rng = np.random.default_rng(seed + 1)
    v0 = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2)); v0 -= v0.mean(0)
    ang = rng.uniform(0, 2 * np.pi, nt); r0 = rng.uniform(22, 26, nt)
    tp = np.stack([r0 * np.cos(ang), r0 * np.sin(ang)], 1)
    tv = 0.95 * c * np.stack([np.cos(ang), np.sin(ang)], 1)
    pos0 = np.concatenate([p0, tp]); vel0 = np.concatenate([v0, tv])
    mass = np.concatenate([np.ones(n), np.zeros(nt)])
    T = n + nt

    def mom_of(v):
        vm = v.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
        return v * (1 - (vm / c) ** 2).clamp(min=1e-6).rsqrt()

    out = {"n": n, "nt": nt, "steps": steps}
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    mom = mom_of(torch.tensor(vel0, dtype=torch.float32, device=dev)[None])
    M = torch.tensor(mass, dtype=torch.float32, device=dev)[None]; mk = torch.ones_like(M)
    f = model.init_field(1, dev)[0]
    traj = [pos[0, n:].cpu().numpy()]
    a, f = model.force(pos, sb.speed(mom, c), M, mk, f)
    for _ in range(steps):
        mom = mom + 0.5 * dt * a
        pos = pos + dt * sb.speed(mom, c)
        a, f = model.force(pos, sb.speed(mom, c), M, mk, f)
        mom = mom + 0.5 * dt * a
        traj.append(pos[0, n:].cpu().numpy())
    out["model_r"] = np.linalg.norm(np.stack(traj), axis=2).tolist()
    out["model_xy"] = np.stack(traj)[:, :40].tolist()
    pos = torch.tensor(pos0, dtype=torch.float64, device=dev)
    mom = mom_of(torch.tensor(vel0, dtype=torch.float64, device=dev))
    Md = torch.tensor(mass, dtype=torch.float64, device=dev)
    traj = [pos[n:].cpu().numpy()]
    h = dt / 4
    a = ops.exact_acc(pos, eps, Md)
    for _ in range(steps):
        for _ in range(4):
            mom = mom + 0.5 * h * a
            pos = pos + h * sb.speed(mom, c)
            a = ops.exact_acc(pos, eps, Md)
            mom = mom + 0.5 * h * a
        traj.append(pos[n:].cpu().numpy())
    out["truth_r"] = np.linalg.norm(np.stack(traj), axis=2).tolist()
    out["truth_xy"] = np.stack(traj)[:, :40].tolist()
    return out


for kind in ("clipgauss", "truncgauss", "disc"):
    r = static_probe(kind, 100000)
    print("STATIC " + json.dumps(r), flush=True)
r = tracer_rollout(20000, 400, 100, 4738)
print("ROLL " + json.dumps(r), flush=True)
print("DONE", flush=True)
