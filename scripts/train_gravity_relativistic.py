"""Can a next-state predictor learn relativistic kinematics from trajectories
alone? TokenFreeDynamics has no built-in speed limit or gamma factor -- it
only sees (pos, vel) each step and must discover from data that a constant
force produces diminishing dv as |v| -> c. Truth comes from scripts/
gravity_sim.py rollout_torch(relativistic=True): momentum state p, dp/dt=F,
v = p/sqrt(1+|p|^2/c^2), same convention as scripts/orbit_bh.py --relativistic.
--model central is included as a structural-failure baseline: CentralForceDynamics'
velocity-Verlet update (v_{t+1} = v_t + dt*a) has no v-dependence in dv at all,
so it cannot represent the saturation even in principle -- confirms the
architecture, not the data, is what's being tested."""
import argparse
import json

import numpy as np
import torch

from model.central_force import CentralForceDynamics
from model.token_free import TokenFreeDynamics
from scripts import gravity_sim as gs
from scripts.train_gravity_dynamics import unroll

DEVICE = torch.device("cpu")


def evaluate(dyn, data, horizon, dt, eps, c):
    err = np.zeros(horizon)
    cv_err = np.zeros(horizon)
    e_model, e_true = np.zeros(horizon), np.zeros(horizon)
    maxv_model, maxv_true = np.zeros(horizon), np.zeros(horizon)
    over_c = np.zeros(horizon)
    with torch.no_grad():
        for P, V in data:
            p0 = torch.tensor(P[0], dtype=torch.float32, device=DEVICE)
            v0 = torch.tensor(V[0], dtype=torch.float32, device=DEVICE)
            ps, vs = unroll(dyn, p0, v0, horizon, dt)
            for t in range(horizon):
                pm, vm = ps[t].cpu().numpy(), vs[t].cpu().numpy()
                err[t] += np.linalg.norm(pm - P[t + 1], axis=1).mean()
                cv_err[t] += np.linalg.norm(P[0] + V[0] * dt * (t + 1) - P[t + 1], axis=1).mean()
                e_model[t] += gs.energy_rel(pm.astype(np.float64), vm.astype(np.float64), eps, c)
                e_true[t] += gs.energy_rel(P[t + 1], V[t + 1], eps, c)
                speed_m = np.linalg.norm(vm, axis=1)
                maxv_model[t] += speed_m.max()
                maxv_true[t] += np.linalg.norm(V[t + 1], axis=1).max()
                over_c[t] += float((speed_m > c).any())
    n = len(data)
    return {"err": (err / n).tolist(), "const_vel_err": (cv_err / n).tolist(),
            "energy_model": (e_model / n).tolist(), "energy_true": (e_true / n).tolist(),
            "max_speed_model": (maxv_model / n).tolist(), "max_speed_true": (maxv_true / n).tolist(),
            "frac_scenes_over_c": (over_c / n).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--min-bodies", type=int, default=3)
    ap.add_argument("--max-bodies", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--neighbor-radius", type=float, default=100.0)
    ap.add_argument("--scale-init", action="store_true")
    ap.add_argument("--eval-scenes", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--no-pair-impulse", action="store_true")
    ap.add_argument("--model", choices=["token", "central"], default="token")
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", type=str, default="results/gravity_relativistic.json")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/gravity_relativistic.pt")
    args = ap.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device)
    if DEVICE.type != "cuda":
        raise RuntimeError("relativistic data generation must run on GPU (Polaris), not CPU")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    kw = dict(dt=args.dt, eps=args.eps, relativistic=True, c=args.c)
    rng_n = (args.min_bodies, args.max_bodies)
    train = gs.make_dataset(args.train, rng_n, args.steps, args.seed, scale=args.scale_init, device=DEVICE, **kw)
    if args.model == "central":
        dyn = CentralForceDynamics(dt=args.dt, neighbor_radius=args.neighbor_radius)
    else:
        dyn = TokenFreeDynamics(n=1000, neighbor_radius=args.neighbor_radius, pair_impulse=not args.no_pair_impulse)
    if args.init:
        dyn.load_state_dict(torch.load(args.init, map_location=DEVICE))
    dyn.to(DEVICE)
    opt = torch.optim.Adam(dyn.parameters(), lr=args.lr)
    for it in range(args.iters):
        k = int(round(args.k_start + (args.k_end - args.k_start) * it / max(1, args.iters - 1)))
        opt.zero_grad()
        total = 0.0
        for _ in range(args.batch):
            P, V = train[rng.integers(len(train))]
            t0 = int(rng.integers(0, args.steps - k + 1))
            p0 = torch.tensor(P[t0], dtype=torch.float32, device=DEVICE)
            v0 = torch.tensor(V[t0], dtype=torch.float32, device=DEVICE)
            ps, vs = unroll(dyn, p0, v0, k, args.dt)
            tp = torch.tensor(P[t0 + 1:t0 + k + 1], dtype=torch.float32, device=DEVICE)
            tv = torch.tensor(V[t0 + 1:t0 + k + 1], dtype=torch.float32, device=DEVICE)
            loss = (((ps - tp) ** 2).mean() + 0.1 * ((vs - tv) ** 2).mean()) / args.batch
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        if it % args.log_every == 0:
            print(f"it {it} k {k} loss {total:.6f}", flush=True)
            torch.save(dyn.state_dict(), args.checkpoint)
    torch.save(dyn.state_dict(), args.checkpoint)
    res = {}
    for seed in (9000, 12000):
        data = gs.make_dataset(args.eval_scenes, rng_n, args.steps, seed, scale=args.scale_init, device=DEVICE, **kw)
        res[str(seed)] = evaluate(dyn, data, 20, args.dt, args.eps, args.c)
        r = res[str(seed)]
        print(seed, "err@5/10/20", [round(r["err"][i], 4) for i in (4, 9, 19)],
              "constvel", [round(r["const_vel_err"][i], 4) for i in (4, 9, 19)],
              "max_speed model/true @20", round(r["max_speed_model"][19], 4), round(r["max_speed_true"][19], 4),
              "c", args.c, "frac_scenes_over_c@20", round(r["frac_scenes_over_c"][19], 4),
              "E true/model @20", round(r["energy_true"][19], 4), round(r["energy_model"][19], 4), flush=True)
    res["args"] = vars(args)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
