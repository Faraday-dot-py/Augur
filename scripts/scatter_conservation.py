"""Long-horizon conservation / stability characterisation of the scatter-field model (plan agent 6).

run: per regime (nr2/nr10/nr100/nr300 = non-relativistic scale-init scenes as Exp B, bh10/bh100/bh300/bh1000 = relativistic c=10 cold collapse as scatter_bh.py)
rolls the model and the exact softened sim (fp64, kick-drift-kick; 4 substeps/tick = the training truth, plus 1 and 16 substep controls for N<=300) for --steps ticks
on a batch of scenes (SEEDS x scenes per seed), computing per tick (fp64, from the state): E, K, U, |P|, L about COM, COM, virial, bound fraction, ejected count,
fraction outside the model grid, min pair distance, close pairs, max speed (and fraction of bodies near c), max coordinate acceleration, finiteness.
Writes results/conservation/<regime>.npz (sampled ticks + extremes over every tick + 4738-seed snapshots every 10 ticks) and <regime>.json (meta).

probe: model-only force probes: two-body force vs truth (radial ratio, tangential fraction, velocity dependence, placement dependence), far-field force on a test
token outside/inside the grid, curl probe.

Usage: PYTHONPATH=. python scripts/scatter_conservation.py run --ckpt <ckpt> --opt kcache,fastio,cl,graph --steps 10000 [--regimes nr2,bh300]
       PYTHONPATH=. python scripts/scatter_conservation.py probe --ckpt <ckpt> --opt kcache,fastio,cl,graph
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

SEEDS = (9100, 9200, 9300, 4738)
C = 10.0
REGIMES = {"nr2": ("nr", 2, 8), "nr10": ("nr", 10, 4), "nr100": ("nr", 100, 1), "nr300": ("nr", 300, 1),
           "bh10": ("bh", 10, 4), "bh100": ("bh", 100, 2), "bh300": ("bh", 300, 1), "bh1000": ("bh", 1000, 1)}
TRUTH_SUBS = {"t4": 4, "t1": 1, "t16": 16}
KEYS = ("E", "K", "U", "P", "L", "comx", "comy", "vir", "bound", "ejected", "outgrid", "rmin", "close", "vmax", "frac90", "fracge", "amax", "rhalf", "rmax", "finite")


def init_scene(kind, n, rng):
    if kind == "nr":
        p, v = gs.init_bodies(n, rng, scale=n > 2)
        return p - gs.CENTER, v
    pos = np.clip(rng.normal(0.0, 8.0, (n, 2)), -29.0, 29.0)
    vel = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
    vel -= vel.mean(0)
    return pos, vel


def make_scenes(kind, n, per_seed):
    ps, vs, seeds = [], [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        for _ in range(per_seed):
            p, v = init_scene(kind, n, rng)
            ps.append(p)
            vs.append(v)
            seeds.append(s)
    return np.stack(ps), np.stack(vs), np.array(seeds)


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


def to_mom(vel, c):
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    return vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()


def metrics(pos, vel, mom, rel, eps, extent, r0):
    pos, vel, mom = pos.double(), vel.double(), mom.double()
    n = pos.shape[1]
    p2 = (mom ** 2).sum(-1)
    ke = C ** 2 * ((1 + p2 / C ** 2).sqrt() - 1) if rel else 0.5 * p2
    d = pos[:, None] - pos[:, :, None]
    r2 = (d ** 2).sum(-1)
    eye = torch.eye(n, device=pos.device, dtype=torch.bool)[None]
    inv = (r2 + eps ** 2).rsqrt().masked_fill(eye, 0.0)
    pot = -inv.sum(-1)
    K, U = ke.sum(1), 0.5 * pot.sum(1)
    ei = ke + pot
    com = pos.mean(1)
    rr = pos - com[:, None]
    rad = rr.norm(dim=-1)
    L = (rr[..., 0] * mom[..., 1] - rr[..., 1] * mom[..., 0]).sum(1)
    sp = vel.norm(dim=-1)
    r2o = r2.masked_fill(eye, 1e30)
    return {"E": K + U, "K": K, "U": U, "P": mom.sum(1).norm(dim=-1), "L": L, "comx": com[:, 0], "comy": com[:, 1],
            "vir": 2 * K / U.abs().clamp_min(1e-12), "bound": (ei < 0).double().mean(1),
            "ejected": ((ei > 0) & (rad > 2 * r0[:, None])).double().sum(1),
            "outgrid": (pos.abs() > extent / 2).any(-1).double().mean(1), "rmin": r2o.amin((1, 2)).sqrt(),
            "close": (r2o < 0.25 ** 2).double().sum((1, 2)) / 2, "vmax": sp.amax(1), "frac90": (sp > 0.9 * C).double().mean(1),
            "fracge": (sp >= C).double().mean(1), "rhalf": rad.median(1).values, "rmax": rad.amax(1),
            "finite": (torch.isfinite(pos).all(-1) & torch.isfinite(vel).all(-1)).all(-1).double()}


class Recorder:
    def __init__(self, rel, eps, extent, r0, steps, B, dev, dt):
        self.rel, self.eps, self.extent, self.r0, self.dt = rel, eps, extent, r0, dt
        self.rows = {k: torch.zeros(steps + 1, B, dtype=torch.float64, device=dev) for k in KEYS + ("amaxfull",)}
        self.prev = None
        self.t = 0
        self.snaps = []

    def __call__(self, t, pos, vel, mom, snap_idx=None):
        m = metrics(pos, vel, mom, self.rel, self.eps, self.extent, self.r0)
        vel = vel.double()
        m["amax"] = ((vel - self.prev).norm(dim=-1) / self.dt).amax(1) if self.prev is not None else torch.zeros_like(m["E"])
        m["amaxfull"] = m["amax"]
        self.prev = vel
        for k, v in m.items():
            self.rows[k][t] = v
        if snap_idx is not None and t % 10 == 0:
            self.snaps.append(pos[snap_idx].float().cpu())
        self.t = t


def roll_model(model, pos0, vel0, rel, steps, rec, snap_idx, dt):
    dev = pos0.device
    B, N = pos0.shape[:2]
    pos, vel = pos0.float(), vel0.float()
    mass, mask = torch.ones(B, N, device=dev), torch.ones(B, N, device=dev)
    with torch.no_grad():
        if rel:
            mom = to_mom(vel, C)
            field = model.init_field(B, dev)[0]
            a, field = model.force(pos, speed(mom, C), mass, mask, field)
            rec(0, pos, speed(mom, C), mom, snap_idx)
            for t in range(1, steps + 1):
                mom = mom + 0.5 * dt * a
                pos = pos + dt * speed(mom, C)
                a, field = model.force(pos, speed(mom, C), mass, mask, field)
                mom = mom + 0.5 * dt * a
                rec(t, pos, speed(mom, C), mom, snap_idx)
                if t % 250 == 0 and not bool(torch.isfinite(pos).any()):
                    return t
        else:
            field = model.init_field(B, dev)
            rec(0, pos, vel, vel, snap_idx)
            for t in range(1, steps + 1):
                pos, vel, field, _ = model.step(pos, vel, mass, mask, field)
                rec(t, pos, vel, vel, snap_idx)
                if t % 250 == 0 and not bool(torch.isfinite(pos).any()):
                    return t
    return steps


def roll_truth(pos0, vel0, rel, steps, subs, eps, rec, snap_idx, dt):
    h = dt / subs
    pos = pos0.double().clone()

    def acc(p):
        d = p[:, None] - p[:, :, None]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
        inv.diagonal(dim1=1, dim2=2).zero_()
        return (d * inv[..., None]).sum(2)

    with torch.no_grad():
        mom = to_mom(vel0.double(), C) if rel else vel0.double().clone()
        vf = (lambda q: speed(q, C)) if rel else (lambda q: q)
        a = acc(pos)
        rec(0, pos, vf(mom), mom, snap_idx)
        for t in range(1, steps + 1):
            for _ in range(subs):
                mom = mom + 0.5 * h * a
                pos = pos + h * vf(mom)
                a = acc(pos)
                mom = mom + 0.5 * h * a
            rec(t, pos, vf(mom), mom, snap_idx)


def sample_ticks(steps):
    t = np.unique(np.concatenate([np.arange(0, 101), np.arange(100, 1001, 5), np.arange(1000, steps + 1, 25), [steps]]))
    return t[t <= steps]


def summarize_set(rec, ticks, sub=""):
    out = {}
    fin = rec.rows["finite"]
    for k in KEYS:
        out[f"{sub}{k}"] = rec.rows[k][ticks].float().cpu().numpy()
    nonfin = (fin < 0.5)
    first = torch.where(nonfin.any(0), nonfin.double().argmax(0), torch.full((fin.shape[1],), -1, device=fin.device))
    out[f"{sub}first_nonfinite"] = first.cpu().numpy()
    good = torch.where(nonfin, torch.zeros_like(rec.rows["vmax"]), rec.rows["vmax"])
    out[f"{sub}vmax_all"] = good.amax(0).cpu().numpy()
    out[f"{sub}rmin_all"] = rec.rows["rmin"].nan_to_num(1e9).amin(0).cpu().numpy()
    out[f"{sub}amax_all"] = rec.rows["amax"].nan_to_num(0).amax(0).cpu().numpy()
    ge = (rec.rows["fracge"] > 0)
    out[f"{sub}first_ge_c"] = torch.where(ge.any(0), ge.double().argmax(0), torch.full((fin.shape[1],), -1, device=fin.device)).cpu().numpy()
    return out


def build_model(args, dev):
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    for o in [x for x in args.opt.split(",") if x]:
        sf.apply_opt(model, o)
    return model


def run(args):
    dev = torch.device("cuda")
    model = build_model(args, dev)
    os.makedirs(args.out_dir, exist_ok=True)
    dt, eps, steps = 0.1, 0.5, args.steps
    ticks = sample_ticks(steps)
    for name in args.regimes.split(","):
        kind, n, per = REGIMES[name]
        pos0, vel0, seeds = make_scenes(kind, n, per)
        P0 = torch.tensor(pos0, device=dev)
        V0 = torch.tensor(vel0, device=dev)
        B = len(seeds)
        rel = kind == "bh"
        com0 = P0.mean(1, keepdim=True)
        r0 = (P0 - com0).norm(dim=-1).pow(2).mean(1).sqrt()
        snap_idx = torch.tensor(np.where(seeds == 4738)[0], device=dev)
        t0 = time.time()
        res = {"ticks": ticks, "seeds": seeds, "r0": r0.cpu().numpy()}
        rec = Recorder(rel, eps, model.extent, r0, steps, B, dev, dt)
        done = roll_model(model, P0, V0, rel, steps, rec, snap_idx, dt)
        torch.cuda.synchronize()
        t_model = time.time() - t0
        res.update(summarize_set(rec, ticks, "m_"))
        res["m_snaps"] = torch.stack(rec.snaps).numpy()
        res["m_done"] = np.array(done)
        print(f"{name} model B={B} N={n} {steps} ticks (stopped {done}) {t_model:.1f}s", flush=True)
        for tag, subs in TRUTH_SUBS.items():
            if subs == 16 and n > 300:
                continue
            if tag != "t4" and n > 300:
                continue
            t0 = time.time()
            rt = Recorder(rel, eps, model.extent, r0, steps, B, dev, dt)
            roll_truth(P0, V0, rel, steps, subs, eps, rt, snap_idx, dt)
            torch.cuda.synchronize()
            res.update(summarize_set(rt, ticks, f"{tag}_"))
            if tag == "t4":
                res["t4_snaps"] = torch.stack(rt.snaps).numpy()
            print(f"{name} truth substeps {subs} {time.time() - t0:.1f}s", flush=True)
        np.savez_compressed(f"{args.out_dir}/{name}.npz", **res)
        json.dump({"regime": name, "kind": kind, "n": n, "per_seed": per, "steps": steps, "ckpt": args.ckpt, "opt": args.opt, "B": B,
                   "model_seconds": t_model, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(dev)}, open(f"{args.out_dir}/{name}.json", "w"))
        print(f"{name} saved", flush=True)
        del rec, rt
        torch.cuda.empty_cache()


def warm(model, pos, vel, mass, mask, hold=3):
    field = model.init_field(pos.shape[0], pos.device)
    field = field[0] if isinstance(field, tuple) else field
    with torch.no_grad():
        for _ in range(hold):
            a, field = model.force(pos, vel, mass, mask, field)
    return a


def probe(args):
    dev = torch.device("cuda")
    model = build_model(args, dev)
    eps = 0.5
    g = torch.Generator(device="cpu").manual_seed(4738)
    out = {}
    rs = [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    P = 64
    rows = []
    for r in rs:
        c = (torch.rand(P, 1, 2, generator=g) - 0.5) * 2 * max(1.0, 30 - r / 2)
        th = torch.rand(P, generator=g) * 2 * np.pi
        u = torch.stack([th.cos(), th.sin()], -1)[:, None]
        pos = torch.cat([c - 0.5 * r * u, c + 0.5 * r * u], 1).float().to(dev)
        vel0 = torch.zeros_like(pos)
        velr = (torch.randn(P, 2, 2, generator=g) * 2.0).float().to(dev)
        mass, mask = torch.ones(P, 2, device=dev), torch.ones(P, 2, device=dev)
        a0 = warm(model, pos, vel0, mass, mask)
        av = warm(model, pos, velr, mass, mask)
        u = u.to(dev)
        d = pos[:, 1] - pos[:, 0]
        at = d * ((d ** 2).sum(-1, keepdim=True) + eps ** 2) ** -1.5
        rad = (a0[:, 0] * u[:, 0]).sum(-1)
        tru = (at * u[:, 0]).sum(-1)
        tan = (a0[:, 0] * torch.stack([-u[:, 0, 1], u[:, 0, 0]], -1)).sum(-1)
        rows.append({"r": r, "radial_ratio_median": float((rad / tru).median()), "radial_ratio_std": float((rad / tru).std()),
                     "tan_over_true_rms": float((tan / tru).pow(2).mean().sqrt()), "rel_err_rms": float(((a0[:, 0] - at).norm(dim=-1) / at.norm(dim=-1)).pow(2).mean().sqrt()),
                     "newton3_resid": float((a0.sum(1)).abs().max()), "vel_dep_rel": float(((av[:, 0] - a0[:, 0]).norm(dim=-1) / at.norm(dim=-1)).mean()),
                     "torque_rel_rms": float(((tan * r) / (tru * r)).pow(2).mean().sqrt())})
        print(rows[-1], flush=True)
    out["two_body"] = rows
    rng = np.random.default_rng(4738)
    cl = torch.tensor(np.clip(rng.normal(0, 3.0, (100, 2)), -12, 12), dtype=torch.float32, device=dev)
    Rs = [5, 10, 20, 28, 31, 33, 40, 60]
    S = len(Rs)
    pos = torch.cat([cl[None].expand(S, -1, -1), torch.tensor([[R, 0.0] for R in Rs], device=dev)[:, None]], 1)
    mass = torch.cat([torch.ones(S, 100, device=dev), torch.full((S, 1), 1e-4, device=dev)], 1)
    mask = torch.ones(S, 101, device=dev)
    a = warm(model, pos, torch.zeros_like(pos), mass, mask)[:, 100]
    d = cl[None] - pos[:, 100:101]
    at = (d * ((d ** 2).sum(-1, keepdim=True) + eps ** 2) ** -1.5).sum(1)
    out["far_field"] = [{"R": R, "a_model_x": float(a[i, 0]), "a_true_x": float(at[i, 0]), "ratio": float(a[i, 0] / at[i, 0]), "a_model_y": float(a[i, 1])} for i, R in enumerate(Rs)]
    for row in out["far_field"]:
        print(row, flush=True)
    try:
        cp = sf.curl_probe(model, dev)
        out["curl"] = {k: v for k, v in cp.items() if k != "a_map"}
        print(out["curl"], flush=True)
    except Exception as e:
        out["curl_error"] = repr(e)
        print("curl probe failed", repr(e), flush=True)
    json.dump(out, open(f"{args.out_dir}/probe.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("run", "probe"):
        p = sub.add_parser(nm)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--opt", default="kcache,fastio,cl,graph")
        p.add_argument("--out-dir", default="results/conservation")
        if nm == "run":
            p.add_argument("--steps", type=int, default=10000)
            p.add_argument("--regimes", default=",".join(REGIMES))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run(args) if args.cmd == "run" else probe(args)


if __name__ == "__main__":
    main()
