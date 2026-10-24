"""Train/evaluate upper tiers (>=1) of the heterogeneous tiered field (model/tiered) on multi-clump 2D gravity scenes in a
128-unit box, with the Exp E scatter field (kernel + pp=2, verlet, grid widened to 256 at h=0.5; weights unchanged) as frozen tier 0.
Modes: S = sequential-frozen (tier k trained on teacher-forced states against truth acc - tier 0 acc, tiers < k frozen, one stage
per tier); R = rollout-in-loop fine-tune of all upper tiers with tier 0 frozen; eval = baseline (tier 0 alone) vs arms on held-out
seeds; compare = pre-registered pass/kill. Seed 4738; held-out seeds 9100/9200/9300; --resume, full-state ckpts, SIGTERM trap."""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from model.tiered import ScatterFieldTier0, TierNet, TieredField, ZeroTier0, nfeat
from scripts import gravity_sim as gs
from scripts import scatter_field as sf

DT, EPS, CENTER = 0.1, 0.5, sf.CENTER
H0, N0, G, HALO, BLEND = 0.5, 256, 16, 4, 2
NMAX = 512
EVAL_SEEDS = (9100, 9200, 9300)
STOP = []
ORDER3 = [False]


def true_acc(pos, mass, mask, eps=EPS):
    d = pos[:, None, :, :] - pos[:, :, None, :]
    inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
    inv = inv * (1 - torch.eye(pos.shape[1], device=pos.device, dtype=pos.dtype))[None]
    return (d * (inv * (mass * mask)[:, None, :])[..., None]).sum(2) * mask[..., None]


def far_acc(pos, mass, mask, r0, eps=EPS):
    d = pos[:, None, :, :] - pos[:, :, None, :]
    r2 = (d ** 2).sum(-1)
    inv = (r2 + eps ** 2) ** -1.5 * (r2 >= r0 ** 2) * (1 - torch.eye(pos.shape[1], device=pos.device, dtype=pos.dtype))[None]
    return (d * (inv * (mass * mask)[:, None, :])[..., None]).sum(2) * mask[..., None]


def make_scenes(num, steps, seed, dev):
    rng = np.random.default_rng(seed)
    P = torch.zeros(num, steps + 1, NMAX, 2, dtype=torch.float64)
    V = torch.zeros_like(P)
    M, mask = torch.zeros(num, NMAX, dtype=torch.float64), torch.zeros(num, NMAX, dtype=torch.float64)
    lab = torch.full((num, NMAX), -1, dtype=torch.long)
    for s in range(num):
        cen, nc = [], int(rng.integers(2, 5))
        while len(cen) < nc:
            c = rng.uniform(-36, 36, 2)
            if all(np.linalg.norm(c - o) >= 24 for o in cen):
                cen.append(c)
        pos, vel, lb = [], [], []
        for i, c in enumerate(cen):
            n = int(rng.integers(40, 111))
            sg = rng.uniform(3.0, 6.0)
            pos.append(np.clip(rng.normal(0.0, sg, (n, 2)) + c, -60, 60))
            vel.append(rng.normal(0.0, rng.uniform(0.1, 0.6) * np.sqrt(n / (4 * sg)), (n, 2)) + rng.normal(0.0, 0.4, 2))
            lb.append(np.full(n, i))
        pos, vel, lb = np.concatenate(pos), np.concatenate(vel), np.concatenate(lb)
        vel -= vel.mean(0)
        p, v = gs.rollout_torch(pos + CENTER, vel, steps, dev, dt=DT, eps=EPS)
        n = len(pos)
        P[s, :, :n], V[s, :, :n] = torch.as_tensor(np.asarray(p) - CENTER), torch.as_tensor(np.asarray(v))
        M[s, :n], mask[s, :n], lab[s, :n] = 1.0, 1.0, torch.as_tensor(lb)
    return dict(P=P.to(dev), V=V.to(dev), M=M.to(dev), mask=mask.to(dev), lab=lab.to(dev))


def load_tier0(path, dev):
    m = sf.ScatterField(grid=256, extent=128.0, net="unet", recurrent=True, hidden_ch=0, momfix=True, dt=DT, in_scale=1.0,
                        potential=True, kernel=True, pp=2.0, split=True, verlet=True).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
    m.eval().requires_grad_(False)
    return ScatterFieldTier0(m)


@torch.no_grad()
def force_series(t0, d, bs=16):
    """Teacher-forced tier-0 and true accelerations along every scene's truth trajectory, (S,T,N,2) float32."""
    S, T = d["P"].shape[:2]
    A0 = torch.zeros(S, T, NMAX, 2, device=d["P"].device)
    AT = torch.zeros_like(A0)
    for s0 in range(0, S, bs):
        sl = slice(s0, min(S, s0 + bs))
        m, mk = d["M"][sl].float(), d["mask"][sl].float()
        st = t0.init_state(m.shape[0], m.device)
        for t in range(T):
            p, v = d["P"][sl, t].float(), d["V"][sl, t].float()
            a, st = t0.force(p, v, m, mk, st)
            A0[sl, t] = a * mk[..., None]
            AT[sl, t] = true_acc(p, m, mk)
    return A0, AT


def make_nets(K, args, dev):
    return nn.ModuleList(TierNet(nfeat(True, False, args.order3), H0 * 2 ** k, args.width, args.layers, residual=args.residual, latent_ch=0, err=False,
                                 out_scale=args.out_scale, log_rho=True) for k in range(1, K + 1)).to(dev)


def make_field(tier0, nets):
    return TieredField(tier0, list(nets), N0, H0, G, HALO, BLEND, static=True, order3=ORDER3[0])


def save(path, **st):
    torch.save(st, path + ".tmp")
    os.replace(path + ".tmp", path)


def on_term(*_):
    STOP.append(1)


def lr_at(args, frac):
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))


def gen(args, it, stage):
    return torch.Generator().manual_seed(args.seed * 1_000_003 + stage * 100_003 + it)


def train_s(args, dev, log):
    d = make_scenes(args.scenes, args.steps, args.seed, dev)
    tpath = os.path.join(args.ckpt_dir, f"targets_{args.seed}_{args.scenes}_{args.steps}{'_far%g' % args.far_radius if args.target == 'far' else ''}.pt")
    if args.target == "far":
        m, mk = d["M"].float(), d["mask"].float()
        AT = torch.stack([far_acc(d["P"][:, t].float(), m, mk, args.far_radius) for t in range(d["P"].shape[1])], 1)
        A0 = torch.zeros_like(AT)
    elif os.path.exists(tpath):
        A0, AT = [x.to(dev) for x in torch.load(tpath)]
    else:
        t0 = load_tier0(args.tier0, dev)
        t = time.time()
        A0, AT = force_series(t0, d)
        torch.save([A0.cpu(), AT.cpu()], tpath)
        del t0
        log(f"targets {tuple(A0.shape)} in {time.time() - t:.0f}s")
    rel0 = ((AT - A0) ** 2 * d["mask"][:, None, :, None]).sum() / (AT ** 2 * d["mask"][:, None, :, None]).sum()
    log(f"train tier-0 rel force err (sq) {float(rel0):.4f}")
    nets = make_nets(args.tiers, args, dev)
    stage, it, ck = 0, 0, os.path.join(args.ckpt_dir, f"tiered_S{args.tag}.pt")
    opt = None
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        nets.load_state_dict(st["nets"])
        stage, it = st["stage"], st["it"]
        log(f"resumed stage {stage} it {it}")
    S, T = d["P"].shape[:2]
    while stage < args.tiers:
        for j, n in enumerate(nets):
            n.requires_grad_(j == stage)
        field = make_field(ZeroTier0(), nets[:stage + 1])
        opt = torch.optim.Adam(nets[stage].parameters(), lr=args.lr)
        if args.resume and os.path.exists(ck) and it > 0:
            opt.load_state_dict(st["opt"])
        t_start = time.time()
        while it < args.iters:
            for pg in opt.param_groups:
                pg["lr"] = lr_at(args, it / max(1, args.iters - 1))
            g = gen(args, it, stage)
            idx = torch.randint(S, (args.batch,), generator=g).to(dev)
            tt = torch.randint(T, (args.batch,), generator=g).to(dev)
            pos, vel = d["P"][idx, tt].float(), d["V"][idx, tt].float()
            m, mk = d["M"][idx].float(), d["mask"][idx].float()
            tgt = (AT[idx, tt] - A0[idx, tt]) * mk[..., None]
            a, _ = field.upper_force(pos, vel, m, mk, field.init_state(args.batch, dev))
            loss = ((a * mk[..., None] - tgt) ** 2).sum() / (tgt ** 2).sum().clamp_min(1e-9)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(nets[stage].parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(gn):
                opt.step()
            it += 1
            if it % args.log_every == 0:
                log(f"S stage {stage + 1} it {it} loss {loss.item():.4f} t {time.time() - t_start:.0f}s")
            if it % args.ckpt_every == 0 or STOP or it == args.iters:
                save(ck, nets=nets.state_dict(), opt=opt.state_dict(), stage=stage + (it == args.iters), it=0 if it == args.iters else it, args=vars(args))
            if STOP:
                log("SIGTERM: checkpoint saved, exiting")
                return
        stage, it = stage + 1, 0
    log("S done")


def train_r(args, dev, log):
    d = make_scenes(args.scenes, args.steps, args.seed, dev)
    t0 = load_tier0(args.tier0, dev)
    nets = make_nets(args.tiers, args, dev)
    nets.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["nets"])
    field = make_field(t0, nets)
    opt = torch.optim.Adam(nets.parameters(), lr=args.lr)
    ck, it = os.path.join(args.ckpt_dir, "tiered_R.pt"), 0
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        nets.load_state_dict(st["nets"])
        opt.load_state_dict(st["opt"])
        it = st["it"]
        log(f"resumed it {it}")
    S, T = d["P"].shape[:2]
    t_start = time.time()
    while it < args.iters:
        frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr_at(args, frac)
        g = gen(args, it, 0)
        idx = torch.randint(S, (args.batch,), generator=g).to(dev)
        s0 = torch.randint(0, T - k, (args.batch,), generator=g).to(dev)
        tt = s0[:, None] + torch.arange(k + 1, device=dev)[None]
        Pw, Vw = d["P"][idx[:, None], tt].float(), d["V"][idx[:, None], tt].float()
        m, mk = d["M"][idx].float(), d["mask"][idx].float()
        pos, vel, st, a_prev = Pw[:, 0], Vw[:, 0], field.init_state(args.batch, dev), None
        w, denom, lp, lv = mk[:, :, None], mk.sum() * 2 * k, 0.0, 0.0
        for s in range(k):
            pos, vel, st, a_prev = field.step_verlet(pos, vel, m, mk, st, a_prev)
            lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / denom
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
        loss = lp + 0.1 * lv
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        it += 1
        if it % args.log_every == 0:
            log(f"R it {it} k {k} loss {loss.item():.6f} t {time.time() - t_start:.0f}s")
        if it % args.ckpt_every == 0 or STOP or it == args.iters:
            save(ck, nets=nets.state_dict(), opt=opt.state_dict(), it=it, args=vars(args))
        if STOP:
            log("SIGTERM: checkpoint saved, exiting")
            return
    log("R done")


def com_err(a_pred, a_true, lab):
    out = []
    for s in range(a_pred.shape[0]):
        for c in range(int(lab[s].max()) + 1):
            sel = lab[s] == c
            at = a_true[s][sel].mean(0)
            out.append(float((a_pred[s][sel].mean(0) - at).norm() / at.norm().clamp_min(1e-9)))
    return float(np.mean(out))


@torch.no_grad()
def eval_arm(t0, nets, d, ticks=(10, 50, 100)):
    field = make_field(t0, nets) if nets is not None else None
    S, T = d["P"].shape[:2]
    m, mk = d["M"].float(), d["mask"].float()
    num = den = 0.0
    coms = []
    st = field.init_state(S, m.device) if field else t0.init_state(S, m.device)
    for t in range(0, T):
        p, v = d["P"][:, t].float(), d["V"][:, t].float()
        a, st_new, _ = field.force(p, v, m, mk, st) if field else (*t0.force(p, v, m, mk, st), None)
        st = st_new
        if t % 10 == 0:
            at = true_acc(p, m, mk)
            num += float(((a - at) ** 2 * mk[..., None]).sum())
            den += float((at ** 2 * mk[..., None]).sum())
            coms.append(com_err(a, at, d["lab"]))
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    st, a_prev, traj = (field.init_state(S, m.device) if field else t0.init_state(S, m.device)), None, {0: (pos, vel)}
    dt = DT
    for t in range(1, T):
        if field:
            pos, vel, st, a_prev = field.step_verlet(pos, vel, m, mk, st, a_prev)
        else:
            if a_prev is None:
                a_prev, st = t0.force(pos, vel, m, mk, st)
            pn = pos + dt * vel + 0.5 * dt ** 2 * a_prev
            a_new, st = t0.force(pn, vel + dt * a_prev, m, mk, st)
            pos, vel, a_prev = pn, vel + 0.5 * dt * (a_prev + a_new), a_new
        if t in ticks:
            traj[t] = (pos, vel)
    out = {"force_rel": (num / den) ** 0.5, "com_err": float(np.mean(coms))}
    cnt = d["mask"].sum(1)
    for k in ticks:
        e = ((traj[k][0].double() - d["P"][:, k]).norm(dim=-1) * d["mask"]).sum(1) / cnt
        out[f"err@{k}"] = float(e.mean())
    Mx = d["M"]
    E0 = sf.energy(d["P"][:, 0], d["V"][:, 0], Mx, d["mask"], EPS)
    Em = sf.energy(traj[ticks[-1]][0].double(), traj[ticks[-1]][1].double(), Mx, d["mask"], EPS)
    Et = sf.energy(d["P"][:, ticks[-1]], d["V"][:, ticks[-1]], Mx, d["mask"], EPS)
    out["dE_model"] = float(((Em - E0).abs() / E0.abs()).mean())
    out["dE_true"] = float(((Et - E0).abs() / E0.abs()).mean())
    return out


@torch.no_grad()
def run_far(args, dev, log):
    res = {}
    for sd in EVAL_SEEDS:
        d = make_scenes(args.eval_scenes, args.eval_steps, sd, dev)
        m, mk = d["M"].float(), d["mask"].float()
        nets = make_nets(args.tiers, args, dev)
        nets.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"tiered_S{args.tag}.pt"), map_location=dev, weights_only=False)["nets"])
        field = make_field(ZeroTier0(), nets)
        num = den = ncl = 0.0
        for t in range(0, d["P"].shape[1], 10):
            p, v = d["P"][:, t].float(), d["V"][:, t].float()
            tg = far_acc(p, m, mk, args.far_radius)
            a, _ = field.upper_force(p, v, m, mk, field.init_state(p.shape[0], dev))
            num += float(((a - tg) ** 2 * mk[..., None]).sum())
            den += float((tg ** 2 * mk[..., None]).sum())
        res[str(sd)] = {"far_rel": (num / den) ** 0.5}
        log(f"seed {sd} far_rel {res[str(sd)]['far_rel']:.4f}")
    json.dump(res, open(args.out, "w"), indent=1)


def run_eval(args, dev, log):
    t0 = load_tier0(args.tier0, dev)
    arms = {"baseline": None}
    for name in args.arms.split(","):
        if name:
            nets = make_nets(args.tiers, args, dev)
            nets.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"tiered_{name}{args.tag}.pt"), map_location=dev, weights_only=False)["nets"])
            nets.eval().requires_grad_(False)
            arms[name] = nets
    res = {a: {} for a in arms}
    for sd in EVAL_SEEDS:
        d = make_scenes(args.eval_scenes, args.eval_steps, sd, dev)
        for a, nets in arms.items():
            res[a][str(sd)] = eval_arm(t0, nets, d)
            log(f"seed {sd} {a} {json.dumps({k: round(v, 5) for k, v in res[a][str(sd)].items()})}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)


def compare(args, log):
    res = json.load(open(args.out))
    b = res["baseline"]
    for a in res:
        if a == "baseline":
            continue
        gains = {k: [1 - res[a][s][k] / b[s][k] for s in b] for k in ("force_rel", "com_err")}
        ok_f = all(g > 0.10 for k in gains for g in gains[k])
        ok_r = True
        for k in ("err@50", "dE_model"):
            base = np.array([b[s][k] for s in b])
            arm = np.array([res[a][s][k] for s in b])
            ok_r &= bool(arm.mean() <= base.mean() + base.std())
        log(f"{a}: force gains {json.dumps({k: [round(x, 3) for x in v] for k, v in gains.items()})} rollout_ok {ok_r} -> {'PASS' if ok_f and ok_r else 'KILL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["S", "R", "eval", "compare", "far"], required=True)
    ap.add_argument("--tier0", default=os.path.expanduser("~/E_ms_kp_pot_v_g128.pt"))
    ap.add_argument("--init", default=None)
    ap.add_argument("--arms", default="S")
    ap.add_argument("--target", choices=["resid", "far"], default="resid")
    ap.add_argument("--far-radius", type=float, default=20.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--order3", action="store_true")
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--out-scale", type=float, default=1.0)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--scenes", type=int, default=160)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--eval-scenes", type=int, default=24)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=2)
    ap.add_argument("--k-end", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-dir", default="checkpoints/tiered")
    ap.add_argument("--out", default="results/tiered_eval.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ORDER3[0] = args.order3
    os.makedirs(args.ckpt_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term)
    torch.manual_seed(args.seed)

    def log(s):
        print(s, flush=True)

    {"S": lambda: train_s(args, dev, log), "R": lambda: train_r(args, dev, log), "eval": lambda: run_eval(args, dev, log), "far": lambda: run_far(args, dev, log),
     "compare": lambda: compare(args, log)}[args.mode]()


if __name__ == "__main__":
    main()
