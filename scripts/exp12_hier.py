"""Experiment 12: S1 tier 0 (scripts/scatter_tier0 ext spec) tiled over a 128-unit span with the upper tiers of model/tiered supplying
each window's external field (Taylor coefficients to order 3 of the acceleration due to sources outside the window).
Modes: smoke (checks), se (train upper tiers on the outside-window field at sample points), je (rollout fine-tune, S1 frozen),
eval (arms OR/ORLS/SE/JE/BL vs tiled Exp E + far kernel), all with seed 4738 and held-out seeds 9100/9200/9300.
Needs the code of the integration worktree (scripts/tiered_adapters.py, scatter_tier0.py, ...) on the path."""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.func import jacfwd, vmap

from model.tiered import TierNet, TieredField, ZeroTier0, nfeat
from scripts import scatter_field as sf
from scripts import train_tiered as ttr

DT, EPS = ttr.DT, ttr.EPS
NT, TILE, WIN = 4, 32.0, 32.0
NW = NT * NT
STOP = []
LOCAL = None


def sample_pts(dev):
    c = torch.linspace(-18.0, 18.0, 7, device=dev)
    y, x = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([x, y], -1).reshape(-1, 2)


def centres(dev):
    c = (torch.arange(NT, device=dev, dtype=torch.float32) + 0.5) * TILE - NT * TILE / 2
    return torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)


def owner(pos):
    ij = ((pos + NT * TILE / 2) / TILE).floor().clamp(0, NT - 1).long()
    return ij[..., 0] * NT + ij[..., 1]


def windows(pos, mask, cen):
    rel = pos[:, None] - cen[None, :, None]
    inside = (rel.abs().amax(-1) < WIN).to(pos.dtype)
    return rel, inside * mask[:, None], (1 - inside) * mask[:, None]


def acc_at(x, spos, sm):
    d = spos[:, None, :] - x[None]
    w = sm[:, None] * ((d ** 2).sum(-1) + EPS ** 2) ** -1.5
    return (d * w[..., None]).sum(0)


def _a(x, spos, sm):
    return acc_at(x[None], spos, sm)[0]


def _taylor(x, spos, sm):
    f = lambda y: _a(y, spos, sm)
    return f(x), jacfwd(f)(x), jacfwd(jacfwd(f))(x), jacfwd(jacfwd(jacfwd(f)))(x)


taylor_b = vmap(_taylor, in_dims=(0, 0, 0))


def ext_from_taylor(a0, J, Q, C):
    H = torch.stack([-J[:, 0, 0], -0.5 * (J[:, 0, 1] + J[:, 1, 0]), -J[:, 1, 1]], -1)
    return -a0, H, Q, C


def exact_ext(rel, out_mask, mass):
    """rel (Bw,N,2) positions relative to the window centre, sources = out_mask*mass."""
    x0 = torch.zeros(rel.shape[0], 2, device=rel.device, dtype=rel.dtype)
    return ext_from_taylor(*taylor_b(x0, rel, out_mask * mass))


def basis(pts):
    x, y = pts[..., 0], pts[..., 1]
    return torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3], -1)


def coef_to_ext(a):
    """a (Bw,P,2) accelerations at sample points (window-local) -> (g, H, Q, C) by least squares on the cubic monomial basis."""
    pts = sample_pts(a.device)
    deg = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], device=a.device, dtype=a.dtype)
    sc = 18.0 ** deg
    c = (torch.linalg.pinv(basis(pts / 18.0).double()).to(a.dtype) @ a) / sc[None, :, None]
    Bw = a.shape[0]
    a0 = c[:, 0]
    J = torch.stack([c[:, 1], c[:, 2]], -1)
    Q = a.new_zeros(Bw, 2, 2, 2)
    Q[:, :, 0, 0], Q[:, :, 0, 1], Q[:, :, 1, 0], Q[:, :, 1, 1] = 2 * c[:, 3], c[:, 4], c[:, 4], 2 * c[:, 5]
    C = a.new_zeros(Bw, 2, 2, 2, 2)
    C[:, :, 0, 0, 0] = 6 * c[:, 6]
    for ix in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        C[:, :, ix[0], ix[1], ix[2]] = 2 * c[:, 7]
    for ix in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        C[:, :, ix[0], ix[1], ix[2]] = 2 * c[:, 8]
    C[:, :, 1, 1, 1] = 6 * c[:, 9]
    return ext_from_taylor(a0, J, Q, C)


def poly_ext_acc(ext, x):
    g, H, Q, C = ext
    a = -torch.stack([g[:, None, 0] + H[:, None, 0] * x[..., 0] + H[:, None, 1] * x[..., 1], g[:, None, 1] + H[:, None, 1] * x[..., 0] + H[:, None, 2] * x[..., 1]], -1)
    a = a + 0.5 * (Q[:, None] * x[:, :, None, :, None] * x[:, :, None, None, :]).sum((-1, -2))
    xj, xk, xl = x[:, :, None, :, None, None], x[:, :, None, None, :, None], x[:, :, None, None, None, :]
    return a + (C[:, None] * xj * xk * xl).sum((-1, -2, -3)) / 6


class S1Windows:
    """S1 (frozen) applied on every 64-unit window of the 4x4 tiling; owned particles take the acceleration of their tile's window."""

    def __init__(self, ckpt, dev, chunk=64):
        from scripts import tiered_adapters as ta

        self.ad = ta.load_t0ext(ckpt, dev)
        self.m, self.t0 = self.ad.m, self.ad.m.t0
        if torch.load(ckpt, map_location='cpu', weights_only=False)['args'].get('blind'):
            self.m.vblind = self.m.norec = True
        self.dev, self.chunk = dev, chunk
        self.cen = centres(dev)

    def init_state(self, B):
        f = self.m.init_field(B * NW, self.dev, torch.float32)
        return f[0] if isinstance(f, tuple) else f

    def force(self, pos, vel, mass, mask, ext, fld):
        B, N = mask.shape
        rel, inw, _ = windows(pos, mask, self.cen)
        rel = rel.reshape(B * NW, N, 2)
        v = vel[:, None].expand(B, NW, N, 2).reshape(B * NW, N, 2)
        mm = mass[:, None].expand(B, NW, N).reshape(B * NW, N)
        k = inw.reshape(B * NW, N)
        g, H, Q, C = ext
        out, nf = [], []
        for i in range(0, B * NW, self.chunk):
            sl = slice(i, i + self.chunk)
            n = rel[sl].shape[0]
            self.t0.set_external(self.m, phi0=torch.zeros(n, device=pos.device), g=g[sl], H=H[sl], high=(Q[sl], C[sl]))
            self.t0.set_boundary(self.m, sdf=None, edge=["open"] * n)
            a, f2 = self.m.force(rel[sl], v[sl], mm[sl], k[sl], fld[sl])
            out.append(a)
            nf.append(f2)
        a = torch.cat(out).reshape(B, NW, N, 2)
        own = owner(pos)
        acc = torch.gather(a, 1, own[:, None, :, None].expand(B, 1, N, 2)).squeeze(1)
        return acc * mask[..., None], torch.cat(nf)


def make_up_nets(args, dev):
    return nn.ModuleList(TierNet(nfeat(True, False, False), ttr.H0 * 2 ** k, args.width, args.layers, residual=False, latent_ch=0, err=False,
                                 out_scale=1.0, log_rho=True) for k in range(1, args.tiers + 1)).to(dev)


def make_up_field(nets):
    f = TieredField(ZeroTier0(), list(nets), ttr.N0, ttr.H0, ttr.G, ttr.HALO, ttr.BLEND, static=True, mean_remove=False)
    f.vblind_up = True
    return f


def learned_samples(field, pos, mass, mask, cen, chunk=32):
    """Upper-tier acceleration at the 49 sample points of every window from sources outside that window. -> (B*NW, 49, 2) window-local."""
    B, N = mask.shape
    _, _, outm = windows(pos, mask, cen)
    P = pos[:, None].expand(B, NW, N, 2).reshape(B * NW, N, 2)
    Mx = mass[:, None].expand(B, NW, N).reshape(B * NW, N)
    K = outm.reshape(B * NW, N)
    pts = sample_pts(pos.device)
    ev = (cen[None, :, None, :] + pts[None, None]).expand(B, NW, -1, -1).reshape(B * NW, -1, 2)
    res = []
    for i in range(0, B * NW, chunk):
        sl = slice(i, i + chunk)
        n = P[sl].shape[0]
        a, _ = field.upper_force(P[sl], torch.zeros_like(P[sl]), Mx[sl], K[sl], field.init_state(n, pos.device), at=ev[sl])
        res.append(a)
    return torch.cat(res)


def true_samples(pos, mass, mask, cen):
    B, N = mask.shape
    _, _, outm = windows(pos, mask, cen)
    pts = sample_pts(pos.device)
    ev = (cen[None, :, None, :] + pts[None, None]).expand(B, NW, -1, -1)
    d = pos[:, None, None, :, :] - ev[:, :, :, None, :]
    w = (mass[:, None] * outm)[:, :, None, :] * ((d ** 2).sum(-1) + EPS ** 2) ** -1.5
    return (d * w[..., None]).sum(3).reshape(B * NW, -1, 2)


def ext_for(kind, pos, vel, mass, mask, cen, field=None):
    B, N = mask.shape
    rel, _, outm = windows(pos, mask, cen)
    if kind == "OR":
        rel_f = rel.reshape(B * NW, N, 2)
        return exact_ext(rel_f, outm.reshape(B * NW, N), mass[:, None].expand(B, NW, N).reshape(B * NW, N))
    if kind == "ORLS":
        return coef_to_ext(true_samples(pos, mass, mask, cen))
    return coef_to_ext(learned_samples(field, pos, mass, mask, cen))


def pair_energy(pos, vel, mass, mask):
    return sf.energy(pos, vel, mass, mask, EPS)


@torch.no_grad()
def eval_generic(force_fn, init_state, d, ticks=(10, 20, 50, 100)):
    S, T = d["P"].shape[:2]
    ticks = tuple(k for k in ticks if k < T)
    m, mk = d["M"].float(), d["mask"].float()
    num = den = 0.0
    coms = []
    st = init_state(S)
    for t in range(0, T):
        p, v = d["P"][:, t].float(), d["V"][:, t].float()
        a, st = force_fn(p, v, m, mk, st)
        if t % 10 == 0:
            at = ttr.true_acc(p, m, mk)
            num += float(((a - at) ** 2 * mk[..., None]).sum())
            den += float((at ** 2 * mk[..., None]).sum())
            coms.append(ttr.com_err(a, at, d["lab"]))
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    st = init_state(S)
    a_prev, st = force_fn(pos, vel, m, mk, st)
    traj = {0: (pos, vel)}
    for t in range(1, T):
        pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
        a_new, st = force_fn(pn, vel + DT * a_prev, m, mk, st)
        pos, vel, a_prev = pn, vel + 0.5 * DT * (a_prev + a_new), a_new
        if t in ticks:
            traj[t] = (pos, vel)
    out = {"force_rel": (num / den) ** 0.5, "com_err": float(np.mean(coms))}
    cnt = d["mask"].sum(1)
    for k in ticks:
        e = ((traj[k][0].double() - d["P"][:, k]).norm(dim=-1) * d["mask"]).sum(1) / cnt
        out[f"err@{k}"] = float(e.mean())
    E0 = sf.energy(d["P"][:, 0], d["V"][:, 0], d["M"], d["mask"], EPS)
    Em = sf.energy(traj[ticks[-1]][0].double(), traj[ticks[-1]][1].double(), d["M"], d["mask"], EPS)
    out["dE_model"] = float(((Em - E0).abs() / E0.abs()).mean())
    return out


def far_ext_rel(kind, field, d, cen, dev, nt=10):
    num = den = 0.0
    for t in range(0, d["P"].shape[1], nt):
        p, v = d["P"][:, t].float(), d["V"][:, t].float()
        m, mk = d["M"].float(), d["mask"].float()
        ext = ext_for(kind, p, v, m, mk, cen, field)
        tgt = true_samples(p, m, mk, cen)
        B, N = mk.shape
        pts = sample_pts(dev)[None].expand(B * NW, -1, -1)
        pr = poly_ext_acc(ext, pts)
        num += float(((pr - tgt) ** 2).sum())
        den += float((tgt ** 2).sum())
    return (num / den) ** 0.5


def save(path, **st):
    ttr.save(path, **st)


def train_se(args, dev, log):
    d = ttr.make_scenes(args.scenes, args.steps, args.seed, dev)
    nets = make_up_nets(args, dev)
    cen = centres(dev)
    pts = sample_pts(dev)
    ck = os.path.join(args.ckpt_dir, "e12_SE.pt")
    stage, it = 0, 0
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        nets.load_state_dict(st["nets"])
        stage, it = st["stage"], st["it"]
        log(f"resumed stage {stage} it {it}")
    S, T = d["P"].shape[:2]
    while stage < args.tiers:
        for j, n in enumerate(nets):
            n.requires_grad_(j == stage)
        field = make_up_field(nets[:stage + 1])
        opt = torch.optim.Adam(nets[stage].parameters(), lr=args.lr)
        if args.resume and os.path.exists(ck) and it > 0:
            opt.load_state_dict(st["opt"])
        t_start = time.time()
        while it < args.iters:
            for pg in opt.param_groups:
                pg["lr"] = ttr.lr_at(args, it / args.iters)
            g = ttr.gen(args, it, stage)
            idx = torch.randint(S, (args.batch,), generator=g).to(dev)
            tt = torch.randint(T, (args.batch,), generator=g).to(dev)
            win = torch.randint(NW, (args.batch,), generator=g).to(dev)
            pos, m, mk = d["P"][idx, tt].float(), d["M"][idx].float(), d["mask"][idx].float()
            c = cen[win]
            outm = mk * ((pos - c[:, None]).abs().amax(-1) >= WIN).float()
            ev = c[:, None] + pts[None]
            dd = pos[:, None] - ev[:, :, None]
            w = (m * outm)[:, None] * ((dd ** 2).sum(-1) + EPS ** 2) ** -1.5
            tgt = (dd * w[..., None]).sum(2)
            a, _ = field.upper_force(pos, torch.zeros_like(pos), m, outm, field.init_state(args.batch, dev), at=ev)
            loss = ((a - tgt) ** 2).sum() / (tgt ** 2).sum().clamp_min(1e-12)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(nets[stage].parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(gn):
                opt.step()
            it += 1
            if it % args.log_every == 0:
                log(f"SE stage {stage + 1} it {it} loss {loss.item():.4f} t {time.time() - t_start:.0f}s")
            if it % args.ckpt_every == 0 or STOP or it == args.iters:
                save(ck, nets=nets.state_dict(), opt=opt.state_dict(), stage=stage + (it == args.iters), it=0 if it == args.iters else it, args=vars(args))
            if STOP:
                log("SIGTERM: saved")
                return
        stage, it = stage + 1, 0
    log("SE done")


def train_je(args, dev, log):
    d = ttr.make_scenes(args.scenes, args.steps, args.seed, dev)
    nets = make_up_nets(args, dev)
    nets.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["nets"])
    nets.requires_grad_(True)
    field = make_up_field(nets)
    s1 = S1Windows(args.s1, dev, chunk=args.chunk)
    cen = centres(dev)
    opt = torch.optim.Adam(nets.parameters(), lr=args.lr)
    ck, it = os.path.join(args.ckpt_dir, "e12_JE.pt"), 0
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        nets.load_state_dict(st["nets"])
        opt.load_state_dict(st["opt"])
        it = st["it"]
        log(f"resumed it {it}")
    S, T = d["P"].shape[:2]
    t_start = time.time()

    def acc(pos, vel, m, mk, fld):
        ext = coef_to_ext(learned_samples(field, pos, m, mk, cen))
        return s1.force(pos, vel, m, mk, ext, fld)

    while it < args.iters:
        frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
        for pg in opt.param_groups:
            pg["lr"] = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        g = ttr.gen(args, it, 0)
        idx = torch.randint(S, (args.batch,), generator=g).to(dev)
        s0 = torch.randint(0, T - k, (args.batch,), generator=g).to(dev)
        tt = s0[:, None] + torch.arange(k + 1, device=dev)[None]
        Pw, Vw = d["P"][idx[:, None], tt].float(), d["V"][idx[:, None], tt].float()
        m, mk = d["M"][idx].float(), d["mask"][idx].float()
        pos, vel = Pw[:, 0], Vw[:, 0]
        fld = s1.init_state(args.batch)
        a_prev, fld = acc(pos, vel, m, mk, fld)
        w, denom, lp, lv, le = mk[:, :, None], mk.sum() * 2 * k, 0.0, 0.0, 0.0

        def one(pos, vel, a_prev, fld):
            pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
            a_new, fld = acc(pn, vel + DT * a_prev, m, mk, fld)
            return pn, vel + 0.5 * DT * (a_prev + a_new), a_new, fld

        for s in range(k):
            pos, vel, a_prev, fld = torch.utils.checkpoint.checkpoint(one, pos, vel, a_prev, fld, use_reentrant=False)
            lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / denom
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
            if args.e_w > 0:
                Et = sf.energy(Pw[:, s + 1], Vw[:, s + 1], m, mk, EPS)
                le = le + (((sf.energy(pos, vel, m, mk, EPS) - Et) / Et.abs().clamp_min(1e-6)) ** 2).mean() / k
        loss = lp + 0.1 * lv + args.e_w * le
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        it += 1
        if it % args.log_every == 0:
            log(f"JE it {it} k {k} pos {float(lp):.6f} e {float(le):.6f} t {time.time() - t_start:.0f}s")
        if it % args.ckpt_every == 0 or STOP or it == args.iters:
            save(ck, nets=nets.state_dict(), opt=opt.state_dict(), it=it, args=vars(args))
        if STOP:
            log("SIGTERM: saved")
            return
    log("JE done")


@torch.no_grad()
def run_eval(args, dev, log):
    cen = centres(dev)
    s1 = S1Windows(args.s1, dev, chunk=args.chunk)
    arms = {}
    for name in args.arms.split(","):
        if name in ("OR", "ORLS"):
            arms[name] = (name, None)
        elif name in ("SE", "JE"):
            nets = make_up_nets(args, dev)
            nets.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"e12_{name}.pt"), map_location=dev, weights_only=False)["nets"])
            nets.eval().requires_grad_(False)
            arms[name] = (name, make_up_field(nets))
        elif name == "BL":
            from scripts import tiered_adapters as ta

            arms[name] = ("BL", ta.load_tiled(f"{args.tier0},nt=4,tile=32,halo=16,far=1", dev))
    res = {a: {} for a in arms}
    for sd in ttr.EVAL_SEEDS:
        d = ttr.make_scenes(args.eval_scenes, args.eval_steps, sd, dev)
        for a, (kind, obj) in arms.items():
            if kind == "BL":
                res[a][str(sd)] = eval_generic(lambda p, v, m, k, st: obj.force(p, v, m, k, st), lambda S: obj.init_state(S, dev), d)
            else:
                f = lambda p, v, m, k, st, kind=kind, obj=obj: s1.force(p, v, m, k, ext_for(kind, p, v, m, k, cen, obj), st)
                res[a][str(sd)] = eval_generic(f, s1.init_state, d)
                res[a][str(sd)]["far_ext_rel"] = far_ext_rel(kind, obj, d, cen, dev)
            log(f"seed {sd} {a} {json.dumps({k: round(v, 5) for k, v in res[a][str(sd)].items()})}")
    json.dump(res, open(args.out, "w"), indent=1)


@torch.no_grad()
def run_smoke(args, dev, log):
    cen = centres(dev)
    d = ttr.make_scenes(4, 12, 4738, dev)
    p, v = d["P"][:, 3].float(), d["V"][:, 3].float()
    m, mk = d["M"].float(), d["mask"].float()
    B, N = mk.shape
    rel, inw, outm = windows(p, mk, cen)
    ex = exact_ext(rel.reshape(B * NW, N, 2), outm.reshape(B * NW, N), m[:, None].expand(B, NW, N).reshape(B * NW, N))
    ts = true_samples(p, m, mk, cen)
    ls = coef_to_ext(ts)
    pts = sample_pts(dev)[None].expand(B * NW, -1, -1)
    for nm, e in (("taylor", ex), ("ls", ls)):
        pr = poly_ext_acc(e, pts)
        log(f"smoke {nm} sample-point rel err {float(((pr - ts) ** 2).sum().sqrt() / (ts ** 2).sum().sqrt()):.4f}")
    s1 = S1Windows(args.s1, dev, chunk=args.chunk)
    fld = s1.init_state(B)
    a, fld = s1.force(p, v, m, mk, ex, fld)
    at = ttr.true_acc(p, m, mk)
    log(f"smoke S1 oracle-ext force_rel {float(((a - at) ** 2).sum().sqrt() / (at ** 2).sum().sqrt()):.4f}")
    z = tuple(torch.zeros_like(x) for x in ex)
    a0, _ = s1.force(p, v, m, mk, z, fld)
    log(f"smoke S1 no-ext force_rel {float(((a0 - at) ** 2).sum().sqrt() / (at ** 2).sum().sqrt()):.4f}")
    nets = make_up_nets(args, dev)
    field = make_up_field(nets)
    ls0 = learned_samples(field, p, m, mk, cen)
    log(f"smoke learned samples {tuple(ls0.shape)} finite {bool(torch.isfinite(ls0).all())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "se", "je", "eval"], required=True)
    ap.add_argument("--s1", default=os.path.expanduser("~/t0s/results/tier0_ext/S1.pt"))
    ap.add_argument("--tier0", default=os.path.expanduser("~/E_ms_kp_pot_v_g128.pt"))
    ap.add_argument("--init", default=None)
    ap.add_argument("--arms", default="OR,ORLS,SE,JE,BL")
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--scenes", type=int, default=1600)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--eval-scenes", type=int, default=24)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--k-start", type=int, default=2)
    ap.add_argument("--k-end", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--e-w", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--ckpt-dir", default="ck_e12")
    ap.add_argument("--out", default="res_e12.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))
    torch.manual_seed(args.seed)
    log = lambda s: print(s, flush=True)
    {"smoke": run_smoke, "se": train_se, "je": train_je, "eval": run_eval}[args.mode](args, dev, log)


if __name__ == "__main__":
    main()
