"""Root-cause decomposition of long-rollout error for scatter-field (docs/debugging/long-rollout-root-cause.md).

  PYTHONPATH=. python scripts/scatter_rootcause.py --e CKPT_E --b CKPT_B --out results/rootcause/rc.json
Models: E (E_ms_kp_pot_v_g128), B (Exp B ckpt, same arch), P2/P4/P8 = analytic P3M (prior kernel, exact complement pair term, no net, pp radius 2/4/8, all neighbours).
Parts: (1) force error at truth states by nn-distance bin, shift-phase variance/bias split, velocity/field dependence; (2) two-body probe by separation;
(3) one-step TF error incl. exact-force Verlet-1 integrator floor; (4) free rollouts incl. shift-averaged models, exact+injected force noise, phase-only hybrid.
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts.scatter_error_growth import DEV, DT, EPS, exact_acc, make_scenes, nn_dist

TR = 100
KS = (5, 20, 50, 100)
NNB = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9)
G = 128
H = 64.0 / G


def exact_tick_m(p, v, m, M):
    h = DT / M
    a = exact_acc(p, m)
    for _ in range(M):
        v = v + 0.5 * h * a
        p = p + h * v
        a = exact_acc(p, m)
        v = v + 0.5 * h * a
    return p, v


def p3m(pp):
    kw = dict(grid=G, net="unet", recurrent=True, hidden_ch=0, momfix=True, kernel=True, pp=float(pp), split=True, potential=True, verlet=True,
              prior=True, nonet=True, knn=400)
    return sf.ScatterField(extent=64.0, dt=DT, in_scale=1.0, **kw).to(DEV).eval()


class Ens:
    """K fixed-shift copies of a model (own recurrent field each); members() -> list of accelerations."""

    def __init__(self, model, sc, shifts):
        self.model, self.sc, self.shifts = model, sc, shifts
        self.m = sc["m"][None].float()
        self.fields = [model.init_field(1, DEV)[0] for _ in shifts]

    def members(self, p, v):
        out = []
        for k, d in enumerate(self.shifts):
            a, f = self.model.force((p + d)[None].float(), v[None].float(), self.m, torch.ones_like(self.m), self.fields[k])
            self.fields[k] = f
            out.append(a[0].double())
        return out


Z = torch.zeros(2, dtype=torch.float64, device=DEV)
SH4 = [Z, torch.tensor([0.5 * H, 0.0], dtype=torch.float64, device=DEV), torch.tensor([0.0, 0.5 * H], dtype=torch.float64, device=DEV),
       torch.tensor([0.5 * H, 0.5 * H], dtype=torch.float64, device=DEV)]


def shifts8(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [Z] + [torch.rand(2, generator=g, device=DEV, dtype=torch.float64) * H for _ in range(7)]


def binidx(nn):
    b = torch.bucketize(nn, torch.tensor(NNB[1:-1], device=DEV, dtype=nn.dtype))
    return b


def force_analysis(name, model, scenes, nsh, log):
    """Teacher-forced along truth. Returns per-nn-bin stats of model force error and decompositions."""
    rec = {k: [] for k in ("nn", "err2", "ae2", "bias2", "var2", "vdep2", "boost2", "fld2", "spd")}
    tf = {"pos": [], "vel": []}
    g = torch.Generator(device=DEV).manual_seed(4738)
    for si, sc in enumerate(scenes):
        m = sc["m"]
        mm = m[None].float()
        ones = torch.ones_like(mm)
        sh = shifts8(4738 + si)[:nsh] if nsh else [Z]
        ens = Ens(model, sc, sh)
        for t in range(TR):
            p, v = sc["P"][t], sc["V"][t]
            mem = ens.members(p, v)
            a0 = mem[0]
            fld = ens.fields[0]
            ae = exact_acc(p, m)
            keep = p.abs().amax(-1) < 26.0
            e2 = ((a0 - ae) ** 2).sum(-1)
            if nsh:
                M = torch.stack(mem)
                abar = M.mean(0)
                err2 = ((M - ae) ** 2).sum(-1).mean(0)
                bias2 = ((abar - ae) ** 2).sum(-1)
                var2 = err2 - bias2
            else:
                err2, bias2, var2 = e2, e2, torch.zeros_like(e2)
            if t >= 2:
                f_in = ens.fields[0]
                av0, _ = model.force(p[None].float(), torch.zeros_like(v)[None].float(), mm, ones, f_in)
                u = torch.randn(2, generator=g, device=DEV, dtype=torch.float64)
                u = 2.0 * u / u.norm()
                av1, _ = model.force(p[None].float(), (v + u)[None].float(), mm, ones, f_in)
                af0, _ = model.force(p[None].float(), v[None].float(), mm, ones, torch.zeros_like(f_in))
                avb, _ = model.force(p[None].float(), v[None].float(), mm, ones, f_in)
                vdep = ((av0[0].double() - avb[0].double()) ** 2).sum(-1)
                boost = ((av1[0].double() - avb[0].double()) ** 2).sum(-1)
                fld2 = ((af0[0].double() - avb[0].double()) ** 2).sum(-1)
                kk = keep
                for k_, x in (("nn", nn_dist(p)), ("err2", err2), ("ae2", (ae ** 2).sum(-1)), ("bias2", bias2), ("var2", var2), ("vdep2", vdep),
                              ("boost2", boost), ("fld2", fld2), ("spd", v.norm(dim=-1))):
                    rec[k_].append(x[kk].cpu())
                # TF one step from truth state with carried field (model's own a at truth state)
                pn = p + DT * v + 0.5 * DT ** 2 * a0
                an, _ = model.force(pn[None].float(), (v + DT * a0)[None].float(), mm, ones, fld)
                vn = v + 0.5 * DT * (a0 + an[0].double())
                tf["pos"].append((pn - sc["P"][t + 1]).norm(dim=-1).cpu())
                tf["vel"].append((vn - sc["V"][t + 1]).norm(dim=-1).cpu())
        if si % 6 == 0:
            log(f"  force_analysis {name} scene {si}/{len(scenes)}")
    R = {k: torch.cat(v).numpy() for k, v in rec.items()}
    b = np.digitize(R["nn"], NNB[1:-1])
    out = {"all": summarize(R, np.ones_like(b, bool))}
    for i in range(len(NNB) - 1):
        out[f"nn {NNB[i]}-{NNB[i + 1]}"] = summarize(R, b == i)
    tot = R["err2"].sum()
    for i in range(len(NNB) - 1):
        out[f"nn {NNB[i]}-{NNB[i + 1]}"]["share_err2"] = float(R["err2"][b == i].sum() / tot)
    out["tf_pos_mean"], out["tf_vel_mean"] = float(torch.cat(tf["pos"]).mean()), float(torch.cat(tf["vel"]).mean())
    return out


def summarize(R, mk):
    if mk.sum() == 0:
        return {}
    ae2 = R["ae2"][mk].mean()
    e = R["err2"][mk].mean()
    return dict(frac=float(mk.mean()), rel_rms=float(np.sqrt(e / ae2)), abs_rms=float(np.sqrt(e)), ae_rms=float(np.sqrt(ae2)),
                var_share=float(R["var2"][mk].mean() / max(e, 1e-30)), bias_share=float(R["bias2"][mk].mean() / max(e, 1e-30)),
                vdep_over_err=float(R["vdep2"][mk].mean() / max(e, 1e-30)), boost_over_err=float(R["boost2"][mk].mean() / max(e, 1e-30)),
                fld_over_err=float(R["fld2"][mk].mean() / max(e, 1e-30)), spd=float(R["spd"][mk].mean()))


def exact_tf(scenes):
    pe, ve = [], []
    for sc in scenes:
        m = sc["m"]
        for t in range(2, TR):
            p, v = sc["P"][t], sc["V"][t]
            a = exact_acc(p, m)
            pn = p + DT * v + 0.5 * DT ** 2 * a
            an = exact_acc(pn, m)
            vn = v + 0.5 * DT * (a + an)
            pe.append((pn - sc["P"][t + 1]).norm(dim=-1).cpu())
            ve.append((vn - sc["V"][t + 1]).norm(dim=-1).cpu())
    return float(torch.cat(pe).mean()), float(torch.cat(ve).mean())


def pair_probe(models, rs, n=64, seed=4738):
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = {}
    for name, model in models.items():
        res = {}
        for vel in (0.0, 1.5):
            rows = []
            for r in rs:
                c = (torch.rand(n, 1, 2, generator=g, device=DEV) - 0.5) * 12
                th = torch.rand(n, generator=g, device=DEV) * 2 * np.pi
                d = torch.stack([torch.cos(th), torch.sin(th)], -1) * r
                pos = torch.cat([c - d[:, None] / 2, c + d[:, None] / 2], 1)
                ph = torch.rand(n, generator=g, device=DEV) * 2 * np.pi
                w = torch.stack([torch.cos(ph), torch.sin(ph)], -1) * vel
                v = torch.stack([w, -w], 1)
                mass, mask = torch.ones(n, 2, device=DEV), torch.ones(n, 2, device=DEV)
                f = model.init_field(n, DEV)[0]
                for _ in range(3):
                    a, f = model.force(pos, v, mass, mask, f)
                dd = d.double()
                rr = (dd ** 2).sum(-1, keepdim=True)
                ae = dd * (rr + EPS ** 2) ** -1.5
                am = a[:, 0].double()
                dh = dd / dd.norm(dim=-1, keepdim=True)
                ex = ae.norm(dim=-1)
                rad = ((am - ae) * dh).sum(-1) / ex
                tan = ((am - ae) * torch.stack([-dh[:, 1], dh[:, 0]], -1)).sum(-1) / ex
                rows.append(dict(r=float(r), rad_mean=float(rad.mean()), rad_std=float(rad.std()), tan_rms=float(tan.pow(2).mean().sqrt()),
                                 rel_rms=float(((am - ae).norm(dim=-1) / ex).pow(2).mean().sqrt())))
            res[f"v{vel}"] = rows
        out[name] = res
    return out


def roll(fa, reset, sc, steps=TR):
    p, v = sc["P"][0], sc["V"][0]
    reset()
    a_prev = fa(p, v)
    P = [p]
    for _ in range(steps):
        pn = p + DT * v + 0.5 * DT ** 2 * a_prev
        an = fa(pn, v + DT * a_prev)
        v = v + 0.5 * DT * (a_prev + an)
        p, a_prev = pn, an
        P.append(p)
    return torch.stack(P)


def model_fa(model, sc):
    st = {}
    m = sc["m"][None].float()
    ones = torch.ones_like(m)

    def reset():
        st["f"] = model.init_field(1, DEV)[0]

    def fa(p, v):
        a, st["f"] = model.force(p[None].float(), v[None].float(), m, ones, st["f"])
        return a[0].double()

    return fa, reset


def ens_fa(model, sc, shifts, mode):
    st = {}

    def reset():
        st["e"] = Ens(model, sc, shifts)

    def fa(p, v):
        mem = st["e"].members(p, v)
        abar = torch.stack(mem).mean(0)
        if mode == "avg":
            return abar
        return exact_acc(p, sc["m"]) + (mem[0] - abar)

    return fa, reset


def noisy_fa(sc, sigma, sel, gen):
    m = sc["m"]

    def fa(p, v):
        a = exact_acc(p, m)
        xi = torch.randn(a.shape, generator=gen, device=DEV, dtype=torch.float64)
        mk = torch.ones(len(m), dtype=torch.bool, device=DEV)
        if sel == "near":
            mk = nn_dist(p) < 1.0
        elif sel == "far":
            mk = nn_dist(p) >= 1.0
        a = a * (1 + sigma * xi * mk[:, None])
        return a - a.mean(0, keepdim=True)

    return fa, lambda: None


def exact_fa(sc):
    return (lambda p, v: exact_acc(p, sc["m"])), (lambda: None)


def exact_sub(sc, M):
    P = [sc["P"][0]]
    p, v = sc["P"][0], sc["V"][0]
    for _ in range(TR):
        p, v = exact_tick_m(p, v, sc["m"], M)
        P.append(p)
    return torch.stack(P)


def curve_stats(errs):
    c = np.mean(errs, 0)
    return {str(k): float(c[k]) for k in KS}


def rollouts(models, scenes, log):
    gen = torch.Generator(device=DEV).manual_seed(4738)
    cur = {}

    def add(name, sc_i, P, sc):
        cur.setdefault(name, []).append((P - sc["P"][:TR + 1]).norm(dim=-1).mean(1).cpu().numpy())

    for si, sc in enumerate(scenes):
        for name, model in models.items():
            fa, rs = model_fa(model, sc)
            add(f"free:{name}", si, roll(fa, rs, sc), sc)
        for name in ("E", "B"):
            fa, rs = ens_fa(models[name], sc, SH4, "avg")
            add(f"shiftavg4:{name}", si, roll(fa, rs, sc), sc)
        fa, rs = ens_fa(models["E"], sc, SH4, "phase")
        add("exact+E_phase_part", si, roll(fa, rs, sc), sc)
        fa, rs = exact_fa(sc)
        add("exact_verlet1", si, roll(fa, rs, sc), sc)
        for M in (2,):
            add(f"exact_kdk_sub{M}", si, exact_sub(sc, M), sc)
        for s in (0.01, 0.02, 0.04, 0.08, 0.16):
            fa, rs = noisy_fa(sc, s, "all", gen)
            add(f"exact+iid_noise{s}", si, roll(fa, rs, sc), sc)
        for sel in ("near", "far"):
            fa, rs = noisy_fa(sc, 0.08, sel, gen)
            add(f"exact+iid_noise0.08_{sel}", si, roll(fa, rs, sc), sc)
        log(f"  rollouts scene {si}/{len(scenes)}")
    return {k: curve_stats(v) for k, v in cur.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(4738)
    log = lambda s: print(s, flush=True)
    models = {"E": ss.build_model("E", DT, DEV, args.e), "B": ss.build_model("E", DT, DEV, args.b), "P2": p3m(2), "P4": p3m(4), "P8": p3m(8)}
    out = {"E": args.e, "B": args.b}
    fams = {"expB": (24, 9300), "cluster": (12, 9200)}
    t0 = time.time()
    with torch.no_grad():
        rs = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
        out["pair_probe"] = pair_probe(models, rs)
        json.dump(out, open(args.out, "w"))
        for fam, (n, seed) in fams.items():
            n = 2 if args.smoke else n
            scenes = make_scenes(fam, n, seed)
            log(f"{fam}: {len(scenes)} scenes N={[len(s['m']) for s in scenes][:6]} {time.time() - t0:.0f}s")
            res = {"N": [len(s["m"]) for s in scenes]}
            res["exact_verlet1_tf"] = exact_tf(scenes)
            res["force"] = {}
            for name, model in models.items():
                res["force"][name] = force_analysis(name, model, scenes, 8 if name in ("E", "B") else 0, log)
                log(f"{fam} force {name} rel_rms all {res['force'][name]['all']['rel_rms']:.4f} tf_pos {res['force'][name]['tf_pos_mean']:.2e} {time.time() - t0:.0f}s")
            res["rollouts"] = rollouts(models, scenes, log)
            out[fam] = res
            json.dump(out, open(args.out, "w"))
            for k, v in res["rollouts"].items():
                log(f"{fam} {k:34s} " + " ".join(f"@{kk} {vv:.4g}" for kk, vv in v.items()))
    log("done")


if __name__ == "__main__":
    main()
