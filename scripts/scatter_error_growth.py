"""Error-growth diagnostic for scatter-field Exp E (E_ms_kp_pot_v_g128) on gravity scenes. See docs/debugging/error-growth-diagnostic.md.

  PYTHONPATH=. python scripts/scatter_error_growth.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/errgrowth/gravity.json
Families: expB (10-100 unit-mass, seed 9000), cluster (80-300, seed 9100), star (heavy + light, OOD for E, seed 9200). Envelope = same generators, train seed 4738.
All coordinates centered (sf.CENTER subtracted). Truth = fp64 KDK, 4 substeps, dt .1, eps .5 (== scatter_scenes.nbody_truth).
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss

DEV = torch.device("cuda")
DT, EPS, T = 0.1, 0.5, 150
KS = (1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150)
BINS = ((1, 10), (11, 30), (31, 60), (61, 100), (101, 150))
EDGE_LO, EDGE_HI = 26.0, 32.0
G = 128


def exact_acc(p, m, rc=None):
    d = p[None] - p[:, None]
    r2 = (d ** 2).sum(-1)
    inv = (r2 + EPS ** 2) ** -1.5
    inv.fill_diagonal_(0.0)
    if rc is not None:
        inv = inv * (r2 < rc ** 2)
    return (d * (inv * m[None])[..., None]).sum(1)


def exact_tick(p, v, m):
    h = DT / 4
    a = exact_acc(p, m)
    for _ in range(4):
        v = v + 0.5 * h * a
        p = p + h * v
        a = exact_acc(p, m)
        v = v + 0.5 * h * a
    return p, v


def exact_traj(p, v, m, steps):
    P, V = [p], [v]
    for _ in range(steps):
        p, v = exact_tick(p, v, m)
        P.append(p)
        V.append(v)
    return torch.stack(P), torch.stack(V)


def nn_dist(p):
    d = torch.cdist(p, p)
    d.fill_diagonal_(1e9)
    return d.min(1).values


def make_scenes(fam, num, seed):
    if fam == "expB":
        data = gs.make_dataset(num, (10, 100), 1, seed, scale=True, device=DEV, dt=DT, eps=EPS)
    elif fam == "cluster":
        data = sf.make_cluster_dataset(num, 1, seed, DEV, dt=DT, eps=EPS)
    else:
        data = sf.make_star_dataset(num, 1, seed, DEV, dt=DT, eps=EPS)
    out = []
    for d in data:
        p = torch.tensor(d[0][0] - sf.CENTER, dtype=torch.float64, device=DEV)
        v = torch.tensor(d[1][0], dtype=torch.float64, device=DEV)
        m = torch.tensor(d[2], dtype=torch.float64, device=DEV) if len(d) > 2 else torch.ones(len(p), dtype=torch.float64, device=DEV)
        P, V = exact_traj(p, v, m, T)
        out.append(dict(P=P, V=V, m=m, L=float((p ** 2).sum(-1).mean().sqrt())))
    return out


class Net:
    def __init__(self, model):
        self.model = model
        self.store = {}
        model.net.register_forward_hook(lambda mod, inp, out: self.store.update(x=inp[0].detach(), out=out.detach()))

    def init(self):
        return self.model.init_field(1, DEV)[0]

    def force(self, sc, p, v, field):
        a, f = self.model.force(p[None].float(), v[None].float(), sc["m"][None].float(), torch.ones(1, len(p), device=DEV), field)
        return a[0].double(), f


def classes(p, m):
    nn = nn_dist(p)
    r = p.abs().amax(-1)
    c = {"all": torch.ones_like(nn, dtype=torch.bool), "nn<1": nn < 1.0, "nn>=1": nn >= 1.0, "r<26": r < EDGE_LO,
         "band26-32": (r >= EDGE_LO) & (r < EDGE_HI), "r>=32": r >= EDGE_HI}
    if float(m.max()) > 5 * float(m.min()):
        c["heavy"] = m >= 8.0
        c["light"] = m < 8.0
    return c


def roll(net, sc, steps=T, reset_k=0, field_reset_k=0, oracle=None, oracle_k=0, tf=False, rec=None, init="model"):
    """Verlet rollout (== ScatterField._step verlet) from the truth IC. tf: reset state to truth at every tick (one-step error).
    reset_k: reset pos/vel to truth every k ticks. oracle in {all, near, iso, edge}: exact force replaces the model's for those bodies (every tick, or only ticks with t % oracle_k == 0).
    rec: list; if given, per-tick net-input/output stats are appended."""
    p, v, m = sc["P"][0], sc["V"][0], sc["m"]
    field = net.init()
    a_prev, field = net.force(sc, p, v, field)
    if init == "exact":
        a_prev = exact_acc(p, m)
    elif init == "warm":
        for _ in range(3):
            a_prev, field = net.force(sc, p, v, field)
    Pm, Vm = [p], [v]
    for t in range(steps):
        if tf or (reset_k and t > 0 and t % reset_k == 0):
            p, v = sc["P"][t], sc["V"][t]
            a_prev, _ = net.force(sc, p, v, field)
        if field_reset_k and t > 0 and t % field_reset_k == 0:
            field = net.init()
        pn = p + DT * v + 0.5 * DT ** 2 * a_prev
        vp = v + DT * a_prev
        an, field = net.force(sc, pn, vp, field)
        if rec is not None:
            rec.append(tick_stats(net, sc, pn, an))
        if oracle and (not oracle_k or t % oracle_k == 0):
            r = pn.abs().amax(-1)
            sel = {"all": torch.ones_like(r, dtype=torch.bool), "near": nn_dist(pn) < 1.0, "iso": nn_dist(pn) >= 1.0, "edge": r >= EDGE_LO}[oracle]
            an = torch.where(sel[:, None], exact_acc(pn, m), an)
        v = v + 0.5 * DT * (a_prev + an)
        p, a_prev = pn, an
        Pm.append(p)
        Vm.append(v)
    return torch.stack(Pm), torch.stack(Vm)


def tick_stats(net, sc, pn, an):
    s = net.store
    x, out = s["x"][0], s["out"][0]
    gacc = net.model.neg_grad(s["out"][:, :1])[0]
    gn = gacc.norm(dim=0)
    idx = int(gn.argmax())
    iy, ix = divmod(idx, G)
    cx, cy = (ix + 0.5) * net.model.h - 32.0, (iy + 0.5) * net.model.h - 32.0
    cell = torch.tensor([cx, cy], device=DEV, dtype=torch.float64)
    r = pn.abs().amax(-1)
    an_n = an.norm(dim=-1)
    w = int(an_n.argmax())
    return dict(rho_max=float(x[0].max()), mom_max=float(x[1:3].abs().max()), ak_max=float(x[3:5].abs().max()), fld_max=float(x[5].abs().max()),
                phi_max=float(out[0].abs().max()), gacc_max=float(gn.max()), a_max=float(an_n.max()), a_p99=float(an_n.quantile(0.99)),
                frac_band=float(((r >= EDGE_LO) & (r < EDGE_HI)).double().mean()), frac_out=float((r >= EDGE_HI).double().mean()),
                pmax=float(r.max()), cell_r=float(cell.abs().max()), cell_nearest_body=float((pn - cell).norm(dim=-1).min()),
                worst_nn=float(nn_dist(pn)[w]), worst_r=float(r[w]))


def acc_err(sums, key_cls, e, cls, t):
    for k, mk in cls.items():
        if bool(mk.any()):
            s = sums.setdefault((key_cls, k), np.zeros((2, T + 1)))
            s[0, t] += float(e[mk].sum())
            s[1, t] += int(mk.sum())


def curve(sums, name, k):
    s = sums.get((name, k))
    return None if s is None else np.where(s[1] > 0, s[0] / np.maximum(s[1], 1), np.nan)


def sel_ks(c, ks=KS):
    return None if c is None else [float(c[k]) for k in ks]


def sel_bins(c):
    return None if c is None else [float(np.nanmean(c[a:b + 1])) for a, b in BINS]


def lyap(e, dt=DT):
    e = np.asarray(e)
    idx = np.where((e > e[0] * 30) & (e < 0.3 * e.max()))[0]
    if len(idx) < 8:
        return None
    return float(np.polyfit((idx + 1) * dt, np.log(e[idx]), 1)[0])


def run_family(net, fam, scenes, env_scenes, args, log):
    t0 = time.time()
    res = {"n_scenes": len(scenes), "N": [len(s["m"]) for s in scenes]}
    sums = {}
    variants = {"free": {}, "reset5": dict(reset_k=5), "reset10": dict(reset_k=10), "reset20": dict(reset_k=20), "fieldreset10": dict(field_reset_k=10),
                "oracle_all": dict(oracle="all"), "oracle_tick_k10": dict(oracle="all", oracle_k=10), "oracle_near": dict(oracle="near"),
                "oracle_iso": dict(oracle="iso"), "oracle_edge": dict(oracle="edge"), "exactinit": dict(init="exact"), "oracle_all_exactinit": dict(oracle="all", init="exact"), "warm3": dict(init="warm"), "reset5_exactinit": dict(reset_k=5, init="exact")}
    free_curves = {}
    recs = []
    for si, sc in enumerate(scenes):
        # exp 1/2: variants
        for name, kw in variants.items():
            rec = [] if name == "free" else None
            Pm, Vm = roll(net, sc, rec=rec, **kw)
            ep = (Pm - sc["P"]).norm(dim=-1)
            ev = (Vm - sc["V"]).norm(dim=-1)
            for t in range(T + 1):
                cls = classes(sc["P"][t], sc["m"]) if name == "free" else {"all": torch.ones(len(sc["m"]), dtype=torch.bool, device=DEV)}
                acc_err(sums, f"{name}:ep", ep[t], cls, t)
                if name == "free":
                    acc_err(sums, f"{name}:ev", ev[t], cls, t)
            if name == "free":
                recs.append(rec)
                free_curves[si] = ep.mean(1).cpu().numpy()
        # TF one-step error + force error at truth states
        field = net.init()
        pT, vT = sc["P"][0], sc["V"][0]
        a_prev, field = net.force(sc, pT, vT, field)
        for t in range(T):
            p, v = sc["P"][t], sc["V"][t]
            pn = p + DT * v + 0.5 * DT ** 2 * a_prev
            an, field_new = net.force(sc, pn, v + DT * a_prev, field)
            vn = v + 0.5 * DT * (a_prev + an)
            cls = classes(sc["P"][t + 1], sc["m"])
            acc_err(sums, "tf:ep", (pn - sc["P"][t + 1]).norm(dim=-1), cls, t + 1)
            acc_err(sums, "tf:ev", (vn - sc["V"][t + 1]).norm(dim=-1), cls, t + 1)
            # force error at truth state t+1, model force with the field from this step
            p1, v1 = sc["P"][t + 1], sc["V"][t + 1]
            am, _ = net.force(sc, p1, v1, field_new)
            ae = exact_acc(p1, sc["m"])
            an_, af_ = exact_acc(p1, sc["m"], rc=2.0), None
            far = ae - an_
            ea = (am - ae).norm(dim=-1)
            rel = ea / ae.norm(dim=-1).clamp_min(1e-6)
            c2 = dict(cls)
            nd = an_.norm(dim=-1) > far.norm(dim=-1)
            c2["near-dominated"], c2["far-dominated"] = nd, ~nd
            acc_err(sums, "fe_abs", ea, c2, t + 1)
            acc_err(sums, "fe_rel", rel, c2, t + 1)
            rb = p1.abs().amax(-1)
            for lo, hi in ((0, 16), (16, 24), (24, 28), (28, 32), (32, 1e9)):
                mk = (rb >= lo) & (rb < hi)
                acc_err(sums, "fe_rel_rbin", rel, {f"{lo}-{hi}": mk}, t + 1)
                acc_err(sums, "fe_abs_rbin", ea, {f"{lo}-{hi}": mk}, t + 1)
            a_prev, field = an, field_new
        log(f"{fam} scene {si}/{len(scenes)} N={len(sc['m'])} free ep@20/50/100/150 {[round(float(free_curves[si][k]), 3) for k in (20, 50, 100, 150)]} {time.time() - t0:.0f}s")
    for key in sorted({k[0] for k in sums}):
        res[key] = {k[1]: dict(at_ks=sel_ks(curve(sums, key, k[1])), bin_means=sel_bins(curve(sums, key, k[1]))) for k in sums if k[0] == key}
    res["ks"], res["bins"] = KS, BINS

    # tf rms one-step error -> noisy-exact
    sp = float(np.nanmean(curve(sums, "tf:ep", "all")[1:]))
    sv = float(np.nanmean(curve(sums, "tf:ev", "all")[1:]))
    res["tf_mean_onestep_pos_vel"] = [sp, sv]
    noisy = {}
    g = torch.Generator(device=DEV).manual_seed(4738)
    for scale in (0.5, 1.0, 2.0):
        cs_ = []
        for sc in scenes:
            p, v = sc["P"][0], sc["V"][0]
            e = [0.0]
            for t in range(T):
                p, v = exact_tick(p, v, sc["m"])
                p = p + scale * sp / 1.4142 * torch.randn(p.shape, generator=g, device=DEV, dtype=torch.float64)
                v = v + scale * sv / 1.4142 * torch.randn(v.shape, generator=g, device=DEV, dtype=torch.float64)
                e.append(float((p - sc["P"][t + 1]).norm(dim=-1).mean()))
            cs_.append(e)
        noisy[str(scale)] = sel_ks(np.mean(cs_, 0))
    res["noisy_exact_ep"] = noisy
    res["free_ep_mean_curve"] = sel_ks(np.mean([free_curves[i] for i in free_curves], 0))

    # exp 4: IC noise amplification, model vs exact
    amp = {}
    for rel in (1e-6, 1e-5, 1e-4, 1e-3):
        ce, cm = [], []
        for si, sc in enumerate(scenes):
            noise = torch.randn(sc["P"][0].shape, generator=g, device=DEV, dtype=torch.float64) * rel * sc["L"]
            sc2 = dict(sc, P=sc["P"].clone(), V=sc["V"])
            sc2["P"][0] = sc["P"][0] + noise
            Pe, _ = exact_traj(sc2["P"][0], sc["V"][0], sc["m"], T)
            ce.append((Pe - sc["P"]).norm(dim=-1).mean(1).cpu().numpy())
            Pm0, _ = roll(net, sc)
            Pm1, _ = roll(net, sc2)
            cm.append((Pm1 - Pm0).norm(dim=-1).mean(1).cpu().numpy())
        ce, cm = np.mean(ce, 0), np.mean(cm, 0)
        amp[str(rel)] = dict(exact=sel_ks(ce), model=sel_ks(cm), init=float(np.mean([rel * s["L"] * 1.2533 for s in scenes])), lyap_exact=lyap(ce), lyap_model=lyap(cm))
    res["ic_noise"] = amp

    # exp 3: stats along free rollout; envelope from train-seed scenes driven by truth states
    env = []
    for sc in env_scenes:
        r = []
        roll(net, sc, tf=True, rec=r)
        env.append(r)
    keys = list(recs[0][0].keys())
    envmax = {k: max(r[t][k] for r in env for t in range(len(r))) for k in keys}
    envp99 = {k: float(np.percentile([r[t][k] for r in env for t in range(len(r))], 99)) for k in keys}
    st = {}
    for k in keys:
        arr = np.array([[r[t][k] for t in range(T)] for r in recs])
        st[k] = dict(mean_by_tick_bin=[float(arr[:, a - 1:b].mean()) for a, b in BINS], max_by_tick_bin=[float(arr[:, a - 1:b].max()) for a, b in BINS],
                     frac_scene_ticks_gt_env_max_by_bin=[float((arr[:, a - 1:b] > envmax[k]).mean()) for a, b in BINS], env_max=envmax[k], env_p99=envp99[k])
    res["ood"] = st
    first = {}
    for k in keys:
        arr = np.array([[r[t][k] for t in range(T)] for r in recs])
        first[k] = [int(np.argmax(a > envmax[k])) + 1 if (a > envmax[k]).any() else None for a in arr]
    res["ood_first_exceed_tick_per_scene"] = first
    res["a_max_curve_free_scene_max"] = [float(max(r[t]["a_max"] for r in recs)) for t in range(T)]
    res["a_max_curve_free_scene_median"] = [float(np.median([r[t]["a_max"] for r in recs])) for t in range(T)]
    res["secs"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fams", default="expB,cluster,star")
    ap.add_argument("--scenes", default="24,12,12")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(4738)
    model = ss.build_model("E", DT, DEV, args.ckpt)
    net = Net(model)
    seeds = {"expB": 9000, "cluster": 9100, "star": 9200}
    out = {"ckpt": args.ckpt, "T": T, "dt": DT, "eps": EPS}
    log = lambda s: print(s, flush=True)
    with torch.no_grad():
        for fam, n in zip(args.fams.split(","), [int(x) for x in args.scenes.split(",")]):
            n = 2 if args.smoke else n
            sc = make_scenes(fam, n, seeds[fam])
            env = make_scenes(fam, 2 if args.smoke else 8, 4738)
            if fam == "expB":
                d = gs.make_dataset(1, (10, 100), T, seeds[fam], scale=True, device=DEV, dt=DT, eps=EPS)[0]
                chk = float(np.abs(sc[0]["P"].cpu().numpy() - (d[0] - sf.CENTER)).max())
                log(f"truth reimplementation max|diff| vs gs.rollout_torch {chk:.2e}")
                out["truth_check_maxdiff"] = chk
            out[fam] = run_family(net, fam, sc, env, args, log)
            json.dump(out, open(args.out, "w"))
    log("done")


if __name__ == "__main__":
    main()
