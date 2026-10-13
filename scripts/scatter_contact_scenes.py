"""Obstacle/contact scenes for scripts/scatter_contact.py: generators for arbitrary static solid shapes, SDF-based truth, train/eval.

  python scripts/scatter_contact_scenes.py truth                                  (build + cache train/eval truth)
  python scripts/scatter_contact_scenes.py train --model B|A --init <E ckpt> --iters 8000 --tag B
  python scripts/scatter_contact_scenes.py eval  --model E|augur1|B|A --ckpt <pt> --tag <tag>
  python scripts/scatter_contact_scenes.py summary
Families: train box15 box_var post_circle post_rect; held out (shape never seen in training) disc post_poly.
Coordinates are CENTERED (sf.CENTER only added for sf.to_tensors). Seed 4738 for training, eval seeds 9100/9200/9300.
"""
import argparse
import json
import math
import os
import time
import types

import numpy as np
import torch

from scripts import scatter_contact as sc
from scripts import scatter_contact_rand as rc
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts import train_scatter_field as tsf

OUT = "results/contact"
CKPT = "checkpoints_contact"
TRAIN_FAMS = ("box15", "box_var", "post_circle", "post_rect")
HOLD_FAMS = ("disc", "post_poly")
EVAL_SEEDS = ss.EVAL_SEEDS
N_EVAL = 24
R0 = ss.BOX_R
V0 = ss.BOX_V
DT = ss.BOX_DT
GRID, EXTENT = 128, 64.0


def _region(prims):
    pr = prims[0]
    if pr["kind"] == "hbox":
        return np.array(pr["lo"]), np.array(pr["hi"])
    c, r = np.array(pr["c"]), pr["r"]
    return c - r, c + r


def place(rng, prims, rad_fn, n):
    lo, hi = _region(prims)
    P, R = [], []
    for _ in range(n):
        r = rad_fn()
        for _ in range(300):
            p = rng.uniform(lo, hi)
            if float(sc.scene_sdf(prims, torch.tensor(p, dtype=torch.float64))) < r + 0.1:
                continue
            if all(np.linalg.norm(p - q) > r + rq + 0.1 for q, rq in zip(P, R)):
                P.append(p)
                R.append(r)
                break
    return np.array(P), np.array(R)


def gen_scene(rng, fam):
    if fam in rc.SPECS:
        return rc.gen_scene(rng, fam)
    n = int(rng.integers(2, 21 if fam == "box15" else 16))
    if fam in ("box15", "post_circle", "post_poly"):
        prims = [sc.hbox((-7, -7), (7, 7))]
    elif fam == "box_var":
        h, c = rng.uniform(4.5, 8.0, 2), rng.uniform(-5, 5, 2)
        prims = [sc.hbox(c - h, c + h)]
    elif fam == "post_rect":
        h, c = rng.uniform(5.0, 8.0, 2), rng.uniform(-3, 3, 2)
        prims = [sc.hbox(c - h, c + h)]
        prims.append(sc.rect_poly(c + rng.uniform(-2.5, 2.5, 2), rng.uniform(2, 5), rng.uniform(1, 3), rng.uniform(0, math.pi)))
    else:
        prims = [sc.hdisc(rng.uniform(-4, 4, 2), rng.uniform(5.0, 8.0))]
    if fam == "post_circle":
        for _ in range(int(rng.integers(1, 3))):
            r = rng.uniform(1.0, 2.5)
            m = 7 - r - 1.8
            prims.append(sc.circle(rng.uniform(-m, m, 2), r))
    if fam == "post_poly":
        prims.append(sc.regular_poly(rng.uniform(-3.5, 3.5, 2), rng.uniform(1.5, 3.0), int(rng.choice([3, 5])), rng.uniform(0, 2 * math.pi)))
    rad_fn = (lambda: float(rng.uniform(0.5, 1.0))) if fam == "box_var" else (lambda: R0)
    pos, rad = place(rng, prims, rad_fn, n)
    return dict(pos=pos, vel=rng.uniform(-V0, V0, (len(pos), 2)), mass=np.ones(len(pos)), rad=rad, prims=prims, fam=fam)


def with_geom(s):
    if "geom" not in s:
        s["geom"] = sc.rasterize(s["prims"], GRID, EXTENT)
    return s


def truth(s, steps, dev, pos=None):
    return sc.contact_truth(s["prims"], s["pos"] if pos is None else pos, s["vel"], s["rad"], steps, dev, dt=DT)


def _rad_pad(scenes, n, fill=R0):
    r = np.full((len(scenes), n), fill, dtype=np.float32)
    for i, s in enumerate(scenes):
        r[i, : len(s["rad"])] = s["rad"]
    return torch.tensor(r)


def _geom_stack(scenes, dev="cpu"):
    out = torch.empty(len(scenes), 4, GRID, GRID, device=dev)
    for i, s in enumerate(scenes):
        out[i] = s["geom"] if "geom" in s else sc.rasterize(s["prims"], GRID, EXTENT, dev=dev)
    return out


def evalset(fam, seed, steps, dev):
    path = f"{OUT}/evalset_{fam}_{seed}.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    rng = np.random.default_rng(seed)
    scenes = [with_geom(gen_scene(rng, fam)) for _ in range(N_EVAL)]
    n = max(len(s["mass"]) for s in scenes)
    trajs = [truth(s, steps, dev) for s in scenes]
    rngp = np.random.default_rng(seed + 1)
    pert = [truth(s, steps, dev, ss.perturb(s, 1e-5, rngp)["pos"])[0] for s in scenes]
    es = dict(scenes=scenes, Pt=ss.pad_traj([t[0] for t in trajs], n), Vt=ss.pad_traj([t[1] for t in trajs], n), Pp=ss.pad_traj(pert, n))
    os.makedirs(OUT, exist_ok=True)
    torch.save(es, path)
    return es


def train_set(num, steps, dev, fams=TRAIN_FAMS):
    path = f"{OUT}/trainset_{num}_{steps}.pt" if fams == TRAIN_FAMS else f"{OUT}/trainset_{'_'.join(fams)}_{num}_{steps}.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    scenes, data = [], []
    for i, fam in enumerate(fams):
        rng = np.random.default_rng(ss.TRAIN_SEED + i)
        for _ in range(num // len(fams)):
            s = gen_scene(rng, fam) if fam in rc.SPECS else with_geom(gen_scene(rng, fam))
            P, V = truth(s, steps, dev)
            scenes.append(s)
            data.append((P + sf.CENTER, V, s["mass"]))
    os.makedirs(OUT, exist_ok=True)
    torch.save((scenes, data), path)
    return scenes, data


def free_space(model):
    """Option-C model reduced to its free-space N-body part: uniform gravity vector and ball-ball contact head zeroed (they are the contact-world physics, not
    the gravity force the harness rows measure); gscale is pinned to 1 (a drifted gscale rescales the base force); the U-Net is kept. Returns gscale."""
    with torch.no_grad():
        model.gvec.zero_()
        model.cmlp[-1].weight.zero_()
        model.cmlp[-1].bias.zero_()
        model.gscale.fill_(1.0)
    model.set_scene(None, None)
    return float(model.gscale)


def build(name, dev, ckpt=None):
    if name == "E":
        return ss.build_model("E", DT, dev, ckpt)
    if name == "augur1":
        return ss.build_model("bounce1", DT, dev, ckpt)
    cfg = tsf.EXPS["E"]
    kw = dict(tsf.BASE)
    kw.update(tsf.VARIANTS["ms_kp_pot_v_g128"])
    model = sc.ContactScatterField(extent=cfg["extent"], dt=DT, in_scale=cfg["in_scale"], obstacle={"B": "sdf", "Bf": "sdf", "Eg": "sdf", "A": "kernel", "C": "chan"}[name], frozen_geo=name in ("Bf", "Eg"), **kw)
    if ckpt and ckpt != "none":
        sd = torch.load(ckpt, map_location=dev, weights_only=False)["model"]
        r = model.load_state_dict(sd, strict=False)
        print(f"loaded {ckpt}: missing {list(r.missing_keys)} unexpected {list(r.unexpected_keys)}", flush=True)
    if name == "Eg":
        with torch.no_grad():
            model.gvec.copy_(torch.tensor([ss.BOX_G / sc.OUT_SCALE, 0.0]))
    return model.to(dev).eval()


def roll(model, scenes, steps, dev):
    if hasattr(model, "set_scene"):
        n = max(len(s["mass"]) for s in scenes)
        model.set_scene(_geom_stack(scenes, dev), _rad_pad(scenes, n).to(dev))
    return ss.model_rollout(model, scenes, steps, dev)


def metrics(scenes, P, V, mask):
    """tunnelled (center beyond surface), deep (sdf < R/2) fraction per tick, mean KE per ball."""
    T = P.shape[1]
    tun, deep = np.zeros(T), np.zeros(T)
    for i, s in enumerate(scenes):
        k = len(s["mass"])
        sd = sc.scene_sdf(s["prims"], torch.tensor(P[i, :, :k])).numpy()
        tun += (sd < 0).mean(1)
        deep += (sd < 0.5 * s["rad"][None]).mean(1)
    ke = (0.5 * (V ** 2).sum(-1) * mask[:, None]).sum(2) / mask.sum(1)[:, None]
    return tun / len(scenes), deep / len(scenes), ke.mean(0)


def overlap_frac(scenes, P, mask):
    n = P.shape[2]
    R = _rad_pad(scenes, n).numpy()
    d = np.linalg.norm(P[:, :, :, None] - P[:, :, None, :], axis=-1)
    mm = mask[:, None, :, None] * mask[:, None, None, :] * (1 - np.eye(n))[None, None]
    thr = 0.7 * (R[:, None, :, None] + R[:, None, None, :])
    return float(((d < thr) * mm).sum() / max(mm.sum(), 1))


def eval_family(model, fam, dev, steps=100):
    seeds, keep = {}, {}
    for seed in EVAL_SEEDS:
        es = evalset(fam, seed, steps, dev)
        scenes, Pt, Vt = es["scenes"], es["Pt"], es["Vt"]
        Pm, Vm, mask = roll(model, scenes, steps, dev)
        t = np.arange(steps + 1)[None, :, None, None] * DT
        ball = Pt[:, :1] + Vt[:, :1] * t
        ball[..., 0] += 0.5 * ss.BOX_G * t[..., 0] ** 2
        r = {"err": ss.at(ss.err_curve(Pm, Pt, mask)), "ballistic_gravity": ss.at(ss.err_curve(ball, Pt, mask)), "floor_pert1e-5": ss.at(ss.err_curve(es["Pp"], Pt, mask))}
        tm, dm, km = metrics(scenes, Pm, Vm, mask)
        tt, dt_, kt = metrics(scenes, Pt, Vt, mask)
        r["tunnel_model"] = {str(k): float(tm[k]) for k in (20, 50, 100)}
        r["tunnel_truth"] = {str(k): float(tt[k]) for k in (20, 50, 100)}
        r["deep_model"] = {str(k): float(dm[k]) for k in (20, 50, 100)}
        r["deep_truth"] = {str(k): float(dt_[k]) for k in (20, 50, 100)}
        r["ke_ratio"] = {str(k): float(km[k] / max(kt[k], 1e-9)) for k in (20, 50, 100)}
        r["pair_overlap_model"], r["pair_overlap_truth"] = overlap_frac(scenes, Pm, mask), overlap_frac(scenes, Pt, mask)
        cnt = mask.sum(1)
        axis = (np.abs(Pm - Pt) * mask[:, None, :, None]).sum(2) / cnt[:, None, None]
        e_sc = (np.linalg.norm(Pm - Pt, axis=-1) * mask[:, None]).sum(2) / cnt[:, None]
        r["err_median"] = {str(k): float(np.median(e_sc[:, k])) for k in (5, 10, 20, 100)}
        r["offgrid100"] = float(((np.abs(Pm[:, 100]) > model.extent / 2).any(-1) * mask).sum() / mask.sum())
        acc = np.linalg.norm(np.diff(Vm, axis=1), axis=-1) / DT * mask[:, None]
        r["runaway"] = float((acc[:, 20:] > 200).any(axis=(1, 2)).mean())
        r["err_x"], r["err_y"] = ({str(k): float(axis[:, k, a].mean()) for k in (5, 20, 100)} for a in (0, 1))
        en = lambda P, V: ((0.5 * (V ** 2).sum(-1) - ss.BOX_G * P[..., 0]) * mask[:, None]).sum(2) / cnt[:, None]
        em, et = en(Pm, Vm), en(Pt, Vt)
        r["dE_model"], r["dE_truth"] = ({str(k): float((e[:, k] - e[:, 0]).mean()) for k in (20, 100)} for e in (em, et))
        r["mom_err"] = {str(k): float((np.linalg.norm(((Vm - Vt) * mask[:, None, :, None]).sum(2), axis=-1) / cnt[:, None])[:, k].mean()) for k in (20, 100)}
        seeds[str(seed)] = r
        if seed == EVAL_SEEDS[0]:
            keep = dict(truth=Pt[:4], model=Pm[:4], mask=mask[:4], rad=_rad_pad(scenes[:4], Pt.shape[2]).numpy(), prims=json.dumps([s["prims"] for s in scenes[:4]]))
    mean = {}
    for k in ("err", "ballistic_gravity", "floor_pert1e-5", "tunnel_model", "tunnel_truth", "deep_model", "deep_truth", "ke_ratio", "err_median", "err_x", "err_y", "dE_model", "dE_truth", "mom_err"):
        mean[k] = {kk: float(np.mean([seeds[s][k][kk] for s in seeds])) for kk in seeds[str(EVAL_SEEDS[0])][k]}
    for k in ("pair_overlap_model", "pair_overlap_truth", "offgrid100", "runaway"):
        mean[k] = float(np.mean([seeds[s][k] for s in seeds]))
    print(fam, "err", {k: round(v, 3) for k, v in mean["err"].items()}, "ballistic@20", round(mean["ballistic_gravity"]["20"], 2), "tunnel", {k: round(v, 3) for k, v in mean["tunnel_model"].items()},
          "ke", {k: round(v, 2) for k, v in mean["ke_ratio"].items()}, flush=True)
    return dict(seeds=seeds, mean=mean), keep


def restitution(model, dev, prims, face, speeds, steps=80):
    """Ball released at rest on the -x side of a face at x=face (gravity +x), impact speed v=sqrt(2 g h); e_eff = sqrt(rebound height / drop height)."""
    out = {}
    wall = face - R0
    for v in speeds:
        x0 = wall - v * v / (2 * ss.BOX_G)
        if x0 < -6.25:
            continue
        s = dict(pos=np.array([[x0, 0.0]]), vel=np.zeros((1, 2)), mass=np.ones(1), rad=np.array([R0]), prims=prims)
        Pt, _ = truth(s, steps, dev)
        Pm, _, _ = roll(model, [s], steps, dev)
        res = {}
        for tag, P in (("truth", Pt[:, 0, 0]), ("model", Pm[0, :, 0, 0])):
            vx = np.diff(P)
            rev = np.nonzero((vx[:-1] > 0) & (vx[1:] <= 0))[0]
            res[tag + "_rev"] = float(math.sqrt(max(wall - P[rev[0] + 1:].min(), 0.0) / max(wall - x0, 1e-9))) if len(rev) and wall - P[rev[0] + 1:].min() > 0 else None
            enter = np.nonzero(P > wall - 0.05)[0]
            back = np.nonzero(P[enter[0]:] < wall - 0.05)[0] if len(enter) else []
            if len(enter) == 0 or len(back) == 0:
                res[tag] = None
                continue
            apex = P[enter[0] + back[0]:].min()
            res[tag] = float(math.sqrt(max(wall - apex, 0.0) / max(wall - x0, 1e-9)))
        out[str(v)] = res
    return out


def fams_of(args):
    return tuple(x for x in args.fams.split(",") if x) or TRAIN_FAMS + HOLD_FAMS


def run_eval(model, tag, args, dev):
    res = dict(model=args.model, ckpt=args.ckpt, tag=tag, families={})
    os.makedirs(OUT, exist_ok=True)
    for fam in fams_of(args):
        r, keep = eval_family(model, fam, dev)
        res["families"][fam] = r
        np.savez_compressed(f"{OUT}/{fam}_{tag}_traj.npz", **keep)
    box = [sc.hbox((-7, -7), (7, 7))]
    res["restitution_wall"] = restitution(model, dev, box, 7.0, (2, 4, 6, 8, 10, 12, 15))
    res["restitution_circle"] = restitution(model, dev, box + [sc.circle((2.0, 0.0), 2.0)], 0.0, (2, 4, 6, 8, 10))
    print("restitution wall", res["restitution_wall"], flush=True)
    print("restitution circle", res["restitution_circle"], flush=True)
    json.dump(res, open(f"{OUT}/eval_{tag}.json", "w"))
    print(f"wrote {OUT}/eval_{tag}.json", flush=True)


def run_train(args, dev):
    model = build(args.model, dev, args.init)
    model.train()
    for o in [x for x in args.opt.split(",") if x]:
        sf.apply_opt(model, o)
    t0 = time.time()
    scenes, data = train_set(args.train, args.steps, dev, ("c_train",) if args.model == "C" else TRAIN_FAMS)
    print(f"train set {len(scenes)} scenes ready in {time.time() - t0:.0f}s", flush=True)
    tensors = sf.to_tensors(data, dev)
    n = tensors[0].shape[2]
    geom = _geom_stack(scenes, dev)
    rad = _rad_pad(scenes, n).to(dev)
    print(f"train data {len(data)} scenes, N max {n}, params {sum(p.numel() for p in model.parameters())}", flush=True)
    os.makedirs(CKPT, exist_ok=True)
    ckpt = f"{CKPT}/{args.tag}.pt"
    targs = types.SimpleNamespace(lr=args.lr, device=str(dev), seed=ss.TRAIN_SEED, opt=args.opt, time_budget=0, iters=args.iters, k_start=args.k_start, k_end=args.k_end,
                                  steps=args.steps, batch=args.batch, log_every=args.log_every, loss_cap=args.loss_cap, train_only=args.train_only, ema=args.ema, lr_warmup=args.lr_warmup, snap_iters=args.snap_iters, w_vel=args.w_vel, long_windows=0, batch_hook=lambda idx: model.set_scene(geom[idx], rad[idx]))
    n_it, secs = tsf.train(model, tensors, targs, ckpt, lambda s: print(s, flush=True))
    print(f"TRAIN DONE iters {n_it} secs {secs:.0f}", flush=True)
    model.set_scene(None, None)
    model.eval()
    args.ckpt = ckpt
    return model


def summary():
    f = lambda x: "-" if x is None else f"{x:.3g}"
    for tag in sorted(x[5:-5] for x in os.listdir(OUT) if x.startswith("eval_") and x.endswith(".json")):
        d = json.load(open(f"{OUT}/eval_{tag}.json"))
        print(f"== {tag} ({d['model']})")
        for fam, r in d["families"].items():
            m = r["mean"]
            print(f"{fam:12s} err@5/10/20/100", [f(m["err"][k]) for k in ("5", "10", "20", "100")], "ballistic@20", f(m["ballistic_gravity"]["20"]), "floor@20", f(m["floor_pert1e-5"]["20"]),
                  "tunnel@100 m/t", f(m["tunnel_model"]["100"]), f(m["tunnel_truth"]["100"]), "ke@100", f(m["ke_ratio"]["100"]))
        for k in ("restitution_wall", "restitution_circle"):
            print(k, {v: f((x or {}).get("model")) for v, x in d[k].items()}, "truth", {v: f((x or {}).get("truth")) for v, x in d[k].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["truth", "train", "eval", "train_eval", "summary"])
    ap.add_argument("--model", default="B")
    ap.add_argument("--ckpt", default="none")
    ap.add_argument("--init", default="checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt")
    ap.add_argument("--tag", default="B")
    ap.add_argument("--opt", default="")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--loss-cap", type=float, default=100.0)
    ap.add_argument("--train-only", default="")
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--lr-warmup", type=int, default=0)
    ap.add_argument("--snap-iters", default="")
    ap.add_argument("--w-vel", type=float, default=0.1)
    ap.add_argument("--fams", default="", help="comma list of eval families (default the six scatter-contact.md families)")
    ap.add_argument("--eval-opt", default="kcache,fastio")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(ss.TRAIN_SEED)
    if args.mode == "summary":
        return summary()
    if args.mode == "truth":
        t0 = time.time()
        for fam in fams_of(args):
            for seed in EVAL_SEEDS:
                evalset(fam, seed, 100, dev)
            print(fam, f"eval truth cached t={time.time() - t0:.0f}s", flush=True)
        train_set(args.train, args.steps, dev, ("c_train",) if args.model == "C" else TRAIN_FAMS)
        print(f"train truth cached t={time.time() - t0:.0f}s", flush=True)
        return
    if args.mode == "eval":
        model = build(args.model, dev, args.ckpt)
    else:
        model = run_train(args, dev)
    for o in [x for x in args.eval_opt.split(",") if x]:
        sf.apply_opt(model, o)
    if args.mode == "train_eval" or args.mode == "eval":
        run_eval(model, args.tag, args, dev)


if __name__ == "__main__":
    main()
