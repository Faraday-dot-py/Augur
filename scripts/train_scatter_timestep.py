"""Step-size-as-input scatter field (experiment timestep-as-input-model): truth generation, frozen-control reference eval, training, evaluation.
Modes: gen (truth at DT_TRAIN + DT_EVAL_ONLY on the Exp E scene distribution, saved as float32 tensors), frozen (Exp E weights, inference only, model.dt set per step size),
train (variant ms_kp_pot_v_g128_dtin, mixed step sizes, Exp E recipe, resumable), eval (a checkpoint on the held-out sets).
Lines: docs/debugging/architecture-push-prereg/timestep-as-input-model.md."""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

DT_TRAIN = (0.05, 0.1, 0.2)
DT_EVAL_ONLY = (0.075,)
EVAL_SEEDS = (9100, 9200, 9300)
VARIANT = "ms_kp_pot_v_g128_dtin"
CFG = tsf.EXPS["E"]
stop = {"flag": False}


def tag(dt):
    return f"dt{dt:g}"


def gen_set(num, steps, seed, dt, dev):
    return tsf.make_data(CFG, num, steps, seed, dev, dt, 0.5)


def save_set(data, path):
    P, V, M, mask = [t.float().cpu() for t in sf.to_tensors(data, "cpu")]
    torch.save(dict(P=P, V=V, M=M, mask=mask), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_set(path, dev):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return [d[k].to(dev) for k in ("P", "V", "M", "mask")]


def gen(args, log):
    dev = torch.device(args.device)
    os.makedirs(args.data_dir, exist_ok=True)
    for dt in DT_TRAIN:
        path = f"{args.data_dir}/train_{tag(dt)}_seed{args.seed}.pt"
        if os.path.exists(path):
            log(f"skip {path}")
            continue
        t0 = time.time()
        save_set(gen_set(args.train_scenes, args.steps, args.seed, dt, dev), path)
        log(f"{path} {time.time() - t0:.0f}s")
    for dt in DT_TRAIN + DT_EVAL_ONLY:
        for seed in args.eval_seeds:
            path = f"{args.data_dir}/eval_{tag(dt)}_seed{seed}.pt"
            if os.path.exists(path):
                log(f"skip {path}")
                continue
            t0 = time.time()
            save_set(gen_set(args.eval_scenes, args.eval_steps, seed, dt, dev), path)
            log(f"{path} {time.time() - t0:.0f}s")
    for dt in DT_TRAIN + DT_EVAL_ONLY:
        for seed in args.eval_seeds:
            assert os.path.getsize(f"{args.data_dir}/eval_{tag(dt)}_seed{seed}.pt") > 0
    log("GEN DONE")


def dE_vs_truth(Pm, Vm, Pt, Vt, M, mask, k, eps=0.5):
    Mk = M
    em = sf.energy(Pm[:, k].double(), Vm[:, k].double(), Mk, mask, eps)
    et = sf.energy(Pt[:, k], Vt[:, k], Mk, mask, eps)
    return float(((em - et).abs() / et.abs().clamp_min(1e-9)).mean())


@torch.no_grad()
def evaluate(model, dt, sets, dev, eps=0.5):
    """sets: {seed: (P, V, M, mask)}. Returns per seed err@5/10/20, dE@100 against truth E(100) (and the E(0)-referenced drift for comparison)."""
    old = model.dt
    model.dt = dt
    out = {}
    for seed, (P, V, M, mask) in sets.items():
        Pm, Vm = sf.rollout(model, P[:, 0].float(), V[:, 0].float(), M.float(), mask.float(), P.shape[1] - 1)
        Pm, Vm = Pm.double(), Vm.double()
        m = sf.traj_metrics(Pm, Vm, P.double(), V.double(), M.double(), mask.double(), eps)
        k = min(100, P.shape[1] - 1)
        out[str(seed)] = dict(err5=m["err"][4], err10=m["err"][9], err20=m["err"][19], err_final=m["err"][k - 1],
                              dE_vs_truth_100=dE_vs_truth(Pm, Vm, P.double(), V.double(), M.double(), mask.double(), k),
                              dE_from_E0_model_100=m["energy_drift_rel_model"][k], dE_from_E0_truth_100=m["energy_drift_rel_true"][k],
                              finite=bool(torch.isfinite(Pm).all()))
    model.dt = old
    return out


def load_eval(args, dt, dev):
    return {s: load_set(f"{args.data_dir}/eval_{tag(dt)}_seed{s}.pt", dev) for s in args.eval_seeds}


def build_model(variant, dev, init_ckpt=None):
    model = tsf.build(CFG, variant, 0.1).to(dev)
    if init_ckpt:
        st = torch.load(init_ckpt, map_location=dev, weights_only=False)
        sf.load_compat(model, st["model"] if "model" in st else st)
    return sf.apply_opts(model, "none", "train")


def frozen(args, log):
    dev = torch.device(args.device)
    model = build_model("ms_kp_pot_v_g128", dev, args.init_ckpt).eval()
    res = {"ckpt": args.init_ckpt, "note": "Exp E weights, no step-size input; inference at each step size only"}
    for dt in DT_TRAIN + DT_EVAL_ONLY:
        res[tag(dt)] = evaluate(model, dt, load_eval(args, dt, dev), dev)
        log(f"frozen {tag(dt)} " + json.dumps({s: (round(v["err20"], 5), round(v["dE_vs_truth_100"], 5)) for s, v in res[tag(dt)].items()}))
    json.dump(res, open(f"{args.out_dir}/frozen_reference.json", "w"), indent=1)


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(args, log):
    dev = torch.device(args.device)
    model = build_model(VARIANT, dev, args.init_ckpt)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt = f"{args.ckpt_dir}/{args.name}.pt"
    start, val_hist = 0, []
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, val_hist = st["it"], st.get("val_hist", [])
        log(f"resumed from it {start}")
    groups = [dict(dt=dt, t=[t.float() for t in load_set(f"{args.data_dir}/train_{tag(dt)}_seed{args.seed}.pt", dev)]) for dt in DT_TRAIN]
    val = [dict(dt=dt, t=load_set(f"{args.data_dir}/eval_{tag(dt)}_seed{args.val_seed}.pt", dev)) for dt in DT_TRAIN]
    for x in val:
        x["t"] = [t.float() for t in x["t"]]

    def save(it):
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), it=it, val_hist=val_hist, args=vars(args)), ckpt)

    def val_loss(k=20):
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for x in val:
                model.dt = x["dt"]
                P, V, M, mask = x["t"]
                sl = slice(0, args.val_scenes)
                pos, vel, m, mk = P[sl, 0], V[sl, 0], M[sl], mask[sl]
                field = model.init_field(pos.shape[0], dev)
                lp = 0.0
                for s in range(k):
                    pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
                    lp = lp + (((pos - P[sl, s + 1]) ** 2) * mk[:, :, None]).sum() / (mk.sum() * 2 * k)
                tot += float(lp) / len(val)
        model.train()
        return tot

    t_start = time.time()
    skipped = 0
    it = start
    last = {}
    while it < args.iters:
        if stop["flag"]:
            save(it)
            log(f"SIGTERM: saved full state at it {it}")
            return it, 143
        g = torch.Generator(device="cpu").manual_seed(args.seed * 1000003 + it)
        gi = int(torch.randint(len(groups), (1,), generator=g))
        grp = groups[gi]
        model.dt = grp["dt"]
        P, V, M, mask = grp["t"]
        frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        B = args.batch
        idx = torch.randint(P.shape[0], (B,), generator=g).to(dev)
        t0 = torch.randint(0, args.steps - k + 1, (B,), generator=g).to(dev)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
        Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
        m, mk = M[idx], mask[idx]
        pos, vel = Pw[:, 0], Vw[:, 0]
        field = model.init_field(B, dev)
        lp = lv = 0.0
        w = mk[:, :, None]
        denom = mk.sum() * 2 * k
        for s in range(k):
            pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
            lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / denom
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
        loss = lp + 0.1 * lv
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            log(f"it {it} dt {grp['dt']} loss {loss.item():.3g} non-finite, step skipped ({skipped} total)")
        last.setdefault(grp["dt"], []).append(loss.item())
        it += 1
        if it % args.val_every == 0:
            val_hist.append((it, val_loss()))
        if it % args.log_every == 0:
            msg = " ".join(f"dt{d:g}:{np.mean(v[-20:]):.5f}" for d, v in sorted(last.items()))
            log(f"it {it} k {k} [{msg}] val {val_hist[-1][1] if val_hist else float('nan'):.6f} t {time.time() - t_start:.0f}s")
        if it % args.ckpt_every == 0 and all(torch.isfinite(p).all() for p in model.parameters()):
            save(it)
    save(it)
    return it, 0


def plateau_falling(val_hist, total, drop=0.03):
    """Plateau guard: mean validation loss over the last 10% of iterations more than `drop` below the mean over the preceding 10% window."""
    w = max(1, round(0.1 * total))
    last = [v for i, v in val_hist if i > total - w]
    prev = [v for i, v in val_hist if total - 2 * w < i <= total - w]
    if not last or not prev:
        return False
    return float(np.mean(last)) < (1.0 - drop) * float(np.mean(prev))


def evaluate_ckpt(args, log):
    dev = torch.device(args.device)
    model = build_model(VARIANT, dev)
    st = torch.load(f"{args.ckpt_dir}/{args.name}.pt", map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    res = {"it": st["it"], "plateau_falling": plateau_falling(st.get("val_hist", []), args.iters), "val_hist_tail": st.get("val_hist", [])[-10:]}
    for dt in DT_TRAIN + DT_EVAL_ONLY:
        res[tag(dt)] = evaluate(model, dt, load_eval(args, dt, dev), dev)
        log(f"eval {tag(dt)} " + json.dumps({s: (round(v["err20"], 5), round(v["dE_vs_truth_100"], 5)) for s, v in res[tag(dt)].items()}))
    json.dump(res, open(f"{args.out_dir}/{args.name}_eval.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "frozen", "train", "eval"], required=True)
    ap.add_argument("--name", default="timestep-as-input-model")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--data-dir", default="data/timestep_truth")
    ap.add_argument("--train-scenes", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eval-scenes", type=int, default=48)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    ap.add_argument("--val-seed", type=int, default=EVAL_SEEDS[0])
    ap.add_argument("--val-scenes", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/timestep_as_input_model")
    ap.add_argument("--ckpt-dir", default="ckpt_timestep_as_input_model")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    logf = open(f"{args.out_dir}/{args.name}_{args.mode}.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    if args.mode == "gen":
        gen(args, log)
    elif args.mode == "frozen":
        frozen(args, log)
    elif args.mode == "eval":
        evaluate_ckpt(args, log)
    else:
        assert args.init_ckpt, "--init-ckpt required for train"
        it, code = train(args, log)
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
