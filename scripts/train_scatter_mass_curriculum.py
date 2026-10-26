"""Mass-ratio curriculum fine-tune of the scatter-field tier-0 model (star:body ratio 1 -> 1000), one arm per run.
Arms: massfeat_curriculum (mass channels + curriculum), curriculum_no_massfeat (curriculum only), massfeat_flat (mass channels, all ratios mixed from the start).
Resumable (--resume or an existing checkpoint is picked up), full state (model, Adam, iteration) every --ckpt-every iterations, SIGTERM saves and exits 143.
Design, pass/kill lines: docs/theory/mass-conditioned-scatter-field.md."""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_ft_data as fd
from scripts import scatter_mass_curriculum_data as cd
from scripts import train_scatter_field as tsf

ARMS = {
    "massfeat_curriculum": dict(variant="ms_kp_pot_v_g128_massfeat", curriculum=True),
    "curriculum_no_massfeat": dict(variant="ms_kp_pot_v_g128", curriculum=True),
    "massfeat_flat": dict(variant="ms_kp_pot_v_g128_massfeat", curriculum=False),
}
EVAL_Q = (10, 100, 1000)
EVAL_TICKS = (20, 50, 100)
LINES = dict(
    planet=dict(pass_da=0.02, pass_de=0.03, pass_dE=0.02, kill_da=0.15, kill_dE=0.2),
    disk=dict(pass_da=0.05, pass_de=0.05, pass_dE=0.05, kill_da=0.25, kill_dE=0.3),
    base_pass=1.15, base_kill=1.5, q1000_kill_da_planet=0.3,
)
stop = {"flag": False}


def build_model(arm, device, dt, init_ckpt=None):
    cfg = dict(extent=64.0, in_scale=1.0)
    model = tsf.build(cfg, ARMS[arm]["variant"], dt).to(device)
    if init_ckpt:
        st = torch.load(init_ckpt, map_location=device, weights_only=False)
        sf.load_compat(model, st["model"] if "model" in st else st)
    return sf.apply_opts(model, "none", "train")


def base_errors(model, device, num=16, steps=50, seed=9000):
    data = fd.base(num, steps, seed, device)
    P, V, M, mask = [t.float() for t in sf.to_tensors(data, device)]
    with torch.no_grad():
        Pm, _ = sf.rollout(model, P[:, 0], V[:, 0], M, mask, steps)
    out = {}
    for k in (5, 10, 20, 50):
        e = (((Pm[:, k] - P[:, k]) ** 2).sum(-1).sqrt() * mask).sum(1) / mask.sum(1)
        out[k] = float(e.mean())
    return out


def evaluate(model, device, num=16, steps=100, seed=9000):
    res = dict(planet={}, disk={}, base=base_errors(model, device))
    for q in EVAL_Q:
        for kind in ("planet", "disk"):
            data = cd.scenes(num, steps, seed + q, device, q, kind)
            res[kind][q] = cd.orbit_metrics(model, data, device, EVAL_TICKS)
    return res


def verdict(res, init):
    """Apply the pre-registered lines. Returns (PASS|KILL|INCONCLUSIVE, list of reasons)."""
    kill, miss = [], []
    for kind in ("planet", "disk"):
        L = LINES[kind]
        for q in EVAL_Q:
            r = res[kind][q][EVAL_TICKS[-1]]
            tag = f"{kind} q{q}"
            kda = L["kill_da"] if not (kind == "planet" and q == 1000) else LINES["q1000_kill_da_planet"]
            if not r["finite"] or r["da_over_a"] > kda or r["dE_model"] > L["kill_dE"]:
                kill.append(f"{tag}: da/a {r['da_over_a']:.3g} dE {r['dE_model']:.3g} finite {r['finite']}")
            elif not (r["da_over_a"] <= L["pass_da"] and r["de"] <= L["pass_de"] and r["dE_model"] <= L["pass_dE"]):
                miss.append(f"{tag}: da/a {r['da_over_a']:.3g} de {r['de']:.3g} dE {r['dE_model']:.3g}")
    for k in (5, 10, 20):
        ratio = res["base"][k] / max(init["base"][k], 1e-12)
        if ratio > LINES["base_kill"]:
            kill.append(f"base err@{k} ratio {ratio:.2f}")
        elif ratio > LINES["base_pass"]:
            miss.append(f"base err@{k} ratio {ratio:.2f}")
    return ("KILL" if kill else "INCONCLUSIVE" if miss else "PASS"), kill + miss


def groups_for(args, stage, device, log):
    if ARMS[args.arm]["curriculum"]:
        raw = cd.stage_groups(stage, args.scenes, args.steps, args.seed, device)
    else:
        raw = [("q1-1000", cd.scenes(args.scenes, args.steps, args.seed, device, (1.0, 1000.0)), 0.7),
               ("unit_base", fd.base(args.scenes, args.steps, args.seed + 200, device), 0.3)]
    out = [dict(name=n, t=[t.float() for t in sf.to_tensors(d, device)], weight=w) for n, d, w in raw]
    for x in out:
        log(f"stage {stage} group {x['name']}: scenes {x['t'][0].shape[0]} N {x['t'][0].shape[2]} weight {x['weight']}")
    return out


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(args, log):
    dev = torch.device(args.device)
    model = build_model(args.arm, dev, args.dt, args.init_ckpt)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt = f"{args.ckpt_dir}/{args.name}.pt"
    start = 0
    results = {}
    rpath = f"{args.out_dir}/{args.name}.json"
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["it"]
        results = st.get("results", {})
        log(f"resumed from it {start}")
    if "init" not in results:
        results["init"] = evaluate(model, dev)
        log(f"init eval {json.dumps(results['init'], default=float)}")
    nst = len(cd.STAGES) if ARMS[args.arm]["curriculum"] else 1
    total = args.stage_iters * nst
    skipped = 0
    cur_stage, groups = -1, None
    last = {}
    t_start = time.time()

    def save(it):
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), it=it, results=results, args=vars(args)), ckpt)

    it = start
    while it < total:
        if stop["flag"]:
            save(it)
            log(f"SIGTERM: saved full state at it {it}")
            return it, 143
        stage = min(it // args.stage_iters, nst - 1)
        if stage != cur_stage:
            groups = groups_for(args, stage, dev, log)
            cur_stage = stage
        g = torch.Generator(device="cpu").manual_seed(args.seed * 1000003 + it)
        gi = int(torch.multinomial(torch.tensor([x["weight"] for x in groups]), 1, generator=g))
        P, V, M, mask = groups[gi]["t"]
        frac = (it % args.stage_iters) / max(1, args.stage_iters - 1) if nst > 1 else it / max(1, total - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * (it / max(1, total - 1))))
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac))) * min(1.0, (it - start + 1) / args.warmup)
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
        if torch.isfinite(loss) and torch.isfinite(gn) and (args.loss_cap <= 0 or loss.item() < args.loss_cap):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            log(f"it {it} {groups[gi]['name']} loss {loss.item():.3g} non-finite or above cap, step skipped ({skipped} total)")
        last.setdefault(groups[gi]["name"], []).append(loss.item())
        it += 1
        if it % args.log_every == 0:
            msg = " ".join(f"{n}:{np.mean(v[-20:]):.5f}" for n, v in last.items())
            log(f"it {it} stage {stage} k {k} [{msg}] t {time.time() - t_start:.0f}s")
        if it % args.ckpt_every == 0 and all(torch.isfinite(p).all() for p in model.parameters()):
            save(it)
        if it % args.stage_iters == 0 or it == total:
            results[f"after_stage_{stage}"] = evaluate(model, dev)
            log(f"stage {stage} eval {json.dumps(results[f'after_stage_{stage}'], default=float)}")
            save(it)
    results["final"] = evaluate(model, dev)
    results["verdict"], results["verdict_reasons"] = verdict(results["final"], results["init"])
    log(f"FINAL {results['verdict']} {results['verdict_reasons']}")
    save(it)
    json.dump(results, open(rpath, "w"), indent=1, default=float)
    return it, 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--init-ckpt", required=True, help="Exp E / tier-0 checkpoint to warm start from (load_compat zero-pads the new input channels)")
    ap.add_argument("--resume", action="store_true", help="accepted for clarity; an existing checkpoint is always resumed")
    ap.add_argument("--stage-iters", type=int, default=600)
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--loss-cap", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/scatter_mass_curriculum")
    ap.add_argument("--ckpt-dir", default="checkpoints/scatter_mass_curriculum")
    args = ap.parse_args(argv)
    args.name = args.name or args.arm
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logf = open(f"{args.out_dir}/{args.name}.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    it, code = train(args, log)
    return code


if __name__ == "__main__":
    sys.exit(main())
