"""Fine-tune the matched-60k tier 0 inside the tiled hierarchy: closed-loop rollouts over the 4x4 tiling of 64-unit windows with the exact external
field and halo exchange of the recurrent field (scripts/tier0_tiled_closed_loop.py), horizon growing to --k-end, per-tick loss normalised to the
reference model's own error at that tick and weighted to the tick-20 scale. Plain single-window loss terms (harvested windows with external field,
isolated Exp E scenes) are mixed in so the plain tier-0 accuracy does not regress. Every loss component is divided by the initial model's value.

  PYTHONPATH=. python scripts/train_tier0_tiled_closed_loop.py --init results/tier0_ext/tier0-matched-60k-iterations.pt --data $HOME/t0n/data4 \
      --pattern "hv3_s*.npz" --tier0 $HOME/E_ms_kp_pot_v_g128.pt --tag tier0-tiled-closed-loop-halo --resume

Early-stop probes (--probe-at 1000 2500): rolls 20 ticks on held-out validation scenes (seed 4739) and compares with the initial model; on failure
writes <tag>.EARLY_KILL and exits 0. --resume restarts from <out-dir>/<tag>.pt (full state: model, optimizer, iteration); SIGTERM saves and exits."""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from scripts import exp12_hier as h
from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m
from scripts import train_scatter_field as tsf
from scripts import train_tier0_ext as tte
from scripts import train_tier0_harvest as th
from scripts import train_tiered as ttr
from scripts import tier0_tiled_closed_loop as tc

STOP = []


def clear_ext(model):
    model.t0.ext_grid = model.t0.ext_poly = model.t0.ext_high = None


def save(path, model, opt, it, args):
    tte.save(path, model, opt, it, args)


@torch.no_grad()
def plain_err20(model, dev, seed=9000, n=16):
    clear_ext(model)
    model.t0.set_material(None)
    data = tsf.make_data(tsf.EXPS["E"], n, 21, seed, dev, 0.1, 0.5)
    m, _ = tsf.evaluate_model(model, data, dev, 0.1, 0.5)
    return float(m["err"][19])


@torch.no_grad()
def tiled_metrics(tt, dval):
    r = h.eval_generic(lambda p, v, m, k, st: tt.force(p, v, m, k, st), tt.init_state, dval, ticks=(10, 20))
    return {k: r[k] for k in ("force_rel", "force_rel_free_rollout", "err@10", "err@20", "dE_model")}


def probe_metrics(model, tt, dval, dev):
    model.eval()
    out = tiled_metrics(tt, dval)
    out["plain_err20"] = plain_err20(model, dev)
    model.train()
    return out


def gate(m, ref, bl, after, a):
    """Pre-registered early-stop gates; returns the list of violated conditions."""
    bad = []
    if not all(np.isfinite(v) for v in m.values()):
        bad.append("non-finite probe metric")
    if m["plain_err20"] > a.gate_plain * ref["plain_err20"]:
        bad.append(f"plain err@20 {m['plain_err20'] / ref['plain_err20']:.3f}x init > {a.gate_plain}")
    if after <= a.probe_at[0]:
        if m["err@20"] > a.gate1_err * ref["err@20"]:
            bad.append(f"tiled err@20 {m['err@20'] / ref['err@20']:.3f}x init > {a.gate1_err}")
        if m["force_rel_free_rollout"] > a.gate1_force * ref["force_rel_free_rollout"]:
            bad.append(f"own-state force error {m['force_rel_free_rollout'] / ref['force_rel_free_rollout']:.3f}x init > {a.gate1_force}")
    else:
        if m["err@20"] > a.gate2_err * ref["err@20"]:
            bad.append(f"tiled err@20 {m['err@20'] / ref['err@20']:.3f}x init > {a.gate2_err}")
        if bl is not None and m["err@20"] > a.gate2_bl * bl["err@20"]:
            bad.append(f"tiled err@20 {m['err@20'] / bl['err@20']:.3f}x baseline > {a.gate2_bl}")
    return bad


def sample(g, S, T, k, batch, p0, dev):
    idx = torch.randint(S, (batch,), generator=g).to(dev)
    t0s = torch.randint(0, T - k, (batch,), generator=g).to(dev)
    return idx, torch.where(torch.rand(batch, generator=g).to(dev) < p0, torch.zeros_like(t0s), t0s)


def schedule(a, it, iters):
    frac = it / max(1, iters - 1)
    return int(round(a.k_start + (a.k_end - a.k_start) * frac)), a.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))


@torch.no_grad()
def reference(model, tt, d, dh, iso, a, dev, g_seed):
    """Initial model's per-tick closed-loop errors (loss normalisers), plain loss values at k-plain, and the early-stop reference metrics."""
    g = torch.Generator(device="cpu").manual_seed(g_seed)
    S, T = d["P"].shape[:2]
    lp, lv = [], []
    for _ in range(a.calib_batches):
        idx, t0s = sample(g, S, T, a.k_end, a.batch, a.p_start0, dev)
        p, v = tc.rollout_terms(tt, d, idx, t0s, a.k_end, grad=False)
        lp.append(torch.stack(p))
        lv.append(torch.stack(v))
    ref = {"ref_p": torch.stack(lp).mean(0).tolist(), "ref_v": torch.stack(lv).mean(0).tolist()}
    if dh is not None:
        S2, T2 = dh["pos"].shape[:2]
        vals = []
        for _ in range(a.calib_batches):
            idx = torch.randint(S2, (a.batch_plain,), generator=g).to(dev)
            t0s = torch.randint(0, T2 - a.k_plain, (a.batch_plain,), generator=g).to(dev)
            vals.append(float(th.rollout_loss(model, dh, idx, t0s, a.k_plain)))
        ref["L_harvest"] = float(np.mean(vals))
    if iso is not None:
        ref["L_iso"] = float(np.mean([float(th.iso_loss(model, iso, torch.randint(len(iso[0]), (a.batch_plain,), generator=g).to(dev), a.k_plain, g, dev))
                                      for _ in range(a.calib_batches)]))
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", default=None, help="harvested windows directory (train_tier0_harvest); omit to drop that loss component")
    ap.add_argument("--pattern", default="hv3_s*.npz")
    ap.add_argument("--tier0", default=None, help="Exp E checkpoint: also scores the tiled baseline at the probes")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="results/tier0_tiled_closed_loop")
    ap.add_argument("--t0", default="ext=6,extapply,taper=8,bnd,halo=8")
    ap.add_argument("--exchange", choices=["halo", "zero", "none"], default="halo")
    ap.add_argument("--scenes", type=int, default=480)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--val-scenes", type=int, default=8)
    ap.add_argument("--val-steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2, help="tiled scenes per iteration (16 windows each)")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--k-start", type=int, default=8)
    ap.add_argument("--k-end", type=int, default=28)
    ap.add_argument("--tb", type=int, default=0)
    ap.add_argument("--p-start0", type=float, default=0.5)
    ap.add_argument("--w-tiled", type=float, default=0.6)
    ap.add_argument("--w-harvest", type=float, default=0.2)
    ap.add_argument("--w-iso", type=float, default=0.2)
    ap.add_argument("--k-plain", type=int, default=16)
    ap.add_argument("--batch-plain", type=int, default=8)
    ap.add_argument("--n-iso", type=int, default=1500)
    ap.add_argument("--calib-batches", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--probe-at", type=int, nargs="*", default=[1000, 2500])
    ap.add_argument("--gate1-err", type=float, default=0.92)
    ap.add_argument("--gate1-force", type=float, default=0.95)
    ap.add_argument("--gate2-err", type=float, default=0.85)
    ap.add_argument("--gate2-bl", type=float, default=1.45)
    ap.add_argument("--gate-plain", type=float, default=1.10)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--val-seed", type=int, default=4739)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--est", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))
    torch.manual_seed(a.seed)
    log = lambda s: print(s, flush=True)
    model = tte.build(argparse.Namespace(t0=a.t0, dt=0.1), dev)
    sf.load_compat(model, torch.load(a.init, map_location=dev, weights_only=False)["model"])
    model.train()
    tt = tc.TiledTier0(model, dev, chunk=a.chunk, exchange=a.exchange)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    path = os.path.join(a.out_dir, a.tag + ".pt")
    ref_path = os.path.join(a.out_dir, a.tag + "_reference.json")
    it = 0
    if a.resume and os.path.exists(path):
        st = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it = st["it"]
        log(f"resumed it {it}")
    t_gen = time.time()
    d = ttr.make_scenes(a.scenes if not a.est else 16, a.steps, a.seed, dev)
    dval = ttr.make_scenes(a.val_scenes, a.val_steps, a.val_seed, dev)
    dh = th.load_data(a.data, dev, a.pattern)[0] if a.data and a.w_harvest > 0 else None
    iso = None
    if a.w_iso > 0:
        iso = [t.float() for t in sf.to_tensors(tsf.make_data(tsf.EXPS["E"], a.n_iso if not a.est else 64, 40, a.seed + 77, dev, 0.1, 0.5), dev)]
    log(f"data ready in {time.time() - t_gen:.0f}s tiled scenes {len(d['P'])} harvest {None if dh is None else len(dh['pos'])} iso {None if iso is None else len(iso[0])} dev {dev}")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
    else:
        assert it == 0, "reference file missing on resume"
        t_ref = time.time()
        init_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        ref = reference(model, tt, d, dh, iso, a, dev, a.seed)
        model.eval()
        ref["probe"] = probe_metrics(model, tt, dval, dev)
        if a.tier0:
            from scripts import tiered_adapters as ta

            bl = ta.load_tiled(f"{a.tier0},nt=4,tile=32,halo=16,far=1", dev)
            ref["probe_baseline"] = {k: v for k, v in h.eval_generic(lambda p, v, m, k, st: bl.force(p, v, m, k, st), lambda S: bl.init_state(S, dev), dval,
                                                                     ticks=(10, 20)).items() if k in ("force_rel", "force_rel_free_rollout", "err@10", "err@20", "dE_model")}
        model.load_state_dict(init_model_state)
        model.train()
        json.dump(ref, open(ref_path, "w"), indent=1)
        log(f"reference in {time.time() - t_ref:.0f}s: {json.dumps({k: v for k, v in ref.items() if k.startswith(('L_', 'probe'))})}")
    ref_p, ref_v = torch.tensor(ref["ref_p"]), torch.tensor(ref["ref_v"])
    omega = tc.step_weights(a.k_end)
    S, T = d["P"].shape[:2]
    g = torch.Generator(device="cpu").manual_seed(a.seed + it)
    t_start, it_start, skipped, streak = time.time(), it, 0, 0
    if a.est:
        per = {}
        for k in (a.k_start, (a.k_start + a.k_end) // 2, a.k_end):
            for rep in range(4):
                if rep == 1:
                    torch.cuda.synchronize() if dev.type == "cuda" else None
                    t_k = time.time()
                idx, t0s = sample(g, S, T, k, a.batch, a.p_start0, dev)
                p, v = tc.rollout_terms(tt, d, idx, t0s, k, a.tb)
                loss = tc.closed_loop_loss(p, v, ref_p, ref_v, omega)
                opt.zero_grad()
                loss.backward()
            torch.cuda.synchronize() if dev.type == "cuda" else None
            per[k] = (time.time() - t_k) / 3
            log(f"k {k}: {per[k]:.2f} s/it tiled part, peak mem {torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else 0:.1f} GB")
        ks = sorted(per)
        tot = sum(np.interp(schedule(a, i, a.iters)[0], ks, [per[k] for k in ks]) for i in range(a.iters))
        log(f"ESTIMATE tiled part of {a.iters} iters = {tot / 3600:.2f} h (plain terms add about 30-50 percent)")
        return
    probes = {}
    while it < a.iters:
        if STOP:
            save(path, model, opt, it, a)
            log(f"SIGTERM: saved it {it}")
            return
        k, lr = schedule(a, it, a.iters)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()
        idx, t0s = sample(g, S, T, k, a.batch, a.p_start0, dev)
        p, v = tc.rollout_terms(tt, d, idx, t0s, k, a.tb)
        lt = tc.closed_loop_loss(p, v, ref_p, ref_v, omega)
        (a.w_tiled * lt).backward()
        parts = {"tiled": float(lt.detach())}
        ok = bool(torch.isfinite(lt))
        if dh is not None:
            S2, T2 = dh["pos"].shape[:2]
            idx2 = torch.randint(S2, (a.batch_plain,), generator=g).to(dev)
            t2 = torch.randint(0, T2 - a.k_plain, (a.batch_plain,), generator=g).to(dev)
            lh = th.rollout_loss(model, dh, idx2, t2, a.k_plain) / ref["L_harvest"]
            (a.w_harvest * lh).backward()
            parts["harvest"] = float(lh)
            ok = ok and bool(torch.isfinite(lh))
        if iso is not None:
            li = th.iso_loss(model, iso, torch.randint(len(iso[0]), (a.batch_plain,), generator=g).to(dev), a.k_plain, g, dev) / ref["L_iso"]
            (a.w_iso * li).backward()
            parts["iso"] = float(li)
            ok = ok and bool(torch.isfinite(li))
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if ok and torch.isfinite(gn):
            opt.step()
            streak = 0
        else:
            opt.zero_grad()
            skipped += 1
            streak += 1
            if streak >= 20 and os.path.exists(path):
                st = torch.load(path, map_location=dev, weights_only=False)
                model.load_state_dict(st["model"])
                opt.load_state_dict(st["opt"])
                log(f"rolled back to {st['it']}")
                streak = 0
        it += 1
        if it % a.log_every == 0:
            el = time.time() - t_start
            log(f"it {it} k {k} lr {lr:.2e} {json.dumps({n: round(x, 4) for n, x in parts.items()})} gn {float(gn):.3f} t {el:.0f}s eta {el / (it - it_start) * (a.iters - it):.0f}s skipped {skipped}")
        if it in a.probe_at:
            pm = probe_metrics(model, tt, dval, dev)
            bad = gate(pm, ref["probe"], ref.get("probe_baseline"), it, a)
            probes[it] = {"metrics": pm, "violations": bad}
            log(f"probe it {it} {json.dumps({n: round(x, 5) for n, x in pm.items()})} ref {json.dumps({n: round(x, 5) for n, x in ref['probe'].items()})} violations {bad}")
            json.dump({"it": it, "metrics": pm, "reference": ref["probe"], "baseline": ref.get("probe_baseline"), "violations": bad},
                      open(os.path.join(a.out_dir, f"{a.tag}_probe_{it}.json"), "w"), indent=1)
            if bad:
                save(path, model, opt, it, a)
                open(os.path.join(a.out_dir, a.tag + ".EARLY_KILL"), "w").write(json.dumps({"it": it, "violations": bad}))
                log("EARLY_KILL")
                return
        if it % a.ckpt_every == 0 or it == a.iters:
            if all(torch.isfinite(q).all() for q in model.parameters()):
                save(path, model, opt, it, a)
    open(os.path.join(a.out_dir, a.tag + ".TRAIN_DONE"), "w").write(str(it))
    log("DONE")


if __name__ == "__main__":
    main()
