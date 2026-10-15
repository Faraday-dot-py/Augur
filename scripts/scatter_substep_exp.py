"""Stage-1 inference-only substep schemes on trained scatter-field checkpoints (docs/debugging/substep-experiments.md).

Schemes (single change each vs v1): v1 = 1 Verlet step per tick (control); u2 / u4 = uniform 2 / 4 Verlet substeps per tick (full force
each substep, existing `subN`); nf2 = symmetric RESPA with near/far cutoff at the pair radius 2: full force once per tick half-kicked at
tick start and end, symmetric pair term (r < 2) on M = 4 KDK substeps (`nf4`, ScatterField._step_respa).
Eval as scripts/scatter_losstest.py: seeds 9100/9200/9300, 48 scenes x 100 ticks each (24 'ab' 10-200-body scale-init, 24 'cl' clusters).
Also: force-call / pair-pass counts per tick, ms per tick, pair counts per body per substep by density (ab, cl, bh1000, bh3000 truth states).

  PYTHONPATH=. python scripts/scatter_substep_exp.py --models E=ckpt,B=ckpt,EW=ckpt,P0=ckpt --out results/substep_exp/acc.json
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import scatter_conservation as scv
from scripts import scatter_field as sf
from scripts import train_scatter_field as ts

EPS, DT = 0.5, 0.1
CFG = ts.EXPS["E"]
SEEDS = (9100, 9200, 9300)
DEV = torch.device("cuda")
VARIANT = {"P0": "ms_kp_pot_v_g128_prior_pp8_full"}
SCHEMES = ("v1", "u2", "u4", "nf2")


def build(name, ckpt):
    model = ts.build(CFG, VARIANT.get(name, "ms_kp_pot_v_g128"), DT).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=False)["model"])
    return model.eval()


def set_scheme(model, scheme):
    model.substeps, model.respa, model.respa_rc = 1, 0, 0.0
    if scheme in ("u2", "u4"):
        model.substeps = int(scheme[1])
    elif scheme == "nf2":
        model.respa = 4
        model.respa_rc = 2.0 if model.pp > 2.0 else 0.0


def evaluate(model, data):
    res = {}
    for seed in SEEDS:
        for tag, sub in (("ab", data[seed][:24]), ("cl", data[seed][24:])):
            m, (Pm, Vm, Pt, Vt) = ts.evaluate_model(model, sub, DEV, DT, EPS)
            P, V, M, mask = sf.to_tensors(sub, DEV)
            e = (Pm[:, 20] - Pt[:, 20]).norm(dim=-1)[mask > 0].double().cpu()
            q = torch.quantile(e, torch.tensor([0.5, 0.9, 0.99], dtype=torch.double))
            res[f"{tag}_{seed}"] = {"err": [m["err"][i] for i in (4, 9, 19, 49, 99)],
                                    "e20": {"mean": float(e.mean()), "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2])},
                                    "dE100": m["energy_drift_rel_model"][100], "mom100": m["mom_drift_model"][100], "fin": bool(torch.isfinite(Pm).all())}
            r = res[f"{tag}_{seed}"]
            print(f"  {tag} {seed} err@5/10/20/50/100 {[round(x, 5) for x in r['err']]} e20 med/mean/p99 {r['e20']['median']:.4f}/{r['e20']['mean']:.4f}/{r['e20']['p99']:.4f}"
                  f" dE100 {r['dE100']:.3g} |dP|100 {r['mom100']:.2g}", flush=True)
    return res


def cost(model, sub):
    """Model.force and pp_acc_sym calls per tick (steady state) and ms per tick on a batch."""
    P, V, M, mask = sf.to_tensors(sub, DEV)
    pos, vel, m, mk = P[:, 0].float(), V[:, 0].float(), M.float(), mask.float()
    cnt = {"force": 0, "pair_sym": 0}
    f0, p0 = model.force, model.pp_acc_sym

    def fw(*a, **k):
        cnt["force"] += 1
        return f0(*a, **k)

    def pw(*a, **k):
        cnt["pair_sym"] += 1
        return p0(*a, **k)

    model.force, model.pp_acc_sym = fw, pw
    with torch.no_grad():
        field = model.init_field(pos.shape[0], DEV)
        for _ in range(3):
            pos, vel, field, _ = model.step(pos, vel, m, mk, field)
        cnt["force"] = cnt["pair_sym"] = 0
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            pos, vel, field, _ = model.step(pos, vel, m, mk, field)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 20 * 1e3
    del model.force, model.pp_acc_sym
    return {"force_per_tick": cnt["force"] / 20, "pair_sym_per_tick": cnt["pair_sym"] / 20, "ms_per_tick": ms, "B": int(pos.shape[0]), "N": int(pos.shape[1])}


def pair_counts(model, pos, mask):
    """Per-body pair counts: all pairs within 2 (uncapped), knn16-capped (what the model's pair term evaluates with nonzero weight),
    symmetric union set (nf2 fast pass), all pairs within 8 (P0's pair term)."""
    with torch.no_grad():
        sel, _, r = model.pp_pairs_sym(pos, mask, 16)
        eye = torch.eye(pos.shape[1], device=DEV, dtype=torch.bool)[None]
        ok = (mask[:, :, None] * mask[:, None, :]) > 0
        w2 = ((r < 2.0) & ok & ~eye).sum(-1).double()
        w8 = ((r < 8.0) & ok & ~eye).sum(-1).double()
        mb = mask > 0
        return {"within2": float(w2[mb].mean()), "within2_p99": float(w2[mb].quantile(0.99)), "knn16_capped": float(w2.clamp(max=16)[mb].mean()),
                "sym_union": float(sel.sum(-1)[mb].mean()), "sym_union_max": float(sel.sum(-1)[mb].max()), "within8": float(w8[mb].mean())}


def bh_states(n, ticks, per=2):
    pos0, vel0, _ = scv.make_scenes("bh", n, per, (9100,))
    P0, V0 = torch.tensor(pos0, device=DEV), torch.tensor(vel0, device=DEV)
    out = {}

    def rec(t, pos, vel, mom, snap_idx=None):
        if t in ticks:
            out[t] = pos.float().clone()

    scv.roll_truth(P0, V0, True, max(ticks), 4, EPS, rec, None, DT)
    return out


def densities(model, data):
    res = {}
    for tag, sl in (("ab", slice(0, 24)), ("cl", slice(24, 48))):
        P, V, M, mask = sf.to_tensors(data[9100][sl], DEV)
        for t in (0, 20, 50, 100):
            res[f"{tag}_t{t}"] = pair_counts(model, P[:, t].float(), mask.float())
    for n in (1000, 3000):
        st = bh_states(n, (0, 50, 100, 200), per=2 if n == 1000 else 1)
        for t, pos in st.items():
            res[f"bh{n}_t{t}"] = pair_counts(model, pos, torch.ones(pos.shape[:2], device=DEV))
    for k, v in res.items():
        print(f"  pairs {k}: " + " ".join(f"{a} {b:.2f}" for a, b in v.items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="name=ckpt,... (P0 = pairfull0, variant prior_pp8_full)")
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(4738)
    data = {s: ts.make_data(CFG, 48, 100, s, DEV, DT, EPS) for s in SEEDS}
    if args.smoke:
        data = {s: d[:2] + d[24:26] for s, d in data.items()}
    out = {"models": args.models, "acc": {}, "cost": {}}
    models = dict(kv.split("=") for kv in args.models.split(","))
    for name, ckpt in models.items():
        model = build(name, ckpt)
        if "densities" not in out and model.pp == 2.0:
            out["densities"] = densities(model, data) if not args.smoke else {}
        for scheme in args.schemes.split(","):
            set_scheme(model, scheme)
            print(f"== {name} {scheme} substeps {model.substeps} respa {model.respa} rc {model.respa_rc}", flush=True)
            t0 = time.time()
            out["acc"][f"{name}:{scheme}"] = evaluate(model, data)
            out["cost"][f"{name}:{scheme}"] = {"cl": cost(model, data[9100][24:] if not args.smoke else data[9100][2:]),
                                              "ab": cost(model, data[9100][:24] if not args.smoke else data[9100][:2])}
            print(f"  cost {out['cost'][f'{name}:{scheme}']} eval {time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(args.out, "w"))
        set_scheme(model, "v1")
    print("done", flush=True)


if __name__ == "__main__":
    main()
