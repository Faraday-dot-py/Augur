"""Stage-1 probe for a separated velocity branch: how much of the frozen replay checkpoint's error power runs through the velocity and history inputs.
Inference only. Along the model's own closed-loop rollout, the net input is re-evaluated with one input group scaled (0 = removed, 0.9 = first-order
finite difference) at the visited states; the carried recurrent field stays that of the unablated run. With v the predictor velocity at the force call (vel + dt a_prev),
  P_err = sum m v.(a_model - a_true)   is the rate at which the model's energy drifts from the true energy (dE_model/dt = P_err),
  dP_g  = sum m v.(a_full - a_without_g)  is the part of the model's power carried by input group g.
S_vh = sum dP_vh / sum P_err, signed, cumulative to tick 100 (pooled over scenes)."""
import argparse
import json
import os

import numpy as np
import torch

from scripts import tier0_stage1_probe_common as pc
from scripts import tier0_tiled_closed_loop as tc
from scripts import train_tiered as ttr
from scripts import exp12_hier as h
from scripts.eval_tier0_tiled_closed_loop import load

GROUPS = ("velocity", "history", "velocity+history", "kernel-acc", "ext+boundary")
ESTIMATORS = {"zero": 0.0, "fd": 0.9}
SNAP = (20, 60, 100)
LINE = {"open": 0.5, "open_each_seed": 0.35, "closed": 0.2, "identity_tol": 0.25}


def group_slices(model):
    cf = model.cf
    return {"velocity": [slice(1, 3)], "history": [slice(5, 5 + cf)], "velocity+history": [slice(1, 3), slice(5, 5 + cf)],
            "kernel-acc": [slice(3, 5)], "ext+boundary": [slice(5 + cf, None)]}


class Scale:
    def __init__(self, net):
        self.sl, self.s = [], 1.0
        net.register_forward_pre_hook(self)

    def __call__(self, mod, args):
        if not self.sl or self.s == 1.0:
            return None
        x = args[0].clone()
        for s in self.sl:
            x[:, s] *= self.s
        return (x,) + tuple(args[1:])


class Tally:
    def __init__(self, tt, d):
        self.tt, self.d = tt, d
        self.m, self.mk = d["M"].float(), d["mask"].float()
        self.sc = Scale(tt.m.net)
        self.slices = group_slices(tt.m)
        B = self.mk.shape[0]
        self.B = B
        self.ser = {k: [] for k in ["P_true", "P_model"] + [f"{e}:{g}" for e in ESTIMATORS for g in GROUPS]}
        self.tile = {k: torch.zeros(B, tc.NW, dtype=torch.float64) for k in ["P_err", "count"] + [f"{e}:{g}" for e in ESTIMATORS for g in GROUPS]}

    def power(self, pos, v, a):
        return (self.m * (v * a).sum(-1) * self.mk).double()

    def per_tile(self, w, own):
        out = torch.zeros(self.B, tc.NW, dtype=torch.float64, device=w.device)
        out.scatter_add_(1, own, w)
        return out.cpu()

    def __call__(self, t, pos, vin, a_model, a_used, fld, fld_out):
        at = ttr.true_acc(pos, self.m, self.mk)
        own = h.owner(pos)
        pt, pm = self.power(pos, vin, at), self.power(pos, vin, a_model)
        self.ser["P_true"].append(pt.sum(1).cpu())
        self.ser["P_model"].append(pm.sum(1).cpu())
        self.tile["P_err"] += self.per_tile(pm - pt, own)
        self.tile["count"] += self.per_tile(self.mk.double(), own)
        for e, lam in ESTIMATORS.items():
            for g in GROUPS:
                self.sc.sl, self.sc.s = self.slices[g], lam
                a_abl, _ = self.tt.force(pos, vin, self.m, self.mk, fld)
                self.sc.sl, self.sc.s = [], 1.0
                dp = (pm - self.power(pos, vin, a_abl)) / ((1.0 - lam))
                self.ser[f"{e}:{g}"].append(dp.sum(1).cpu())
                self.tile[f"{e}:{g}"] += self.per_tile(dp, own)


def cumulate(tally):
    cum = {}
    for k, v in tally.ser.items():
        c = torch.stack(v).cumsum(0) * pc.DT
        cum[k] = c
    return cum


def run(tt, d):
    tl = Tally(tt, d)
    r = pc.rollout(tt, d, observe=tl)
    cum = cumulate(tl)
    en = r["energy"]
    out = {"snap": {}, "tile": {k: (v * pc.DT).tolist() for k, v in tl.tile.items() if k != "count"}, "tile_count": tl.tile["count"].tolist()}
    drift = (en - en[0:1])
    perr = (cum["P_model"] - cum["P_true"])
    for t in SNAP:
        s = {"measured_dE": drift[t].tolist(), "P_err_integral": perr[t - 1].tolist(), "E0": en[0].tolist()}
        for k in cum:
            s[k] = cum[k][t - 1].tolist()
        out["snap"][str(t)] = s
    out["err"] = r["err"]
    return out


def summarise(res):
    out = {}
    for t in SNAP:
        pe = np.concatenate([np.array(res[str(sd)]["snap"][str(t)]["P_err_integral"]) for sd in pc.SEEDS])
        md = np.concatenate([np.array(res[str(sd)]["snap"][str(t)]["measured_dE"]) for sd in pc.SEEDS])
        o = {"identity_ratio_sum_Perr_over_sum_measured_dE": float(pe.sum() / md.sum()) if abs(md.sum()) > 0 else None,
             "identity_abs_ratio": float(np.abs(pe - md).sum() / max(np.abs(md).sum(), 1e-30)), "pooled_sum_P_err": float(pe.sum())}
        for e in ESTIMATORS:
            for g in GROUPS:
                num = np.concatenate([np.array(res[str(sd)]["snap"][str(t)][f"{e}:{g}"]) for sd in pc.SEEDS])
                o[f"S[{e}:{g}]"] = float(num.sum() / pe.sum())
                o[f"S_per_seed[{e}:{g}]"] = [float(np.sum(res[str(sd)]["snap"][str(t)][f"{e}:{g}"]) / np.sum(res[str(sd)]["snap"][str(t)]["P_err_integral"])) for sd in pc.SEEDS]
            nv = np.concatenate([np.array(res[str(sd)]["snap"][str(t)][f"{e}:velocity"]) for sd in pc.SEEDS]).sum()
            nh = np.concatenate([np.array(res[str(sd)]["snap"][str(t)][f"{e}:history"]) for sd in pc.SEEDS]).sum()
            nj = np.concatenate([np.array(res[str(sd)]["snap"][str(t)][f"{e}:velocity+history"]) for sd in pc.SEEDS]).sum()
            o[f"non_additive_residual_over_P_err[{e}]"] = float((nj - nv - nh) / pe.sum())
        out[str(t)] = o
    tb = {}
    cnt = np.concatenate([np.array(res[str(sd)]["tile_count"]).ravel() for sd in pc.SEEDS])
    pe = np.concatenate([np.array(res[str(sd)]["tile"]["P_err"]).ravel() for sd in pc.SEEDS])
    live = cnt > 0
    q = np.quantile(cnt[live], [1 / 3, 2 / 3])
    for e in ESTIMATORS:
        dp = np.concatenate([np.array(res[str(sd)]["tile"][f"{e}:velocity+history"]).ravel() for sd in pc.SEEDS])
        for name, sel in (("low", live & (cnt <= q[0])), ("mid", live & (cnt > q[0]) & (cnt <= q[1])), ("high", live & (cnt > q[1]))):
            tb[f"S_vh[{e}]_{name}_occupancy_tiles"] = float(dp[sel].sum() / pe[sel].sum()) if abs(pe[sel].sum()) > 0 else None
    out["tile_occupancy_terciles"] = tb
    return out


def verdict(summary, ckpt_ok=True):
    s = summary[str(SNAP[-1])]
    if not (1 - LINE["identity_tol"] <= (s["identity_ratio_sum_Perr_over_sum_measured_dE"] or 0) <= 1 + LINE["identity_tol"]):
        return "VOID"
    z, f = s["S[zero:velocity+history]"], s["S[fd:velocity+history]"]
    zs, fs = s["S_per_seed[zero:velocity+history]"], s["S_per_seed[fd:velocity+history]"]
    if z >= LINE["open"] and f >= LINE["open"] and min(zs) >= LINE["open_each_seed"] and min(fs) >= LINE["open_each_seed"]:
        return "PASS"
    if z < LINE["closed"] and f < LINE["closed"]:
        return "KILL"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path; the first is the verdict checkpoint, the rest are reported replicates")
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(pc.SEEDS), help="held-out scene seeds; 12000 12100 12200 is the fresh set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pc.SEEDS = tuple(a.seeds)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, paths = zip(*(c.split("=", 1) for c in a.ckpt))
    for p in paths:
        assert os.path.exists(p), f"MISSING {p}"
    out = {"args": vars(a), "line": LINE, "groups": GROUPS, "estimators": ESTIMATORS, "ckpt": {}}
    for n, p in zip(names, paths):
        model = load(p, dev)
        n_in = 5 + model.cf + model.t0.cfg.n_extra()
        tt = tc.TiledTier0(model, dev, chunk=a.chunk, exchange="halo", ckpt=False)
        seen = []
        tt.m.net.register_forward_pre_hook(lambda mod, args: seen.append(args[0].shape[1]) if not seen else None)
        res = {}
        for sd in pc.SEEDS:
            d = ttr.make_scenes(a.scenes, a.steps, sd, dev)
            res[str(sd)] = run(tt, d)
            assert seen[0] == n_in, f"net input channels {seen[0]} != {n_in}; channel slices are wrong"
            print(n, sd, "done", flush=True)
        summ = summarise(res)
        out["ckpt"][n] = {"raw": pc.tolist(res), "summary": summ, "verdict_rule_applied": verdict(summ)}
        print(n, json.dumps({k: v for k, v in summ[str(SNAP[-1])].items() if k.startswith("S[") or k.startswith("identity")}), "verdict-rule", out["ckpt"][n]["verdict_rule_applied"], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
