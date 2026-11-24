"""Ceiling probe for nested refinement levels: on the frozen replay checkpoint (tiled tier 0, halo exchange), replace the model's force by the exact direct
sum for bodies in the densest region and measure how much drift and error disappear. Inference only. Arms, each a full 100-tick closed-loop rollout:
  base          unchanged model
  dense-exact   exact force for bodies with at least DENSE_MIN neighbours within R_DENS (decided at every tick on the model's own state)
  random-exact  exact force for the same number of randomly chosen bodies per scene and tick (is the gain specific to dense bodies)
  all-exact     exact force for every body, integrated with the truth's own substepped integrator (harness check: must land near the exact trajectory)
Usage: PYTHONPATH=. python scripts/dense_region_exact_force_ceiling_probe.py --ckpt replay=<pt> [more name=path] --out <json>"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import replay_mechanism_force_error_breakdown as rb
from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import tier0_stage1_probe_common as pc
from scripts import tier0_tiled_closed_loop as tc
from scripts import train_tiered as ttr
from scripts.eval_tier0_tiled_closed_loop import load

DENSE_MIN = 20
ARMS = ("base", "dense-exact", "random-exact", "all-exact")
FORCE_TICKS = tuple(range(20, 101, 5))
LATE = 60
SUBSTEPS = gs.rollout_torch.__defaults__[1]
LINE = {"dE_pass": 0.60, "err20_pass": 0.95, "random_margin": 0.10, "dE_kill": 0.75, "err20_kill": 1.00, "harness": 0.25}


def div(a, b):
    return a / b if b else None


def make_hook(arm, m, mk, gen):
    def hook(t, pos, vel, a):
        if arm == "base":
            return a
        live = mk > 0
        if arm == "all-exact":
            sel = live
        else:
            sel = (rb.density(pos, mk) >= DENSE_MIN) & live
            if arm == "random-exact":
                k = sel.sum(1)
                score = torch.rand(mk.shape, generator=gen).to(mk.device).masked_fill(~live, -1.0)
                rank = score.argsort(1, descending=True).argsort(1)
                sel = (rank < k[:, None]) & live
        return torch.where(sel[..., None], ttr.true_acc(pos, m, mk), a)
    return hook


class ForceRows:
    """Squared force error of the unreplaced model force on the arm's own visited states, split by density and early/late."""

    def __init__(self, m, mk):
        self.m, self.mk = m, mk
        self.acc = {f"{w}_{b}": [0.0, 0, 0.0] for w in ("early", "late") for b in ("dense", "nondense")}

    def __call__(self, t, pos, vin, a_model, a_used, fld, fld_out):
        if t not in FORCE_TICKS:
            return
        at = ttr.true_acc(pos, self.m, self.mk)
        live = self.mk > 0
        dn = rb.density(pos, self.mk) >= DENSE_MIN
        e2 = ((a_model - at) ** 2).sum(-1)
        t2 = (at ** 2).sum(-1)
        for b, s in (("dense", dn & live), ("nondense", ~dn & live)):
            r = self.acc[f"{'late' if t >= LATE else 'early'}_{b}"]
            r[0] += float(e2[s].sum())
            r[1] += int(s.sum())
            r[2] += float(t2[s].sum())


@torch.no_grad()
def rollout_exact(tt, d, observe, substeps=SUBSTEPS):
    """All-exact harness check: the truth's own integrator (substeps Verlet sub-steps per tick, exact force, float32) so the only residual is float32 noise.
    A single tick-sized Verlet step with exact force carries a discretisation error against the substepped truth that the trained model absorbs."""
    m, mk = d["M"].float(), d["mask"].float()
    T = d["P"].shape[1]
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    fld = tt.init_state(pos.shape[0])
    _, fld = tt.force(pos, vel, m, mk, fld)
    h = pc.DT / substeps
    a = ttr.true_acc(pos, m, mk)
    cnt = d["mask"].sum(1)
    err, en = {}, [sf.energy(pos.double(), vel.double(), d["M"], d["mask"], pc.EPS).cpu()]
    for t in range(1, T):
        for _ in range(substeps):
            pos = pos + h * vel + 0.5 * h * h * a
            an = ttr.true_acc(pos, m, mk)
            vel = vel + 0.5 * h * (a + an)
            a = an
        a_model, fld_out = tt.force(pos, vel, m, mk, fld)
        observe(t, pos, vel, a_model, a, fld, fld_out)
        fld = fld_out
        en.append(sf.energy(pos.double(), vel.double(), d["M"], d["mask"], pc.EPS).cpu())
        if t in pc.ERR_TICKS:
            err[t] = (((pos.double() - d["P"][:, t]).norm(dim=-1) * d["mask"]).sum(1) / cnt).cpu().numpy()
    return {"err": err, "energy": torch.stack(en)}


def run_arm(tt, d, arm, seed):
    m, mk = d["M"].float(), d["mask"].float()
    gen = torch.Generator().manual_seed(4738 + seed)
    rows = ForceRows(m, mk)
    if arm == "all-exact":
        r = rollout_exact(tt, d, rows)
        return {"err": r["err"], "dE": pc.energy_vs_truth(r["energy"], pc.truth_energy(d)), "force": rows.acc}
    r = pc.rollout(tt, d, hook=make_hook(arm, m, mk, gen), observe=rows)
    en = pc.energy_vs_truth(r["energy"], pc.truth_energy(d))
    return {"err": r["err"], "dE": en, "force": rows.acc}


def pooled(res, arm, key):
    return float(np.mean(np.concatenate([res[str(sd)][arm][key[0]][key[1]] if key[0] == "err" else res[str(sd)][arm]["dE"][key[1]][key[2]] for sd in pc.SEEDS])))


def summarise(res):
    s = {}
    for arm in ARMS:
        a = {"err@20": pooled(res, arm, ("err", 20)), "err@100": pooled(res, arm, ("err", pc.ERR_TICKS[-1])), "dE@100_abs": pooled(res, arm, ("dE", pc.ENERGY_TICKS[-1], "abs")),
             "dE@100_signed": pooled(res, arm, ("dE", pc.ENERGY_TICKS[-1], "signed")), "dE@60_abs": pooled(res, arm, ("dE", pc.ENERGY_TICKS[1], "abs"))}
        for k in ("early_dense", "early_nondense", "late_dense", "late_nondense"):
            sse = sum(res[str(sd)][arm]["force"][k][0] for sd in pc.SEEDS)
            n = sum(res[str(sd)][arm]["force"][k][1] for sd in pc.SEEDS)
            tr = sum(res[str(sd)][arm]["force"][k][2] for sd in pc.SEEDS)
            a[f"{k}_mean_sq_force_err"] = sse / max(n, 1)
            a[f"{k}_n_body_ticks"] = n
            a[f"{k}_rel_force_err"] = (sse / max(tr, 1e-30)) ** 0.5
            a[f"{k}_sse"] = sse
        tot = a["late_dense_sse"] + a["late_nondense_sse"]
        a["late_dense_share_of_sse"] = a["late_dense_sse"] / max(tot, 1e-30)
        s[arm] = a
    base = s["base"]
    ratios = {}
    for arm in ARMS[1:]:
        a = s[arm]
        ratios[arm] = {"dE@100": a["dE@100_abs"] / base["dE@100_abs"], "err@20": a["err@20"] / base["err@20"], "err@100": a["err@100"] / base["err@100"],
                       "late_dense_mean_sq_force_err_knock_on": div(a["late_dense_mean_sq_force_err"], base["late_dense_mean_sq_force_err"]),
                       "late_nondense_mean_sq_force_err_knock_on": div(a["late_nondense_mean_sq_force_err"], base["late_nondense_mean_sq_force_err"]),
                       "dE@100_per_seed": [div(float(np.mean(res[str(sd)][arm]["dE"][pc.ENERGY_TICKS[-1]]["abs"])), float(np.mean(res[str(sd)]["base"]["dE"][pc.ENERGY_TICKS[-1]]["abs"]))) for sd in pc.SEEDS]}
    return s, ratios


def verdict(ratios):
    r, rr, ra = ratios["dense-exact"], ratios["random-exact"], ratios["all-exact"]
    if ra["err@20"] > LINE["harness"]:
        return "VOID"
    if r["dE@100"] <= LINE["dE_pass"] and r["err@20"] <= LINE["err20_pass"] and r["dE@100"] <= rr["dE@100"] - LINE["random_margin"]:
        return "PASS"
    if r["dE@100"] > LINE["dE_kill"] or r["err@20"] > LINE["err20_kill"]:
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
    out = {"args": vars(a), "line": LINE, "dense_min": DENSE_MIN, "ckpt": {}}
    for n, p in zip(names, paths):
        tt = tc.TiledTier0(load(p, dev), dev, chunk=a.chunk, exchange="halo", ckpt=False)
        res = {}
        for sd in pc.SEEDS:
            d = ttr.make_scenes(a.scenes, a.steps, sd, dev)
            res[str(sd)] = {}
            for arm in ARMS:
                res[str(sd)][arm] = run_arm(tt, d, arm, sd)
                print(f"{n} {sd} {arm} err@20 {res[str(sd)][arm]['err'][20].mean():.5f} dE@100 {res[str(sd)][arm]['dE'][pc.ENERGY_TICKS[-1]]['abs'].mean():.5f}", flush=True)
        summ, ratios = summarise(res)
        out["ckpt"][n] = {"raw": pc.tolist(res), "summary": summ, "ratios_vs_base": ratios, "verdict_rule_applied": verdict(ratios)}
        print(n, json.dumps(ratios["dense-exact"]), "verdict-rule", out["ckpt"][n]["verdict_rule_applied"], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
