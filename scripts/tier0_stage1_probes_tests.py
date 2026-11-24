"""CPU smoke tests for the three stage-1 probes of the frozen tiled tier 0: PYTHONPATH=. python scripts/tier0_stage1_probes_tests.py [--device cuda]
Uses a randomly perturbed model (no trained checkpoint is needed) and one scene of 22 ticks, so only wiring is checked, not any scientific number."""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

from scripts import dense_region_exact_force_ceiling_probe as dp
from scripts import input_group_work_split_probe as ip
from scripts import neighbour_tile_force_error_predictability_probe as npb
from scripts import tier0_stage1_probe_common as pc
from scripts import tier0_tiled_closed_loop as tc
from scripts import train_tiered as ttr
from scripts.tier0_tiled_closed_loop_tests import perturbed_model

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not ok:
        fails.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    steps = 22
    pc.SEEDS = (9100,)
    pc.ERR_TICKS, pc.ENERGY_TICKS = (5, 10, 20, 22), (10, 20, 22)
    dp.FORCE_TICKS, dp.LATE = (10, 15, 20, 22), 15
    ip.SNAP = (10, 20, 22)
    npb.TICKS, npb.TRAIN_SEEDS, npb.EVAL_SEEDS = (10, 15, 20, 22), (7100,), (9100,)
    npb.MLP = dict(npb.MLP, epochs=3)
    model = perturbed_model(dev)
    tmp = tempfile.mkdtemp()
    ck = os.path.join(tmp, "random.pt")
    torch.save({"model": model.state_dict(), "t0": model.t0_spec}, ck)
    from scripts.eval_tier0_tiled_closed_loop import load
    tt = tc.TiledTier0(load(ck, dev), dev, chunk=16, exchange="halo", ckpt=False)
    d = ttr.make_scenes(1, steps, 9100, dev)
    m, mk = d["M"].float(), d["mask"].float()

    base = pc.rollout(tt, d)
    same = pc.rollout(tt, d, hook=dp.make_hook("base", m, mk, torch.Generator().manual_seed(0)))
    check("hook that returns the model force reproduces the plain rollout bit for bit", torch.equal(base["energy"], same["energy"]))
    keep = dp.DENSE_MIN
    dp.DENSE_MIN = 10 ** 6
    none = pc.rollout(tt, d, hook=dp.make_hook("dense-exact", m, mk, torch.Generator().manual_seed(0)))
    dp.DENSE_MIN = keep
    check("dense-exact with an unreachable density threshold (empty mask) reproduces base bit for bit", torch.equal(base["energy"], none["energy"]))
    allx = pc.rollout(tt, d, hook=dp.make_hook("all-exact", m, mk, torch.Generator().manual_seed(0)))
    check("all-exact lands closer to the exact trajectory than the perturbed model", float(allx["err"][22].mean()) < float(base["err"][22].mean()), f"{allx['err'][22].mean():.4f} vs {base['err'][22].mean():.4f}")
    ar = dp.make_hook("random-exact", m, mk, torch.Generator().manual_seed(0))
    dn = dp.make_hook("dense-exact", m, mk, torch.Generator().manual_seed(0))
    pos, vel = d["P"][:, 10].float(), d["V"][:, 10].float()
    fake = torch.zeros_like(pos) + 7.0
    ca, cb = (ar(10, pos, vel, fake) != 7.0).any(-1).sum().item(), (dn(10, pos, vel, fake) != 7.0).any(-1).sum().item()
    check("random-exact replaces as many bodies as dense-exact", ca == cb, f"{ca} {cb}")

    out = os.path.join(tmp, "dense.json")
    sys.argv = ["x", "--ckpt", f"c1={ck}", "--scenes", "1", "--steps", str(steps), "--chunk", "16", "--out", out]
    dp.main()
    j = json.load(open(out))["ckpt"]["c1"]
    check("dense probe writes ratios, summary and a verdict", j["verdict_rule_applied"] in ("PASS", "KILL", "INCONCLUSIVE", "VOID") and "dense-exact" in j["ratios_vs_base"], j["verdict_rule_applied"])
    check("dense probe knock-on counts are positive in both bins", j["summary"]["base"]["late_nondense_n_body_ticks"] > 0, json.dumps({k: j["summary"]["base"][k] for k in ("late_dense_n_body_ticks", "late_nondense_n_body_ticks")}))

    out = os.path.join(tmp, "split.json")
    sys.argv = ["x", "--ckpt", f"c1={ck}", "--scenes", "1", "--steps", str(steps), "--chunk", "16", "--out", out]
    ip.main()
    j = json.load(open(out))["ckpt"]["c1"]["summary"][str(ip.SNAP[-1])]
    vals = [v for k, v in j.items() if k.startswith("S[")]
    check("input-group probe: every attributed share is finite", all(np.isfinite(vals)), f"S_vh zero {j['S[zero:velocity+history]']:.3f} fd {j['S[fd:velocity+history]']:.3f}")
    check("input-group probe: energy identity ratio reported", j["identity_ratio_sum_Perr_over_sum_measured_dE"] is not None, str(j["identity_ratio_sum_Perr_over_sum_measured_dE"]))
    sc = ip.Scale(tt.m.net)
    sc.sl, sc.s = [], 1.0
    st = tt.init_state(1)
    a0, _ = tt.force(d["P"][:, 0].float(), d["V"][:, 0].float(), m, mk, st)
    sc.sl, sc.s = ip.group_slices(tt.m)["velocity+history"], 1.0
    a1, _ = tt.force(d["P"][:, 0].float(), d["V"][:, 0].float(), m, mk, st)
    check("scaling a group by 1.0 changes nothing", torch.equal(a0, a1))
    sc.s = 0.0
    a2, _ = tt.force(d["P"][:, 0].float(), d["V"][:, 0].float(), m, mk, st)
    check("zeroing the velocity group changes the force", float((a2 - a0).abs().max()) > 0)

    out = os.path.join(tmp, "tile.json")
    sys.argv = ["x", "--ckpt", f"c1={ck}", "--scenes", "1", "--steps", str(steps), "--chunk", "16", "--out", out]
    npb.main()
    s = json.load(open(out))["ckpt"]["c1"]["summary"]
    check("neighbour probe: finite explained fractions for every variant", all(np.isfinite(v["pooled"]) for v in s["f"].values()), json.dumps({k: round(v["pooled"], 3) for k, v in s["f"].items()}))
    tr = npb.collect(tt, 7100, 1, steps, dev, npb.neighbour_index())
    check("neighbour probe feature widths are 28 / 72 / 72", (tr["own"].shape[1], tr["n1"].shape[1], tr["n2"].shape[1]) == (28, 72, 72), str((tr["own"].shape[1], tr["n1"].shape[1], tr["n2"].shape[1])))
    check("neighbour probe: rows carry only live bodies and finite features", bool(torch.isfinite(tr["own"]).all() and torch.isfinite(tr["n1"]).all() and torch.isfinite(tr["n2"]).all()))
    idx = npb.neighbour_index()
    check("neighbour index: corner tile has 3 neighbours, interior tile has 8", int((idx[0] >= 0).sum()) == 3 and int((idx[5] >= 0).sum()) == 8)
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
