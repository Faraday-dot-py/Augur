"""Unit tests for scripts/scatter_regress_contact.py (run on Polaris: python scripts/scatter_regress_contact_test.py [baseline json])."""
import io
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import torch

from scripts import scatter_contact as sc
from scripts import scatter_contact_scenes as cs
from scripts import scatter_regress as sr
from scripts import scatter_regress_contact as rc
from scripts import scatter_scenes as ss

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = sys.argv[1] if len(sys.argv) > 1 else "results/scatter_baseline.json"
OK = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info, flush=True)
    OK.append(bool(cond))


def test_energy():
    P = np.array([[[[1.0, 0.0], [3.0, 0.0]]]])
    V = np.array([[[[2.0, 0.0], [0.0, 0.0]]]])
    E, K = rc.energy(P, V, np.ones((1, 2)))
    check("energy = mean(0.5 v^2 - g x)", abs(E[0, 0] - ((2.0 - 9.0) + (-27.0)) / 2) < 1e-12 and abs(K[0, 0] - 1.0) < 1e-12, str(E))


def test_similarity_not_conservation():
    t = np.arange(101)[None]
    Et = 5.0 * np.exp(-0.01 * t)
    K0 = np.array([5.0])
    tracks = rc.track_err(Et * 1.0, Et, K0)
    conserve = rc.track_err(np.full_like(Et, 5.0), Et, K0)
    check("model tracking a decaying truth energy scores ~0", tracks.max() < 1e-12, f"{tracks.max():.2e}")
    check("conserving model is penalised against a dissipative truth", conserve[0, 100] > 0.3, f"{conserve[0, 100]:.3f}")
    check("non-finite model energy scores 1e6", rc.track_err(np.array([[np.nan, 1.0]]), np.array([[1.0, 1.0]]), K0)[0, 0] == 1e6)


def test_apex():
    x = np.concatenate([np.linspace(0.0, 6.0, 31), np.linspace(6.0, 0.0, 31)[1:]])
    check("elastic bounce apex ratio ~1", abs(rc.apex_ratio(x, 6.0, 0.0) - 1.0) < 1e-6, str(rc.apex_ratio(x, 6.0, 0.0)))
    check("no reversal -> None", rc.apex_ratio(np.linspace(0.0, 6.0, 40), 6.0, 0.0) is None)


def test_truth_vs_truth_and_frozen():
    box = [sc.hbox((-7, -7), (7, 7))]
    orig = cs.roll

    def truth_roll(model, scenes, steps, dev_):
        Ps, Vs = zip(*[cs.truth(s, steps, dev_) for s in scenes])
        n = max(len(s["mass"]) for s in scenes)
        mask = np.zeros((len(scenes), n))
        for i, s in enumerate(scenes):
            mask[i, : len(s["mass"])] = 1
        return ss.pad_traj(Ps, n), ss.pad_traj(Vs, n), mask

    def frozen_roll(model, scenes, steps, dev_):
        Pm, Vm, mask = truth_roll(model, scenes, steps, dev_)
        return np.repeat(Pm[:, :1], steps + 1, 1), np.zeros_like(Vm), mask

    try:
        cs.roll = truth_roll
        rows = rc.bounce_rows(None, dev, box, 7.0, (6, 10))
        check("truth as model: restitution ratio exactly 1 and gain 0", all(abs(r["e_model"] / r["e_truth"] - 1) < 1e-9 and abs(r["gain"]) < 1e-9 for r in rows.values()), str(rows))
        dev_, miss = rc.rest_summary(rows)
        check("rest_summary truth-vs-truth", dev_ < 1e-9 and miss == 0)
        cs.roll = frozen_roll
        rows = rc.bounce_rows(None, dev, box, 7.0, (6, 10))
        dev_, miss = rc.rest_summary(rows)
        check("static (frozen-gas) model: no rebound counted as miss", miss == 1.0 and dev_ == float("inf"), str(rows))
    finally:
        cs.roll = orig


def test_score_and_gating():
    good = {"e_track@20": 0.05, "e_track@100": 0.1, "rest_dev": 0.1, "rest_miss": 0.0, "gain": 0.0, "runaway": 0.0, "offgrid": 0.0, "e_floor@20": 0.0, "e_floor@100": 0.0,
            "rest_dev_wall": 0.1, "rest_dev_circle": 0.1}
    check("score all PASS", all(r[3] == "PASS" for r in rc.score(good)))
    bad = dict(good, gain=0.5, runaway=float("nan"))
    check("score FAIL on gain and non-finite runaway", [r[0] for r in rc.score(bad) if r[3] == "FAIL"] == ["gain", "runaway"])
    meta = {"ckpt": "x", "git": "g"}
    cand = {"meta": meta, "contact_energy": {"summary": good}}
    buf = io.StringIO()
    with redirect_stdout(buf):
        r_ok = sr.print_contact_energy({}, cand, True)
        r_nogate = sr.print_contact_energy({}, cand, False)
    out = buf.getvalue()
    check("contact rows counted when gate passes, not counted when it fails", r_ok and not r_nogate and "NOT COUNTED" in out and "CONTACT OVERALL PASS" in out)
    check("no contact_energy block -> no effect", sr.print_contact_energy({}, {"meta": meta}, True) is True)


def test_replicates_and_exit_policy():
    reps = [{"model": {"dE": 0.006, "frac_ge_c": None}, "vir_gap": 0.1}, {"model": {"dE": 0.019, "frac_ge_c": None}, "vir_gap": 0.3}, {"model": {"dE": 0.017, "frac_ge_c": None}, "vir_gap": 0.2}]
    med, sp = sr.median_reps(reps), sr.spread_reps(reps)
    check("median_reps takes the leaf-wise median, keeps None", abs(med["model"]["dE"] - 0.017) < 1e-12 and abs(med["vir_gap"] - 0.2) < 1e-12 and med["model"]["frac_ge_c"] is None, str(med))
    check("spread_reps records [min, max]", sp["model"]["dE"] == [0.006, 0.019] and sp["vir_gap"] == [0.1, 0.3], str(sp))
    ns = lambda **k: type("A", (), {"require_contact": False, "no_require_contact": False, **k})()
    check("contact ckpt gates exit by default", sr.contact_gates_exit(ns(), {"contact": True}))
    check("gravity-only run unaffected", not sr.contact_gates_exit(ns(), {"contact": False}) and not sr.contact_gates_exit(ns(), {}))
    check("--no-require-contact overrides", not sr.contact_gates_exit(ns(no_require_contact=True), {"contact": True}))
    check("--require-contact forces it on", sr.contact_gates_exit(ns(require_contact=True), {"contact": False}))
    check("multi-ball gain is report-only (not in THRESH or score)", "gain_multi@50" not in rc.THRESH and all(r[0] in rc.THRESH for r in rc.score({**dict.fromkeys(rc.THRESH, 0.0), "gain_multi@50": 9e9})))
    K0 = np.array([2.0, 2.0])
    Et = np.full((2, 101), 2.0)
    Em = Et.copy()
    Em[0, 10] = 3.0
    Em[1, 60] = 9.0
    g = rc.gain_multi(Em, Et, K0)
    check("gain_multi: gain within t<=50 counted (0.5), after 50 and losses ignored", abs(g[0] - 0.5) < 1e-12 and g[1] == 0.0, str(g))
    check("gain_multi: non-finite model energy = 1e6", rc.gain_multi(np.full((1, 101), np.nan), np.ones((1, 101)), np.ones(1))[0] == 1e6)


def test_baseline_compare_unchanged():
    if not os.path.exists(BASE):
        print("SKIP baseline compare (no", BASE, ")")
        return
    import subprocess

    p = subprocess.run([sys.executable, "scripts/scatter_regress.py", "compare", "--baseline", BASE, "--candidate", BASE], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    check("compare baseline vs itself (no contact block) still OVERALL PASS, exit 0", p.returncode == 0 and "OVERALL PASS (geometry-free gate)" in p.stdout and "CONTACT OVERALL" not in p.stdout, p.stdout[-200:])
    json.load(open(BASE))


if __name__ == "__main__":
    test_energy()
    test_similarity_not_conservation()
    test_apex()
    test_truth_vs_truth_and_frozen()
    test_score_and_gating()
    test_replicates_and_exit_policy()
    test_baseline_compare_unchanged()
    print("ALL PASS" if all(OK) else "SOME FAILED", len(OK), "checks")
    sys.exit(0 if all(OK) else 1)
