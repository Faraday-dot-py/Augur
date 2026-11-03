"""scripts/train_tier0_harvest.py with one extra soft loss term on the window rollouts (scripts/loss_terms_local.py). Same flags plus
--work-bias (design 1) or --coherent (design 2); both off = control. The term weight is set once so that weight x term = --term-share (default 0.1)
of the position loss lp of the initial model at k_end, then written to <out-dir>/<tag>_termweight.json and reused on resume (recorded, not swept).
Usage: PYTHONPATH=. python scripts/train_tier0_harvest_loss_terms.py --work-bias <th flags>"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from scripts import loss_terms_local as lt
from scripts import scatter_field as sf
from scripts import train_tier0_harvest as th

CFG = {}
_STATE = {"calls": 0, "w": None, "last": {}}


def rollout(model, d, idx, t0, k, with_ext=True, terms=False):
    mk = d["mask_f"][idx, t0].float()
    m = d["mass"][idx].float()
    pos, vel = d["pos"][idx, t0], d["vel"][idx, t0]
    v0 = vel
    field = model.init_field(len(idx), pos.device)
    lp = lv = work = coh = 0.0
    denom = mk.sum() * 2 * k
    bins = lt.density_bins(pos, mk) if terms and CFG["work"] else None
    for s in range(k):
        tt = t0 + s
        if with_ext:
            th.set_ext(model, d, idx, tt)
        else:
            model.t0.ext_grid = model.t0.ext_poly = None
        pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
        w2 = mk * d["mask_f"][idx, t0 + s + 1].float()
        w = w2[:, :, None]
        pt, vt = d["pos"][idx, t0 + s + 1], d["vel"][idx, t0 + s + 1]
        lp = lp + (((pos - pt) ** 2) * w).sum() / denom
        lv = lv + (((vel - vt) ** 2) * w).sum() / denom
        if terms and CFG["work"]:
            work = work + lt.work_bias_step(vel, vt, v0, m, w2, bins) / k
        if terms and CFG["coh"]:
            coh = coh + lt.coherent_step(pos, pt, vel, vt, w2) / k
    return lp, lv, work, coh


@torch.no_grad()
def calibrate(model, d, k):
    g = torch.Generator(device="cpu").manual_seed(CFG["seed"])
    S, T = d["pos"].shape[:2]
    lps, tms = [], []
    for _ in range(8):
        idx = torch.randint(S, (16,), generator=g).to(d["pos"].device)
        t0 = torch.randint(0, T - k, (16,), generator=g).to(d["pos"].device)
        lp, lv, work, coh = rollout(model, d, idx, t0, k, terms=True)
        lps.append(float(lp))
        tms.append(float(work if CFG["work"] else coh))
    w = CFG["share"] * float(np.mean(lps)) / max(float(np.mean(tms)), 1e-20)
    info = {"weight": w, "k": k, "mean_lp": float(np.mean(lps)), "mean_term": float(np.mean(tms)), "share": CFG["share"],
            "term": "work_bias" if CFG["work"] else "coherent"}
    json.dump(info, open(CFG["wfile"], "w"))
    print(f"term weight calibrated: {json.dumps(info)}", flush=True)
    return w


def rollout_loss(model, d, idx, t0, k, with_ext=True):
    if not torch.is_grad_enabled() or not (CFG["work"] or CFG["coh"]):
        lp, lv, _, _ = rollout(model, d, idx, t0, k, with_ext)
        return lp + 0.1 * lv
    if _STATE["w"] is None:
        if os.path.exists(CFG["wfile"]):
            _STATE["w"] = json.load(open(CFG["wfile"]))["weight"]
            print(f"term weight from file: {_STATE['w']:.4g}", flush=True)
        else:
            _STATE["w"] = calibrate(model, d, min(CFG["k_end"], d["pos"].shape[1] - 1))
    lp, lv, work, coh = rollout(model, d, idx, t0, k, with_ext, terms=True)
    term = work if CFG["work"] else coh
    _STATE["calls"] += 1
    if _STATE["calls"] % CFG["log_every"] == 0:
        print(f"terms lp {float(lp):.5f} lv {float(lv):.5f} {'work_bias' if CFG['work'] else 'coherent'} {float(term):.5f} weighted share {_STATE['w'] * float(term) / max(float(lp), 1e-12):.3f}", flush=True)
    return lp + 0.1 * lv + _STATE["w"] * term


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--work-bias", action="store_true")
    ap.add_argument("--coherent", action="store_true")
    ap.add_argument("--term-share", type=float, default=0.1)
    ap.add_argument("--out-dir", default="results/tier0_ext")
    ap.add_argument("--tag", default="H1")
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--log-every", type=int, default=100)
    a, rest = ap.parse_known_args()
    assert not (a.work_bias and a.coherent), "one term per arm"
    CFG.update(work=a.work_bias, coh=a.coherent, share=a.term_share, k_end=a.k_end, seed=a.seed, log_every=a.log_every,
               wfile=os.path.join(a.out_dir, a.tag + "_termweight.json"))
    os.makedirs(a.out_dir, exist_ok=True)
    sys.argv = [sys.argv[0]] + rest + ["--out-dir", a.out_dir, "--tag", a.tag, "--k-end", str(a.k_end), "--seed", str(a.seed), "--log-every", str(a.log_every)]
    th.rollout_loss = rollout_loss
    th.main()


if __name__ == "__main__":
    main()
