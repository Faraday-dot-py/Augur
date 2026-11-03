import argparse
import json

import torch

from scripts import train_tier0_harvest_loss_terms as tl
from scripts import train_tier0_ext as tte


def fake_windows(S=4, T=8, N=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(S, T, N, 2, generator=g) - 0.5) * 20
    vel = torch.randn(S, T, N, 2, generator=g) * 0.3
    return {"pos": pos, "vel": vel, "mask_f": torch.ones(S, T, N), "mass": torch.ones(S, N), "a0": torch.zeros(S, T, 2),
            "J": torch.zeros(S, T, 2, 2), "Q": torch.zeros(S, T, 2, 2, 2), "C": torch.zeros(S, T, 2, 2, 2, 2)}


def run_term(tmp_path, work, coh):
    ns = argparse.Namespace(t0="ext=6,extapply,taper=8,bnd,halo=8", dt=0.1)
    model = tte.build(ns, torch.device("cpu"))
    d = fake_windows()
    tl.CFG.update(work=work, coh=coh, share=0.1, k_end=3, seed=4738, log_every=1000, wfile=str(tmp_path / "w.json"))
    tl._STATE.update(w=None, calls=0)
    idx = torch.arange(2)
    t0 = torch.zeros(2, dtype=torch.long)
    loss = tl.rollout_loss(model, d, idx, t0, 2)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
    info = json.load(open(tmp_path / "w.json"))
    assert info["weight"] > 0
    with torch.no_grad():
        lp, lv, _, _ = tl.rollout(model, d, idx, t0, 2)
    assert float(loss) > float(lp + 0.1 * lv)


def test_work_bias_smoke(tmp_path):
    run_term(tmp_path, True, False)


def test_coherent_smoke(tmp_path):
    run_term(tmp_path, False, True)
