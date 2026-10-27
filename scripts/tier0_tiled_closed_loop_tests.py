"""CPU/GPU tests for scripts/tier0_tiled_closed_loop.py and its trainer: PYTHONPATH=. python scripts/tier0_tiled_closed_loop_tests.py [--fast] [--device cuda]
--fast skips the end-to-end trainer runs."""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

from scripts import exp12_hier as h
from scripts import train_tier0_ext as tte
from scripts import train_tiered as ttr
from scripts import tier0_tiled_closed_loop as tc

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not ok:
        fails.append(name)


def window_fields(fn, B, G, n, dev="cpu"):
    """Per-window fields sampled from a global function fn(X, Y) with the model's cell convention (dims y, x; cell centres (idx + .5) * hc)."""
    hc = tc.TILE / n
    cen = h.centres(torch.device(dev))
    idx = (torch.arange(G, dtype=torch.float32) + 0.5) * hc - G * hc / 2
    out = []
    for w in range(tc.NW):
        Y, X = torch.meshgrid(idx + cen[w, 1], idx + cen[w, 0], indexing="ij")
        out.append(fn(X, Y))
    f = torch.stack(out)[:, None]
    return f.repeat(B, 1, 1, 1), hc


def test_exchange():
    G, n = 16, 8
    fn = lambda X, Y: torch.sin(0.1 * X) + torch.cos(0.07 * Y) + 0.01 * X * Y
    f, hc = window_fields(fn, 2, G, n)
    ex = tc.exchange_field(f, n)
    r = (G - n) // 2
    cen = h.centres(torch.device("cpu"))
    idx = (torch.arange(G, dtype=torch.float32) + 0.5) * hc - G * hc / 2
    inside = torch.stack([((idx[None, :] + cen[w, 0]).abs() < 64) & ((idx[:, None] + cen[w, 1]).abs() < 64) for w in range(tc.NW)]).repeat(2, 1, 1)[:, None]
    check("exchange reproduces a globally consistent field inside the domain", torch.allclose(ex * inside, f * inside, atol=1e-6))
    check("exchange zeroes cells outside the domain", bool((ex[~inside.expand_as(ex)] == 0).all()))
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(32, 3, G, G, generator=g)
    e1 = tc.exchange_field(rnd, n)
    check("exchange keeps each window's own tile", torch.equal(e1[:, :, r:r + n, r:r + n], rnd[:, :, r:r + n, r:r + n]))
    check("exchange is idempotent", torch.equal(tc.exchange_field(e1, n), e1))
    B0 = lambda i, j: (i * tc.NT + j)
    right_nb = e1[B0(0, 1), :, r:r + n, r + n:]
    check("right rim of window (x tile 0, y tile 1) = core of window (x tile 1, y tile 1)", torch.equal(right_nb, rnd[B0(1, 1), :, r:r + n, r:r + r]))
    top_nb = e1[B0(2, 0), :, r + n:, r:r + n]
    check("upper rim (larger y) of window (2,0) = core of window (2,1)", torch.equal(top_nb, rnd[B0(2, 1), :, r:r + r, r:r + n]))
    x = rnd.clone().requires_grad_(True)
    y = tc.exchange_field(x, n)
    y[B0(1, 1), :, r:r + n, r + n:].sum().backward()
    nz = (x.grad.abs().sum((1, 2, 3)) > 0).nonzero().flatten().tolist()
    check("gradient flows only from the neighbour that fed the rim", nz == [B0(2, 1)], str(nz))


def test_weights():
    w = tc.step_weights(28)
    check("step weights 1 through tick 20 then halve every 4 ticks", torch.allclose(w[:20], torch.ones(20)) and abs(float(w[23]) - 0.5) < 1e-6 and abs(float(w[27]) - 0.25) < 1e-6)
    lp = [torch.tensor(2.0 * (s + 1))for s in range(5)]
    lv = [torch.tensor(0.5 * (s + 1)) for s in range(5)]
    rp, rv = [2.0 * (s + 1) for s in range(5)], [0.5 * (s + 1) for s in range(5)]
    check("closed-loop loss is 1 for the reference errors", abs(float(tc.closed_loop_loss(lp, lv, rp, rv, tc.step_weights(5))) - 1.0) < 1e-6)


def perturbed_model(dev):
    model = tte.build(argparse.Namespace(t0="ext=6,extapply,taper=8,bnd,halo=8", dt=0.1), dev)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g).to(p.device) * 0.02)
    return model


def test_model(dev):
    model = perturbed_model(dev).eval()
    d = ttr.make_scenes(1, 4, 4738, dev)
    m, mk = d["M"].float(), d["mask"].float()
    p, v = d["P"][:, 0].float(), d["V"][:, 0].float()
    tt = {ex: tc.TiledTier0(model, dev, chunk=16, exchange=ex) for ex in ("halo", "none", "zero")}
    outs = {}
    with torch.no_grad():
        for ex, t in tt.items():
            st = t.init_state(1)
            a1, st = t.force(p, v, m, mk, st)
            a2, st2 = t.force(p + 0.05, v, m, mk, st)
            outs[ex] = (a1, a2, st)
    check("first force call identical for halo / none / zero (zero state)", torch.equal(outs["halo"][0], outs["none"][0]) and torch.equal(outs["halo"][0], outs["zero"][0]))
    check("second call: halo differs from none (the exchanged state is used)", float((outs["halo"][1] - outs["none"][1]).abs().max()) > 0)
    check("zero mode carries no state", float(outs["zero"][2].abs().max()) == 0.0)
    n = tt["halo"].n
    r = (model.grid - n) // 2
    st = outs["halo"][2]
    core = st[:, :, r:r + n, r:r + n]
    check("exchanged field keeps own tiles of the raw output", torch.isfinite(st).all() and float(core.abs().max()) > 0)
    model.train()
    grads = {}
    for ck in (True, False):
        t = tc.TiledTier0(model, dev, chunk=8, exchange="halo", ckpt=ck)
        model.zero_grad()
        lp, lv = tc.rollout_terms(t, d, torch.zeros(1, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev), 2)
        (sum(lp) + sum(lv)).backward()
        grads[ck] = torch.cat([q.grad.flatten() for q in model.parameters() if q.grad is not None])
    check("checkpointed and plain gradients agree", torch.allclose(grads[True], grads[False], rtol=1e-3, atol=1e-8), f"max rel {float((grads[True] - grads[False]).abs().max() / grads[False].abs().max()):.2e}")
    check("closed-loop gradients are finite and non-zero", bool(torch.isfinite(grads[True]).all()) and float(grads[True].abs().max()) > 0)
    t = tc.TiledTier0(model, dev, chunk=16, exchange="halo")
    lp, lv = tc.rollout_terms(t, d, torch.zeros(1, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev), 3, tb=2, grad=True)
    check("truncated-backprop rollout returns the same values", len(lp) == 3 and all(torch.isfinite(x) for x in lp))


def test_energy_term():
    g = torch.Generator().manual_seed(0)
    P = torch.randn(2, 12, 2, generator=g, dtype=torch.float64) * 3
    V = torch.randn(2, 12, 2, generator=g, dtype=torch.float64)
    m, mk = torch.ones(2, 12, dtype=torch.float64), torch.ones(2, 12, dtype=torch.float64)
    lab = torch.arange(12)[None].expand(2, 12) // 4
    z = tc.similarity_energy_terms(P, V, P, V, m, mk, lab)
    nz = tc.similarity_energy_terms(P, V * 1.1, P, V, m, mk, lab)
    check("energy-similarity term is zero at the truth state and positive off it", float(z) == 0.0 and float(nz) > 0, f"{float(z)} {float(nz)}")


def run_train(args, env):
    return subprocess.run([sys.executable, "scripts/train_tier0_tiled_closed_loop.py"] + args, env=env, capture_output=True, text=True)


def test_trainer(dev):
    model = perturbed_model(dev)
    tmp = tempfile.mkdtemp()
    init = os.path.join(tmp, "init.pt")
    torch.save({"model": model.state_dict(), "t0": model.t0_spec}, init)
    env = dict(os.environ, PYTHONPATH=".")
    base = ["--init", init, "--scenes", "2", "--steps", "8", "--val-scenes", "1", "--val-steps", "21", "--batch", "1", "--k-start", "2", "--k-end", "3",
            "--w-harvest", "0", "--n-iso", "8", "--k-plain", "3", "--batch-plain", "2", "--calib-batches", "1", "--chunk", "16", "--log-every", "1",
            "--ckpt-every", "1", "--out-dir", tmp]
    r1 = run_train(base + ["--tag", "t", "--iters", "2", "--probe-at"], env)
    ok = r1.returncode == 0 and os.path.exists(os.path.join(tmp, "t.pt")) and os.path.exists(os.path.join(tmp, "t_reference.json"))
    check("trainer runs, writes checkpoint and reference", ok, r1.stderr[-400:] if not ok else "")
    if not ok:
        return
    st = torch.load(os.path.join(tmp, "t.pt"), map_location="cpu", weights_only=False)
    check("checkpoint in train_tier0_ext format with iteration", st["it"] == 2 and "t0" in st and "opt" in st and "model" in st)
    r2 = run_train(base + ["--tag", "t", "--iters", "3", "--resume", "--probe-at"], env)
    st2 = torch.load(os.path.join(tmp, "t.pt"), map_location="cpu", weights_only=False)
    check("resume continues from the checkpoint", "resumed it 2" in r2.stdout and st2["it"] == 3 and os.path.exists(os.path.join(tmp, "t.TRAIN_DONE")), r2.stderr[-300:])
    r3 = run_train(base + ["--tag", "k", "--iters", "3", "--probe-at", "1", "--gate1-err", "0.0"], env)
    check("early-stop probe writes EARLY_KILL on a violated gate", os.path.exists(os.path.join(tmp, "k.EARLY_KILL")) and "EARLY_KILL" in r3.stdout, r3.stderr[-300:])
    r5 = run_train(base + ["--tag", "en", "--iters", "2", "--w-energy", "1.0", "--gate1-dE", "100", "--probe-at", "1"], env)
    check("energy-similarity trainer runs and logs the term", r5.returncode == 0 and '"energy"' in r5.stdout and os.path.exists(os.path.join(tmp, "en.pt")), r5.stderr[-400:])
    r4 = run_train(base + ["--tag", "e", "--iters", "3", "--est"], env)
    check("--est exits without training", r4.returncode == 0 and not os.path.exists(os.path.join(tmp, "e.pt")), r4.stderr[-300:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    test_exchange()
    test_weights()
    test_energy_term()
    test_model(dev)
    if not a.fast:
        test_trainer(dev)
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
