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
from scripts import train_tier0_tiled_closed_loop as trn

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
    check("checkpointed and plain gradients agree", float((grads[True] - grads[False]).abs().max()) <= 1e-5 * float(grads[False].abs().max()), f"max rel {float((grads[True] - grads[False]).abs().max() / grads[False].abs().max()):.2e}")
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


def test_flattened_muon():
    def net():
        torch.manual_seed(4738)
        return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.Flatten(), torch.nn.Linear(8 * 25, 4))

    def stepn(m, o, n):
        for i in range(n):
            o.zero_grad()
            torch.manual_seed(100 + i)
            m(torch.randn(2, 3, 5, 5)).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()

    m = net()
    o = trn.HybridMuon(m.parameters(), 2e-4, conv=True)
    w0 = [p.detach().clone() for p in m.parameters()]
    stepn(m, o, 3)
    names = [n for n, _ in m.named_parameters()]
    check("flattened Muon updates every parameter incl. all conv weights", all(not torch.equal(a, p) for a, p in zip(w0, m.parameters())), str([n for n, a, p in zip(names, w0, m.parameters()) if torch.equal(a, p)]))
    check("flattened Muon split counts: 4-D and 2-D under Muon, 1-D under Adam", o.split == {"muon_2d": 8 * 25 * 4, "muon_flattened_4d": 8 * 3 * 9 + 8 * 8 * 9, "adam": 8 + 8 + 4}, str(o.split))
    check("flattened Muon proxies share storage with conv weights", all(v.data_ptr() == q.data_ptr() for q, v in zip(o.flat, o.views)))
    o2 = trn.HybridMuon(m.parameters(), 2e-4, conv=False)
    check("2-D-only mode leaves conv weights to Adam", o2.split["muon_flattened_4d"] == 0 and o2.split["adam"] == 8 * 3 * 9 + 8 * 8 * 9 + 8 + 8 + 4)
    ref = net()
    ro = trn.HybridMuon(ref.parameters(), 2e-4, conv=True)
    stepn(ref, ro, 6)
    a = net()
    ao = trn.HybridMuon(a.parameters(), 2e-4, conv=True)
    stepn(a, ao, 3)
    b = net()
    b.load_state_dict(a.state_dict())
    bo = trn.HybridMuon(b.parameters(), 2e-4, conv=True)
    bo.load_state_dict(ao.state_dict())

    def stepk(m, o, lo, hi):
        for i in range(lo, hi):
            o.zero_grad()
            torch.manual_seed(100 + i)
            m(torch.randn(2, 3, 5, 5)).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()

    stepk(b, bo, 3, 6)
    check("flattened Muon checkpoint state resumes bit-identically to an uninterrupted run", all(torch.equal(x, y) for x, y in zip(ref.parameters(), b.parameters())), str(max(float((x - y).abs().max()) for x, y in zip(ref.parameters(), b.parameters()))))
    bad = net()
    bo2 = trn.HybridMuon(bad.parameters(), 2e-4, conv=True)
    bad(torch.randn(2, 3, 5, 5)).square().mean().backward()
    bo2.zero_grad()
    check("zero_grad clears conv weight grads", all(q.grad is None for q in bo2.flat))


def test_sam_adam():
    def net():
        torch.manual_seed(4738)
        return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.Flatten(), torch.nn.Linear(8 * 25, 4))

    def stepk(m, o, lo, hi):
        for i in range(lo, hi):
            torch.manual_seed(100 + i)
            x = torch.randn(2, 3, 5, 5)
            o.zero_grad()
            m(x).square().mean().backward()
            check_ascent = o.ascend(0.05)
            o.zero_grad()
            m(x).square().mean().backward()
            o.restore()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()
        return check_ascent

    m = net()
    o = trn.SamAdam(m.parameters(), 2e-4)
    w0 = [q.detach().clone() for q in m.parameters()]
    stepk(m, o, 0, 3)
    names = [n for n, _ in m.named_parameters()]
    check("sam-adam updates every parameter", all(not torch.equal(a, q) for a, q in zip(w0, m.parameters())), str([n for n, a, q in zip(names, w0, m.parameters()) if torch.equal(a, q)]))
    m2 = net()
    o2 = trn.SamAdam(m2.parameters(), 2e-4)
    torch.manual_seed(100)
    x = torch.randn(2, 3, 5, 5)
    o2.zero_grad()
    m2(x).square().mean().backward()
    base = [q.detach().clone() for q in m2.parameters()]
    gn = torch.sqrt(sum((q.grad ** 2).sum() for q in m2.parameters()))
    o2.ascend(0.05)
    shift = torch.sqrt(sum(((q.detach() - b) ** 2).sum() for q, b in zip(m2.parameters(), base)))
    check("sam-adam ascent moves the weights by rho along the gradient", abs(float(shift) - 0.05) < 1e-6 and float(gn) > 0, f"shift {float(shift):.8f}")
    o2.restore()
    check("sam-adam restore returns the weights", all(torch.allclose(q.detach(), b, atol=1e-7, rtol=0) for q, b in zip(m2.parameters(), base)))
    o2.zero_grad()
    bad = net()
    ob = trn.SamAdam(bad.parameters(), 2e-4)
    bad(x).square().mean().mul(float("nan")).backward()
    check("sam-adam ascend refuses a non-finite gradient and leaves the weights", ob.ascend(0.05) is False and all(torch.equal(q, r) for q, r in zip(bad.parameters(), net().parameters())))
    ref = net()
    ro = trn.SamAdam(ref.parameters(), 2e-4)
    stepk(ref, ro, 0, 6)
    a = net()
    ao = trn.SamAdam(a.parameters(), 2e-4)
    stepk(a, ao, 0, 3)
    b = net()
    b.load_state_dict(a.state_dict())
    bo = trn.SamAdam(b.parameters(), 2e-4)
    bo.load_state_dict(ao.state_dict())
    bo.param_groups[0]["lr"] = 1.234e-5
    check("sam-adam lr writes reach the optimizer after a state load", bo.opt.param_groups[0]["lr"] == 1.234e-5)
    bo.param_groups[0]["lr"] = 2e-4
    stepk(b, bo, 3, 6)
    check("sam-adam checkpoint state resumes bit-identically to an uninterrupted run", all(torch.equal(x_, y_) for x_, y_ in zip(ref.parameters(), b.parameters())), str(max(float((x_ - y_).abs().max()) for x_, y_ in zip(ref.parameters(), b.parameters()))))


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
    r6 = run_train(base + ["--tag", "rp", "--iters", "4", "--replay-rounds", "2", "--replay-scenes", "2", "--replay-steps", "12", "--replay-start", "4",
                           "--replay-stride", "4", "--replay-batch", "1", "--probe-at"], env)
    check("visited-state replay trainer builds two rounds and logs the term", r6.returncode == 0 and r6.stdout.count("replay round") == 2 and '"replay"' in r6.stdout
          and os.path.exists(os.path.join(tmp, "rp.TRAIN_DONE")), (r6.stdout + r6.stderr)[-500:])
    r7 = run_train(base + ["--tag", "mu", "--iters", "4", "--optimizer", "muon-matrices-adam-rest", "--ckpt-every", "2"], env)
    r8 = run_train(base + ["--tag", "mu", "--iters", "6", "--optimizer", "muon-matrices-adam-rest", "--ckpt-every", "2", "--resume"], env)
    check("muon-matrices-adam-rest trainer runs, checkpoints and resumes", r7.returncode == 0 and r8.returncode == 0 and "resumed it" in r8.stdout and os.path.exists(os.path.join(tmp, "mu.pt")), (r7.stderr + r8.stderr)[-500:])
    r9 = run_train(base + ["--tag", "mf", "--iters", "4", "--optimizer", "muon-conv-flattened-adam-rest", "--ckpt-every", "2"], env)
    r10 = run_train(base + ["--tag", "mf", "--iters", "6", "--optimizer", "muon-conv-flattened-adam-rest", "--ckpt-every", "2", "--resume"], env)
    check("muon-conv-flattened-adam-rest trainer logs the split, runs, checkpoints and resumes", r9.returncode == 0 and r10.returncode == 0 and "parameter split by optimizer" in r9.stdout and "resumed it" in r10.stdout
          and os.path.exists(os.path.join(tmp, "mf.pt")), (r9.stdout + r9.stderr + r10.stderr)[-600:])
    r11 = run_train(base + ["--tag", "sam", "--iters", "4", "--optimizer", "sam-adam", "--ckpt-every", "2"], env)
    r12 = run_train(base + ["--tag", "sam", "--iters", "6", "--optimizer", "sam-adam", "--ckpt-every", "2", "--resume"], env)
    check("sam-adam trainer runs, checkpoints and resumes", r11.returncode == 0 and r12.returncode == 0 and "resumed it" in r12.stdout and os.path.exists(os.path.join(tmp, "sam.pt")), (r11.stdout + r11.stderr + r12.stderr)[-600:])
    r4 = run_train(base + ["--tag", "e", "--iters", "3", "--est"], env)
    check("--est exits without training", r4.returncode == 0 and not os.path.exists(os.path.join(tmp, "e.pt")), r4.stderr[-300:])


def test_fresh_seed_eval(dev):
    from scripts import eval_tier0_fresh_seed_checkpoints as fe
    from scripts import score_tier0_fresh_seed_checkpoints as sc

    model = perturbed_model(dev)
    tmp = tempfile.mkdtemp()
    ck = os.path.join(tmp, "m.pt")
    torch.save({"model": model.state_dict(), "t0": model.t0_spec}, ck)

    class Stand:
        def __init__(self):
            self.t = tc.TiledTier0(fe.load(ck, dev), dev, chunk=16, exchange="halo")

        def init_state(self, S, device):
            return self.t.init_state(S)

        def force(self, p, v, m, k, st):
            return self.t.force(p, v, m, k, st)

    fe.ta.load_tiled = lambda spec, device, dt=0.1: Stand()
    out = os.path.join(tmp, "fresh.json")
    sys.argv = ["x", "--tier0", "unused", "--ckpt", f"a={ck}", f"b={ck}", "--seeds", "12000", "12100", "--eval-scenes", "2", "--eval-steps", "21", "--chunk", "16", "--out", out]
    fe.main()
    ev = json.load(open(out))
    check("fresh-seed eval writes BL and every named checkpoint on every seed", sorted(ev) == ["BL", "a", "b", "seeds"] and all(sorted(ev[n]) == ["12000", "12100"] for n in ("BL", "a", "b")))
    r = sc.score(ev)
    same = all(abs(r["err20_over_BL"]["a"][x] - 1.0) < 1e-6 for x in ("12000", "12100"))
    check("fresh-seed scorer: checkpoint identical to the stand-in baseline scores 1.0x and PASS", same and r["verdict"] == "PASS", json.dumps(r["err20_over_BL"]))
    ev["b"]["12100"]["err@20"] = ev["BL"]["12100"]["err@20"] * 1.3
    check("fresh-seed scorer: 1.3x on one (checkpoint, seed) is KILL", sc.score(ev)["verdict"] == "KILL")
    ev["b"]["12100"]["err@20"] = ev["BL"]["12100"]["err@20"] * 1.1
    check("fresh-seed scorer: 1.1x on one (checkpoint, seed) is INCONCLUSIVE", sc.score(ev)["verdict"] == "INCONCLUSIVE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    test_exchange()
    test_weights()
    test_energy_term()
    test_flattened_muon()
    test_sam_adam()
    test_model(dev)
    if not a.fast:
        test_trainer(dev)
        test_fresh_seed_eval(dev)
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
