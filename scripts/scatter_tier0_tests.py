"""Tests for the opt-in tier-0 extensions (scripts/scatter_tier0.py). Run on a GPU node:
PYTHONPATH=. python scripts/scatter_tier0_tests.py [--ckpt <scatter ckpt>] [--device cuda]

Regression reference (flags off must reproduce the pre-change ScatterField): results/tier0_regress_ref.pt was written by
PYTHONPATH=. python scripts/scatter_tier0_tests.py --make-ref --orig <pre-change scatter_field.py> [--ckpt <ckpt>]
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m
from scripts import train_scatter_field as tsf

REF = "results/tier0_regress_ref.pt"
CONFIGS = {
    "local_rec_h2": dict(grid=32, extent=32.0, net="local", recurrent=True, hidden_ch=2, width=8, layers=3),
    "local_pot_verlet": dict(grid=32, extent=32.0, net="local", potential=True, verlet=True, width=8, layers=3),
    "unet_kp_pot_v": dict(grid=64, extent=64.0, net="unet", potential=True, verlet=True, kernel=True, pp=2.0, split=True, width=8),
}
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not ok:
        fails.append(name)


def perturb(model, seed=4738, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_((scale * torch.randn(p.shape, generator=g)).to(p.device))
    return model


def scene(extent, B=2, N=40, seed=4738, dev="cpu", frac=0.6):
    g = torch.Generator().manual_seed(seed)
    pos = ((torch.rand(B, N, 2, generator=g) - 0.5) * extent * frac).to(dev)
    vel = torch.randn(B, N, 2, generator=g).to(dev)
    mass = (0.5 + torch.rand(B, N, generator=g)).to(dev)
    mask = torch.ones(B, N).to(dev)
    mask[:, -3:] = 0.0
    return pos, vel, mass, mask


def run_steps(model, pos, vel, mass, mask, n=3):
    st = model.init_field(pos.shape[0], pos.device)
    outs = []
    for _ in range(n):
        pos, vel, st, _ = model.step(pos, vel, mass, mask, st)
        outs.append((pos, vel))
    return outs


def flat(outs):
    return torch.stack([torch.stack(o) for o in outs]).detach().cpu()


def build_cfg(mod, name):
    torch.manual_seed(4738)
    return perturb(mod.ScatterField(dt=0.1, **CONFIGS[name]))


def make_ref(args):
    spec = importlib.util.spec_from_file_location("sf_orig", args.orig)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    ref = {}
    for name in CONFIGS:
        m = build_cfg(orig, name)
        ref[name] = flat(run_steps(m, *scene(m.extent), n=3))
    if args.ckpt:
        torch.manual_seed(4738)
        m = orig.ScatterField(grid=128, extent=64.0, net="unet", potential=True, verlet=True, kernel=True, pp=2.0, split=True, dt=0.1)
        sf.load_compat(m, torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
        ref["ckpt"] = flat(run_steps(m, *scene(64.0, N=60), n=3))
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    torch.save(ref, REF)
    print("wrote", REF, list(ref))


def regress(args, dev):
    ref = torch.load(REF)
    for name in CONFIGS:
        m = build_cfg(sf, name)
        assert m.t0 is None
        cpu = flat(run_steps(m, *scene(m.extent), n=3))
        check(f"regress {name} cpu bitwise vs pre-change", torch.equal(cpu, ref[name]))
        if dev != "cpu":
            md = build_cfg(sf, name).to(dev)
            g = flat(run_steps(md, *scene(md.extent, dev=dev), n=3))
            d = float((g - ref[name]).abs().max())
            check(f"regress {name} {dev} vs pre-change", d < 2e-3, f"max abs {d:.2e}")
        me = t0m.attach_tier0(build_cfg(sf, name), "")
        e = flat(run_steps(me, *scene(me.extent), n=3))
        check(f"regress {name} t0 attached, all flags off, bitwise", torch.equal(e, ref[name]))
    if "ckpt" in ref and args.ckpt:
        m = sf.ScatterField(grid=128, extent=64.0, net="unet", potential=True, verlet=True, kernel=True, pp=2.0, split=True, dt=0.1)
        sf.load_compat(m, torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
        c = flat(run_steps(m, *scene(64.0, N=60), n=3))
        check("regress real ckpt cpu bitwise vs pre-change", torch.equal(c, ref["ckpt"]))
        m2 = t0m.attach_tier0(sf.ScatterField(grid=128, extent=64.0, net="unet", potential=True, verlet=True, kernel=True, pp=2.0,
                                              split=True, dt=0.1), "moments=2,vdisp,cross,head=16,phid=4,phscatter,mat=2,ext=6,bnd")
        sf.load_compat(m2, torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
        m2.t0.set_material(torch.randn(2, 60, 2))
        m2.t0.set_external(m2, g=torch.randn(2, 2) * 0.1, H=torch.randn(2, 3) * 0.01)
        c2 = flat(run_steps(m2, *scene(64.0, N=60), n=3))
        d = float((c2 - ref["ckpt"]).abs().max())
        check("real ckpt + all extras, load_compat reproduces pre-change at init", d < 1e-4, f"max abs {d:.2e}")


def warm_start(dev):
    kw = CONFIGS["local_rec_h2"]
    base = build_cfg(sf, "local_rec_h2")
    spec = "moments=2,vdisp,cross,head=16,phid=4,phscatter,mat=2,ext=6,bnd"
    m = t0m.attach_tier0(sf.ScatterField(dt=0.1, **kw), spec)
    sf.load_compat(m, base.state_dict())
    m.t0.set_material(torch.randn(2, 40, 2))
    m.t0.set_external(m, phi0=torch.randn(2), g=torch.randn(2, 2), H=torch.randn(2, 3))
    s = scene(32.0)
    a = flat(run_steps(base, *s))
    b = flat(run_steps(m, *s))
    d = float((a - b).abs().max())
    check("warm start: all extras on, zero-padded plain weights reproduce plain model", d < 1e-5, f"max abs {d:.2e} cin {m.net.inp.in_channels} (+{m.t0.cfg.n_extra()})")


def moments_exact():
    m = sf.ScatterField(grid=32, extent=32.0, net="local", dt=0.1, width=8, layers=2).double()
    pos, vel, mass, mask = [t.double() for t in scene(32.0, B=2, N=50)]
    mg = t0m.moment_scatter(m, pos, vel, mass, mask)
    d = mg.data
    h, G = m.h, m.grid
    xs = (torch.arange(G, dtype=torch.float64) + 0.5) * h - m.extent / 2
    XC, YC = torch.meshgrid(xs, xs, indexing="xy")
    mm = mass * mask
    g = lambda n: mg.get(n)[:, 0]
    tot = lambda t: t.sum((-1, -2))
    x, y, vx, vy = pos[..., 0], pos[..., 1], vel[..., 0], vel[..., 1]
    errs = {
        "mass": (tot(g("m")) - mm.sum(1)).abs().max(),
        "px": (tot(g("px")) - (mm * vx).sum(1)).abs().max(),
        "s1x": (tot(h * g("s1x") + g("m") * XC) - (mm * x).sum(1)).abs().max(),
        "s1y": (tot(h * g("s1y") + g("m") * YC) - (mm * y).sum(1)).abs().max(),
        "s2xx": (tot(h * h * g("s2xx") + 2 * h * XC * g("s1x") + XC ** 2 * g("m")) - (mm * x * x).sum(1)).abs().max(),
        "s2xy": (tot(h * h * g("s2xy") + h * XC * g("s1y") + h * YC * g("s1x") + XC * YC * g("m")) - (mm * x * y).sum(1)).abs().max(),
        "vvxy": (tot(g("vvxy")) - (mm * vx * vy).sum(1)).abs().max(),
        "cxy": (tot(h * g("cxy") + XC * g("py")) - (mm * x * vy).sum(1)).abs().max(),
        "cyx": (tot(h * g("cyx") + YC * g("px")) - (mm * y * vx).sum(1)).abs().max(),
    }
    check("moments: global mass/momentum/first/second/cross sums exact (parallel axis)", all(float(v) < 1e-9 for v in errs.values()),
          " ".join(f"{k} {float(v):.1e}" for k, v in errs.items()))
    one = torch.zeros(1, 1, 2, dtype=torch.float64)
    one[0, 0] = torch.tensor([0.37, -1.21])
    mg1 = t0m.moment_scatter(m, one, torch.tensor([[[0.7, -0.3]]], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64))
    mc = mg1.get("m")[0, 0]
    d1 = {n: mg1.get(n)[0, 0] for n in t0m.MOMENT_NAMES}
    vd = (d1["vvxx"] - d1["px"] ** 2 / mc.clamp_min(1e-9)).abs().max()
    s2c = (d1["s2xx"] - d1["s1x"] ** 2 / mc.clamp_min(1e-9)).abs().max()
    check("moments: single particle has zero central second moment and zero velocity dispersion in every cell", float(vd) < 1e-12 and float(s2c) < 1e-12)


def periodic_open_taper():
    kw = dict(grid=32, extent=32.0, net="local", width=8, layers=2, dt=0.1, hidden_ch=0)
    m = t0m.attach_tier0(perturb(sf.ScatterField(**kw)), "boundary=periodic,halo=4")
    L = m.t0.core(m)
    g = torch.Generator().manual_seed(4738)
    pos = (torch.rand(1, 30, 2, generator=g) - 0.5) * L
    pos[0, 0] = torch.tensor([L / 2 - 0.4, 0.2])
    pos[0, 1] = torch.tensor([-L / 2 + 0.3, 0.5])
    vel = torch.randn(1, 30, 2, generator=g)
    mass, mask = torch.ones(1, 30), torch.ones(1, 30)
    a0, _ = m.force(pos, vel, mass, mask, m.init_field(1, "cpu"))
    k = 5 * m.h
    sh = torch.remainder(pos + k + L / 2, L) - L / 2
    a1, _ = m.force(sh, vel, mass, mask, m.init_field(1, "cpu"))
    shy = torch.remainder(pos + torch.tensor([-3 * m.h, 7 * m.h]) + L / 2, L) - L / 2
    a2, _ = m.force(shy, vel, mass, mask, m.init_field(1, "cpu"))
    sc = float(a0.detach().abs().max())
    check("periodic: force invariant to integer-cell shifts with wrap (incl. boundary-straddling pair)",
          float((a0 - a1).abs().max()) < 1e-4 * max(sc, 1e-6) and float((a0 - a2).abs().max()) < 1e-4 * max(sc, 1e-6),
          f"|a| {sc:.2e} diffs {float((a0 - a1).abs().max()):.1e} {float((a0 - a2).abs().max()):.1e}")
    x = m.scatter(*m.t0.geometry(m, pos, mass)[:1], vel, m.t0.geometry(m, pos, mass)[1], mask)
    x = t0m.wrap_halo(x, 4, fold=True)
    tot = float(x[0, 0].sum() * m.h ** 2)
    core = float(x[0, 0, 4:-4, 4:-4].sum() * m.h ** 2)
    check("periodic: wrapped scatter conserves mass in the core", abs(core - 30.0) < 1e-4 and abs(tot - 30.0) > 1e-3, f"core {core:.4f} total-with-halo {tot:.3f}")
    mo = t0m.attach_tier0(perturb(sf.ScatterField(**kw, momfix=False)), "boundary=open,halo=4")
    p2 = torch.cat([pos, torch.tensor([[[L / 2 + 1.5, 0.0]]])], 1)
    v2 = torch.cat([vel, torch.zeros(1, 1, 2)], 1)
    m2, k2 = torch.ones(1, 31), torch.ones(1, 31)
    ao, _ = mo.force(pos, vel, mass, mask, mo.init_field(1, "cpu"))
    ap, _ = mo.force(p2, v2, m2, k2, mo.init_field(1, "cpu"))
    check("open: halo particle neither scatters nor feels force", float((ao - ap[:, :30]).abs().max()) < 1e-6 and float(ap[:, 30].abs().max()) == 0.0)
    mt = t0m.attach_tier0(sf.ScatterField(**kw), "taper=4")
    edge = torch.tensor([[[15.95, 0.0], [0.0, 0.0], [14.0, 0.0], [12.0, 0.0]]])
    _, sm = mt.t0.geometry(mt, edge, torch.ones(1, 4))
    w = sm[0].tolist()
    check("taper: ~0 at grid edge, 1 at >= taper cells in, monotone", w[0] < 0.01 and w[1] == 1.0 and w[3] == 1.0 and 0 < w[2] < 1, str([round(v, 3) for v in w]))


def external_boundary():
    kw = dict(grid=16, extent=16.0, net="local", width=8, layers=2, dt=0.1)
    m = t0m.attach_tier0(perturb(sf.ScatterField(**kw)), "ext=6,extapply")
    B = 2
    g = torch.randn(B, 2) * 0.3
    H = torch.randn(B, 3) * 0.05
    phi0 = torch.randn(B)
    m.t0.set_external(m, phi0=phi0, g=g, H=H)
    e = m.t0.ext_grid
    iy, ix = 3, 11
    x, y = (ix + 0.5) * m.h - 8.0, (iy + 0.5) * m.h - 8.0
    phi = phi0 + g[:, 0] * x + g[:, 1] * y + 0.5 * (H[:, 0] * x * x + 2 * H[:, 1] * x * y + H[:, 2] * y * y)
    gx, gy = g[:, 0] + H[:, 0] * x + H[:, 1] * y, g[:, 1] + H[:, 1] * x + H[:, 2] * y
    ok = torch.allclose(e[:, 0, iy, ix], phi, atol=1e-5) and torch.allclose(e[:, 1, iy, ix], gx, atol=1e-5) \
        and torch.allclose(e[:, 2, iy, ix], gy, atol=1e-5) and torch.allclose(e[:, 3:, iy, ix], H, atol=1e-6)
    check("ext: value/grad/Hessian channels equal the polynomial at cell centres", ok)
    pos, vel, mass, mask = scene(16.0, B=B, N=20)
    st = m.init_field(B, "cpu")
    a_on, _ = m.force(pos, vel, mass, mask, st)
    m.t0.cfg.extapply = False
    a_off, _ = m.force(pos, vel, mass, mask, st)
    m.t0.cfg.extapply = True
    ea = m.t0.ext_acc(m, pos) * mask[..., None]
    check("ext: extapply adds exactly -(g + H x) after momfix (uniform field survives)", torch.allclose(a_on - a_off, ea, atol=1e-6))
    eps = 1e-3
    pe = pos[:, :1] + torch.tensor([eps, 0.0])
    pm = pos[:, :1] - torch.tensor([eps, 0.0])
    f = lambda p: phi0[:, None] + g[:, None, 0] * p[..., 0] + g[:, None, 1] * p[..., 1] + 0.5 * (H[:, None, 0] * p[..., 0] ** 2 + 2 * H[:, None, 1] * p[..., 0] * p[..., 1] + H[:, None, 2] * p[..., 1] ** 2)
    fd = -(f(pe) - f(pm)) / (2 * eps)
    check("ext: ext_acc is minus the gradient of the potential", torch.allclose(m.t0.ext_acc(m, pos[:, :1])[..., 0], fd, atol=1e-4))
    gridonly = t0m.attach_tier0(sf.ScatterField(**kw), "ext=3")
    gridonly.t0.set_external(gridonly, grid=e[:, :3])
    check("ext: grid passthrough (parent lagged expansion)", gridonly.t0.ext_grid.shape == (B, 3, 16, 16) and torch.allclose(gridonly.t0.ext_acc(gridonly, pos[:, :1]), m.t0.ext_acc(m, pos[:, :1]), atol=1e-4))
    mb = t0m.attach_tier0(sf.ScatterField(**kw), "bnd,halo=2")
    sdf = torch.full((1, 1, 16, 16), 5.0)
    sdf[..., 6:9, 6:9] = -1.0
    mb.t0.set_boundary(mb, sdf=sdf, edge="periodic")
    bg = mb.t0.bnd_grid
    ring_ok = bg[0, 3, 0, 5] == 1 and bg[0, 3, 8, 0] == 1 and bg[0, 3, 8, 8] == 0 and bg[0, 2:5].sum(0)[0, 0] == 1
    obst_ok = bg[0, 0, 7, 7] == 1 and bg[0, 4, 7, 7] == 1 and bg[0, 0, 0, 0] == 0
    check("bnd: occupancy, clipped SDF and one-hot type mask (periodic ring, wall obstacle)", bool(ring_ok and obst_ok) and bg.shape == (1, 5, 16, 16) and float(bg[0, 1].abs().max()) <= 1.0)
    pos, vel, mass, mask = scene(16.0, B=2, N=10)
    mb.t0.bnd_grid = None
    mb.force(pos, vel, mass, mask, mb.init_field(2, "cpu"))
    check("bnd: default boundary channels are built and broadcast over the batch", mb.t0.bnd_grid.shape[0] == 1)


def head_state():
    for verlet in (False, True):
        kw = dict(grid=32, extent=32.0, net="local", width=8, layers=2, dt=0.1, verlet=verlet, recurrent=True, hidden_ch=1)
        m = t0m.attach_tier0(sf.ScatterField(**kw), "moments=2,vdisp,head=16,phid=4,phscatter,mat=1")
        perturb(m, scale=0.1)
        m.t0.set_material(torch.rand(2, 30, 1))
        pos, vel, mass, mask = scene(32.0, N=30)
        pos0, vel0 = pos.clone().requires_grad_(True), vel.clone().requires_grad_(True)
        st = m.init_field(2, "cpu")
        p, v = pos0, vel0
        for _ in range(3):
            p, v, st, _ = m.step(p, v, mass, mask, st)
        loss = (p ** 2).sum() + (v ** 2).sum()
        loss.backward()
        ph = (st[0] if verlet else st)[1]
        gn = float(sum(q.grad.abs().sum() for q in m.t0.parameters() if q.grad is not None))
        det = tsf.detach_state(st)
        check(f"per-particle hidden (verlet={verlet}): state carried, nonzero, masked, grads reach the head", ph.shape == (2, 30, 4) and float(ph.abs().max()) > 0
              and float(ph[:, -3:].abs().max()) == 0.0 and gn > 0 and isinstance(det, tuple))
        a, _ = m.force(pos, vel, mass, mask, m.init_field(2, "cpu"))
        P = ((mass * mask)[..., None] * a).sum(1).abs().max()
        check(f"head + momfix: total momentum change zero (verlet={verlet})", float(P) < 1e-5, f"{float(P):.1e}")
    kw = dict(grid=32, extent=32.0, net="local", width=8, layers=2, dt=0.1)
    m = t0m.attach_tier0(sf.ScatterField(**kw), "head=8")
    cap = []
    m.t0.headnet.register_forward_pre_hook(lambda mod, inp: cap.append(inp[0].detach()))
    pos = torch.tensor([[[-8.0, -8.0], [5.2, 5.3], [5.4, 5.1]]])
    vel = torch.tensor([[[1.0, 0.5], [2.0, 0.0], [-2.0, 0.0]]])
    m.force(pos, vel, torch.ones(1, 3), torch.ones(1, 3), m.init_field(1, "cpu"))
    rel = cap[0][0, :, 4]
    check("head input: relative velocity ~0 for an isolated particle, large for colliding pair", float(rel[0]) < 1e-5 and float(rel[1]) > 0.5 and float(rel[2]) > 0.5, str([round(float(r), 3) for r in rel]))
    m.t0.record = True
    m.t0.headnet[-1].weight.data.normal_(std=0.1)
    m.force(pos, vel, torch.ones(1, 3), torch.ones(1, 3), m.init_field(1, "cpu"))
    check("head penalty available from aux", float(m.t0.head_penalty(torch.ones(1, 3))) > 0)


def interface(dev):
    kw = dict(grid=32, extent=32.0, net="local", width=8, layers=2, dt=0.1, potential=True, hidden_ch=1)
    m = perturb(sf.ScatterField(**kw)).to(dev)
    pos, vel, mass, mask = scene(32.0, dev=dev)
    out = t0m.tier0_forward(m, pos, vel, mass, mask)
    a_ref, f_ref = m.force(pos, vel, mass, mask, m.init_field(2, dev))
    ok = out.moments.data.shape == (2, 15, 32, 32) and out.acc_grid.shape == (2, 2, 32, 32) and torch.allclose(out.a_tok, a_ref, atol=1e-6) \
        and out.field.shape == f_ref.shape and m.t0 is None
    check("interface: tier0_forward returns moments (B,15,G,G), mesh force grid, particle accel equal to force(); t0 left detached", ok)
    m2 = t0m.attach_tier0(perturb(sf.ScatterField(**{**kw, "momfix": False})).to(dev), "head=8,phid=4")
    o2 = t0m.tier0_forward(m2, pos, vel, mass, mask)
    g2 = m2.gather(o2.acc_grid, pos) * mask[..., None]
    check("interface: gather(acc_grid, pos) == a_tok with momfix off", torch.allclose(g2, o2.a_tok, atol=1e-5) and isinstance(o2.state, tuple) and o2.state[1].shape == (2, 40, 4))


def opts_and_roundtrip(dev):
    kw = dict(grid=32, extent=32.0, net="local", width=8, layers=2, dt=0.1, verlet=True)
    spec = "moments=2,vdisp,head=8,phid=2,ext=3,bnd,halo=2"
    m = t0m.attach_tier0(perturb(sf.ScatterField(**kw)), spec).to(dev)
    m.t0.headnet[-1].weight.data.normal_(std=0.05)
    s = scene(32.0, dev=dev)
    a = flat(run_steps(m, *s))
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    m2 = t0m.attach_tier0(sf.ScatterField(**kw), m.t0_spec).to(dev)
    m2.load_state_dict(sd)
    b = flat(run_steps(m2, *s))
    check("round trip: state_dict + spec string rebuilds identical model", torch.equal(a, b), f"spec '{m.t0_spec}'")
    sf.apply_opts(m2, "default")
    c = flat(run_steps(m2, *s))
    check("apply_opts(default) with t0: graph path bypassed, outputs match eager", float((a - c).abs().max()) < 1e-4)
    check("apply_opts: graphtrain skipped when t0 attached", "graphtrain" not in sf.apply_opts(t0m.attach_tier0(sf.ScatterField(**kw), "moments=1"), "default", "train").opt)


def smoke_train(dev):
    with tempfile.TemporaryDirectory() as d:
        cmd = [sys.executable, "scripts/train_scatter_field.py", "--exp", "A", "--variant", "ss_rec", "--t0", "moments=2,vdisp,head=16,phid=4",
               "--train", "32", "--steps", "8", "--iters", "30", "--k-start", "2", "--k-end", "4", "--batch", "8", "--eval-scenes", "4",
               "--eval-steps", "24", "--log-every", "10", "--device", dev, "--out-dir", d, "--ckpt-dir", d]
        r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
        tail = "\n".join(r.stdout.strip().splitlines()[-6:])
        losses = [float(l.split("loss ")[1].split()[0]) for l in r.stdout.splitlines() if " loss " in l]
        ck = torch.load(os.path.join(d, "A_ss_rec.pt"), map_location="cpu", weights_only=False) if r.returncode == 0 else {}
        check("smoke train via trainer --t0 (30 its, moments+vdisp+head+phid)", r.returncode == 0 and len(losses) >= 3 and all(l == l and l < 1e6 for l in losses)
              and any(k.startswith("t0.headnet") for k in ck.get("model", {})), f"rc {r.returncode} losses {losses}\n{tail}\n{r.stderr[-800:] if r.returncode else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--make-ref", action="store_true")
    ap.add_argument("--orig", default=None)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    if args.make_ref:
        make_ref(args)
        return
    torch.manual_seed(4738)
    regress(args, args.device)
    warm_start(args.device)
    moments_exact()
    periodic_open_taper()
    external_boundary()
    head_state()
    interface(args.device)
    opts_and_roundtrip(args.device)
    if not args.skip_train:
        smoke_train(args.device)
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
