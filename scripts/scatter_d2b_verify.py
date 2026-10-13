"""D2b step 0: on C.pt (runaway state) check the natural field scale, that the growth penalty is nonzero under the runaway, that gate params receive gradient and move
(gate-only steps, base frozen), and what the trained gate does to a free rollout.
  PYTHONPATH=. python scripts/scatter_d2b_verify.py --ckpt checkpoints_contact/C.pt [--steps 40]"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_contact_scenes as cs
from scripts import scatter_d2b as d2b
from scripts import scatter_field as sf


def load(dev):
    scenes, data = cs.train_set(4000, 80, dev, ("c_train",))
    tensors = [t.float() for t in sf.to_tensors(data, dev)]
    n = tensors[0].shape[2]
    return tensors, cs._geom_stack(scenes, dev), cs._rad_pad(scenes, n).to(dev)


def roll(model, state, ticks, record=None):
    pos, vel, m, mk, field = state
    for t in range(ticks):
        pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
        if record is not None:
            f = d2b._f(field)
            a, _ = model._force(pos, vel, m, mk, f)
            record.append((float(f.abs().amax()), float(d2b.rms(f).max()), float(a.norm(dim=-1).amax())))
    return pos, vel, m, mk, field


def series(model, P, V, M, mask, geom, rad, idx, ticks):
    model.set_scene(geom[idx], rad[idx])
    rec = []
    st = (P[idx, 0], V[idx, 0], M[idx], mask[idx], model.init_field(len(idx), P.device))
    with torch.no_grad():
        roll(model, st, ticks, rec)
    return np.array(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_contact/C.pt")
    ap.add_argument("--ref", default="checkpoints_contact/D5ctrl.pt")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--ext", type=int, default=12)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=-6.0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(4738)
    (P, V, M, mask), geom, rad = load(dev)
    idx = torch.arange(args.batch, device=dev) * 7
    out = {}
    for tag, path in (("C", args.ckpt), ("ref", args.ref)):
        mdl = d2b.build(path, dev, gate=False)
        s = series(mdl, P, V, M, mask, geom, rad, idx, 60)
        out[f"natural_{tag}"] = {t: s[t - 1, :2].tolist() + [float(s[t - 1, 2])] for t in (5, 10, 20, 28, 40, 60)}
        print(tag, "field max / rms-max / |a| max by tick", json.dumps(out[f"natural_{tag}"]), flush=True)
    model = d2b.build(args.ckpt, dev, gate=True, bias=args.bias)
    base = d2b.build(args.ckpt, dev, gate=False)
    s0 = series(model, P, V, M, mask, geom, rad, idx, 60)
    sb = series(base, P, V, M, mask, geom, rad, idx, 60)
    print("gated-at-init vs ungated field max t20/t40/t60", s0[[19, 39, 59], 0].tolist(), sb[[19, 39, 59], 0].tolist(), flush=True)
    out["init_equiv"] = dict(gated=s0[[19, 39, 59], 0].tolist(), ungated=sb[[19, 39, 59], 0].tolist())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fgate.parameters():
        p.requires_grad_(True)
    w0, b0 = model.fgate.weight.detach().clone().flatten(), model.fgate.bias.detach().clone()
    opt = torch.optim.Adam(model.fgate.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(4738)
    g = torch.Generator(device="cpu").manual_seed(4738)
    hist = []
    for it in range(args.steps):
        bi = torch.randint(P.shape[0], (args.batch,), generator=g).to(dev)
        model.set_scene(geom[bi], rad[bi])
        st = (P[bi, 0], V[bi, 0], M[bi], mask[bi], model.init_field(args.batch, dev))
        with torch.no_grad():
            st = roll(model, st, args.k)
        pos, vel, m, mk, field = st
        opt.zero_grad()
        pen, stt = d2b.growth_pen(model, pos, vel, m, mk, field, args.ext, args.rho, args.kappa, 0.5, 1000.0, gen)
        pen.backward()
        gnorm = float(torch.cat([p.grad.flatten() for p in model.fgate.parameters()]).norm())
        opt.step()
        hist.append(dict(it=it, pen=float(pen), gnorm=gnorm, **stt, wn=float(model.fgate.weight.norm()), b=float(model.fgate.bias)))
        if it % 5 == 0 or it == args.steps - 1:
            print(json.dumps(hist[-1]), flush=True)
    out["gate_hist"] = hist
    out["gate_moved"] = dict(w_delta=float((model.fgate.weight.flatten() - w0).norm()), b_delta=float(model.fgate.bias - b0))
    s1 = series(model, P, V, M, mask, geom, rad, idx, 100)
    out["rollout_after_gate_only"] = {t: s1[t - 1].tolist() for t in (10, 20, 28, 40, 60, 100)}
    sbb = series(base, P, V, M, mask, geom, rad, idx, 100)
    out["rollout_base"] = {t: sbb[t - 1].tolist() for t in (10, 20, 28, 40, 60, 100)}
    print("after gate-only: field max, rms max, |a| max", json.dumps(out["rollout_after_gate_only"]), flush=True)
    print("base            :", json.dumps(out["rollout_base"]), flush=True)
    print("gate moved", out["gate_moved"], flush=True)
    json.dump(out, open("results/d2b_verify.json", "w"), indent=1)


if __name__ == "__main__":
    main()
