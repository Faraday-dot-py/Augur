"""Diagnose why the arm A net ignores the external-field channels: input-weight norms, channel-only gradient, field amplitude vs target scale.
Usage: PYTHONPATH=. python scripts/tier0_extdiag.py --ckpt results/tier0_ext/armA.pt --t0 SPEC --out results/tier0_ext/armA_extdiag.json"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m
from scripts import scatter_tier0_data as td
from scripts import train_scatter_field as tsf
from scripts import train_tier0_ext as tte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ns = argparse.Namespace(t0=a.t0, dt=0.1)
    model = tte.build(ns, dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    c = model.t0.cfg
    W = model.net.inp.weight.detach()
    n_old = W.shape[1] - c.n_extra()
    names = ["phi", "gx", "gy", "Hxx", "Hxy", "Hyy"][:c.ext] if c.ext else []
    ext0 = n_old + c.mat + 0
    res = {"n_in": W.shape[1], "n_old": n_old}
    res["w_norm_old_channels_mean"] = float(W[:, :n_old].pow(2).sum((0, 2, 3)).sqrt().mean())
    tail = W[:, n_old:]
    res["w_norm_new_channels"] = [float(tail[:, i].pow(2).sum().sqrt()) for i in range(tail.shape[1])]
    data, ext, kinds = td.make_scenes(48, 12, a.seed + 2000, dev)
    P, V, M, mask = [t.float() for t in sf.to_tensors(data, dev)]
    acc = ((V[:, 1:] - V[:, :-1]) / 0.1)
    w = mask[:, None, :, None]
    res["target_acc_rms"] = float(((acc ** 2 * w).sum() / (mask.sum() * 2 * acc.shape[1])).sqrt())
    phi0, g, H = [torch.tensor(x, dtype=torch.float32, device=dev) for x in ext]
    res["g_rms"] = float(g.pow(2).mean().sqrt())
    res["Hx_rms_at_20"] = float((H.pow(2).mean().sqrt()) * 20)
    res["frac_scenes_ext_zero"] = float(((g ** 2).sum(-1) == 0).float().mean())
    model.t0.cfg.extapply = False
    idx = torch.arange(16, device=dev)
    gl = g[idx].clone().requires_grad_(True)
    Hl = H[idx].clone().requires_grad_(True)
    model.t0.set_external(model, phi0=phi0[idx], g=gl, H=Hl)
    t0s = torch.zeros(16, dtype=torch.long, device=dev)
    tt = t0s[:, None] + torch.arange(5, device=dev)[None]
    Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
    pos, vel, m, mk = Pw[:, 0], Vw[:, 0], M[idx], mask[idx]
    field = model.init_field(16, dev)
    loss = 0.0
    for s in range(4):
        pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
        loss = loss + (((pos - Pw[:, s + 1]) ** 2) * mk[:, :, None]).sum() / (mk.sum() * 8)
    loss.backward()
    res["channel_only_grad_norm_g"] = float(gl.grad.norm())
    res["channel_only_grad_norm_H"] = float(Hl.grad.norm())
    model.t0.cfg.extapply = True
    outs = {}
    for name, scale in (("ext", 1.0), ("ext_x10", 10.0), ("ext_zero_channel", None)):
        model.t0.set_external(model, phi0=phi0[idx], g=g[idx] * (scale or 1.0), H=H[idx] * (scale or 1.0))
        if scale is None:
            model.t0.ext_grid = torch.zeros_like(model.t0.ext_grid)
        with torch.no_grad():
            pos, vel = P[idx, 0], V[idx, 0]
            field, m = model.init_field(16, dev), M[idx]
            for s in range(4):
                pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
        outs[name] = pos
    res["pos_diff_ext_vs_zero_channel"] = float(((outs["ext"] - outs["ext_zero_channel"]) ** 2).sum(-1).sqrt().mean())
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
