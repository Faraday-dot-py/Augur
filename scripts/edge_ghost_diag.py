"""Diagnose the learned ghost-cell closure: print trained stencils, fit the best linear 3x3 stencil to the exact kernel ghost value phi[G] by least squares
(oracle for what a linear closure can do), and measure the static edge probe with each stencil."""
import argparse
import json

import numpy as np
import torch

from scripts import edge_probe as ep
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf
from scripts import scatter_scaling_ops as ops


def build(ck, dev):
    m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128_ghost", 0.1).to(dev)
    sf.load_compat(m, torch.load(ck, map_location=dev, weights_only=False)["model"])
    return m.eval()


@torch.no_grad()
def kernel_phi(m, pos, dev):
    P = torch.tensor(pos, dtype=torch.float32, device=dev)[None]
    M = torch.ones(1, P.shape[1], device=dev)
    mk = torch.ones_like(M)
    x = m.scatter(P, torch.zeros_like(P), M, mk)[:, :1]
    G = m.grid
    rf = torch.fft.rfft2(x, s=(2 * G, 2 * G))
    return torch.fft.irfft2(rf * m._kernel_fft(x), s=(2 * G, 2 * G)) * m.h ** 2


@torch.no_grad()
def strips(full, G):
    """per boundary cell: 9 strip values (3 deep x 3 tangent, ordered as the conv sees them: tangent rows, depth columns, outermost last) and target phi[G]/phi[-1]."""
    phi = full[..., :G, :G]
    X, Y = [], []
    sides = [(phi[..., G - 3:], full[..., :G, G]), (phi[..., :3].flip(-1), full[..., :G, 2 * G - 1]),
             (phi[..., G - 3:, :].transpose(-1, -2), full[..., G, :G]), (phi[..., :3, :].flip(-2).transpose(-1, -2), full[..., 2 * G - 1, :G])]
    for s, t in sides:
        s = torch.nn.functional.pad(s, (0, 0, 1, 1))[0, 0]
        for j in range(G):
            X.append(s[j:j + 3].reshape(-1))
        Y.append(t[0, 0])
    return torch.stack(X), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = {}
    names = dict(x.split("=") for x in args.ckpts.split(","))
    m = build(next(iter(names.values())), dev)
    Xs, Ys = [], []
    for seed in range(8):
        for n, sg in ((300, 8.0), (2000, 8.0), (20000, 12.0)):
            rng = np.random.default_rng(1000 + seed)
            pos = np.clip(rng.normal(0, sg, (n, 2)), -29, 29)
            full = kernel_phi(m, pos, dev)
            X, Y = strips(full, m.grid)
            sc = Y.abs().max().clamp_min(1e-9)
            Xs.append(X / sc)
            Ys.append(Y / sc)
    X, Y = torch.cat(Xs), torch.cat(Ys)
    w = torch.linalg.lstsq(X.double(), Y.double()[:, None]).solution[:, 0].float()
    res = (X @ w - Y).abs().mean() / Y.abs().mean()
    print("LSQ stencil (tangent rows x depth cols, outermost last):", w.view(3, 3).cpu().numpy().round(3).tolist(), "rel resid", float(res), flush=True)
    out["lsq"] = {"w": w.view(3, 3).tolist(), "rel_resid": float(res)}
    ec = {}
    for name, ck in list(names.items()) + [("lsq", None)]:
        mm = build(next(iter(names.values())) if ck is None else ck, dev)
        if ck is None:
            mm.gpad.weight.data = w.view(1, 1, 3, 3).clone()
        print(f"STENCIL {name}", mm.gpad.weight.data[0, 0].cpu().numpy().round(3).tolist(), flush=True)
        ops.chunked_pp(mm)
        sts = [ep.static_probe(mm, dev, 100000, s, ec) for s in (9000, 12000)]
        for st in sts:
            print(f"DIAG {name} static seed {st['seed']} ratio {st['band_ratio']:.1f} signmatch {st['sign_match']:.2f} interior_err {st['interior_rel_err']:.3f}", flush=True)
        out[name] = {"w": mm.gpad.weight.data[0, 0].tolist(), "ratio": [s["band_ratio"] for s in sts]}
    json.dump(out, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
