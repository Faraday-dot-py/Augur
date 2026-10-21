"""Dump float64 PyTorch references for the tiled scatter-field page (web/scatter-tiled/tests/vectors.json), for web/scatter-tiled/tests/verify.mjs.

Tiled model = scripts/scatter_tiling.py TiledScatter (tiles of 32 + halo 16, G=128 copies, ghost bodies, halo field exchange, coarse far term,
far_c 4) with the R1b far kernel installed through scripts/scatter_far.py. Rollouts are free (Verlet), relative coordinates.
Scenes: nt=2 (domain 64) and nt=4 (domain 128). Dumps tiled rollouts, a single-grid rollout of the same scene (G=128 / G=256 whole-domain grid,
same weights and kernel), single-force accelerations from a zero field for several flag settings, and one tile's potential.

Usage:
    PYTHONPATH=<repo with scatter_tiling.py, scatter_far.py>:. python3 scripts/export_web_testvectors_scatter_tiled.py \
        --checkpoint E_ms_kp_pot_v_g128.pt --kernel results/far_kernels/r1b_mlp.pt --out web/scatter-tiled/tests/vectors.json
"""
import argparse
import json

import numpy as np
import torch

from scripts.scatter_far import install, load_kernel
from scripts.scatter_field import ScatterField
from scripts.scatter_tiling import TiledScatter

DT = 0.1
DEV = "cpu"


def clusters(rng, specs):
    ps, vs = [], []
    for n, c, sigma, v, vfac in specs:
        ps.append(rng.normal(0.0, sigma, (n, 2)) + np.array(c))
        vs.append(rng.normal(0.0, vfac * np.sqrt(n / (4 * sigma)), (n, 2)) + np.array(v))
    p, v = np.concatenate(ps), np.concatenate(vs)
    return p, v - v.mean(0)


SCENES = {
    "nt2_cluster": (2, [(40, (0.0, 0.0), 5.0, (0.0, 0.0), 0.6)]),
    "nt2_cross": (2, [(20, (-14.0, -3.0), 3.0, (2.0, 0.6), 0.3), (20, (14.0, 3.0), 3.0, (-2.0, -0.6), 0.3)]),
    "nt4_cluster": (4, [(60, (0.0, 0.0), 8.0, (0.0, 0.0), 0.3)]),
    "nt4_cross": (4, [(30, (-30.0, -6.0), 5.0, (3.0, 0.8), 0.3), (30, (30.0, 6.0), 5.0, (-3.0, -0.8), 0.3)]),
}
VARIANTS = {"full": (True, True, True), "noghost": (False, True, True), "nofar": (True, True, False), "nofx": (True, False, True),
            "naive": (False, False, False)}


def build(ckpt, kernel, dev):
    m = ScatterField(grid=128, extent=64.0, net="unet", dt=DT, kernel=True, pp=2.0, split=True, potential=True, verlet=True).double().to(dev)
    m.load_state_dict({k: v.double() for k, v in torch.load(ckpt, map_location=dev, weights_only=False)["model"].items()})
    m.eval()
    install(m, load_kernel(kernel, dev).double(), cache=True)
    return m


@torch.no_grad()
def rollout(tl, pos, vel, mass, steps):
    st = (tl.init_field(DEV, torch.float64), None)
    ps, vs = [pos.tolist()], [vel.tolist()]
    owners, fields = [], {}
    for t in range(steps):
        pos, vel, st = tl.step(pos, vel, mass, st)
        ps.append(pos.tolist())
        vs.append(vel.tolist())
        if t in (0, steps - 1):
            fields[str(t + 1)] = st[0]
    return ps, vs, fields


@torch.no_grad()
def single(m, nt, pos, vel, mass, steps):
    G, E = round(nt * 32 / 0.5), nt * 32.0
    old = (m.grid, m.extent, m.h)
    m.grid, m.extent, m.h = G, E, 0.5
    try:
        p, v = pos[None].clone(), vel[None].clone()
        mk = torch.ones_like(p[..., 0])
        st = m.init_field(1, DEV, torch.float64)
        ps, vs = [p[0].tolist()], [v[0].tolist()]
        for _ in range(steps):
            p, v, st, _ = m.step(p, v, mass[None], mk, st)
            ps.append(p[0].tolist())
            vs.append(v[0].tolist())
    finally:
        m.grid, m.extent, m.h = old
    return ps, vs, st[0][0, 0]


def r9(x):
    return [float(f"{v:.9g}") for v in x.reshape(-1).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="E_ms_kp_pot_v_g128.pt")
    ap.add_argument("--kernel", default="results/far_kernels/r1b_mlp.pt")
    ap.add_argument("--out", default="web/scatter-tiled/tests/vectors.json")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    global DEV
    DEV = args.device
    torch.set_num_threads(8)
    m = build(args.checkpoint, args.kernel, DEV)
    out = {"steps": args.steps, "scenes": {}}
    rng = np.random.default_rng(4738)
    for name, (nt, specs) in SCENES.items():
        p, v = clusters(rng, specs)
        pos, vel = torch.tensor(p, device=DEV), torch.tensor(v, device=DEV)
        mass = torch.ones(len(p), dtype=torch.float64, device=DEV)
        sc = {"nt": nt, "pos0": p.tolist(), "vel0": v.tolist(), "force": {}}
        for vn, (g, fx, far) in VARIANTS.items():
            tl = TiledScatter(m, nt, 32.0, 16.0, ghosts=g, field_x=fx, far=far)
            a, f = tl.force(pos, vel, mass, tl.init_field(DEV, torch.float64))
            sc["force"][vn] = a.tolist()
            if vn == "full":
                sc["field_force"] = r9(f[nt * nt // 2 + nt // 2])
                sc["ghost_stats"] = dict(tl.stats)
        tl = TiledScatter(m, nt, 32.0, 16.0, ghosts=True, field_x=True, far=True)
        ps, vs, fields = rollout(tl, pos, vel, mass, args.steps)
        sc["tiled"] = {"pos": ps, "vel": vs}
        if name == "nt4_cross":
            k = nt * nt // 2 + nt // 2
            sc["tile_k"] = k
            sc["tile_field"] = {s: r9(f[0][k] if isinstance(f, tuple) else f[k]) for s, f in fields.items()}
        sp, sv, sf = single(m, nt, pos, vel, mass, args.steps)
        sc["single"] = {"pos": sp, "vel": sv}
        if name in ("nt2_cluster", "nt4_cross"):
            tl2 = TiledScatter(m, nt, 32.0, 16.0, ghosts=False, field_x=False, far=False)
            ps2, vs2, _ = rollout(tl2, pos, vel, mass, args.steps)
            sc["naive"] = {"pos": ps2, "vel": vs2}
        out["scenes"][name] = sc
        d = np.abs(np.array(ps[-1]) - np.array(sp[-1]))
        print(name, len(p), "bodies; tiled vs single at step", args.steps, "max", d.max())
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
