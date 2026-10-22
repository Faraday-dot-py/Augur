"""TGSim (global-deposit tiled mesh) vs scatter_tiling.TiledScatter (per-tile ghost packing), pair term off, one force evaluation from zero field, galaxy IC."""
import argparse
import os

import torch

from scripts import galaxy_ic
from scripts import scatter_galaxy as sg
from scripts import scatter_far
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf
from scripts.scatter_tiling import TiledScatter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--preset", default="g1")
    ap.add_argument("--kernel", default="old")
    a = ap.parse_args()
    dev = torch.device("cuda")
    P = dict(galaxy_ic.PRESETS[a.preset])
    n = a.n
    mass = galaxy_ic.total_mass(P) / n
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    model.load_state_dict(torch.load(os.path.expanduser(a.ckpt), map_location=dev, weights_only=False)["model"])
    model.eval()
    ops.no_pp(model)
    if a.kernel != "old":
        scatter_far.install_spec(model, a.kernel)
    pos, mom = galaxy_ic.make_galaxy(n, P, 4738, n, 100.0, dev)
    v = mom / (1 + (mom ** 2).sum(-1, keepdim=True) / 100.0 ** 2).sqrt()
    with torch.no_grad():
        ref_t = TiledScatter(model, 4, 32.0, 16.0, ghosts=True, field_x=True, far=True)
        ms = torch.full((n,), mass, device=dev)
        a_ref, _ = ref_t.force(pos, v, ms, ref_t.init_field(dev))
        sim = sg.TGSim(model, n, n, 100.0, 0.1, torch.float32, torch.float32, mass, dev)
        x = sim.scatter_pass(pos, mom, False)
        acc, abar = sim.grid_force(x)
        a_new = model.gather(acc, pos[None])[0] - abar
    rel = (a_new - a_ref).norm() / a_ref.norm()
    ins = (pos.abs() < 64).all(-1)
    print(f"N {n} frac_in_domain {float(ins.float().mean()):.4f} rel diff all {float(rel):.2e} inside "
          f"{float((a_new[ins] - a_ref[ins]).norm() / a_ref[ins].norm()):.2e} |a_ref| mean {float(a_ref.norm(dim=-1).mean()):.4f}")


if __name__ == "__main__":
    main()
