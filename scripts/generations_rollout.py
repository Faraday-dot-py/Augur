"""CPU rollouts of several scatter-field gravity generations on one shared scene, saved to npz for scripts/render_generations_video.py.
Stage "main" (run in this tree) writes the scene, ground truth and the scatter-gravity-clustered-baseline / tier0 arms;
stage "dt-half" (run with PYTHONPATH at the tree that has ScatterField.nsub, branch hole1-potential-field) adds the half-timestep model on the saved scene.
Usage:
  PYTHONPATH=. python3 scripts/generations_rollout.py --stage main --ic multiclump --n 300 --span 64 --baseline B.pt --tier0 T.pt --out results/gen_multiclump64.npz
  PYTHONPATH=<hole1 tree> python3 scripts/generations_rollout.py --stage dt-half --scene results/gen_multiclump64.npz --ckpt H.pt --out results/gen_multiclump64_dthalf.npz"""
import argparse

import numpy as np
import torch


def energy_curve(phys, P, V, mass, mask):
    return torch.stack([phys.energy(P[:, k], V[:, k], mass, mask) for k in range(P.shape[1])], 1)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "dt-half", "central"], required=True)
    ap.add_argument("--ic", default="multiclump")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--span", type=float, default=64.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--baseline")
    ap.add_argument("--tier0")
    ap.add_argument("--scene")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    if a.stage == "main":
        from scripts import tiered_adapters as ta
        from scripts import tiered_eval as te
        from scripts import tiered_iface as ti

        b = te.grav_batch(a.ic, a.n, a.span, a.seed, 1, dev)
        rt = te.truth_roll(b, 0.1, a.steps)
        out = dict(truth_pos=rt["pos"][0].numpy(), truth_vel=rt["vel"][0].numpy(), mass=b["mass"][0].numpy(), span=a.span, ic=a.ic)
        out["truth_E"] = energy_curve(b["phys"], rt["pos"], rt["vel"], b["mass"], b["mask"])
        arms = {}
        if a.baseline:
            m = te.load_model(f"scatter:{a.baseline}", dev, 0.1)
            arms["baseline"] = te.wrap(m, a.span)
        if a.tier0:
            m = ta.load_t0ext(a.tier0, dev, 0.1)
            arms["tier0"] = te.wrap(m, a.span)
            if a.span <= m.reach:
                arms["tier0_tiled64"] = ti.Tiled(m, a.span, tile=32.0, sparse=True)
            else:
                arms["tier0_single_window"] = m
        for k, mod in arms.items():
            rm = te.model_roll(mod, b, a.steps)
            out[f"{k}_pos"], out[f"{k}_vel"] = rm["pos"][0].numpy(), rm["vel"][0].numpy()
            out[f"{k}_E"] = energy_curve(b["phys"], rm["pos"], rm["vel"], b["mass"], b["mask"])
            out[f"{k}_tiled"] = type(mod).__name__ == "Tiled"
            print(k, type(mod).__name__, float((rm["pos"] - rt["pos"]).norm(dim=-1)[:, 20].mean()), flush=True)
        np.savez(a.out, **out)
    elif a.stage == "central":
        from model.central_force import CentralForceDynamics
        from scripts.train_gravity_dynamics import unroll

        sc = np.load(a.scene)
        dyn = CentralForceDynamics(dt=0.1)
        dyn.load_state_dict(torch.load(a.ckpt, map_location=dev))
        p0 = torch.as_tensor(sc["truth_pos"][0], dtype=torch.float32)
        v0 = torch.as_tensor(sc["truth_vel"][0], dtype=torch.float32)
        with torch.no_grad():
            ps, vs = unroll(dyn, p0, v0, sc["truth_pos"].shape[0] - 1, 0.1)
        P = torch.cat([p0[None], ps]).double()
        V = torch.cat([v0[None], vs]).double()
        m = torch.as_tensor(sc["mass"], dtype=torch.float64)
        eye = torch.eye(m.shape[0], dtype=torch.float64)

        def energy(p, v):
            r = torch.sqrt(((p[:, None] - p[None]) ** 2).sum(-1) + 0.25)
            return float(0.5 * (m * (v ** 2).sum(-1)).sum() - 0.5 * (m[:, None] * m[None] / r * (1 - eye)).sum())

        np.savez(a.out, central_pos=P.numpy(), central_vel=V.numpy(), central_E=np.array([energy(P[k], V[k]) for k in range(P.shape[0])]))
        print("central err@20", float((P[20] - torch.as_tensor(sc["truth_pos"][20])).norm(dim=-1).mean()), flush=True)
    else:
        from scripts import scatter_field as sf
        from scripts import train_scatter_field as tsf

        sc = np.load(a.scene)
        model = tsf.build(tsf.EXPS["E"], "gv_NG", 0.1)
        model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
        model.nsub = 2
        model.eval().requires_grad_(False)
        P0 = torch.as_tensor(sc["truth_pos"][0:1], dtype=torch.float32)
        V0 = torch.as_tensor(sc["truth_vel"][0:1], dtype=torch.float32)
        M = torch.as_tensor(sc["mass"][None], dtype=torch.float32)
        mask = torch.ones_like(M)
        with torch.no_grad():
            Pm, Vm = sf.rollout(model, P0, V0, M, mask, sc["truth_pos"].shape[0] - 1)
        Pm, Vm = Pm.double(), Vm.double()
        m64 = M.double()[0]
        eye = torch.eye(m64.shape[0], dtype=torch.float64)

        def energy(p, v):
            r = torch.sqrt(((p[:, None] - p[None]) ** 2).sum(-1) + 0.25)
            return float(0.5 * (m64 * (v ** 2).sum(-1)).sum() - 0.5 * (m64[:, None] * m64[None] / r * (1 - eye)).sum())

        out = dict(dthalf_pos=Pm[0].numpy(), dthalf_vel=Vm[0].numpy(), dthalf_E=np.array([energy(Pm[0, k], Vm[0, k]) for k in range(Pm.shape[1])]))
        np.savez(a.out, **out)
        print("dt-half err@20", float((Pm[0, 20] - torch.as_tensor(sc["truth_pos"][20])).norm(dim=-1).mean()), flush=True)


if __name__ == "__main__":
    main()
