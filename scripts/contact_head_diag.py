"""Diagnose the tier-0 head divergence in the contact arm (Bh2): (1) step-0 identity of a zero-init head vs the same model without it,
(2) per-step head output / velocity / distance-to-surface along a model rollout of eval scenes with a trained Bh ckpt, (3) training-range
statistics of the head's velocity input.
Usage: PYTHONPATH=. python scripts/contact_head_diag.py --ckpt checkpoints_contact/Bh2.pt --t0 head=32,headmax=5 --out results/contact/head_diag.json"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_contact as sc
from scripts import scatter_contact_scenes as cs
from scripts import scatter_field as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t0", default="head=32")
    ap.add_argument("--init", default="")
    ap.add_argument("--fam", default="post_circle")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="results/contact/head_diag.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    es = cs.evalset(a.fam, cs.EVAL_SEEDS[0], 100, dev)
    scenes = es["scenes"][:8]
    if a.init:
        torch.manual_seed(4738)
        m0 = cs.build("B", dev, a.init, "")
        torch.manual_seed(4738)
        m1 = cs.build("B", dev, a.init, a.t0)
        for m in (m0, m1):
            m.eval()
            m.set_scene(cs._geom_stack(scenes).to(dev), cs._rad_pad(scenes, max(len(s["mass"]) for s in scenes)).to(dev))
        n = max(len(s["mass"]) for s in scenes)
        pos = torch.zeros(len(scenes), n, 2, device=dev)
        vel = torch.zeros_like(pos)
        mass = torch.zeros(len(scenes), n, device=dev)
        mask = torch.zeros_like(mass)
        for i, s in enumerate(scenes):
            k = len(s["mass"])
            pos[i, :k], vel[i, :k] = torch.tensor(s["pos"], dtype=torch.float32), torch.tensor(s["vel"], dtype=torch.float32)
            mass[i, :k], mask[i, :k] = torch.tensor(s["mass"], dtype=torch.float32), 1.0
        with torch.no_grad():
            f0 = m0.init_field(len(scenes), dev)[0]
            f1 = m1.init_field(len(scenes), dev)[0]
            a0, _ = m0.force(pos, vel, mass, mask, f0)
            a1, _ = m1.force(pos, vel, mass, mask, f1)
        res["step0_identity_max_abs"] = float((a0 - a1).abs().max())
    model = cs.build("B", dev, "none", a.t0)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    res["ckpt_it"] = st.get("it")
    n = max(len(s["mass"]) for s in scenes)
    model.set_scene(cs._geom_stack(scenes).to(dev), cs._rad_pad(scenes, n).to(dev))
    pos = torch.zeros(len(scenes), n, 2, device=dev)
    vel = torch.zeros_like(pos)
    mass = torch.zeros(len(scenes), n, device=dev)
    mask = torch.zeros_like(mass)
    for i, s in enumerate(scenes):
        k = len(s["mass"])
        pos[i, :k], vel[i, :k] = torch.tensor(s["pos"], dtype=torch.float32), torch.tensor(s["vel"], dtype=torch.float32)
        mass[i, :k], mask[i, :k] = torch.tensor(s["mass"], dtype=torch.float32), 1.0
    model.t0.record = True
    field = model.init_field(len(scenes), dev)
    rows = []
    with torch.no_grad():
        for t in range(a.steps):
            pos, vel, mass, field, _ = sf._step_m(model, pos, vel, mass, mask, field)
            ha = model.t0.aux["head_acc"]
            mag = ha.norm(dim=-1) * mask
            sp = vel.norm(dim=-1) * mask
            j = int(mag.flatten().argmax())
            b, i = divmod(j, n)
            sd = float(sc.scene_sdf(scenes[b]["prims"], pos[b, i:i + 1].double().cpu()).numpy()[0]) if i < len(scenes[b]["mass"]) else None
            rows.append(dict(step=t + 1, head_max=float(mag.max()), head_mean=float(mag.sum() / mask.sum()), speed_max=float(sp.max()),
                             speed_at_headmax=float(sp[b, i]), sdf_at_headmax=sd, rad=float(scenes[b]["rad"][i]) if i < len(scenes[b]["mass"]) else None,
                             finite=bool(torch.isfinite(pos).all())))
    res["rollout"] = rows
    ts = cs.train_set(800, 80, dev)
    sp_all = np.concatenate([np.linalg.norm(d[1], axis=-1).ravel() for d in ts[1][:200]])
    res["train_speed_pctl_50_99_100"] = [float(np.percentile(sp_all, p)) for p in (50, 99, 100)]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rollout"}, indent=1))
    for r in rows[::3]:
        print(r)


if __name__ == "__main__":
    main()
