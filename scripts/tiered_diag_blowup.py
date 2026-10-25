"""Per-step blow-up diagnostic for a model rollout: finite flags, max |pos|/|vel| per channel, fractions outside the tile span / the view,
first bad step, which mass terciles escape. Usage: python scripts/tiered_diag_blowup.py --model t0ext:<ckpt> --ic multiclump --out results/tiered/diag_blowup.json"""
import argparse
import json

import torch

from scripts import tiered_eval as te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ic", default="multiclump")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--span", type=float, default=128.0)
    ap.add_argument("--wrap", type=float, default=128.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="results/tiered/diag_blowup.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    model = te.load_model(a.model, dev, 0.1)
    b = te.grav_batch(a.ic, a.n, a.span, a.seed, 4, dev)
    tl = te.wrap(model, a.wrap)
    rt = te.truth_roll(b, model.dt, a.steps)
    rm = te.model_roll(tl, b, a.steps)
    mk = b["mask"] > 0
    rows = []
    for k in range(a.steps + 1):
        row = dict(step=k)
        for nm, r in (("model", rm), ("truth", rt)):
            P, V = r["pos"][:, k], r["vel"][:, k]
            row[nm] = dict(max_abs_pos=[float(P[..., c][mk].abs().max()) for c in range(2)], max_abs_vel=[float(V[..., c][mk].abs().max()) for c in range(2)],
                           frac_outside_64=float((P.abs().amax(-1)[mk] > 64).float().mean()), frac_outside_200=float((P.abs().amax(-1)[mk] > 200).float().mean()),
                           median_r=float(P.norm(dim=-1)[mk].median()))
        rows.append(row)
    d = (rm["pos"] - rt["pos"]).norm(dim=-1)
    first_bad = [int(((d[i] > 10) & mk[i]).any(-1).nonzero()[0]) if bool(((d[i] > 10) & mk[i]).any()) else -1 for i in range(d.shape[0])]
    edges = te.mass_bins(b["mass"], b["mask"])
    esc = None
    if edges is not None:
        bk = torch.bucketize(b["mass"], edges)
        esc = [float(((rm["pos"][:, -1].abs().amax(-1) > 200) & mk & (bk == i)).sum() / (mk & (bk == i)).sum().clamp_min(1)) for i in range(3)]
    out = dict(first_nonfinite=rm["first_nonfinite"].tolist(), first_step_pos_err_gt_10_per_scene=first_bad, escaped_frac_by_mass_tercile=esc, rows=rows)
    json.dump(out, open(a.out, "w"), indent=1)
    for r in rows[::10]:
        print(r["step"], "model", {k: (round(v, 2) if not isinstance(v, list) else [round(x, 2) for x in v]) for k, v in r["model"].items()}, flush=True)
    print("first_nonfinite", out["first_nonfinite"], "first err>10", first_bad, "escaped by mass tercile", esc)


if __name__ == "__main__":
    main()
