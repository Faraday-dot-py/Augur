"""Train-style k-step position loss (as train_scatter_field.train: mean over steps of per-body squared position error, cold field) of trained vs untrained
checkpoints on the TRAINING scenes (seed 4738) and on held-out scenes (seed 9000). Eval only. See long-rollout-root-cause.md."""
import argparse
import json

import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as ts

DEV = torch.device("cuda")
CFG = ts.EXPS["E"]


def loss_at(model, data, k, starts, bs=16):
    P, V, M, mask = [t.float() for t in sf.to_tensors(data, DEV)]
    out = {s: [] for s in (1, 5, 10, 20)}
    tot = []
    errs = []
    with torch.no_grad():
        for t0 in starts:
            for i in range(0, P.shape[0], bs):
                Pw, Vw, m, mk = P[i:i + bs, t0:t0 + k + 1], V[i:i + bs, t0:t0 + k + 1], M[i:i + bs], mask[i:i + bs]
                pos, vel = Pw[:, 0], Vw[:, 0]
                field = model.init_field(pos.shape[0], DEV)
                w = mk[:, :, None]
                acc = 0.0
                for s in range(k):
                    pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
                    e = (((pos - Pw[:, s + 1]) ** 2) * w).sum() / (mk.sum() * 2)
                    acc = acc + e
                    if s + 1 in out:
                        out[s + 1].append(float(e))
                    if s + 1 == 20:
                        errs.append(((pos - Pw[:, s + 1]).norm(dim=-1)[mk > 0]).cpu())
                tot.append(float(acc) / k)
    e = torch.cat(errs)
    q = torch.quantile(e, torch.tensor([0.5, 0.9, 0.99]))
    top = e.pow(2).sort(descending=True).values
    return {"loss": sum(tot) / len(tot), **{f"step{s}": sum(v) / len(v) for s, v in out.items()}, "e20_mean": float(e.mean()), "e20_q50_90_99": [float(x) for x in q], "sqshare_top1pct": float(top[: max(1, len(top) // 100)].sum() / top.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="name:variant:ckpt,...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--train-steps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=21)
    args = ap.parse_args()
    torch.manual_seed(4738)
    tr = ts.make_data(CFG, args.train_n, args.train_steps, 4738, DEV, 0.1, 0.5)[:: args.stride][:96]
    he = ts.make_data(CFG, 96, 30, 9000, DEV, 0.1, 0.5)
    res = {}
    for spec in args.specs.split(","):
        name, variant, ck = spec.split(":")
        model = ts.build(CFG, variant, 0.1).to(DEV)
        model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False)["model"])
        model.eval()
        res[name] = {"train_scenes": loss_at(model, tr, 20, (0, 5, 10)), "heldout": loss_at(model, he, 20, (0, 5, 10))}
        print(name, json.dumps(res[name]), flush=True)
        json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
