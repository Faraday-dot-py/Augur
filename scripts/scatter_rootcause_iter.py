"""Force/rollout error of Exp E intermediate checkpoints (it0..it60000) on expB and cluster scenes (eval only). See long-rollout-root-cause.md."""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_scenes as ss
from scripts import train_scatter_field as tsf
from scripts.scatter_error_growth import DEV, DT, make_scenes
from scripts.scatter_rootcause import KS, curve_stats, force_analysis, model_fa, roll, TR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="")
    ap.add_argument("--expb-seed", type=int, default=9300)
    ap.add_argument("--cluster-seed", type=int, default=9200)
    args = ap.parse_args()
    torch.manual_seed(4738)
    log = lambda s: print(s, flush=True)
    out = {}
    fams = {"expB": (24, args.expb_seed), "cluster": (12, args.cluster_seed)}
    with torch.no_grad():
        scenes = {f: make_scenes(f, n, s) for f, (n, s) in fams.items()}
        for ck in args.ckpts.split(","):
            if args.variant:
                model = tsf.build(tsf.EXPS["E"], args.variant, DT).to(DEV)
                model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False)["model"])
                model.eval()
            else:
                model = ss.build_model("E", DT, DEV, ck)
            out[ck] = {}
            for fam, sc_list in scenes.items():
                fa = force_analysis(ck, model, sc_list, 0, lambda s: None)
                errs = []
                for sc in sc_list:
                    f, r = model_fa(model, sc)
                    P = roll(f, r, sc)
                    errs.append((P - sc["P"][: TR + 1]).norm(dim=-1).mean(1).cpu().numpy())
                out[ck][fam] = dict(force={k: v for k, v in fa.items() if k in ("all", "nn 0.0-0.25", "nn 0.25-0.5", "nn 0.5-1.0", "nn 1.0-2.0", "nn 2.0-4.0", "nn 4.0-1000000000.0")},
                                    tf_pos=fa["tf_pos_mean"], tf_vel=fa["tf_vel_mean"], rollout=curve_stats(errs))
                log(f"{ck} {fam} rel_rms {fa['all']['rel_rms']:.4f} tf_pos {fa['tf_pos_mean']:.2e} rollout {out[ck][fam]['rollout']}")
            json.dump(out, open(args.out, "w"))
    log("done")


if __name__ == "__main__":
    main()
