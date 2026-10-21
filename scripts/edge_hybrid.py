"""Where do tracers get thrown from? Tracer rollout where the tracer force is replaced by the exact force (from the MODEL's evolved massive bodies) beyond a
max-norm coordinate threshold (None = pure model, 0 = all tracers exact, 28/32 = exact outside that coordinate)."""
import argparse
import json

import torch

from scripts import edge_probe as ep
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    tc = {}
    for opts in ("kext", "kext,far@0.05"):
        m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128_ghost", 0.1).to(dev)
        sf.load_compat(m, torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
        m.eval()
        sf.apply_opts(m, opts)
        ops.chunked_pp(m)
        for hy in (None, 0.0, 28.0, 32.0):
            r = [ep.rollout_probe(m, dev, 20000, 2000, 100, s, tc, hy) for s in (9000, 12000)]
            res[f"{opts}|{hy}"] = r
            for x in r:
                print(f"HYB {opts} hybrid>{hy} seed {x['seed']} beyond70/400 {x['model']['beyond70_per400']:.1f} median_r {x['model']['median_r']:.1f} p99_r {x['model']['p99_r']:.1f} "
                      f"rev {x['model']['reversal_frac']:.4f} (truth {x['truth']['beyond70_per400']:.1f} {x['truth']['median_r']:.1f} {x['truth']['reversal_frac']:.4f})", flush=True)
    json.dump(res, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
