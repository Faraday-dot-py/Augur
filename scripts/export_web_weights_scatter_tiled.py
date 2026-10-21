"""Export the tiled scatter-field model (Exp E, E_ms_kp_pot_v_g128, plus the R1b far kernel from scripts/scatter_far.py) to
web/scatter-tiled/weights.bin (fp32 little-endian, tensors concatenated) + web/scatter-tiled/weights.json (manifest + config).

The tile copies share the Exp E weights; the far kernel is a separate 2 -> 64 -> 64 -> 1 MLP K(r) (exported as kfar.*) used for the
coarse global term and for every tile's local kernel convolution.

Usage:
    PYTHONPATH=. python3 scripts/export_web_weights_scatter_tiled.py --checkpoint E_ms_kp_pot_v_g128.pt --kernel results/far_kernels/r1b_mlp.pt
"""
import argparse
import hashlib
import json

import numpy as np
import torch

CONFIG = {"grid": 128, "extent": 64.0, "dt": 0.1, "levels": 5, "pp": 2.0, "knn": 16, "eps": 0.5,
          "in_scale": 1.0, "momfix": True, "potential": True, "kernel": True, "split": True, "verlet": True,
          "tile": 32.0, "halo": 16.0, "far_c": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="E_ms_kp_pot_v_g128.pt")
    ap.add_argument("--kernel", default="results/far_kernels/r1b_mlp.pt")
    ap.add_argument("--out", default="web/scatter-tiled/weights")
    args = ap.parse_args()

    st = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ker = torch.load(args.kernel, map_location="cpu", weights_only=False)
    assert ker["arch"] == "mlp" and ker["kw"] == {"pp": CONFIG["pp"], "split": True}
    state = dict(st["model"])
    for k, v in ker["state"].items():
        state["kfar." + k[len("net."):]] = v
    tensors, chunks, offset = {}, [], 0
    for name, t in state.items():
        a = t.detach().numpy().astype("<f4")
        tensors[name] = {"shape": list(a.shape), "offset": offset}
        chunks.append(a.reshape(-1))
        offset += a.size
    np.concatenate(chunks).tofile(args.out + ".bin")
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
    with open(args.out + ".json", "w") as f:
        json.dump({"checkpoint": args.checkpoint.split("/")[-1], "iters": st.get("it"), "checkpoint_sha256": sha(args.checkpoint),
                   "kernel": args.kernel.split("/")[-1], "kernel_sha256": sha(args.kernel), "config": CONFIG, "tensors": tensors}, f, indent=1)
    print(f"{len(tensors)} tensors, {offset} floats -> {args.out}.bin")


if __name__ == "__main__":
    main()
