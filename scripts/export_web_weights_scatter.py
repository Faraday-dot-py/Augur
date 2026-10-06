"""Export a ScatterField checkpoint (scripts/scatter_field.py, variant ms_kp_pot_v_g128) to
web/scatter/weights.bin (fp32 little-endian, tensors concatenated) + web/scatter/weights.json
(manifest + config).

Usage:
    PYTHONPATH=. python3 scripts/export_web_weights_scatter.py --checkpoint checkpoints/budgetB/sfv/B_ms_kp_pot_v_g128.pt
"""
import argparse
import json

import numpy as np
import torch

CONFIG = {"grid": 128, "extent": 64.0, "dt": 0.1, "levels": 5, "pp": 2.0, "knn": 16, "eps": 0.5,
          "in_scale": 1.0, "momfix": True, "potential": True, "kernel": True, "split": True, "verlet": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/budgetB/sfv/B_ms_kp_pot_v_g128.pt")
    ap.add_argument("--out", default="web/scatter/weights")
    args = ap.parse_args()

    st = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = st["model"]
    tensors, chunks, offset = {}, [], 0
    for name, t in state.items():
        a = t.detach().numpy().astype("<f4")
        tensors[name] = {"shape": list(a.shape), "offset": offset}
        chunks.append(a.reshape(-1))
        offset += a.size
    np.concatenate(chunks).tofile(args.out + ".bin")
    with open(args.out + ".json", "w") as f:
        json.dump({"checkpoint": args.checkpoint.split("/")[-1], "iters": st.get("it"), "config": CONFIG, "tensors": tensors}, f, indent=1)
    print(f"{len(tensors)} tensors, {offset} floats -> {args.out}.bin")


if __name__ == "__main__":
    main()
