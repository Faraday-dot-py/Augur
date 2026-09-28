"""Export a CentralForceDynamics checkpoint (model/central_force.py) to
web/gravity/weights.bin (fp32 little-endian, tensors concatenated) +
web/gravity/weights.json (manifest + config). Only the pairwise force MLP
(3 Linear layers) is written -- there is no wall/gravity head, gravity has
no walls or box during training.

Usage:
    PYTHONPATH=. python3 scripts/export_web_weights_central.py --checkpoint checkpoints/gravity_central_v1.pt
"""
import argparse
import json

import numpy as np
import torch

CONFIG = {"dt": 0.1, "neighbor_radius": 100.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--out", default="web/gravity/weights")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("model", state)
    tensors, chunks, offset = {}, [], 0
    for name, t in state.items():
        a = t.detach().numpy().astype("<f4")
        tensors[name] = {"shape": list(a.shape), "offset": offset}
        chunks.append(a.reshape(-1))
        offset += a.size
    np.concatenate(chunks).tofile(args.out + ".bin")
    with open(args.out + ".json", "w") as f:
        json.dump({"checkpoint": args.checkpoint.split("/")[-1], "config": CONFIG, "tensors": tensors}, f, indent=1)
    print(f"{len(tensors)} tensors, {offset} floats -> {args.out}.bin")


if __name__ == "__main__":
    main()
