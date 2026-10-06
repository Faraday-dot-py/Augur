"""Export a conservative-contact checkpoint (scripts/train_conservative.py) to
web/token/weights.bin (fp32 little-endian, tensors concatenated) + web/token/weights.json
(manifest + config). Only the tensors the conservative step uses are written.

Usage:
    PYTHONPATH=. python3 scripts/export_web_weights.py --checkpoint results/cons_pure.pt
"""
import argparse
import json

import numpy as np
import torch

USED = ("gravity", "pair_force.", "wall_force.")
CONFIG = {"neighbor_radius": 4.0, "radius": 0.75, "dt": 0.15, "force_scale": 100.0, "gravity": 9.0,
          "n": 100, "max_speed": 30.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/cons_pure.pt")
    ap.add_argument("--out", default="web/token/weights")
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu")
    flags = blob["flags"]
    if flags["residual"]:
        raise ValueError("contact_residual checkpoints are not supported by the web port")
    config = dict(CONFIG, contact_substeps=flags["substeps"], conservative_contact=True)
    tensors, chunks, offset = {}, [], 0
    for name, t in blob["model"].items():
        name = name.removeprefix("dynamics.")
        if not name.startswith(USED):
            continue
        a = t.detach().numpy().astype("<f4")
        tensors[name] = {"shape": list(a.shape), "offset": offset}
        chunks.append(a.reshape(-1))
        offset += a.size
    np.concatenate(chunks).tofile(args.out + ".bin")
    with open(args.out + ".json", "w") as f:
        json.dump({"checkpoint": args.checkpoint.split("/")[-1], "config": config, "tensors": tensors}, f, indent=1)
    print(f"{len(tensors)} tensors, {offset} floats -> {args.out}.bin")


if __name__ == "__main__":
    main()
