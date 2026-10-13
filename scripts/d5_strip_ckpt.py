"""Keep only the keys of a plain ScatterField checkpoint (drops the contact heads and gin) so older harness code loads it. usage: d5_strip_ckpt.py ref.pt in.pt out.pt"""
import sys

import torch

ref = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
st = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
out = {k: v for k, v in st["model"].items() if k in ref}
print("kept", len(out), "dropped", sorted(set(st["model"]) - set(ref)))
torch.save({"model": out}, sys.argv[3])
