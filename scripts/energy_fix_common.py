"""Model construction shared by the energy-fix scripts: soup B (optionally with
adaptive/fixed-large pair radius) and conservative-contact checkpoints.

kind: "soupb" (checkpoints/token_model_soup_b.pt flags), "cons" (a checkpoint saved by
scripts/train_conservative.py, {"model": state, "flags": {...}}).
"""
import torch

from model.token_model import TokenModel
from scripts.eval_free_rollout import load_model


def build_model(kind, ckpt, n, device, adaptive_radius="off", neighbor_radius=4.0):
    if kind == "soupb":
        model = load_model(ckpt, "free", n, 32, neighbor_radius, False, True, True, True, True, True, True)
        model.dynamics.adaptive_radius = adaptive_radius
    else:
        blob = torch.load(ckpt, map_location="cpu")
        flags = blob["flags"]
        model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=32, neighbor_radius=neighbor_radius, free_rollout=True,
                           conservative_contact=True, contact_substeps=flags["substeps"],
                           contact_residual=flags["residual"])
        model.load_state_dict(blob["model"])
    model.dynamics.cell_graph = False
    return model.to(device).eval()
