"""Wrapper around scripts/nbody_rollout.py adding --perturb EPS (uniform position noise, seed 4738) for the exact-run chaos noise floor.

Usage: PYTHONPATH=. python scripts/genic_rollout.py --perturb 1e-6 <nbody_rollout args>
"""
import sys

import torch

from scripts import nbody_ic
from scripts import nbody_rollout

argv = sys.argv[1:]
eps = 0.0
if "--perturb" in argv:
    i = argv.index("--perturb")
    eps = float(argv[i + 1])
    del argv[i:i + 2]
if eps > 0:
    for name, fn in list(nbody_ic.IC.items()):
        def wrapped(n, kernel, dev, seed=4738, _fn=fn):
            pos, vel = _fn(n, kernel, dev, seed)
            g = torch.Generator(device=dev).manual_seed(seed + 1)
            return pos + eps * (torch.rand(pos.shape, generator=g, device=dev, dtype=pos.dtype) - 0.5) * 2, vel
        nbody_ic.IC[name] = wrapped
sys.argv = [sys.argv[0]] + argv
nbody_rollout.main()
