import sys

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import three_dimensional_sparse_voxel_scatter_field as t

ckpt, n, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = "cpu"
model = t.build(dev)
st = torch.load(ckpt, map_location=dev, weights_only=False)
model.load_state_dict(st["model"])
model.eval()
print("ckpt it", st["it"], flush=True)
for fam, (a, q) in t.FAMILIES.items():
    p3, v3 = t.plummer(n, a, q, np.random.default_rng([9100, n, 0]))
    P = torch.tensor(p3, dtype=torch.float32)[None]
    V = torch.tensor(v3, dtype=torch.float32)[None]
    M = torch.ones(1, n)
    for k in range(1, steps + 1):
        try:
            with torch.no_grad():
                P, V, _, _, _ = sf._step_m(model, P, V, M, M, model.init_field(1, dev))
        except RuntimeError as e:
            print(fam, "step", k, "RuntimeError", str(e)[:60], "last max|pos|", float(P.abs().max()), flush=True)
            break
        fin = bool(torch.isfinite(P).all())
        mx = float(P.abs().max()) if fin else float("nan")
        if k % 5 == 0 or not fin or mx > 200:
            print(fam, "step", k, "finite", fin, "max|pos|", mx, "max|vel|", float(V.abs().max()), flush=True)
        if not fin or mx > 1e4:
            break
