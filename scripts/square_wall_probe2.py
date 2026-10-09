import json
import sys

import numpy as np
import torch

from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

dev = torch.device("cuda")
model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
model.load_state_dict(torch.load(sys.argv[1], map_location=dev, weights_only=False)["model"])
model.eval()
ops.no_pp(model)
n = 20000
rng = np.random.default_rng(4738)
p0 = np.clip(rng.normal(0.0, 8.0, (n, 2)), -29.0, 29.0)
xs = np.arange(14.0, 70.01, 0.25)
tp = np.stack([xs, np.zeros_like(xs)], 1)
pos = np.concatenate([p0, tp])
mass = np.concatenate([np.ones(n), np.zeros(len(tp))])


@torch.no_grad()
def parts(grid, extent, shift):
    ops.retarget(model, grid, extent)
    P = torch.tensor(pos + np.array([shift, 0.0]), dtype=torch.float32, device=dev)[None]
    M = torch.tensor(mass, dtype=torch.float32, device=dev)[None]
    mk = torch.ones_like(M)
    vel = torch.zeros_like(P)
    x = model.scatter(P, vel, M, mk)
    ns = model._nscale(x, M, mk) if model.norm else None
    if ns is not None:
        x = x / ns
    ak = model.kernel_acc(x[:, :1])
    x2 = torch.cat([x, ak], 1)
    if model.recurrent:
        x2 = torch.cat([x2, model.init_field(1, dev)[0]], 1)
    if model.epsin:
        import math
        x2 = torch.cat([x2, x2.new_full((1, 1, grid, grid), math.log(model.eps_cur / tsf.sf.EPS0))], 1)
    field = model.net(x2)
    an = model.neg_grad(field[:, :1])
    g = lambda a: (model.gather(a, P)[0, n:, 0] * (ns.view(-1)[0] if ns is not None else 1)).cpu().numpy()
    return {"kernel_only": g(ak), "net_only": g(an), "total": g(ak + an), "phi_net_row": field[0, 0, grid // 2, :].cpu().numpy().tolist()}


res = {"xs": xs.tolist()}
ex = ops.exact_acc(torch.tensor(pos, dtype=torch.float64, device=dev), 0.5, torch.tensor(mass, dtype=torch.float64, device=dev))[n:, 0].cpu().numpy()
res["exact"] = ex.tolist()
for name, grid, extent, shift in (("g128_e64", 128, 64.0, 0.0), ("g256_e128", 256, 128.0, 0.0), ("g128_e64_shift10", 128, 64.0, 10.0)):
    r = parts(grid, extent, shift)
    res[name] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
    if shift:
        res[name]["xs_shifted_note"] = "scene and tracers shifted +x by shift; x label is unshifted coordinate"
print("RES " + json.dumps(res))
print("DONE")
