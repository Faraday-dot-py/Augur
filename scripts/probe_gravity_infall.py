import numpy as np
import torch

from model.token_free import TokenFreeDynamics
from scripts import gravity_sim as gs

p0, v0 = gs.init_bodies(1000, np.random.default_rng(9000), scale=True)
dyn = TokenFreeDynamics(n=1000, neighbor_radius=100.0, pair_impulse=True)
dyn.load_state_dict(torch.load("checkpoints/gravity_dynamics_v2.pt", map_location="cpu"))
mp, mv = torch.tensor(p0, dtype=torch.float32), torch.tensor(v0, dtype=torch.float32)
with torch.no_grad():
    dp, dv, _ = dyn(mp, mv, torch.zeros(1000, dyn.hidden_dim))
dv = dv.numpy()
a = gs.accel(p0, 0.5) * 0.1
c = p0.mean(0)
rad = p0 - c
r = np.linalg.norm(rad, axis=1)
u = rad / r[:, None]
tr, mr = (a * u).sum(1), (dv * u).sum(1)
for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 90)]:
    s = (r >= lo) & (r < hi)
    print(f"r {lo}-{hi} n={s.sum()}: truth radial dv {tr[s].mean():.4f} model {mr[s].mean():.4f} model tangential rms {np.sqrt(((dv[s]-mr[s,None]*u[s])**2).mean()):.4f}")
print("corr", np.corrcoef(a.ravel(), dv.ravel())[0, 1], "|truth| rms", np.sqrt((a**2).mean()), "|model| rms", np.sqrt((dv**2).mean()))
