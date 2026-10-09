import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("results/scatter_bh_1b_snaps.npz")
P = z["pos"]
print(P.shape, P.dtype, "t0 absmax", np.abs(P[0]).max())
x0 = P[0]
cx = np.abs(x0[:, 0]) == 29.0; cy = np.abs(x0[:, 1]) == 29.0
clip = cx | cy
print("clipped in subsample", clip.sum(), "expected", 1e6 * (1 - (1 - 2 * 1.4e-4 * 1.0) ** 2) if False else None)
out = {"n_clipped_subsample": int(clip.sum()), "frac_clipped": float(clip.mean())}
V = (P[1:] - P[:-1]) / 0.1
sp = np.linalg.norm(V, axis=2)
out["mean_speed_t1_t10_t50"] = [float(sp[0].mean()), float(sp[9].mean()), float(sp[49].mean())]
out["frac_outside_box32"] = {int(t): float((np.abs(P[t]) > 32).any(1).mean()) for t in (0, 5, 14, 29, 57, 71, 100)}
rev = {}
for lo, hi in ((20, 28), (28, 31), (31, 33), (33, 36), (36, 45)):
    cnt = tot = 0
    for ax in (0, 1):
        x = P[1:-1, :, ax]; v1 = V[:-1, :, ax]; v2 = V[1:, :, ax]
        inb = (np.abs(x) >= lo) & (np.abs(x) < hi)
        og = inb & (np.sign(v1) == np.sign(x)) & (np.abs(v1) > 1.0)
        fl = og & (np.sign(v2) != np.sign(v1)) & (np.abs(v2) > 1.0)
        cnt += int(fl.sum()); tot += int(og.sum())
    rev[f"{lo}-{hi}"] = dict(outgoing=tot, reversed=cnt, frac=cnt / max(tot, 1))
out["reversal_axis"] = rev
prof = []
for lo in range(20, 50, 2):
    s = []
    for ax in (0, 1):
        x = np.abs(P[:-1, :, ax]); vr = V[:, :, ax] * np.sign(P[:-1, :, ax])
        s.append(vr[(x >= lo) & (x < lo + 2)])
    s = np.concatenate(s)
    prof.append([lo, float(s.mean()), float((s < 0).mean()), int(s.size)])
out["axis_outward_v_profile[lo,mean,frac_inward,count]"] = prof
json.dump(out, open("results/square_wall/square_wall_1b.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:2500])

fig, ax = plt.subplots(2, 4, figsize=(16, 8))
def dens(Q, hw, a, title):
    h, _, _ = np.histogram2d(Q[:, 0], Q[:, 1], bins=256, range=[[-hw, hw], [-hw, hw]])
    a.imshow(np.log1p(h).T, origin="lower", cmap="viridis", extent=[-hw, hw, -hw, hw]); a.set_title(title, fontsize=9)
for i, (t, hw) in enumerate(((14, 32), (14, 60), (29, 32), (29, 100))):
    dens(P[t], hw, ax[0, i], f"1e9 subsample t{t} +-{hw}")
t = 14
ax[1, 0].scatter(P[t][~clip][::20, 0], P[t][~clip][::20, 1], s=.2, c="gray")
ax[1, 0].scatter(P[t][clip][:, 0], P[t][clip][:, 1], s=4, c="r"); ax[1, 0].set_xlim(-60, 60); ax[1, 0].set_ylim(-60, 60); ax[1, 0].set_title("t14: IC-clipped bodies (red) vs 5% others")
ax[1, 1].hist(P[14][:, 0], bins=np.linspace(-60, 60, 241), log=True, histtype="step"); ax[1, 1].axvline(32); ax[1, 1].axvline(-32); ax[1, 1].set_title("t14 x-marginal")
ax[1, 2].hist(P[0][:, 0], bins=np.linspace(-34, 34, 273), log=True, histtype="step"); ax[1, 2].set_title("t0 x-marginal")
p = np.array(out["axis_outward_v_profile[lo,mean,frac_inward,count]"]); ax[1, 3].plot(p[:, 0] + 1, p[:, 2]); ax[1, 3].axvline(32); ax[1, 3].set_title("frac inward vs |coord|")
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_1b.png", dpi=70)
