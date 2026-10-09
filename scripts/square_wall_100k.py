import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("results/scatter_bh_100000_snaps.npz", mmap_mode="r")
M = np.asarray(z["pos"]); T = np.asarray(z["truth_pos"])
n = M.shape[1]
rng = np.random.default_rng(4738)
pos0 = np.clip(rng.normal(0.0, 8.0, (n, 2)), -29.0, 29.0)
print("IC match model t0:", np.abs(pos0 - M[0]).max(), "truth t0:", np.abs(pos0 - T[0]).max())
clipx = np.abs(pos0[:, 0]) == 29.0
clipy = np.abs(pos0[:, 1]) == 29.0
clip = clipx | clipy
print("clipped bodies", clip.sum(), "of", n, "x", clipx.sum(), "y", clipy.sum())
out = {"n_clipped": int(clip.sum()), "n_clipx": int(clipx.sum()), "n_clipy": int(clipy.sum())}

def radial(P, t, m=None):
    r = np.linalg.norm(P[t], axis=1)
    return r if m is None else r[m]

for name, P in (("model", M), ("truth", T)):
    out[name] = {}
    for t in (14, 29, 43, 57, 71, 86, 100):
        r = radial(P, t)
        rc = radial(P, t, clip)
        rn = radial(P, t, ~clip)
        out[name][t] = dict(rmax=float(r.max()), r99=float(np.percentile(r, 99)), r_clip_med=float(np.median(rc)), r_clip_p5=float(np.percentile(rc, 5)), r_clip_p95=float(np.percentile(rc, 95)),
                            r_nonclip_p99=float(np.percentile(rn, 99)),
                            frac_inside_box32=float((np.abs(P[t]) <= 32).all(1).mean()),
                            frac_clip_inside_box32=float((np.abs(P[t][clip]) <= 32).all(1).mean()))
    # speeds from position differences
    V = (P[1:] - P[:-1]) / 0.1
    sp = np.linalg.norm(V, axis=2)
    out[name]["speed_mean_t1_10_50"] = [float(sp[0].mean()), float(sp[9].mean()), float(sp[49].mean())]
    # axis-velocity sign reversal near edges: body at |x| in [28,36] at t, vx sign reverses between t-1..t and t..t+1 with sign(vx)==sign(x) before (outgoing) and opposite after
    rev = {}
    for lo, hi in ((20, 28), (28, 31), (31, 33), (33, 36), (36, 45)):
        cnt = 0; tot = 0
        for ax in (0, 1):
            x = P[1:-1, :, ax]; v1 = V[:-1, :, ax]; v2 = V[1:, :, ax]
            inb = (np.abs(x) >= lo) & (np.abs(x) < hi)
            outgoing = inb & (np.sign(v1) == np.sign(x)) & (np.abs(v1) > 1.0)
            flipped = outgoing & (np.sign(v2) != np.sign(v1)) & (np.abs(v2) > 1.0)
            cnt += int(flipped.sum()); tot += int(outgoing.sum())
        rev[f"{lo}-{hi}"] = dict(outgoing=tot, reversed=cnt, frac=cnt / max(tot, 1))
    out[name]["reversal_axis"] = rev

# net radial velocity of bodies as function of |x| at |x| (axis-wise) , outgoing speed ratio
for name, P in (("model", M), ("truth", T)):
    V = (P[1:] - P[:-1]) / 0.1
    bins = np.arange(20, 50, 2)
    prof = []
    for lo in bins:
        sel = []
        for ax in (0, 1):
            x = np.abs(P[:-1, :, ax]); vr = V[:, :, ax] * np.sign(P[:-1, :, ax])
            m = (x >= lo) & (x < lo + 2)
            sel.append(vr[m])
        s = np.concatenate(sel)
        prof.append([float(lo), float(s.mean()), float((s < 0).mean()), int(s.size)])
    out[name]["axis_outward_v_profile[lo,mean,frac_inward,count]"] = prof

json.dump(out, open("results/square_wall/square_wall_100k.json", "w"), indent=1)

fig, ax = plt.subplots(3, 4, figsize=(16, 12))
def dens(P, hw, ax_, title, lw=None):
    h, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=256, range=[[-hw, hw], [-hw, hw]])
    ax_.imshow(np.log1p(h).T, origin="lower", cmap="inferno", extent=[-hw, hw, -hw, hw])
    ax_.set_title(title, fontsize=9)
for i, (t, hw) in enumerate(((57, 40), (71, 40), (71, 100), (86, 100))):
    dens(T[t], hw, ax[0, i], f"truth t{t} window +-{hw}")
    dens(M[t], hw, ax[1, i], f"model t{t} window +-{hw}")
for a in ax[:2].ravel():
    a.axvline(32, c="c", lw=.5); a.axvline(-32, c="c", lw=.5); a.axhline(32, c="c", lw=.5); a.axhline(-32, c="c", lw=.5)
r = np.linspace(0, 100, 101)
for name, P, col in (("truth", T, "k"), ("model", M, "r")):
    for t, ls in ((57, "-"), (71, "--"), (86, ":")):
        ax[2, 0].plot(r[:-1] + .5, np.histogram(np.linalg.norm(P[t], axis=1), bins=r)[0], col, ls=ls, label=f"{name} t{t}")
ax[2, 0].set_yscale("log"); ax[2, 0].legend(fontsize=6); ax[2, 0].set_title("radial count histogram")
for name, P, col in (("truth", T, "k"), ("model", M, "r")):
    ax[2, 1].plot(np.linspace(-60, 60, 241)[:-1], np.histogram(P[71][:, 0], bins=np.linspace(-60, 60, 241))[0], col, label=name)
ax[2, 1].set_yscale("log"); ax[2, 1].axvline(32); ax[2, 1].axvline(-32); ax[2, 1].set_title("t71 x-marginal"); ax[2, 1].legend()
for name, P, col in (("truth", T, "k"), ("model", M, "r")):
    ax[2, 2].plot(np.linalg.norm(P[:, clip], axis=2).mean(1), col, label=f"{name} clipped bodies mean r")
    ax[2, 2].plot(np.linalg.norm(P[:, ~clip], axis=2).mean(1), col + "--", label=f"{name} others mean r")
ax[2, 2].legend(fontsize=6); ax[2, 2].set_xlabel("tick")
prof = out
for name, col in (("truth", "k"), ("model", "r")):
    p = np.array(out[name]["axis_outward_v_profile[lo,mean,frac_inward,count]"])
    ax[2, 3].plot(p[:, 0] + 1, p[:, 2], col, label=name)
ax[2, 3].axvline(32); ax[2, 3].set_title("frac of axis-velocity samples moving inward vs |coord|"); ax[2, 3].legend()
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_100k.png", dpi=70)
