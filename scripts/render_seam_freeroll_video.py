"""Short free-rollout video with the 32-unit tile seams drawn: truth, tiled baseline, tier 0 windows with exact external field (matched-60k and no-velocity seed 4738).
Particles that changed owner window in the last tick are drawn red. Reads the history npz from scripts/exp12_seam_freeroll.py --hist-out; writes frames to a temp dir and encodes with ffmpeg.
Usage: python3 scripts/render_seam_freeroll_video.py hist_all.npz videos/seam_freeroll_tiled_hierarchy.mp4 --seed 9200 --scene 0"""
import argparse
import os
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=9200)
ap.add_argument("--scene", type=int, default=0)
ap.add_argument("--stride", type=int, default=2)
a = ap.parse_args()
d = np.load(a.npz)
arms = [("truth", None), ("baseline", "tiled scatter baseline + far kernel"), ("matched60k", "tier 0 matched-60k, exact ext field"), ("novel4738", "tier 0 no-velocity/no-recurrent s4738")]
mk = d[f"mask_{a.seed}"][a.scene] > 0
T = d[f"truth_{a.seed}_P"].shape[1]
tmp = tempfile.mkdtemp()
n = 0
for t in range(0, T, a.stride):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for j, (k, title) in enumerate(arms):
        P = (d[f"truth_{a.seed}_P"] if k == "truth" else d[f"{k}_{a.seed}_P"])[a.scene, t][mk]
        col = "k"
        if k != "truth" and t > 0:
            own = d[f"{k}_{a.seed}_own"][a.scene]
            ch = (own[t] != own[t - 1])[mk]
            col = np.where(ch, "r", "tab:blue")
        ax[j].scatter(P[:, 0], P[:, 1], s=4, c=col)
        for s in (-32, 0, 32):
            ax[j].axvline(s, color="gray", lw=0.6)
            ax[j].axhline(s, color="gray", lw=0.6)
        ax[j].set_xlim(-64, 64)
        ax[j].set_ylim(-64, 64)
        ax[j].set_aspect("equal")
        ax[j].set_title(("ground truth" if k == "truth" else title) + f"  tick {t}", fontsize=8)
        if k != "truth":
            e = np.linalg.norm(d[f"{k}_{a.seed}_P"][a.scene, t][mk] - d[f"truth_{a.seed}_P"][a.scene, t][mk], axis=-1).mean()
            ax[j].text(-62, 56, f"mean pos err {e:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(tmp, f"f{n:04d}.png"), dpi=70)
    plt.close(fig)
    n += 1
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", os.path.join(tmp, "f%04d.png"), "-pix_fmt", "yuv420p", a.out], check=True)
print(n, "frames", a.out, tmp)
