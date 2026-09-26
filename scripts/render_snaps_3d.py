"""3D video for an orbit_3d.py snapshot npz: camera orbits the scene (azimuth --turns full turns over the video, fixed elevation --elev), orthographic projection.
Brightness is log density; colour is the mean depth of the bodies in each pixel (near = warm, far = cool). Fixed cube of half-width --half around the
median start centre, with its edges drawn for depth reference. One frame image in memory at a time.

Usage: PYTHONPATH=. python3 scripts/render_snaps_3d.py results/orbit_X_snaps.npz videos/X_3d.mp4 [--res 1080] [--half 900] [--elev 25] [--turns 1] [--dt 0.01]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.ndimage import gaussian_filter

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out")
ap.add_argument("--res", type=int, default=1080)
ap.add_argument("--half", type=float, default=900)
ap.add_argument("--elev", type=float, default=25)
ap.add_argument("--turns", type=float, default=1.0)
ap.add_argument("--blur", type=float, default=1.5)
ap.add_argument("--dt", type=float, default=0.01)
args = ap.parse_args()
d = np.load(args.npz)
frames, steps = d["pos"], d["steps"]
n, res, half = frames.shape[1], args.res, args.half
centre = np.median(frames[0], axis=0)
el = np.radians(args.elev)
corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if (corners[a] != corners[b]).sum() == 1]


def view(p, i):
    az = 2 * np.pi * args.turns * i / max(len(frames) - 1, 1)
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    xr, yr = ca * x - sa * y, sa * x + ca * y
    return np.stack([xr, ce * z - se * yr, ce * yr + se * z], 1)


def image(i):
    v = view(frames[i] - centre, i)
    e = np.linspace(-half, half, res + 1)
    w = np.clip((v[:, 2] + half) / (2 * half), 0, 1)
    h = np.histogram2d(v[:, 0], v[:, 1], bins=[e, e])[0]
    hw = np.histogram2d(v[:, 0], v[:, 1], bins=[e, e], weights=w)[0]
    h, hw = gaussian_filter(h, args.blur), gaussian_filter(hw, args.blur)
    depth = hw / np.maximum(h, 1e-9)
    return h.T, depth.T


peak = max(np.percentile(image(i)[0], 99.95) for i in range(0, len(frames), 8))
fig = plt.figure(figsize=(res / 100, res / 100), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
ax.set_xlim(0, res)
ax.set_ylim(0, res)
im = ax.imshow(np.zeros((res, res, 3), dtype=np.float32), origin="lower", extent=(0, res, 0, res))
lines = [ax.plot([0, 0], [0, 0], color="white", alpha=0.25, lw=0.8)[0] for _ in edges]
label = ax.text(0.02, 0.98, "", color="white", transform=ax.transAxes, va="top", fontsize=16)
note = ax.text(0.02, 0.02, "colour = depth (warm near, cool far)", color="white", transform=ax.transAxes, va="bottom", fontsize=12)


def update(i):
    h, depth = image(i)
    rgb = cm.coolwarm_r(depth)[..., :3] * (np.log1p(h) / np.log1p(0.5 * peak)).clip(0, 1)[..., None]
    im.set_data(rgb.astype(np.float32))
    cv = view(corners, i)
    for ln, (a, b) in zip(lines, edges):
        ln.set_data((cv[[a, b], 0] + half) / (2 * half) * res, (cv[[a, b], 1] + half) / (2 * half) * res)
    label.set_text(f"{n:,} bodies, 3D, step {steps[i]}, t = {steps[i] * args.dt:.0f}")


ani = animation.FuncAnimation(fig, update, frames=len(frames))
ani.save(args.out, writer=animation.FFMpegWriter(fps=30, bitrate=12000), savefig_kwargs={"facecolor": "black"})
