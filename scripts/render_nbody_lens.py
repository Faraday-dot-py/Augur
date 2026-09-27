"""Gravitational-lensing render of a 2D nbody/black-hole snapshot npz (pos, steps). Same window/brightness as render_nbody_snaps.py, then the image plane is warped
with a softened point-mass lens centred on the window centre: source position beta = u (1 - E^2 / (|u|^2 + a^2)), u = image position in window half-widths
(E = --einstein, a = --core). A fixed starfield (window coordinates, so it does not jitter with the moving window) is warped the same way, so the bending is visible;
surface brightness is conserved (no extra magnification factor). A black disk of radius --shadow hides the centre. Artistic, not a GR ray trace: E is a fraction of the view.

Usage: PYTHONPATH=. python3 scripts/render_nbody_lens.py snaps.npz out.mp4 [--fps 30] [--res 1080] [--preview IDX out.png] [--einstein 0.35] [--core 0.08] [--shadow 0.05] [--dt 0.005] [--min-half 3] [--label TEXT]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out")
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--res", type=int, default=1080)
ap.add_argument("--blur", type=float, default=1.2)
ap.add_argument("--einstein", type=float, default=0.35)
ap.add_argument("--core", type=float, default=0.08)
ap.add_argument("--shadow", type=float, default=0.05)
ap.add_argument("--stars", type=int, default=6000)
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--min-half", type=float, default=3.0)
ap.add_argument("--label", default="learned force, c=40")
ap.add_argument("--preview", type=int, default=-1)
args = ap.parse_args()
d = np.load(args.npz)
frames, steps = d["pos"], d["steps"]
n, res = frames.shape[1], args.res
centre = np.stack([np.median(f, axis=0) for f in frames])
radius = np.array([np.percentile(np.linalg.norm(f - c, axis=1), 45) for f, c in zip(frames, centre)])
centre = gaussian_filter1d(centre, 4, axis=0, mode="nearest")
half = np.maximum(3.5 * gaussian_filter1d(radius, 4, mode="nearest"), args.min_half)


def density(i):
    e = [np.linspace(centre[i, k] - half[i], centre[i, k] + half[i], res + 1) for k in range(2)]
    h = np.histogram2d(frames[i][:, 0], frames[i][:, 1], bins=e)[0].astype(np.float32)
    return gaussian_filter(h, args.blur).T / (2 * half[i] / res) ** 2


peaks = [np.percentile(density(i), 99.95) for i in range(0, len(frames), max(len(frames) // 40, 1))]
vmax = np.log1p(0.5 * max(peaks))

rng = np.random.default_rng(4738)
ts = 2 * res
tex = np.zeros((ts, ts), dtype=np.float32)
sx, sy = rng.integers(0, ts, args.stars), rng.integers(0, ts, args.stars)
np.add.at(tex, (sy, sx), (rng.pareto(2.0, args.stars) + 1).clip(max=12).astype(np.float32))
tex = gaussian_filter(tex, 1.1)
tex = (tex / np.percentile(tex, 99.99)).clip(0, 1) ** 0.6

yy, xx = (np.mgrid[0:res, 0:res].astype(np.float64) + 0.5) / res * 2 - 1
th2 = xx ** 2 + yy ** 2
fac = 1 - args.einstein ** 2 / (th2 + args.core ** 2)
bx, by = xx * fac, yy * fac
img_coords = np.stack([(by + 1) / 2 * res - 0.5, (bx + 1) / 2 * res - 0.5])
tex_coords = np.stack([(by + 2) / 4 * ts - 0.5, (bx + 2) / 4 * ts - 0.5])
stars = map_coordinates(tex, tex_coords, order=1, cval=0.0)
shadow = th2 > args.shadow ** 2


def render(i):
    s = np.log1p(density(i)) / vmax
    warped = map_coordinates(s, img_coords, order=1, cval=0.0).clip(0, 1)
    rgb = plt.get_cmap("inferno")(warped)[..., :3] * warped[..., None]
    rgb = rgb + stars[..., None] * np.array([0.55, 0.65, 1.0]) * 0.7
    return (rgb.clip(0, 1) * shadow[..., None]).astype(np.float32)


fig = plt.figure(figsize=(res / 100, res / 100), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
im = ax.imshow(np.zeros((res, res, 3), dtype=np.float32), origin="lower")
label = ax.text(0.02, 0.98, "", color="white", transform=ax.transAxes, va="top", fontsize=16)


def update(i):
    im.set_data(render(i))
    label.set_text(f"{n:,} bodies ({args.label}), step {steps[i]}, t = {steps[i] * args.dt:.1f}")


if args.preview >= 0:
    update(args.preview)
    fig.savefig(args.out, dpi=100, facecolor="black")
else:
    ani = animation.FuncAnimation(fig, update, frames=len(frames))
    ani.save(args.out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=12000), savefig_kwargs={"facecolor": "black"})
