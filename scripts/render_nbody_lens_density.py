"""NOTE: per explicit user request (2026-09-28), no blur (see docs/debugging/render-lensing-no-blur.md).

Same lens as render_nbody_lens.py (no-blur path) but the per-frame density comes from precomputed histograms of ALL bodies (scatter_bh_1b.py --lens-out) instead of particle positions.
hist: .npy memmap [F, res, res] float32 counts, indexed [y, x]; windows: .npy [F, 3] (centre x, centre y, half-width), the same tracked-window rule as render_nbody_lens.py.

Usage: python3 scripts/render_nbody_lens_density.py hist.npy windows.npy out.mp4 --n 1000000000 [--fps 15] [--preview IDX out.png]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

ap = argparse.ArgumentParser()
ap.add_argument("hist")
ap.add_argument("windows")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=10 ** 9)
ap.add_argument("--fps", type=float, default=15)
ap.add_argument("--einstein", type=float, default=0.35)
ap.add_argument("--core", type=float, default=0.08)
ap.add_argument("--stars", type=int, default=6000)
ap.add_argument("--dt", type=float, default=0.1)
ap.add_argument("--label", default="scatter field mesh-only, N=1e9 (all bodies), c=10")
ap.add_argument("--preview", type=int, default=-1)
ap.add_argument("--title", default=None, help="clean overlay: TITLE | t = ... (replaces the diagnostic label)")
args = ap.parse_args()
H = np.load(args.hist, mmap_mode="r")
win = np.load(args.windows)
nf, res = H.shape[0], H.shape[1]
half = win[:, 2]


def density(i):
    return np.asarray(H[i], dtype=np.float32) / (2 * half[i] / res) ** 2


step = max(nf // 40, 1)
vmax = np.log1p(0.5 * max(np.percentile(density(i), 99.95) for i in range(0, nf, step)))

rng = np.random.default_rng(4738)
ts = 2 * res
tex = np.zeros((ts, ts), dtype=np.float32)
sx, sy = rng.integers(0, ts, args.stars), rng.integers(0, ts, args.stars)
np.add.at(tex, (sy, sx), (rng.pareto(2.0, args.stars) + 1).clip(max=12).astype(np.float32))
tex = gaussian_filter(tex, 1.1)
tex = (tex / np.percentile(tex, 99.99)).clip(0, 1) ** 0.6

yy, xx = (np.mgrid[0:res, 0:res].astype(np.float64) + 0.5) / res * 2 - 1
pad = 2 * res
off = np.fft.fftfreq(pad) * pad * (2.0 / res)
oy, ox = np.meshgrid(off, off, indexing="ij")
den = ox ** 2 + oy ** 2 + args.core ** 2
kx_f, ky_f = np.fft.rfft2(ox / den), np.fft.rfft2(oy / den)


def deflection(dens):
    m = np.zeros((pad, pad))
    m[:res, :res] = dens / dens.sum()
    mf = np.fft.rfft2(m)
    return (args.einstein ** 2 * np.fft.irfft2(mf * kx_f, s=(pad, pad))[:res, :res],
            args.einstein ** 2 * np.fft.irfft2(mf * ky_f, s=(pad, pad))[:res, :res])


def render(i):
    dens = density(i)
    ax_, ay_ = deflection(dens)
    bx, by = xx - ax_, yy - ay_
    img_coords = np.stack([(by + 1) / 2 * res - 0.5, (bx + 1) / 2 * res - 0.5])
    tex_coords = np.stack([(by + 2) / 4 * ts - 0.5, (bx + 2) / 4 * ts - 0.5])
    stars = map_coordinates(tex, tex_coords, order=1, cval=0.0)
    warped = map_coordinates(np.log1p(dens) / vmax, img_coords, order=1, cval=0.0).clip(0, 1)
    rgb = plt.get_cmap("inferno")(warped)[..., :3] * warped[..., None]
    rgb = rgb + stars[..., None] * np.array([0.55, 0.65, 1.0]) * 0.7
    return rgb.clip(0, 1).astype(np.float32)


fig = plt.figure(figsize=(res / 100, res / 100), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
im = ax.imshow(np.zeros((res, res, 3), dtype=np.float32), origin="lower")
label = ax.text(0.02, 0.98, "", color="white", transform=ax.transAxes, va="top", fontsize=16)


def update(i):
    im.set_data(render(i))
    label.set_text(f"{args.title}   |   {args.n:,} bodies   |   t = {i * args.dt:.1f}" if args.title else f"{args.n:,} bodies ({args.label}), step {i}, t = {i * args.dt:.1f}")


if args.preview >= 0:
    update(args.preview)
    fig.savefig(args.out, dpi=100, facecolor="black")
else:
    ani = animation.FuncAnimation(fig, update, frames=nf)
    ani.save(args.out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=12000), savefig_kwargs={"facecolor": "black"})
