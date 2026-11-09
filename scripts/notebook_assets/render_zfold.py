import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

INK, BRICK = "#1f1b17", "#a43a2b"
OUT = "docs/design/assets/"
ink_map = LinearSegmentedColormap.from_list("ink", [(0.12, 0.11, 0.09, 0.0), (0.12, 0.11, 0.09, 1.0)])
red_map = LinearSegmentedColormap.from_list("red", [(0.64, 0.23, 0.17, 0.0), (0.64, 0.23, 0.17, 1.0)])
div_map = LinearSegmentedColormap.from_list("div", [(0.64, 0.23, 0.17, 1.0), (0.64, 0.23, 0.17, 0.0), (0.12, 0.11, 0.09, 0.0), (0.12, 0.11, 0.09, 1.0)])


def save(fig, name, dpi=200):
    fig.savefig(OUT + name, dpi=dpi, transparent=True)
    plt.close(fig)


def fig_ax(lim):
    fig = plt.figure(figsize=(2.4, 2.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.axis("off")
    return fig, ax


def frame(ax, lim):
    ax.plot([-lim, lim, lim, -lim, -lim], [-lim, -lim, lim, lim, -lim], c=INK, lw=0.8)


def trace_panels():
    d = json.load(open("results/notebook_assets/trace_n100_tick21.json"))
    G = d["G"]; E = G * d["h"] / 2
    pos = np.array(d["pos"]).reshape(-1, 2)
    acc = np.array(d["accel"]).reshape(-1, 2)
    v = 29.0
    def grid(k): return np.array(d[k]).reshape(G, G)
    c0, c1 = int(G / 2 - v / d["h"]), int(G / 2 + v / d["h"])
    ext = [-E, E, -E, E]
    fig, ax = fig_ax(v); frame(ax, v - 0.2)
    ax.scatter(pos[:, 0], pos[:, 1], s=14, c=INK, alpha=0.9, linewidths=0)
    save(fig, "zfold_bodies.png")
    fig, ax = fig_ax(v); frame(ax, v - 0.2)
    rho = grid("density")
    ax.imshow(np.log1p(rho / rho.max() * 30) / np.log(31), cmap=ink_map, origin="lower", extent=ext, interpolation="nearest")
    save(fig, "zfold_density.png")
    fig, ax = fig_ax(v); frame(ax, v - 0.2)
    phi = grid("phi")
    phi = phi - float(np.median(phi[c0:c1, c0:c1]))
    s = float(np.percentile(np.abs(phi[c0:c1, c0:c1]), 99))
    ax.imshow(phi, cmap=div_map, origin="lower", extent=ext, interpolation="bilinear", vmin=-s, vmax=s)
    save(fig, "zfold_potential.png")
    fig, ax = fig_ax(v); frame(ax, v - 0.2)
    mag = np.hypot(np.array(d["aGrid"][0]).reshape(G, G), np.array(d["aGrid"][1]).reshape(G, G))
    s = float(np.percentile(mag[c0:c1, c0:c1], 99))
    ax.imshow(np.clip(mag / s, 0, 1) ** 0.6, cmap=red_map, origin="lower", extent=ext, interpolation="bilinear")
    save(fig, "zfold_grid_force.png")
    fig, ax = fig_ax(v); frame(ax, v - 0.2)
    ax.scatter(pos[:, 0], pos[:, 1], s=14, c=INK, alpha=0.9, linewidths=0)
    sc = 3.0 / np.percentile(np.hypot(acc[:, 0], acc[:, 1]), 90)
    ax.quiver(pos[:, 0], pos[:, 1], acc[:, 0] * sc, acc[:, 1] * sc, angles="xy", scale_units="xy", scale=1, color=BRICK, width=0.006, headwidth=3.5, headlength=4)
    save(fig, "zfold_body_forces.png")


def loop_panel():
    fig, ax = fig_ax(1.0)
    th = np.linspace(0.35, 2 * np.pi - 0.35, 200)
    ax.plot(0.62 * np.cos(th), 0.62 * np.sin(th), c=BRICK, lw=3.5, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((0.62 * np.cos(th[-1]), 0.62 * np.sin(th[-1])), (0.62 * np.cos(th[-1] + 0.1), 0.62 * np.sin(th[-1] + 0.1)), arrowstyle="-|>", mutation_scale=34, color=BRICK, lw=0))
    ax.scatter([0.62 * np.cos(a) for a in (0.9, 2.4, 4.0)], [0.62 * np.sin(a) for a in (0.9, 2.4, 4.0)], s=60, c=INK)
    ax.set_aspect("equal")
    save(fig, "zfold_repeat.png")


def seam_panel():
    d = np.load("results/seamvideo8.npz")
    nt, tile = int(d["nt"]), float(d["tile"])
    L = nt * tile; lim = 90
    p = d["tiled"][30]
    fig, ax = fig_ax(lim)
    for s in range(nt + 1):
        c = -L / 2 + s * tile
        ax.plot([c, c], [-L / 2, L / 2], c=BRICK, lw=0.8 if 0 < s < nt else 1.4)
        ax.plot([-L / 2, L / 2], [c, c], c=BRICK, lw=0.8 if 0 < s < nt else 1.4)
    ax.scatter(p[:, 0], p[:, 1], s=3.5, c=INK, alpha=0.8, linewidths=0)
    save(fig, "zfold_tiles.png")


def density_panel():
    h = np.load("results/showcase_bh_1e10_kpadext_hist.npz")
    a = np.log1p(h["full"][-1].astype(np.float64))
    a = (a / a.max()) ** 0.35
    fig = plt.figure(figsize=(2.4, 2.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(a, cmap=ink_map, origin="lower", interpolation="bilinear")
    save(fig, "zfold_1e10.png")


trace_panels()
loop_panel()
seam_panel()
density_panel()
