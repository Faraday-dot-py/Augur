import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, BRICK = "#1f1b17", "#a43a2b"
OUT = "docs/design/assets/"
ink_map = LinearSegmentedColormap.from_list("ink", [(0.12, 0.11, 0.09, 0.0), (0.12, 0.11, 0.09, 1.0)])
div_map = LinearSegmentedColormap.from_list("div", [(0.64, 0.23, 0.17, 1.0), (0.64, 0.23, 0.17, 0.0), (0.12, 0.11, 0.09, 0.0), (0.12, 0.11, 0.09, 1.0)])


def save(fig, name, dpi=200):
    fig.savefig(OUT + name, dpi=dpi, transparent=True)
    plt.close(fig)


def fig_ax(w, h, lim_x, lim_y):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y); ax.axis("off")
    return fig, ax


def bodies_frames():
    d = np.load("results/showcase_bh_100k_kpadext_snaps.npz")
    m, t = d["pos"], d["truth_pos"]
    lim = float(np.percentile(np.abs(t[0]), 99.5)) * 1.05
    print("100k lim", lim, "steps", d["steps"][[0, 10, 20, 50, 100]])
    for tick in (0, 20, 50, 100):
        for name, p, c in (("exact", t, INK), ("learned", m, BRICK)):
            fig, ax = fig_ax(2.4, 2.4, lim, lim)
            ax.scatter(p[tick, :, 0], p[tick, :, 1], s=0.35, c=c, alpha=0.55, linewidths=0, rasterized=True)
            save(fig, f"bh100k_{name}_t{tick:03d}.png")
        fig, ax = fig_ax(2.4, 2.4, lim, lim)
        ax.scatter(m[tick, :, 0], m[tick, :, 1], s=0.25, c=BRICK, alpha=0.4, linewidths=0, rasterized=True)
        ax.scatter(t[tick, :, 0], t[tick, :, 1], s=0.25, c=INK, alpha=0.4, linewidths=0, rasterized=True)
        save(fig, f"bh100k_overlay_t{tick:03d}.png")
    fig, ax = fig_ax(6, 6, lim, lim)
    ax.scatter(m[50, :, 0], m[50, :, 1], s=0.6, c=BRICK, alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(t[50, :, 0], t[50, :, 1], s=0.6, c=INK, alpha=0.4, linewidths=0, rasterized=True)
    save(fig, "bh100k_hero_t050.png", dpi=220)


def density(arr, name, gamma=0.35):
    a = np.log1p(arr.astype(np.float64))
    a = (a / a.max()) ** gamma if a.max() > 0 else a
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(a.T[::-1] if False else a, cmap=ink_map, origin="lower", interpolation="bilinear")
    save(fig, name)


def scale_panels():
    d = np.load("results/bh_1e9_edge_free_ticks.npz")
    print("1e9 ticks", d["ticks"], "n", d["n"])
    density(d["full"][-1], "density_1e9.png")
    h = np.load("results/showcase_bh_1e10_kpadext_hist.npz")
    print("1e10 n", h["n"], "steps", h["steps"][[0, -1]])
    density(h["full"][-1], "density_1e10.png")


def trace_panels():
    d = json.load(open("results/notebook_assets/trace_n100_tick21.json"))
    G = d["G"]; E = G * d["h"] / 2
    pos = np.array(d["pos"]).reshape(-1, 2)
    acc = np.array(d["accel"]).reshape(-1, 2)
    W, H = 3.0, 2.07
    vx, vy = 29.0, 29.0 * H / W
    def grid(k): return np.array(d[k]).reshape(G, G)
    def base():
        fig, ax = fig_ax(W, H, vx, vy)
        return fig, ax
    fig, ax = base()
    ax.scatter(pos[:, 0], pos[:, 1], s=7, c=INK, alpha=0.9, linewidths=0)
    save(fig, "trace_bodies.png")
    fig, ax = base()
    rho = grid("density")
    ax.imshow(np.log1p(rho / rho.max() * 30) / np.log(31), cmap=ink_map, origin="lower", extent=[-E, E, -E, E], interpolation="nearest")
    save(fig, "trace_density.png")
    fig, ax = base()
    phi = grid("phi")
    c0, c1 = int(G / 2 - vx / d["h"]), int(G / 2 + vx / d["h"])
    phi = phi - float(np.median(phi[c0:c1, c0:c1]))
    v = float(np.percentile(np.abs(phi[c0:c1, c0:c1]), 99))
    ax.imshow(phi, cmap=div_map, origin="lower", extent=[-E, E, -E, E], interpolation="bilinear", vmin=-v, vmax=v)
    save(fig, "trace_phi.png")
    fig, ax = base()
    ax_, ay_ = grid("aGrid") if False else (np.array(d["aGrid"][0]).reshape(G, G), np.array(d["aGrid"][1]).reshape(G, G))
    mag = np.hypot(ax_, ay_)
    v = float(np.percentile(mag[c0:c1, c0:c1], 99))
    ax.imshow(np.clip(mag / v, 0, 1) ** 0.6, cmap=ink_map, origin="lower", extent=[-E, E, -E, E], interpolation="bilinear")
    save(fig, "trace_accel_grid.png")
    fig, ax = base()
    ax.scatter(pos[:, 0], pos[:, 1], s=7, c=INK, alpha=0.9, linewidths=0)
    sc = 1.0 / np.percentile(np.hypot(acc[:, 0], acc[:, 1]), 90) * 3.0
    ax.quiver(pos[:, 0], pos[:, 1], acc[:, 0] * sc, acc[:, 1] * sc, angles="xy", scale_units="xy", scale=1, color=BRICK, width=0.004, headwidth=3.5, headlength=4)
    save(fig, "trace_accel_bodies.png")
    print("trace: tick", d["tick"], "ckpt", d["checkpoint"], d["iters"], "max|phi|", float(np.abs(phi).max()), "momfix", d["momfix"])


bodies_frames()
scale_panels()
trace_panels()
