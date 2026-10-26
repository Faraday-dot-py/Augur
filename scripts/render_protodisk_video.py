"""Video of the planetesimal-disk test: real design reference (face-on, edge-on) plus the in-range proxy
(direct N-body reference vs CentralForceDynamics), with per-frame element/energy overlays.
Stages: real | proxy (simulate, cache to npz) | render (stream frames to ffmpeg)."""
import argparse
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts")
from protodisk_gravity_test import G_REAL, accel, energy, ang_mom, real_ic, helio_elements, proxy_ic

NFRAMES = 500
DEV = torch.device('cpu')


def sim_real(args):
    rng = np.random.default_rng(args.seed)
    pos, vel, mass = real_ic(rng, args.n_disk, [5.0])
    p, v, m = (torch.tensor(x, dtype=torch.float64, device=DEV) for x in (pos, vel, mass))
    h = args.dt
    nsteps = int(round(args.years / h))
    per = nsteps // NFRAMES
    a = accel(p, m, args.eps, G_REAL)
    e0, l0 = energy(p, v, m, args.eps, G_REAL), ang_mom(p, v, m)
    P, A, E, DE, DL, T = [], [], [], [], [], []
    t0 = time.time()
    for s in range(0, nsteps + 1):
        if s % per == 0:
            pn, vn = p.cpu().numpy(), v.cpu().numpy()
            P.append(pn.astype(np.float32))
            aa, ee = helio_elements(pn[None], vn[None], mass, G_REAL)
            A.append(aa[0])
            E.append(ee[0])
            DE.append((energy(p, v, m, args.eps, G_REAL) - e0) / abs(e0))
            DL.append(float((ang_mom(p, v, m) - l0).norm() / l0.norm()))
            T.append(s * h)
            if len(P) % 50 == 0:
                print(f"real frame {len(P)} wall {time.time() - t0:.0f}s dE {DE[-1]:.2e}", flush=True)
        if s == nsteps:
            break
        v = v + 0.5 * h * a
        p = p + h * v
        a = accel(p, m, args.eps, G_REAL)
        v = v + 0.5 * h * a
    np.savez_compressed(args.out, pos=np.stack(P), a=np.stack(A), e=np.stack(E), dE=np.array(DE), dL=np.array(DL), t=np.array(T), mass=mass)


def sim_proxy(args):
    from model.central_force import CentralForceDynamics
    rng = np.random.default_rng(args.seed)
    ncore = args.n_core
    pos, vel = proxy_ic(rng, ncore, args.n_disk)
    n = len(pos)
    mass = torch.ones(n, dtype=torch.float64, device=DEV)
    sub = 4
    h = 0.1 / sub
    ticks = args.ticks
    per = ticks // NFRAMES
    p, v = torch.tensor(pos, device=DEV), torch.tensor(vel, device=DEV)
    a = accel(p, mass, 0.5, 1.0)
    e0 = energy(p, v, mass, 0.5, 1.0)
    RP, RDE = [p.cpu().numpy().astype(np.float32)], [0.0]
    t0 = time.time()
    for s in range(1, ticks * sub + 1):
        v = v + 0.5 * h * a
        p = p + h * v
        a = accel(p, mass, 0.5, 1.0)
        v = v + 0.5 * h * a
        if s % (per * sub) == 0:
            RP.append(p.cpu().numpy().astype(np.float32))
            RDE.append((energy(p, v, mass, 0.5, 1.0) - e0) / abs(e0))
            if len(RP) % 50 == 0:
                print(f"proxy ref frame {len(RP)} wall {time.time() - t0:.0f}s", flush=True)
    dyn = CentralForceDynamics(dt=0.1, neighbor_radius=100.0).to(DEV)
    dyn.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    dyn.to(DEV)
    mp, mv = torch.tensor(pos, dtype=torch.float32, device=DEV), torch.tensor(vel, dtype=torch.float32, device=DEV)
    hid = torch.zeros(n, 1, device=DEV)
    MP, MDE, MV = [mp.cpu().numpy().copy()], [0.0], [mv.cpu().numpy().copy()]
    me0 = energy(mp.double(), mv.double(), mass, 0.5, 1.0)
    with torch.no_grad():
        for s in range(1, ticks + 1):
            dpp, dvv, hid = dyn(mp, mv, hid)
            mp = mp + mv * 0.1 + dpp
            mv = mv + dvv
            if s % per == 0:
                MP.append(mp.cpu().numpy().copy())
                MV.append(mv.cpu().numpy().copy())
                MDE.append((energy(mp.double(), mv.double(), mass, 0.5, 1.0) - me0) / abs(me0))
                if len(MP) % 50 == 0:
                    print(f"proxy model frame {len(MP)} wall {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(args.out, ref=np.stack(RP), model=np.stack(MP), refdE=np.array(RDE), modeldE=np.array(MDE),
                        ncore=ncore, ticks_per_frame=per)


def proxy_stats(P, ncore):
    com = P[:, :ncore].mean(1, keepdims=True)
    r = np.linalg.norm(P[:, ncore:] - com, axis=-1)
    return np.median(r, 1), (r < 150).mean(1)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    R, X = np.load(args.real), np.load(args.proxy)
    nf = min(len(R["pos"]), len(X["ref"]), len(X["model"]), args.frames)
    ncore = int(X["ncore"])
    jup_a = R["a"][:, 0]
    med_e = np.nanmedian(R["e"][:, 1:], 1)
    med_a_ratio = None
    tyr = R["t"]
    disk_a0 = R["a"][0, 1:]
    bound_frac = (R["a"][:, 1:] > 0).mean(1)
    ref_med_r, ref_surv = proxy_stats(X["ref"][:nf], ncore)
    mod_med_r, mod_surv = proxy_stats(X["model"][:nf], ncore)
    disp = np.linalg.norm(X["model"][:nf] - X["ref"][:nf], axis=-1).mean(1)
    tk = np.arange(nf) * int(X["ticks_per_frame"])
    bg, fg, dim = "#10161a", "#c9d4cf", "#6f7f79"
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor=bg)
    gs = GridSpec(2, 3, figure=fig, left=0.04, right=0.98, top=0.9, bottom=0.07, wspace=0.18, hspace=0.3)
    ax = {k: fig.add_subplot(gs[i // 3, i % 3]) for i, k in enumerate(["face", "edge", "ts_real", "ref", "model", "ts_proxy"])}
    for a_ in ax.values():
        a_.set_facecolor(bg)
        for sp in a_.spines.values():
            sp.set_color(dim)
        a_.tick_params(colors=dim, labelsize=10)
    cmap = plt.get_cmap("viridis")
    col = cmap(np.clip(np.log10(np.clip(disk_a0, 0.3, 30)) / np.log10(30) * 0.5 + 0.5 - 0.15, 0, 1))
    n_d = len(disk_a0)
    L = 32
    P0 = R["pos"][0]
    sc_face = ax["face"].scatter(P0[1 + 1:, 0], P0[2:, 1], s=2, c=col, lw=0)
    ax["face"].scatter([0], [0], s=40, c="#ffd27a", marker="*", zorder=5)
    jf = ax["face"].scatter([P0[1, 0]], [P0[1, 1]], s=40, c="#ff7a59", zorder=6)
    ax["face"].set_xlim(-L, L)
    ax["face"].set_ylim(-L, L)
    ax["face"].set_aspect("equal")
    ax["face"].set_title("REAL DESIGN reference, face-on (AU): star, 1000 bodies, Jupiter at 5 AU", color=fg, fontsize=10)
    sc_edge = ax["edge"].scatter(P0[2:, 0], P0[2:, 2], s=2, c=col, lw=0)
    ax["edge"].scatter([0], [0], s=40, c="#ffd27a", marker="*", zorder=5)
    je = ax["edge"].scatter([P0[1, 0]], [P0[1, 2]], s=40, c="#ff7a59", zorder=6)
    ax["edge"].set_xlim(-L, L)
    ax["edge"].set_ylim(-1.5, 1.5)
    ax["edge"].set_title("REAL DESIGN reference, edge-on (AU, z stretched 20x)", color=fg, fontsize=10)
    ax["ts_real"].set_title("real reference: dE/E0 (Jupiter a, median disk e below)", color=fg, fontsize=10)
    ax["ts_real"].plot(tyr[:nf], np.abs(R["dE"][:nf]) + 1e-12, color="#7fb7a0", lw=1)
    ax["ts_real"].set_yscale("log")
    ax["ts_real"].set_ylabel("|dE/E0| (energy drift)", color=dim, fontsize=10)
    ax["ts_real"].set_xlabel("year", color=dim, fontsize=10)
    t2 = ax["ts_real"].twinx()
    t2.plot(tyr[:nf], med_e[:nf], color="#d9a35f", lw=1, label="median disk e")
    t2.plot(tyr[:nf], jup_a[:nf] / 5.0 - 1 + 0.05, color="#e8736a", lw=1, label="Jupiter a/5AU-0.95")
    t2.tick_params(colors=dim, labelsize=10)
    t2.legend(loc="upper left", fontsize=9, facecolor=bg, labelcolor=fg, frameon=False)
    cur_r = ax["ts_real"].axvline(0, color=fg, lw=0.8)
    PR, PM = X["ref"][0], X["model"][0]
    pcol = np.array(["#6fa58f"] * ncore + ["#8aa9d6"] * (len(PR) - ncore))
    sr = ax["ref"].scatter(PR[:, 0], PR[:, 1], s=2, c=pcol, lw=0)
    sm = ax["model"].scatter(PM[:, 0], PM[:, 1], s=2, c=pcol, lw=0)
    for k, ttl in (("ref", "PROXY direct N-body reference (equal mass, 300 core + 700 disk)"),
                   ("model", "PROXY CentralForceDynamics (learned central force), same start")):
        ax[k].set_xlim(-80, 80)
        ax[k].set_ylim(-80, 80)
        ax[k].set_aspect("equal")
        ax[k].set_title(ttl, color=fg, fontsize=10)
    ax["ts_proxy"].set_title("proxy: |dE/E0| ref vs model (line), model-ref mean displacement", color=fg, fontsize=10)
    ax["ts_proxy"].plot(tk, np.abs(X["refdE"][:nf]) + 1e-6, color="#7fb7a0", lw=1, label="reference |dE/E0|")
    ax["ts_proxy"].plot(tk, np.abs(X["modeldE"][:nf]) + 1e-6, color="#e8736a", lw=1, label="model |dE/E0|")
    ax["ts_proxy"].set_yscale("log")
    ax["ts_proxy"].set_xlabel("tick", color=dim, fontsize=10)
    ax["ts_proxy"].legend(loc="lower right", fontsize=9, facecolor=bg, labelcolor=fg, frameon=False)
    t3 = ax["ts_proxy"].twinx()
    t3.plot(tk, disp, color="#d9a35f", lw=1)
    t3.set_ylabel("mean |model-ref| position", color=dim, fontsize=10)
    t3.tick_params(colors=dim, labelsize=10)
    cur_p = ax["ts_proxy"].axvline(0, color=fg, lw=0.8)
    title = fig.text(0.5, 0.965, "", color=fg, ha="center", fontsize=15)
    txt_r = ax["face"].text(0.02, 0.02, "", transform=ax["face"].transAxes, color=fg, fontsize=9, family="monospace", va="bottom")
    txt_e = ax["edge"].text(0.02, 0.02, "", transform=ax["edge"].transAxes, color=fg, fontsize=9, family="monospace", va="bottom")
    txt_pr = ax["ref"].text(0.02, 0.02, "", transform=ax["ref"].transAxes, color=fg, fontsize=9, family="monospace", va="bottom")
    txt_pm = ax["model"].text(0.02, 0.02, "", transform=ax["model"].transAxes, color=fg, fontsize=9, family="monospace", va="bottom")
    fig.text(0.5, 0.012, "units: real = AU, solar masses, years (G=4 pi^2); proxy = G=1, softening 0.5, tick 0.1. e = eccentricity, a = semi-major axis, dE/E0 = relative total-energy drift. "
             "Proxy elements approximate (about core centre of mass, mu=G*300).", color=dim, ha="center", fontsize=9)
    W, H = fig.canvas.get_width_height()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", "25", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.video], stdin=subprocess.PIPE)
    for f in range(nf):
        P = R["pos"][f]
        sc_face.set_offsets(P[2:, :2])
        jf.set_offsets(P[1:2, :2])
        sc_edge.set_offsets(P[2:, [0, 2]])
        je.set_offsets(P[1:2, [0, 2]])
        sr.set_offsets(X["ref"][f][:, :2])
        sm.set_offsets(X["model"][f][:, :2])
        cur_r.set_xdata([tyr[f]])
        cur_p.set_xdata([tk[f]])
        title.set_text(f"Planetesimal disk test   real reference t = {tyr[f]:6.2f} yr   proxy tick = {tk[f]:5d}")
        ev = R["e"][f, 1:]
        txt_r.set_text(f"|dE/E0| {abs(R['dE'][f]):.1e}  dL/L0 {R['dL'][f]:.0e}\nJupiter a {jup_a[f]:.3f} AU  e {R['e'][f, 0]:.3f}\nmedian disk e {med_e[f]:.4f}  bound {bound_frac[f]*100:.0f}%")
        txt_e.set_text(f"disk e 10/50/90%: {np.nanpercentile(ev, 10):.3f}/{np.nanpercentile(ev, 50):.3f}/{np.nanpercentile(ev, 90):.3f}")
        txt_pr.set_text(f"|dE/E0| {abs(X['refdE'][f]):.2e}\nmedian disk r {ref_med_r[f]:.1f}  within 150: {ref_surv[f]*100:.0f}%")
        txt_pm.set_text(f"|dE/E0| {abs(X['modeldE'][f]):.2e}\nmedian disk r {mod_med_r[f]:.1f}  within 150: {mod_surv[f]*100:.0f}%\nmean |model-ref| {disp[f]:.2f}")
        fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["real", "proxy", "render"])
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--n-disk", type=int, default=1000)
    ap.add_argument("--n-core", type=int, default=300)
    ap.add_argument("--years", type=float, default=100.0)
    ap.add_argument("--dt", type=float, default=1 / 640)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--ticks", type=int, default=15000)
    ap.add_argument("--ckpt", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real", default="results/protodisk_video_real.npz")
    ap.add_argument("--proxy", default="results/protodisk_video_proxy.npz")
    ap.add_argument("--video", default="videos/protodisk_real_reference_faceon_edgeon_plus_central_force_proxy.mp4")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--frames", type=int, default=10 ** 9)
    a = ap.parse_args()
    DEV = torch.device(a.device)
    if a.stage == "real":
        a.out = a.out or a.real
        sim_real(a)
    elif a.stage == "proxy":
        a.out = a.out or a.proxy
        sim_proxy(a)
    else:
        render(a)
