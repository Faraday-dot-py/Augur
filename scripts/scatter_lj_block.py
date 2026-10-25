"""Bonded-block stability and collision test for the scatter field on LJ MD (hole-8/objects link, docs/debugging/experiment-log.md).
A cold 2D hex-lattice block (self-bonded by the LJ attraction) sits in vacuum in a periodic box. Scenes: `rest` = block alone (T = 0.05 per
particle, zero net momentum); `hit` = a small hex cluster fired at the block with speed --speed and impact offset --offset. Truth is shifted-force
LJ velocity Verlet; each ckpt is rolled out from the same frame-0 state for --frames frames (20 truth steps each).
Metrics per scene and run (truth and models): dE/N, total-momentum drift |P(t)-P(0)|/N, block centre-of-mass displacement, radius of gyration
of the biggest bonded cluster (bond = pair distance < 1.5), number of particles outside it (evaporated/fragmented), psi6 order of the block.
Usage: PYTHONPATH=. python -m scripts.scatter_lj_block --ckpts P=ckpt/P_final.pt,V=ckpt/V_final.pt --tag blk --seed 4738"""
import argparse
import json
import math

import numpy as np
import torch

from scripts import lj_sim
from scripts import scatter_lj as sl

BOX = 60.0
BOND = 1.5


def hex_patch(nx, ny, a, center):
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    p = np.stack([((ix + 0.5 * (iy % 2)) * a).ravel(), (iy * a * math.sqrt(3) / 2).ravel()], 1)
    return p - p.mean(0) + center


def build(scene, speed, offset, seed):
    rng = np.random.default_rng(seed)
    a = 2 ** (1 / 6) * 0.97
    c = np.array([BOX / 2, BOX / 2])
    blk = hex_patch(20, 16, a, c)
    vb = rng.normal(0, math.sqrt(0.05), blk.shape)
    vb -= vb.mean(0)
    if scene == "rest":
        return blk, vb
    cl = hex_patch(4, 5, a, c + np.array([-26.0, offset]))
    vc = rng.normal(0, math.sqrt(0.05), cl.shape)
    vc -= vc.mean(0)
    vc += np.array([speed, 0.0])
    pos = np.concatenate([blk, cl])
    vel = np.concatenate([vb, vc])
    return pos, vel


def cluster_stats(p, box):
    d = p[:, None, :] - p[None, :, :]
    d -= box * np.round(d / box)
    r = np.sqrt((d ** 2).sum(-1))
    adj = (r < BOND) & (r > 0)
    n = len(p)
    lab = -np.ones(n, int)
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        st = [i]
        lab[i] = k
        while st:
            u = st.pop()
            for w in np.nonzero(adj[u] & (lab < 0))[0]:
                lab[w] = k
                st.append(w)
        k += 1
    big = np.bincount(lab).argmax()
    m = lab == big
    ref = p[m][0]
    q = p[m] - ref
    q -= box * np.round(q / box)
    com = q.mean(0) + ref
    rg = math.sqrt(((q - q.mean(0)) ** 2).sum(-1).mean())
    nb = adj[m][:, m].sum(1)
    inner = nb >= 4
    ang = np.arctan2(d[..., 1], d[..., 0])
    z = np.where(adj, np.exp(6j * ang), 0).sum(1) / np.maximum(adj.sum(1), 1)
    psi6 = float(np.abs(z[m][inner]).mean()) if inner.any() else 0.0
    return {"n_big": int(m.sum()), "outside": int(n - m.sum()), "rg": rg, "com": com.tolist(), "psi6": psi6}


def summarize(pos, vel, stats, box, n_every=40):
    out = {"dE_per_N": float(stats[-1, 3] - stats[0, 3]), "T_end": float(stats[-1, 1]), "frames": len(pos) - 1}
    p0 = vel[0].sum(0)
    out["dP_per_N_max"] = float(max(np.linalg.norm(v.sum(0) - p0) for v in vel) / len(vel[0]))
    for f in range(0, len(pos), n_every):
        out[f"f{f}"] = cluster_stats(pos[f].astype(np.float64), box)
    out[f"f{len(pos) - 1}"] = cluster_stats(pos[-1].astype(np.float64), box)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="label=path,label=path")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--offset", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    box = np.array([BOX, BOX])
    models = {}
    for kv in args.ckpts.split(","):
        k, v = kv.split("=")
        m = sl.load_model(v, dev)
        m.eval()
        models[k] = m
    res = {}
    for scene in ("rest", "hit"):
        pos0, vel0 = build(scene, args.speed, args.offset, args.seed)
        fp, fv, st = lj_sim.run(pos0 % box, vel0, box, args.frames * 20, 20, 0.005, dev)
        np.savez_compressed(f"results/block_{args.tag}_{scene}_truth.npz", pos=fp, vel=fv, box=box, stats=st)
        res[scene] = {"truth": summarize(fp, fv, st, box)}
        print(scene, "truth", {k: v for k, v in res[scene]["truth"].items() if not k.startswith("f")}, flush=True)
        for k, m in models.items():
            mp, mv, mst, ok = sl.rollout(m, fp[0].astype(np.float64), fv[0].astype(np.float64), box, args.frames, dev)
            r = summarize(mp, mv, mst, box)
            r["finite"] = bool(ok)
            res[scene][k] = r
            np.savez_compressed(f"results/block_{args.tag}_{scene}_{k}.npz", pos=mp, vel=mv, box=box, stats=mst)
            last = r[f"f{r['frames']}"]
            print(scene, k, {a: b for a, b in r.items() if not a.startswith("f") or a == "frames"}, "last", last, flush=True)
    json.dump(res, open(f"results/scatter_lj_block_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
