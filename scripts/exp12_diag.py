"""Per-channel diagnosis of the tiled window hierarchy with an exact external field vs the tiled baseline (no training).
Teacher-forced on truth states: per-particle force error and energy-work error rate (dt * m * v . (a_model - a_true)) binned by local density,
distance to the nearest internal tile boundary, distance to the outer boundary, clump vs field particles, and tick group.
Arms: BL (Exp E tiled + far kernel), OR (exact external field from sources outside the 64 window), ORRING (external field from sources
outside the 56 unit region the tier 0 actually scatters; sources in the outer 4 unit ring are otherwise dropped), OR0 (no external field)."""
import argparse
import json
import os

import numpy as np
import torch

from scripts import exp12_hier as h
from scripts import train_tiered as ttr

DT = ttr.DT
DENS = [0, 2, 5, 12, 1e9]
SEAM = [0, 2, 4, 8, 16.01]
OUT = [-1e9, 0, 8, 16, 1e9]
TICKS = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 101)]


def outside(pos, mask, cen, edge):
    rel = pos[:, None] - cen[None, :, None]
    return (rel.abs().amax(-1) >= edge).to(pos.dtype) * mask[:, None]


def ext_with_edge(pos, mass, mask, cen, edge):
    B, N = mask.shape
    rel = pos[:, None] - cen[None, :, None]
    outm = outside(pos, mask, cen, edge)
    return h.exact_ext(rel.reshape(B * h.NW, N, 2), outm.reshape(B * h.NW, N), mass[:, None].expand(B, h.NW, N).reshape(B * h.NW, N)), outm


def bin_of(x, edges):
    e = torch.tensor(edges, device=x.device, dtype=x.dtype)
    return (torch.bucketize(x, e[1:-1], right=True)).clamp(0, len(edges) - 2)


def truth_bins(p, mk, lab):
    d2 = ((p[:, :, None] - p[:, None]) ** 2).sum(-1)
    near = ((d2 < 4.0) & (mk[:, None] > 0)).sum(-1).float() - 1.0
    b = ttr_edges(p.device)
    ds = (p[..., None] - b).abs().amin((-1, -2))
    dout = (h.NT * h.TILE / 2 - p.abs()).amin(-1)
    return bin_of(near, DENS), bin_of(ds, SEAM), bin_of(dout, OUT), (lab >= 0).long()


def ttr_edges(dev):
    return -h.NT * h.TILE / 2 + h.TILE * torch.arange(1, h.NT, device=dev, dtype=torch.float32)


def run_arm(force_fn, init_state, d, acc):
    S, T = d["P"].shape[:2]
    m, mk = d["M"].float(), d["mask"].float()
    st = init_state(S)
    for t in range(T):
        p, v = d["P"][:, t].float(), d["V"][:, t].float()
        a, st = force_fn(p, v, m, mk, st)
        at = ttr.true_acc(p, m, mk)
        da = a - at
        w = DT * m * (v * da).sum(-1)
        bd, bs, bo, bc = truth_bins(p, mk, d["lab"].to(p.device))
        tg = [i for i, (lo, hi) in enumerate(TICKS) if lo <= t < hi][0]
        e2, t2 = (da ** 2).sum(-1) * mk, (at ** 2).sum(-1) * mk
        for name, b in (("density", bd), ("seam", bs), ("outer", bo), ("clump", bc), ("all", torch.zeros_like(bd))):
            for k in b.unique().tolist():
                sel = (b == k) & (mk > 0)
                if not sel.any():
                    continue
                key = (name, k, tg)
                x = acc.setdefault(key, np.zeros(4))
                x += [float(e2[sel].sum()), float(t2[sel].sum()), float((w * mk)[sel].sum()), float(sel.sum())]


def summarize(acc):
    out = {}
    for (name, k, tg), x in acc.items():
        out.setdefault(name, {}).setdefault(str(k), {})[f"{TICKS[tg][0]}-{TICKS[tg][1]}"] = {
            "rel": (x[0] / max(x[1], 1e-12)) ** 0.5, "rms": (x[0] / max(x[3], 1)) ** 0.5, "work_per_particle": x[2] / max(x[3], 1), "n": x[3]}
    return out


@torch.no_grad()
def ext_at_owned(d, cen, edge, dev, nsc=6):
    m, mk = d["M"][:nsc].float(), d["mask"][:nsc].float()
    res = []
    for t in (0, min(50, d["P"].shape[1] - 1)):
        p = d["P"][:nsc, t].float()
        ext, outm = ext_with_edge(p, m, mk, cen, edge)
        B, N = mk.shape
        rel = (p[:, None] - cen[None, :, None]).reshape(B * h.NW, N, 2)
        pr = h.poly_ext_acc(ext, rel).reshape(B, h.NW, N, 2)
        own = h.owner(p)
        pr = torch.gather(pr, 1, own[:, None, :, None].expand(B, 1, N, 2)).squeeze(1)
        tr = torch.zeros_like(pr)
        for b in range(B):
            for w_i in range(h.NW):
                sel = (own[b] == w_i) & (mk[b] > 0)
                if not sel.any():
                    continue
                dx = p[b][None] - p[b][sel][:, None]
                ww = (m[b] * outm[b, w_i])[None] * ((dx ** 2).sum(-1) + ttr.EPS ** 2) ** -1.5
                tr[b][sel] = (dx * ww[..., None]).sum(1)
        at = ttr.true_acc(p, m, mk)
        k = mk > 0
        res.append({"tick": t, "ext_err_rel_to_far": float(((pr - tr)[k] ** 2).sum().sqrt() / (tr[k] ** 2).sum().sqrt()),
                    "far_to_total_rms": float((tr[k] ** 2).sum().sqrt() / (at[k] ** 2).sum().sqrt())})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--tier0", default=os.path.expanduser("~/E_ms_kp_pot_v_g128.pt"))
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--arms", default="BL,OR,ORRING,OR0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = h.centres(dev)
    s1 = h.S1Windows(args.s1, dev, chunk=args.chunk)
    cfg = s1.t0.cfg
    info = {"momfix": bool(s1.m.momfix), "t0_core_units": float(s1.t0.core(s1.m)), "extent": float(s1.m.extent), "halo": cfg.halo, "taper": cfg.taper,
            "recurrent": bool(s1.m.recurrent)}
    print("info", json.dumps(info), flush=True)
    res = {"info": info}
    for sd in ttr.EVAL_SEEDS:
        d = ttr.make_scenes(args.scenes, args.steps, sd, dev)
        res[str(sd)] = {}
        for a in args.arms.split(","):
            acc = {}
            if a == "BL":
                from scripts import tiered_adapters as ta
                bl = ta.load_tiled(f"{args.tier0},nt=4,tile=32,halo=16,far=1", dev)
                with torch.no_grad():
                    run_arm(lambda p, v, m, k, st: bl.force(p, v, m, k, st), lambda S: bl.init_state(S, dev), d, acc)
            else:
                edge = {"OR": 32.0, "ORRING": 28.0}.get(a, 32.0)

                def f(p, v, m, k, st, a=a, edge=edge):
                    if a in ("ORV0", "ORS0", "ORV0S0"):
                        ext, _ = ext_with_edge(p, m, k, cen, edge)
                        st = s1.init_state(p.shape[0]) if "S0" in a else st
                        return s1.force(p, torch.zeros_like(v) if "V0" in a else v, m, k, ext, st)
                    if a == "OR0":
                        z = tuple(torch.zeros_like(x) for x in h.ext_for("OR", p, v, m, k, cen))
                        return s1.force(p, v, m, k, z, st)
                    ext, _ = ext_with_edge(p, m, k, cen, edge)
                    return s1.force(p, v, m, k, ext, st)

                with torch.no_grad():
                    run_arm(f, s1.init_state, d, acc)
                if a in ("OR", "ORRING"):
                    res[str(sd)][a + "_ext_at_owned"] = ext_at_owned(d, cen, edge, dev)
            res[str(sd)][a] = summarize(acc)
            tot = res[str(sd)][a]["all"]["0"]
            print("seed", sd, a, json.dumps({k: {x: round(y, 5) for x, y in v.items() if x != "n"} for k, v in tot.items()}), flush=True)
    json.dump(res, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
