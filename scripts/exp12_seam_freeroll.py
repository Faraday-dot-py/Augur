"""Free-rollout seam / owner-window diagnosis of the tiled window hierarchy (4x4 tiles of 32 units, 64-unit windows, span 128).
Reuses scripts/exp12_hier.py (S1Windows, ext_for, centres, owner) and scripts/train_tiered.py (scenes, true_acc); run from a tree that has them.
Per free-rollout step (Verlet, model positions): flag whether the owner window of a particle changed, its distance to the nearest internal tile seam,
the work error w = dt * m * v . (a_model - a_true) with a_true taken at the model's own state, |a_model - a_true| / |a_true|, the cross-window
force discrepancy (owner window vs every other window that also contains the particle, relative to |a_true|), the growth of the position error
against truth, and the kinetic-energy error growth.
Probe mode: a massless probe on a line across a seam in a truth state; force from each window containing it vs exact.
Fields: relative difference of the recurrent field of neighbouring windows over their overlap.
Arms: name=OR:<ckpt> (tier 0 windows with exact external field), name=BL:<ckpt> (tiled Exp E baseline)."""
import argparse
import json

import numpy as np
import torch

from scripts import exp12_hier as h
from scripts import scatter_field as sf
from scripts import train_tiered as ttr

DT, EPS = ttr.DT, ttr.EPS
SEAMS = [0, 1, 2, 4, 8, 16, 1e9]
NAMES = ["w", "abs_w", "rel_force_err", "cross_win_rel", "dpos_err", "dKE_err"]


def seam_dist(p):
    lines = (-h.NT * h.TILE / 2 + h.TILE * torch.arange(1, h.NT, dtype=torch.float32)).to(p.device)
    return (p[..., None] - lines).abs().amin((-1, -2))


class S1All(h.S1Windows):
    def force(self, pos, vel, mass, mask, ext, fld):
        B, N = mask.shape
        rel, inw, _ = h.windows(pos, mask, self.cen)
        rel = rel.reshape(B * h.NW, N, 2)
        v = vel[:, None].expand(B, h.NW, N, 2).reshape(B * h.NW, N, 2)
        mm = mass[:, None].expand(B, h.NW, N).reshape(B * h.NW, N)
        k = inw.reshape(B * h.NW, N)
        g, H, Q, C = ext
        out, nf = [], []
        for i in range(0, B * h.NW, self.chunk):
            sl = slice(i, i + self.chunk)
            n = rel[sl].shape[0]
            self.t0.set_external(self.m, phi0=torch.zeros(n, device=pos.device), g=g[sl], H=H[sl], high=(Q[sl], C[sl]))
            self.t0.set_boundary(self.m, sdf=None, edge=["open"] * n)
            a, f2 = self.m.force(rel[sl], v[sl], mm[sl], k[sl], fld[sl])
            out.append(a)
            nf.append(f2)
        a = torch.cat(out).reshape(B, h.NW, N, 2)
        self.last_all, self.last_in = a, inw
        own = h.owner(pos)
        acc = torch.gather(a, 1, own[:, None, :, None].expand(B, 1, N, 2)).squeeze(1)
        return acc * mask[..., None], torch.cat(nf)


@torch.no_grad()
def free_roll(force_fn, init_state, d, s1=None, hist=False):
    S, T = d["P"].shape[:2]
    m, mk = d["M"].float(), d["mask"].float()
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    st = init_state(S)
    a_prev, st = force_fn(pos, vel, m, mk, st)
    acc = {}
    ev = dict(P=[pos.clone()], own=[h.owner(pos)], E=[sf.energy(pos.double(), vel.double(), d["M"], d["mask"], EPS)])
    pe_prev = (pos.double() - d["P"][:, 0]).norm(dim=-1)
    own_prev = h.owner(pos)
    fields = []
    edges = torch.tensor(SEAMS[1:-1], device=pos.device)
    for t in range(1, T):
        pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
        a_new, st = force_fn(pn, vel + DT * a_prev, m, mk, st)
        vn = vel + 0.5 * DT * (a_prev + a_new)
        at = ttr.true_acc(pn, m, mk)
        own = h.owner(pn)
        changed = (own != own_prev) & (mk > 0)
        sb = torch.bucketize(seam_dist(pn), edges, right=True)
        anorm = at.norm(dim=-1).clamp_min(1e-6)
        w = DT * m * (vn * (a_new - at)).sum(-1)
        rel = (a_new - at).norm(dim=-1) / anorm
        cross = torch.zeros_like(rel)
        if s1 is not None:
            A, inw = s1.last_all, s1.last_in
            ao = torch.gather(A, 1, own[:, None, :, None].expand(-1, 1, -1, 2))
            other = inw.clone()
            other.scatter_(1, own[:, None, :], 0.0)
            cross = (A - ao).norm(dim=-1).mul(other).amax(1) / anorm
        pe = (pn.double() - d["P"][:, t]).norm(dim=-1)
        dpe = (pe - pe_prev).float()
        ke = 0.5 * m * (vn ** 2).sum(-1) - 0.5 * m * (d["V"][:, t].float() ** 2).sum(-1)
        ke_prev = 0.5 * m * (vel ** 2).sum(-1) - 0.5 * m * (d["V"][:, t - 1].float() ** 2).sum(-1)
        vals = torch.stack([w, w.abs(), rel, cross, dpe, ke - ke_prev])
        for ch, nm in ((True, "changed"), (False, "kept")):
            for b in range(len(SEAMS) - 1):
                sel = (changed == ch) & (mk > 0) & (sb == b)
                if sel.any():
                    x = acc.setdefault((nm, b), np.zeros(len(NAMES) + 1))
                    x[:-1] += vals[:, sel].sum(1).double().cpu().numpy()
                    x[-1] += int(sel.sum())
        pos, vel, a_prev, pe_prev, own_prev = pn, vn, a_new, pe, own
        if hist:
            ev["P"].append(pos.clone())
            ev["own"].append(own)
            ev["E"].append(sf.energy(pos.double(), vel.double(), d["M"], d["mask"], EPS))
        if s1 is not None and t in (1, 10, 50):
            fields.append((t, st[:h.NW].clone()))
    return acc, ev, fields


def summarize(acc):
    out = {}
    for (k, b), x in acc.items():
        out.setdefault(k, {})[f"seam<{SEAMS[b + 1]:g}"] = dict(zip(NAMES, (x[:-1] / x[-1]).tolist()), n=int(x[-1]), sum_w=float(x[0]))
    return out


def field_overlap(fld, shift):
    n = h.NT
    f = fld.reshape(n, n, *fld.shape[1:])
    res = {}
    for ax, name in ((2, "dim_a"), (3, "dim_b")):
        num = den = 0.0
        for i in range(n - 1):
            for j in range(n):
                a_, b_ = f[i, j], f[i + 1, j]
                G = a_.shape[-1]
                x, y = (a_[:, shift:, :], b_[:, :G - shift, :]) if ax == 2 else (a_[:, :, shift:], b_[:, :, :G - shift])
                num += float(((x - y) ** 2).sum())
                den += float((x ** 2).sum() + (y ** 2).sum()) / 2
        res[name] = (num / max(den, 1e-12)) ** 0.5
    return res


@torch.no_grad()
def probe_scan(s1, d, scene, t, y, xs):
    dev = d["P"].device
    P = d["P"][scene:scene + 1, t].float()
    V = d["V"][scene:scene + 1, t].float()
    M = d["M"][scene:scene + 1].float()
    K = d["mask"][scene:scene + 1].float()
    n = int(K.sum())
    rows = []
    for x in xs:
        p, v, mm, kk = P.clone(), V.clone(), M.clone(), K.clone()
        p[0, n] = torch.tensor([x, y], device=dev)
        v[0, n], mm[0, n], kk[0, n] = 0, 0.0, 1.0
        s1.force(p, v, mm, kk, h.ext_for("OR", p, v, mm, kk, s1.cen), s1.init_state(1))
        A, inw = s1.last_all[0, :, n], s1.last_in[0, :, n]
        at = ttr.true_acc(p, mm, kk)[0, n]
        rows.append(dict(x=float(x), owner=int(h.owner(p)[0, n]), a={str(i): A[i].tolist() for i in range(h.NW) if inw[i] > 0}, exact=at.tolist()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tier0", default="")
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(ttr.EVAL_SEEDS))
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--hist-out", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = h.centres(dev)
    res, hist = {}, {}
    for spec in a.arms.split(","):
        name, _, rest = spec.partition("=")
        kind, _, ck = rest.partition(":")
        res[name] = {}
        if kind == "OR":
            s1 = S1All(ck, dev, chunk=a.chunk)
            cells = int(round(h.TILE / (s1.m.extent / s1.m.grid)))
        else:
            from scripts import tiered_adapters as ta
            bl = ta.load_tiled(f"{ck},nt=4,tile=32,halo=16,far=1", dev)
        for sd in a.seeds:
            d = ttr.make_scenes(a.scenes, a.steps, sd, dev)
            if kind == "OR":
                acc, ev, fl = free_roll(lambda p, v, m, k, st: s1.force(p, v, m, k, h.ext_for("OR", p, v, m, k, cen), st), s1.init_state, d, s1=s1, hist=bool(a.hist_out))
                res[name][str(sd)] = summarize(acc)
                res[name][str(sd)]["field_overlap"] = {str(t): field_overlap(fo, cells) for t, fo in fl}
                if a.probe and sd == a.seeds[0]:
                    res[name][str(sd)]["probe"] = {f"t{t}_y{y}": probe_scan(s1, d, 0, t, y, np.arange(-6, 6.01, 1.0)) for t in (0, 50) for y in (0.0, 8.0)}
            else:
                acc, ev, fl = free_roll(lambda p, v, m, k, st: bl.force(p, v, m, k, st), lambda S: bl.init_state(S, dev), d, hist=bool(a.hist_out))
                res[name][str(sd)] = summarize(acc)
            if a.hist_out:
                hist[f"{name}_{sd}_P"] = torch.stack(ev["P"], 1).cpu().numpy()
                hist[f"{name}_{sd}_own"] = torch.stack(ev["own"], 1).cpu().numpy()
                hist[f"{name}_{sd}_E"] = torch.stack(ev["E"], 1).cpu().numpy()
                hist[f"truth_{sd}_P"] = d["P"].cpu().numpy()
                hist[f"mask_{sd}"] = d["mask"].cpu().numpy()
            r = res[name][str(sd)]
            print(name, sd, {k: (sum(v["sum_w"] for v in r[k].values()), sum(v["n"] for v in r[k].values())) for k in ("kept", "changed") if k in r}, flush=True)
            json.dump(res, open(a.out, "w"))
    if a.hist_out:
        np.savez_compressed(a.hist_out, **hist)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
