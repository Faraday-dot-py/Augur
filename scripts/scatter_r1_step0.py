"""R1 Step 0 (eval only, no training): exact discrete mesh-pair complement (docs/debugging/close-pair-design-proposals.md, R1).

  PYTHONPATH=. python scripts/scatter_r1_step0.py --e checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/r1_step0/r1_step0.json
Parts: (U) unit checks; (a) attribution probe on Exp E two-body scenes (kernel-only ak / U-Net-only / pair-only / momfix vs exact softened gravity);
(b) analytic P3M (prior kernel, pp=2) with the exact CIC-FFT-central-difference pair complement vs the analytic baseline, force error by nn bin on expB seed 9300, cluster seed 9200.
Definitions: M_ij = acceleration on body i per unit m_j produced by the mesh pipeline alone (j deposits on its 4 CIC nodes, potential via the prior kernel K at integer node
offsets, 2h central difference with zero padding outside the grid, i gathers with its 4 CIC weights); F = exact softened pair force r/q^3; a_cont = continuous mesh share;
win = (1-u)^2, u = min(r/pp,1)^2. Variants (pair term per unit m_j; total = mesh + pair, then momfix):
  base: (F - a_cont) rhat          [current analytic P3M]
  V1  : win (F rhat - M_ij)        [literal proposal text]
  V2  : (F - a_cont) rhat + win (a_cont rhat - M_ij)   [same as V1 with F_hat chosen so the blend is exact for an exact mesh]
  V2r : V2 with only the radial part of M_ij subtracted
  V3  : (F rhat - M_ij) for r < pp   [full complement, hard edge]
"""
import argparse
import json
import math
import time

import numpy as np
import torch

from scripts import scatter_error_growth as sg
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sg.DEV = DEV
DT, EPS = sg.DT, sg.EPS
G, EXT, PP = 128, 64.0, 2.0
H = EXT / G
NNB = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9)
VARS = ("base", "V1", "V2", "V2r", "V3")


def kfun(r, pp=PP, eps=EPS):
    u = (r / pp).clamp(max=1.0) ** 2
    return -(1 - (1 - u) ** 2) / torch.sqrt(r ** 2 + eps ** 2)


def corners(pos):
    u = (pos + EXT / 2) / H - 0.5
    i0 = torch.floor(u)
    w = u - i0
    off = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=pos.device, dtype=pos.dtype)
    nodes = i0[..., None, :] + off
    wt = torch.where(off.bool(), w[..., None, :], 1 - w[..., None, :]).prod(-1)
    ok = ((nodes >= 0) & (nodes < G)).all(-1).to(pos.dtype)
    return nodes, wt * ok


def mesh_pair_acc(xi, xj):
    """Acceleration on i per unit m_j through the mesh pipeline (in_scale 1). xi, xj broadcastable (..., 2) -> (..., 2)."""
    xi, xj = torch.broadcast_tensors(xi, xj)
    ni, wi = corners(xi)
    nj, wj = corners(xj)
    e = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=xi.device, dtype=xi.dtype)
    s = ni[..., :, None, :] + e[:, :]
    ingrid = ((s >= 0) & (s < G)).all(-1).to(xi.dtype)
    d = (s[..., :, :, None, :] - nj[..., None, None, :, :]) * H
    phi = (kfun(d.norm(dim=-1)) * wj[..., None, None, :]).sum(-1) * ingrid
    gx = (phi[..., 0] - phi[..., 1]) / (2 * H)
    gy = (phi[..., 2] - phi[..., 3]) / (2 * H)
    acc = -torch.stack([gx, gy], -1)
    return (acc * wi[..., None]).sum(-2)


def pair_terms(p):
    """p (B,N,2). Returns d, r, rhat, F, a_cont, win, Mall (per unit m_j), eye."""
    B, N = p.shape[:2]
    d = p[:, None, :, :] - p[:, :, None, :]
    eye = torch.eye(N, device=p.device, dtype=torch.bool)[None]
    r = torch.sqrt((d ** 2).sum(-1) + 1e-30)
    rhat = d / r[..., None]
    q = torch.sqrt(r ** 2 + EPS ** 2)
    u = (r / PP).clamp(max=1.0) ** 2
    win = (1 - u) ** 2
    F = r / q ** 3
    a_cont = -4 * (1 - u) * r / PP ** 2 / q + (1 - win) * r / q ** 3
    Mall = mesh_pair_acc(p[:, :, None, :], p[:, None, :, :])
    return d, r, rhat, F, a_cont, win, Mall, eye


def pair_vec(var, rhat, F, a_cont, win, Mall, r):
    if var == "base":
        return (F - a_cont)[..., None] * rhat
    if var == "V1":
        return win[..., None] * (F[..., None] * rhat - Mall)
    if var == "V2":
        return (F - a_cont)[..., None] * rhat + win[..., None] * (a_cont[..., None] * rhat - Mall)
    if var == "V2r":
        mr = (Mall * rhat).sum(-1)
        return (F - a_cont)[..., None] * rhat + win[..., None] * (a_cont - mr)[..., None] * rhat
    if var == "V3":
        return (r < PP).to(F.dtype)[..., None] * (F[..., None] * rhat - Mall)
    raise ValueError(var)


def momfix(a, m):
    return a - (m[..., None] * a).sum(1, keepdim=True) / m.sum(1)[:, None, None]


def forces(p, m, with_parts=False):
    """p (B,N,2), m (B,N) fp64. Returns {var: acc (B,N,2)} after momfix; with_parts also residual components per var."""
    d, r, rhat, F, a_cont, win, Mall, eye = pair_terms(p)
    mj = m[:, None, :, None]
    off = (~eye)[..., None].to(p.dtype)
    mesh = (Mall * mj).sum(2)
    exact = (F[..., None] * rhat * mj * off).sum(2)
    out, parts = {}, {}
    inr = ((r < PP) & ~eye)[..., None].to(p.dtype)
    far1 = ((r >= PP) & (r < 2 * PP) & ~eye)[..., None].to(p.dtype)
    far2 = ((r >= 2 * PP) & ~eye)[..., None].to(p.dtype)
    selfm = eye[..., None].to(p.dtype)
    for v in VARS:
        pv = pair_vec(v, rhat, F, a_cont, win, Mall, r) * inr
        out[v] = momfix(mesh + (pv * mj).sum(2), m)
        if with_parts:
            Fv = F[..., None] * rhat
            parts[v] = dict(inrange=momfix(((Mall + pv - Fv) * inr * mj).sum(2), m), far_pp_2pp=momfix(((Mall - Fv) * far1 * mj).sum(2), m),
                            far_gt_2pp=momfix(((Mall - Fv) * far2 * mj).sum(2), m), self=momfix((Mall * selfm * mj).sum(2), m))
    return out, exact, mesh, parts


def binidx(nn):
    return np.digitize(nn, NNB[1:-1])


def force_scenes(scenes, log, tag):
    acc = {}
    for si, sc in enumerate(scenes):
        m = sc["m"][None]
        for t in range(2, 100):
            p = sc["P"][t][None]
            keep = (p[0].abs().amax(-1) < 26.0).cpu().numpy()
            nn = sg.nn_dist(p[0]).cpu().numpy()
            fo, ex, mesh, parts = forces(p, m, True)
            ex = momfix(ex, m)
            rec = {"nn": nn[keep], "ae2": (ex[0] ** 2).sum(-1).cpu().numpy()[keep]}
            for v in VARS:
                res = fo[v] - ex
                rec[f"{v}:err2"] = (res[0] ** 2).sum(-1).cpu().numpy()[keep]
                for k, x in parts[v].items():
                    rec[f"{v}:{k}"] = (x[0] * res[0]).sum(-1).cpu().numpy()[keep]
                rec[f"{v}:checkres"] = ((sum(parts[v].values()) - res)[0] ** 2).sum(-1).cpu().numpy()[keep]
            for k, x in rec.items():
                acc.setdefault(k, []).append(x)
        if si % 6 == 0:
            log(f"  {tag} scene {si}/{len(scenes)}")
    R = {k: np.concatenate(v) for k, v in acc.items()}
    b = binidx(R["nn"])
    out = {}
    names = ["all"] + [f"nn {NNB[i]}-{NNB[i + 1]}" for i in range(len(NNB) - 1)]
    for ni, nm in enumerate(names):
        mk = np.ones_like(b, bool) if ni == 0 else b == ni - 1
        if mk.sum() == 0:
            continue
        ae2 = R["ae2"][mk].mean()
        row = {"frac": float(mk.mean()), "count": int(mk.sum())}
        for v in VARS:
            e2 = R[f"{v}:err2"][mk].mean()
            row[v] = dict(rel_rms=float(np.sqrt(e2 / ae2)), decomposition_check=float(R[f"{v}:checkres"][mk].mean() / max(e2, 1e-30)),
                          share={k: float(R[f"{v}:{k}"][mk].mean() / e2) for k in ("inrange", "far_pp_2pp", "far_gt_2pp", "self")})
        out[nm] = row
    tot = R["base:err2"].sum()
    for i in range(len(NNB) - 1):
        out[f"nn {NNB[i]}-{NNB[i + 1]}"]["share_of_base_err2"] = float(R["base:err2"][b == i].sum() / tot)
    return out


def two_body_pos(r, n, gen, dtype):
    c = (torch.rand(n, 1, 2, generator=gen, device=DEV, dtype=dtype) - 0.5) * 12
    th = torch.rand(n, generator=gen, device=DEV, dtype=dtype) * 2 * math.pi
    d = torch.stack([torch.cos(th), torch.sin(th)], -1) * r
    return torch.cat([c - d[:, None] / 2, c + d[:, None] / 2], 1), d


def rad_stats(a0, d):
    dd = d.double()
    r2 = (dd ** 2).sum(-1, keepdim=True)
    ae = dd * (r2 + EPS ** 2) ** -1.5
    dh = dd / dd.norm(dim=-1, keepdim=True)
    ex = ae.norm(dim=-1)
    rad = (a0.double() * dh).sum(-1) / ex * 100
    tan = (a0.double() * torch.stack([-dh[:, 1], dh[:, 0]], -1)).sum(-1) / ex * 100
    return rad, tan, ex


def stat(x):
    return dict(mean=float(x.mean()), std=float(x.std()))


def analytic_two_body(rs, n, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    res = []
    for r in rs:
        pos, d = two_body_pos(r, n, gen, torch.float64)
        m = torch.ones(n, 2, device=DEV, dtype=torch.float64)
        fo, ex, mesh, _ = forces(pos, m)
        row = {"r": r}
        rad, tan, _ = rad_stats(momfix(mesh, m)[:, 0], d)
        row["mesh_only"] = dict(rad_bias_pct=stat(rad - 100), tan_rms_pct=float(tan.pow(2).mean().sqrt()))
        for v in VARS:
            rad, tan, _ = rad_stats(fo[v][:, 0], d)
            row[v] = dict(rad_bias_pct=stat(rad - 100), tan_rms_pct=float(tan.pow(2).mean().sqrt()))
        res.append(row)
    return res


def components(model, pos, vel, mass, mask, field):
    """E's _force split: ak (kernel mesh), net (U-Net potential gradient), pair, momfix; per-body accelerations."""
    x = model.scatter(pos, vel, mass, mask)
    ak = model.kernel_acc(x[:, :1])
    x = torch.cat([x, ak], 1)
    x = torch.cat([x, field], 1)
    fld = model.net(x)
    net = model.neg_grad(fld[:, :1])
    ak_g, net_g = model.gather(ak, pos), model.gather(net, pos)
    pair = model.pp_acc(pos, mass, mask, model.knn)
    tot = ak_g + net_g + pair
    mm = (mass * mask)[..., None]
    mf = -(mm * tot).sum(1, keepdim=True) / mm.sum(1, keepdim=True)
    return ak_g, net_g, pair, mf, fld


def attribution(model, rs, n, seed, log):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    res = []
    maxdiff = 0.0
    for r in rs:
        pos64, d = two_body_pos(r, n, gen, torch.float32)
        pos = pos64.float()
        v = torch.zeros_like(pos)
        mass, mask = torch.ones(n, 2, device=DEV), torch.ones(n, 2, device=DEV)
        f_prev = model.init_field(n, DEV)[0]
        for _ in range(2):
            _, f_prev = model.force(pos, v, mass, mask, f_prev)
        ak_g, net_g, pair, mf, fld = components(model, pos, v, mass, mask, f_prev)
        a_ref, _ = model.force(pos, v, mass, mask, f_prev)
        tot = ak_g + net_g + pair + mf
        maxdiff = max(maxdiff, float((tot - a_ref).abs().max()))
        row = {"r": r}
        for nm, c in (("kernel_ak", ak_g), ("unet", net_g), ("pair", pair), ("momfix", mf), ("total", tot)):
            rad, tan, _ = rad_stats(c[:, 0], d)
            row[nm] = dict(rad_pct=stat(rad - (100 if nm == "total" else 0)), tan_rms_pct=float(tan.pow(2).mean().sqrt()))
        ake = ak_g - (ak_g.sum(1, keepdim=True) / 2)
        rad, _, _ = rad_stats(ake[:, 0], d)
        row["kernel_ak_after_momfix"] = stat(rad)
        res.append(row)
        log(f"  attribution r={r}: total {row['total']['rad_pct']['mean']:+.1f}% (std {row['total']['rad_pct']['std']:.1f}) ak {row['kernel_ak']['rad_pct']['mean']:+.1f} "
            f"unet {row['unet']['rad_pct']['mean']:+.1f} pair {row['pair']['rad_pct']['mean']:+.1f} momfix {row['momfix']['rad_pct']['mean']:+.1f}")
    return res, maxdiff


def p3m_model():
    kw = dict(grid=G, net="unet", recurrent=True, hidden_ch=0, momfix=True, kernel=True, pp=PP, split=True, potential=True, verlet=True, prior=True, nonet=True, knn=400)
    return sf.ScatterField(extent=EXT, dt=DT, in_scale=1.0, **kw).to(DEV).eval()


def unit_checks(log):
    out = {}
    model = p3m_model()
    g = torch.Generator(device=DEV).manual_seed(4738)
    worst_mesh, worst_v3, worst_mom, worst_v2id, worst_antisym = 0.0, 0.0, 0.0, 0.0, 0.0
    rows = []
    for r in (0.1, 0.25, 0.5, 1.0, 1.9, 3.0):
        pos, d = two_body_pos(r, 64, g, torch.float64)
        m = torch.ones(64, 2, device=DEV, dtype=torch.float64)
        # mesh from the model (fp32, zero field, nonet): gather of kernel acceleration only
        x = model.scatter(pos.float(), torch.zeros_like(pos).float(), m.float(), torch.ones_like(m).float())
        ak = model.kernel_acc(x[:, :1])
        mod_mesh = model.gather(ak, pos.float()).double()
        d_, r_, rhat, F, a_cont, win, Mall, eye = pair_terms(pos)
        mine_mesh = (Mall * m[:, None, :, None]).sum(2)
        scale = mod_mesh.abs().amax().clamp_min(1e-12)
        e_mesh = float((mod_mesh - mine_mesh).abs().amax() / scale)
        worst_mesh = max(worst_mesh, e_mesh)
        # V3 total equals exact pair force inside the window (self-force and momfix excluded, both ~0)
        pv = pair_vec("V3", rhat, F, a_cont, win, Mall, r_) * (~eye)[..., None]
        tot3 = mine_mesh + (pv * m[:, None, :, None]).sum(2)
        ex = (F[..., None] * rhat * (~eye)[..., None] * m[:, None, :, None]).sum(2)
        e_v3 = float((tot3 - ex).abs().amax() / ex.abs().amax()) if r < PP else None
        if e_v3 is not None:
            worst_v3 = max(worst_v3, e_v3)
        # momentum sum of mesh + pair (before momfix), all variants
        mom = 0.0
        for v in VARS:
            pv = pair_vec(v, rhat, F, a_cont, win, Mall, r_) * (~eye)[..., None]
            t = mine_mesh + (pv * m[:, None, :, None]).sum(2)
            mom = max(mom, float((m[..., None] * t).sum(1).abs().amax() / t.abs().amax()))
        worst_mom = max(worst_mom, mom)
        anti = float((Mall[:, 0, 1] + Mall[:, 1, 0]).abs().amax() / Mall[:, 0, 1].abs().amax().clamp_min(1e-12))
        worst_antisym = max(worst_antisym, anti)
        # model's P3M pair term (analytic complement) equals my base pair term
        pb = model.pp_acc(pos.float(), m.float(), torch.ones_like(m).float(), 400).double()
        base_pair = (pair_vec("base", rhat, F, a_cont, win, Mall, r_) * (~eye)[..., None] * m[:, None, :, None]).sum(2)
        e_base = float((pb - base_pair).abs().amax() / base_pair.abs().amax().clamp_min(1e-12))
        # V2 residual vs F at this r: (1-win)(M - a_cont rhat)
        pv2 = pair_vec("V2", rhat, F, a_cont, win, Mall, r_) * (~eye)[..., None]
        t2 = mine_mesh + (pv2 * m[:, None, :, None]).sum(2)
        rel2 = float(((t2 - ex).norm(dim=-1) / ex.norm(dim=-1)).pow(2).mean().sqrt())
        pvb = pair_vec("base", rhat, F, a_cont, win, Mall, r_) * (~eye)[..., None]
        tb = mine_mesh + (pvb * m[:, None, :, None]).sum(2)
        relb = float(((tb - ex).norm(dim=-1) / ex.norm(dim=-1)).pow(2).mean().sqrt())
        rows.append(dict(r=r, mesh_vs_model_relmax=e_mesh, v3_vs_exact_relmax=e_v3, momentum_relmax=mom, antisym_relmax=anti, base_pair_vs_model_relmax=e_base,
                         base_rel_rms_pct=100 * relb, V2_rel_rms_pct=100 * rel2))
        log(f"  unit r={r}: mesh vs model {e_mesh:.2e}, V3 vs exact {e_v3}, momentum {mom:.2e}, antisym {anti:.2e}, base pair vs model {e_base:.2e}, rel_rms base {100 * relb:.2f}% V2 {100 * rel2:.2f}%")
    out["rows"] = rows
    out["worst"] = dict(mesh_vs_model=worst_mesh, v3_vs_exact=worst_v3, momentum=worst_mom, antisym=worst_antisym)
    # many-body: complement mesh sum equals the model's mesh on a real scene incl. edge handling
    sc = sg.make_scenes("expB", 1, 9300)[0]
    p, m = sc["P"][10][None], sc["m"][None]
    x = model.scatter(p.float(), torch.zeros_like(p).float(), m.float(), torch.ones_like(m).float())
    mod_mesh = model.gather(model.kernel_acc(x[:, :1]), p.float()).double()
    mine = (pair_terms(p)[6] * m[:, None, :, None]).sum(2)
    out["scene_mesh_vs_model_relmax"] = float((mod_mesh - mine).abs().amax() / mod_mesh.abs().amax())
    fp, _, _, _ = forces(p, m)
    out["scene_base_vs_model_force_relmax"] = float(
        (model.force(p.float(), torch.zeros_like(p).float(), m.float(), torch.ones_like(m).float(), model.init_field(1, DEV)[0])[0].double() - fp["base"]).abs().amax() / fp["base"].abs().amax())
    log(f"  unit scene: mesh vs model {out['scene_mesh_vs_model_relmax']:.2e}, base force vs model.force {out['scene_base_vs_model_force_relmax']:.2e}")
    return out


def fmt_table(res):
    lines = []
    for nm, row in res.items():
        if "base" not in row:
            continue
        lines.append(f"{nm:14s} frac {row['frac']:.3f} " + " ".join(f"{v} {100 * row[v]['rel_rms']:.2f}%" for v in VARS))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(4738)
    log = lambda s: print(s, flush=True)
    out = {"e": args.e}
    t0 = time.time()
    with torch.no_grad():
        log("unit checks")
        out["unit"] = unit_checks(log)
        json.dump(out, open(args.out, "w"))
        rs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        n = 32 if args.smoke else 256
        log(f"analytic two-body probe {time.time() - t0:.0f}s")
        out["analytic_two_body"] = analytic_two_body(rs, n, 4738)
        for row in out["analytic_two_body"]:
            log(f"  r={row['r']}: " + " ".join(f"{v} {row[v]['rad_bias_pct']['mean']:+.1f}/{row[v]['rad_bias_pct']['std']:.1f}" for v in VARS))
        json.dump(out, open(args.out, "w"))
        log(f"attribution probe on E {time.time() - t0:.0f}s")
        model = ss.build_model("E", DT, DEV, args.e)
        out["attribution_E"], out["attribution_maxdiff_vs_model_force"] = attribution(model, rs, n, 4738, log)
        log(f"attribution component sum vs model.force max abs diff {out['attribution_maxdiff_vs_model_force']:.2e}")
        json.dump(out, open(args.out, "w"))
        for fam, (num, seed) in {"expB": (24, 9300), "cluster": (12, 9200)}.items():
            num = 2 if args.smoke else num
            scenes = sg.make_scenes(fam, num, seed)
            log(f"{fam}: {len(scenes)} scenes N={[len(s['m']) for s in scenes][:6]} {time.time() - t0:.0f}s")
            out[fam] = force_scenes(scenes, log, fam)
            log(fam + "\n" + fmt_table(out[fam]))
            json.dump(out, open(args.out, "w"))
    log("done")


if __name__ == "__main__":
    main()
