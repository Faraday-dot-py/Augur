"""R1 Step 1 (eval only): exact discrete mesh-pair complement built from the LEARNED kernel table of Exp E, swapped into the frozen model.

  PYTHONPATH=. python scripts/scatter_r1_step1.py --e checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/r1_step1/r1_step1.json
M_ij = acceleration on i per unit m_j through the deployed kernel mesh path (CIC deposit of j, FFT-conv with the deployed K table = kmlp(feat(r)) * split mask,
2h central difference, CIC gather at i). Variants (all after momfix, same teacher-forced truth states and recurrent field as the baseline):
  base : frozen E as deployed (ak + U-Net + learned pair g*win over the knn nearest)
  V3   : ak + U-Net + sum_j m_j (F rhat - M_ij) over r < pp   (learned pair term removed)
  V3g  : V3 + learned pair term g*win kept
  V3n  : ak + V3 pair, U-Net term removed (diagnostic: in-range exactness without the U-Net compensation)
Truth: exact softened gravity (eps 0.5) on the same bodies, momfix.
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import scatter_error_growth as sg
from scripts import scatter_r1_step0 as s0
from scripts import scatter_scenes as ss

DEV = s0.DEV
EPS, PP, G, EXT = s0.EPS, s0.PP, s0.G, s0.EXT
VARS = ("base", "V3", "V3g", "V3n")
NNB = s0.NNB


def deployed_K(model):
    n = torch.arange(2 * G, device=DEV, dtype=torch.float32)
    d = torch.minimum(n, 2 * G - n) * model.h
    r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
    K = model.kmlp(model._feat(r, "k"))[..., 0]
    if model.prior:
        K = -(1 + K) / torch.sqrt(r ** 2 + model.eps ** 2)
    if model.split:
        K = K * (1 - (1 - (r / model.ppe()).clamp(max=1.0) ** 2) ** 2)
    return K.double()


def mesh_pair_tab(xi, xj, Kt):
    xi, xj = torch.broadcast_tensors(xi, xj)
    ni, wi = s0.corners(xi)
    nj, wj = s0.corners(xj)
    e = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=xi.device, dtype=xi.dtype)
    s = ni[..., :, None, :] + e
    ingrid = ((s >= 0) & (s < G)).all(-1).to(xi.dtype)
    di = (s[..., :, :, None, :] - nj[..., None, None, :, :]).abs().long().clamp(max=2 * G - 1)
    phi = (Kt[di[..., 0], di[..., 1]] * wj[..., None, None, :]).sum(-1) * ingrid
    gx = (phi[..., 0] - phi[..., 1]) / (2 * s0.H)
    gy = (phi[..., 2] - phi[..., 3]) / (2 * s0.H)
    acc = -torch.stack([gx, gy], -1)
    return (acc * wi[..., None]).sum(-2)


def components(model, pos, vel, mass, mask, field):
    x = model.scatter(pos, vel, mass, mask)
    ak = model.kernel_acc(x[:, :1])
    x = torch.cat([x, ak], 1)
    x = torch.cat([x, field], 1)
    fld = model.net(x)
    net = model.neg_grad(fld[:, :1])
    ak_g, net_g = model.gather(ak, pos), model.gather(net, pos)
    pair = model.pp_acc(pos, mass, mask, model.knn)
    return ak_g, net_g, pair, fld


def v3_pair(p, m, Kt):
    d = p[:, None, :, :] - p[:, :, None, :]
    N = p.shape[1]
    eye = torch.eye(N, device=p.device, dtype=torch.bool)[None]
    r = torch.sqrt((d ** 2).sum(-1) + 1e-30)
    rhat = d / r[..., None]
    F = r / (r ** 2 + EPS ** 2) ** 1.5
    M = mesh_pair_tab(p[:, :, None, :], p[:, None, :, :], Kt)
    inr = ((r < PP) & ~eye)[..., None].to(p.dtype)
    pv = (F[..., None] * rhat - M) * inr
    mesh_all = (M * m[:, None, :, None]).sum(2)
    exact = (F[..., None] * rhat * (~eye)[..., None].to(p.dtype) * m[:, None, :, None]).sum(2)
    return (pv * m[:, None, :, None]).sum(2), mesh_all, exact


def one_state(model, Kt, p, v, m, field):
    pf, vf, mf_ = p[None].float(), v[None].float(), m[None].float()
    mask = torch.ones(1, len(p), device=DEV)
    ak_g, net_g, pair, fld = components(model, pf, vf, mf_, mask, field)
    a_ref, _ = model.force(pf, vf, mf_, mask, field)
    pv3, mesh_all, exact = v3_pair(p[None], m[None], Kt)
    mm = m[None]
    tot = {}
    tot["base"] = s0.momfix(ak_g.double() + net_g.double() + pair.double(), mm)
    tot["V3"] = s0.momfix(ak_g.double() + net_g.double() + pv3, mm)
    tot["V3g"] = s0.momfix(ak_g.double() + net_g.double() + pv3 + pair.double(), mm)
    tot["V3n"] = s0.momfix(ak_g.double() + pv3, mm)
    chk = float((tot["base"] - a_ref.double()).abs().max())
    mesh_chk = float((mesh_all - ak_g.double()).abs().max() / ak_g.double().abs().amax().clamp_min(1e-12))
    return tot, s0.momfix(exact, mm), chk, mesh_chk, fld


def run_scenes(model, Kt, scenes, log, tag):
    acc = {}
    chk, mchk = 0.0, 0.0
    for si, sc in enumerate(scenes):
        m = sc["m"]
        field = model.init_field(1, DEV)[0]
        for t in range(0, 100):
            p, v = sc["P"][t], sc["V"][t]
            tot, ex, c1, c2, fld = one_state(model, Kt, p, v, m, field)
            field = fld
            if t < 2:
                continue
            chk, mchk = max(chk, c1), max(mchk, c2)
            keep = (p.abs().amax(-1) < 26.0).cpu().numpy()
            nn = sg.nn_dist(p).cpu().numpy()
            rec = {"nn": nn[keep], "ae2": (ex[0] ** 2).sum(-1).cpu().numpy()[keep]}
            for k in VARS:
                rec[f"{k}:err2"] = ((tot[k] - ex)[0] ** 2).sum(-1).cpu().numpy()[keep]
            for k, x in rec.items():
                acc.setdefault(k, []).append(x)
        if si % 6 == 0:
            log(f"  {tag} scene {si}/{len(scenes)}")
    R = {k: np.concatenate(v) for k, v in acc.items()}
    b = np.digitize(R["nn"], NNB[1:-1])
    out = {"check_base_vs_model_force_maxabs": chk, "check_sum_Mij_vs_ak_relmax": mchk}
    sets = {"all": np.ones_like(b, bool), "nn<1": b <= 2, "nn>=2": b >= 4}
    for i in range(len(NNB) - 1):
        sets[f"nn {NNB[i]}-{NNB[i + 1]}"] = b == i
    for nm, mk in sets.items():
        if mk.sum() == 0:
            continue
        ae2 = R["ae2"][mk].mean()
        row = {"frac": float(mk.mean()), "count": int(mk.sum())}
        for k in VARS:
            row[k] = float(np.sqrt(R[f"{k}:err2"][mk].mean() / ae2))
        row["V3_reduction"] = 1 - row["V3"] / row["base"]
        out[nm] = row
    return out


def two_body(model, Kt, rs, n, seed, log):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    res = []
    for r in rs:
        pos64, d = s0.two_body_pos(r, n, gen, torch.float32)
        pos = pos64.float()
        v = torch.zeros_like(pos)
        mass, mask = torch.ones(n, 2, device=DEV), torch.ones(n, 2, device=DEV)
        f = model.init_field(n, DEV)[0]
        for _ in range(2):
            _, f = model.force(pos, v, mass, mask, f)
        ak_g, net_g, pair, _ = components(model, pos, v, mass, mask, f)
        pv3, _, _ = v3_pair(pos.double(), mass.double(), Kt)
        m64 = mass.double()
        tb = s0.momfix(ak_g.double() + net_g.double() + pair.double(), m64)
        t3 = s0.momfix(ak_g.double() + net_g.double() + pv3, m64)
        row = {"r": r}
        for nm, t in (("base", tb), ("V3", t3)):
            rad, tan, _ = s0.rad_stats(t[:, 0], d)
            row[nm] = dict(rad_bias_pct=s0.stat(rad - 100), tan_rms_pct=float(tan.pow(2).mean().sqrt()))
        res.append(row)
        log(f"  r={r}: base {row['base']['rad_bias_pct']['mean']:+.1f}/{row['base']['rad_bias_pct']['std']:.1f} V3 {row['V3']['rad_bias_pct']['mean']:+.1f}/{row['V3']['rad_bias_pct']['std']:.1f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(4738)
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    out = {"e": args.e}
    with torch.no_grad():
        model = ss.build_model("E", s0.DT, DEV, args.e)
        log(f"model prior={model.prior} split={model.split} knn={model.knn} pp_mode(200)={model.pp_mode(200)}")
        out["model"] = dict(prior=model.prior, split=model.split, knn=model.knn, pp_mode=str(model.pp_mode(200)))
        Kt = deployed_K(model)
        rs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        out["two_body"] = two_body(model, Kt, rs, 32 if args.smoke else 256, 4738, log)
        json.dump(out, open(args.out, "w"))
        for fam, (num, seed) in {"expB": (24, 9300), "cluster": (12, 9200), "expB_s9400": (24, 9400)}.items():
            num = 2 if args.smoke else num
            scenes = sg.make_scenes(fam.split("_")[0], num, seed)
            log(f"{fam}: {len(scenes)} scenes {time.time() - t0:.0f}s")
            out[fam] = run_scenes(model, Kt, scenes, log, fam)
            log(f"{fam} checks: base-vs-force {out[fam]['check_base_vs_model_force_maxabs']:.2e} sumMij-vs-ak {out[fam]['check_sum_Mij_vs_ak_relmax']:.2e}")
            for nm, row in out[fam].items():
                if isinstance(row, dict):
                    log(f"  {nm:14s} frac {row['frac']:.3f} base {100 * row['base']:.2f}% V3 {100 * row['V3']:.2f}% V3g {100 * row['V3g']:.2f}% V3n {100 * row['V3n']:.2f}% red {100 * row['V3_reduction']:+.1f}%")
            json.dump(out, open(args.out, "w"))
    log("done")


if __name__ == "__main__":
    main()
