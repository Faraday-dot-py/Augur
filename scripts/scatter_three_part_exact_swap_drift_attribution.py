"""Closed-loop exact-swap drift attribution for the three scatter-field parts (kernel, U-Net potential residual, pair term). Evaluation only, no training.
Pre-registration: docs/debugging/architecture-push-prereg/scatter-field-three-part-exact-swap-drift-attribution.md

  PYTHONPATH=. python scripts/scatter_three_part_exact_swap_drift_attribution.py --model exp-e-frozen --ckpt kern/E_frozen.pt --out-dir results/scatter_field_three_part_exact_swap_drift_attribution

Per body, a = U + K + P (+ momentum fix for Exp E), U the gathered U-Net potential gradient, K the gathered FFT-kernel mesh acceleration, P the pair term.
Exact parts (fp64, direct sums, unit mass, eps 0.5, G 1): F = exact softened force; P_exact = sum over pairs with r < pp of (pair force minus the analytic
continuous mesh share a_cont); K_exact = the model's own mesh pipeline (CIC scatter, FFT, 2h central difference, CIC gather) with the analytic split kernel;
U_exact = F - K_exact - P_exact. The U-Net always receives the learned kernel acceleration as input and its recurrent field is always advanced, so a swap
changes only the force that part adds. An arm is a (kernel, U-Net, pair) triple of exact flags; arm index = kernel + 2 U-Net + 4 pair.
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_error_growth as sg
from scripts import scatter_field as sf
from scripts import scatter_multi as sm
from scripts import single_law_network_sizing as sl

DT, EPS, T = sg.DT, sg.EPS, sg.T
PP = 2.0
STOP = {"flag": False}
PARTS = ("kernel", "unet", "pair")
ARM_FLAGS = [(a & 1, (a >> 1) & 1, (a >> 2) & 1) for a in range(8)]
GROUPS = [("heldout", "expB", 9100), ("heldout", "expB", 9200), ("heldout", "expB", 9300), ("heldout", "cluster", 9100), ("heldout", "cluster", 9200),
          ("heldout", "cluster", 9300), ("fresh", "expB", 12000), ("fresh", "expB", 12200), ("fresh", "expB", 12100), ("fresh", "cluster", 12000),
          ("fresh", "cluster", 12200), ("fresh", "cluster", 12100)]
UNIT_FORCE_TOL = 1e-5
UNIT_ROLL_TOL = 1e-4
UNIT_SUM_TOL = 1e-5


def on_term(*_):
    STOP["flag"] = True


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def build_model(kind, ckpt, dev):
    if kind == "exp-e-frozen":
        m = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=DT, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=PP, split=True, verlet=True)
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        return m.to(dev).eval()
    m = sl.build_model(8, 4738, dev)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    m.set_law(torch.zeros(1, sm.DC, device=dev), None)
    return m.to(dev).eval()


def kernel_exact(model, rho):
    G = model.grid
    idx = torch.arange(2 * G, device=rho.device, dtype=rho.dtype)
    d = torch.minimum(idx, 2 * G - idx) * model.h
    r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
    K = -1.0 / torch.sqrt(r ** 2 + EPS ** 2) * (1 - (1 - (r / PP).clamp(max=1.0) ** 2) ** 2)
    full = torch.fft.irfft2(torch.fft.rfft2(rho, s=(2 * G, 2 * G)) * torch.fft.rfft2(K), s=(2 * G, 2 * G)) * model.h ** 2
    return model.neg_grad(full[..., :G, :G])


def exact_parts(pos, mask):
    """pos (B,N,2) fp64, mask (B,N) fp64 -> F, P_exact (B,N,2)."""
    N = pos.shape[1]
    d = pos[:, None, :, :] - pos[:, :, None, :]
    r2 = (d ** 2).sum(-1)
    valid = mask[:, None, :] * mask[:, :, None] * (~torch.eye(N, device=pos.device, dtype=torch.bool))[None].to(pos.dtype)
    q = torch.sqrt(r2 + EPS ** 2)
    F = ((valid / q ** 3)[..., None] * d).sum(2)
    r = torch.sqrt(r2 + 1e-30)
    u = (r / PP).clamp(max=1.0) ** 2
    win = (1 - u) ** 2
    a_mesh = -4 * (1 - u) * r / PP ** 2 / q + (1 - win) * r / q ** 3
    w = (r / q ** 3 - a_mesh) * valid * (r < PP).to(pos.dtype)
    return F, (w[..., None] * d / r[..., None]).sum(2)


def learned_parts(model, kind, pos, vel, mass, mask, field):
    x = model.scatter(pos, vel, mass, mask)
    if kind == "exp-e-frozen":
        ak = model.kernel_acc(x[:, :1])
        fld = model.net(torch.cat([x, ak, field], 1))
        k_acc = ak
    else:
        B, G = pos.shape[0], model.grid
        ak = model.kernel_acc_multi(x[:, :1], torch.zeros_like(x[:, :1]))
        ak = torch.cat([ak[:, :2], ak[:, 2:] * 0], 1)
        cmap = model.cond.to(pos.device, pos.dtype).view(-1, sm.DC, 1, 1).expand(B, sm.DC, G, G)
        fld = model.net(torch.cat([x, torch.zeros_like(x[:, :1]), ak, field, cmap], 1))
        k_acc = ak[:, :2]
    U = model.gather(model.neg_grad(fld[:, :1]), pos)
    K = model.gather(k_acc, pos)
    P = model.pp_acc(pos, mass, mask, model.knn)
    Ke = model.gather(kernel_exact(model, x[:, :1]), pos)
    return U, K, P, Ke, fld


def arm_force(model, kind, pos, vel, mask, field, sel):
    """pos, vel, mask fp64; sel (B,3) bool exact flags (kernel, unet, pair). Returns a (fp64), field, F, per-part learned-minus-exact (dK, dU, dP)."""
    m32 = mask.float()
    U, K, P, Ke, fld = learned_parts(model, kind, pos.float(), vel.float(), m32, m32, field)
    F, Pe = exact_parts(pos, mask)
    Ke, U, K, P = Ke.double(), U.double(), K.double(), P.double()
    Ue = F - Ke - Pe
    s = sel[:, :, None, None]
    a = torch.where(s[:, 0], Ke, K) + torch.where(s[:, 1], Ue, U) + torch.where(s[:, 2], Pe, P)
    if model.momfix:
        mm = mask[..., None]
        a = a - (mm * a).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)
    return a * mask[..., None], fld, F, (K - Ke, U - Ue, P - Pe)


def stats(a, F, d, pos, mask):
    w = ((pos.abs() < 32).all(-1) * mask)[..., None]
    sq = lambda x: ((x * w) ** 2).sum((1, 2))
    return torch.stack([sq(d[0]), sq(d[1]), sq(d[2]), sq(a - F), sq(F)], -1)


def pad_group(scenes, dev):
    S = len(scenes)
    N = max(s["P"].shape[1] for s in scenes)
    P = torch.zeros(S, T + 1, N, 2, dtype=torch.float64, device=dev)
    V = torch.zeros_like(P)
    mask = torch.zeros(S, N, dtype=torch.float64, device=dev)
    for i, s in enumerate(scenes):
        n = s["P"].shape[1]
        P[i, :, :n], V[i, :, :n], mask[i, :n] = s["P"], s["V"], 1.0
    return P, V, mask


def pos_err(p, Pt, mask):
    return ((p - Pt).norm(dim=-1) * mask).sum(1) / mask.sum(1)


@torch.no_grad()
def rollout_arms(model, kind, P, V, mask):
    A, S = 8, P.shape[0]
    sel = torch.tensor(ARM_FLAGS, dtype=torch.bool, device=P.device).repeat_interleave(S, 0)
    rep = lambda x: x.repeat(A, *([1] * (x.dim() - 1)))
    mk = rep(mask)
    p, v = rep(P[:, 0]), rep(V[:, 0])
    field = model.init_field(A * S, P.device)[0]
    a_prev, field, F, d = arm_force(model, kind, p, v, mk, field, sel)
    err = torch.zeros(A, S, T + 1, dtype=torch.float64)
    rec = torch.zeros(A, S, T + 1, 5, dtype=torch.float64)
    rec[:, :, 0] = stats(a_prev, F, d, p, mk).view(A, S, 5).cpu()
    for t in range(T):
        pn = p + DT * v + 0.5 * DT ** 2 * a_prev
        an, field, F, d = arm_force(model, kind, pn, v + DT * a_prev, mk, field, sel)
        v = v + 0.5 * DT * (a_prev + an)
        p, a_prev = pn, an
        err[:, :, t + 1] = pos_err(p, rep(P[:, t + 1]), mk).view(A, S).cpu()
        rec[:, :, t + 1] = stats(an, F, d, p, mk).view(A, S, 5).cpu()
    return err, rec


@torch.no_grad()
def rollout_reference(P, V, mask):
    p, v = P[:, 0], V[:, 0]
    a_prev = exact_parts(p, mask)[0]
    out = torch.zeros(P.shape[0], T + 1, dtype=torch.float64)
    for t in range(T):
        pn = p + DT * v + 0.5 * DT ** 2 * a_prev
        an = exact_parts(pn, mask)[0]
        v = v + 0.5 * DT * (a_prev + an)
        p, a_prev = pn, an
        out[:, t + 1] = pos_err(p, P[:, t + 1], mask).cpu()
    return out


@torch.no_grad()
def teacher_forced(model, kind, P, V, mask):
    S = P.shape[0]
    zero = torch.zeros(S, 3, dtype=torch.bool, device=P.device)
    one = torch.ones_like(zero)
    field = model.init_field(S, P.device)[0]
    rec = torch.zeros(2, S, T + 1, 5, dtype=torch.float64)
    sum_vs_model = 0.0
    for t in range(T + 1):
        a0, fld, F, d = arm_force(model, kind, P[:, t], V[:, t], mask, field, zero)
        a1, _, F1, d1 = arm_force(model, kind, P[:, t], V[:, t], mask, field, one)
        rec[0, :, t] = stats(a0, F, d, P[:, t], mask).cpu()
        rec[1, :, t] = stats(a1, F1, d1, P[:, t], mask).cpu()
        if t in (0, 50, 100):
            m32 = mask.float()
            ref, _ = model.force(P[:, t].float(), V[:, t].float(), m32, m32, field)
            sum_vs_model = max(sum_vs_model, float((a0 - ref.double()).abs().max() / ref.double().abs().max()))
        field = fld
    return rec, sum_vs_model


@torch.no_grad()
def unit_checks(model, kind, P, V, mask, dev):
    out = {}
    rho_pos = P[:, 5].float()
    m32 = mask.float()
    x = model.scatter(rho_pos, V[:, 5].float(), m32, m32)
    prior = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=DT, momfix=False, in_scale=1.0, potential=True, kernel=True,
                            pp=PP, split=True, verlet=True, prior=True, nonet=True).to(dev).eval()
    ref = prior.kernel_acc(x[:, :1])
    mine = kernel_exact(model, x[:, :1])
    out["kernel_exact_vs_prior_model_relmax"] = float((ref - mine).abs().max() / ref.abs().max())
    F, Pe = exact_parts(P[:, 5], mask)
    Ke = model.gather(mine, rho_pos).double()
    Ue = F - Ke - Pe
    w = mask[..., None]
    out["unet_exact_over_force_relrms"] = float(((Ue * w) ** 2).sum().sqrt() / ((F * w) ** 2).sum().sqrt())
    out["pair_exact_over_force_relrms"] = float(((Pe * w) ** 2).sum().sqrt() / ((F * w) ** 2).sum().sqrt())
    out["kernel_exact_over_force_relrms"] = float(((Ke * w) ** 2).sum().sqrt() / ((F * w) ** 2).sum().sqrt())
    return out


def run_group(model, kind, grp, num, out_dir, dev):
    tag, fam, seed = grp
    path = f"{out_dir}/{kind}_{tag}_{fam}_{seed}.npz"
    if os.path.exists(path):
        log(f"skip {path}")
        return None
    t0 = time.time()
    scenes = sg.make_scenes(fam, num, seed)
    P, V, mask = pad_group(scenes, dev)
    log(f"{tag} {fam} {seed}: {P.shape[0]} scenes, N {[int(s['P'].shape[1]) for s in scenes]}, truth {time.time() - t0:.0f}s")
    chk = unit_checks(model, kind, P, V, mask, dev)
    tf, sum_vs_model = teacher_forced(model, kind, P, V, mask)
    log(f"  teacher-forced done {time.time() - t0:.0f}s; parts sum vs model.force {sum_vs_model:.2e}")
    err, rec = rollout_arms(model, kind, P, V, mask)
    ref = rollout_reference(P, V, mask)
    log(f"  rollout done {time.time() - t0:.0f}s; err@20 all learned {float(err[0, :, min(20, T)].mean()):.4g} all exact {float(err[7, :, min(20, T)].mean()):.4g}")
    allex_force = float((tf[1, :, :, 3].sum() / tf[1, :, :, 4].sum()).sqrt())
    roll_gap = {str(k): float((err[7, :, k] - ref[:, k]).abs().max()) for k in (20, 100, 150) if k <= T}
    unit = dict(chk, parts_sum_vs_model_force_relmax=sum_vs_model, all_exact_teacher_forced_force_rel=allex_force, all_exact_vs_reference_err_gap=roll_gap)
    unit["pass"] = bool(sum_vs_model <= UNIT_SUM_TOL and allex_force <= UNIT_FORCE_TOL and roll_gap[min(roll_gap, key=int)] <= UNIT_ROLL_TOL
                        and chk["kernel_exact_vs_prior_model_relmax"] <= UNIT_SUM_TOL)
    np.savez_compressed(path + ".tmp.npz", err=err.numpy(), rec=rec.numpy(), tf=tf.numpy(), ref=ref.numpy(), n=np.array([int(s["P"].shape[1]) for s in scenes]),
                        unit=json.dumps(unit), group=json.dumps(grp), scene_l=np.array([s["L"] for s in scenes]))
    os.replace(path + ".tmp.npz", path)
    log(f"  unit {json.dumps(unit)}")
    log(f"  wrote {path} in {time.time() - t0:.0f}s")
    return unit


def main():
    global T
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["exp-e-frozen", "width8-single-law-control"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenes-per-seed", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ticks", type=int, default=T)
    ap.add_argument("--groups", type=int, default=len(GROUPS))
    a = ap.parse_args()
    T = a.ticks
    sg.T = a.ticks
    signal.signal(signal.SIGTERM, on_term)
    dev = torch.device(a.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sg.DEV = dev
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(4738)
    model = build_model(a.model, a.ckpt, dev)
    log(f"{a.model}: {sum(p.numel() for p in model.parameters())} parameters, momfix {model.momfix}, knn {model.knn}, kpad {model.kpad}")
    assert model.kpad == "zero"
    for grp in GROUPS[:a.groups]:
        unit = run_group(model, a.model, grp, a.scenes_per_seed, a.out_dir, dev)
        if unit is not None and not unit["pass"]:
            log("UNIT CHECK FAILED, stopping")
            sys.exit(3)
        if STOP["flag"]:
            log("SIGTERM: exiting 143 after group")
            sys.exit(143)
    open(f"{a.out_dir}/{a.model}.done", "w").write(json.dumps({"groups": a.groups}))
    log("done")


if __name__ == "__main__":
    main()
