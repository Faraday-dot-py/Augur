"""Probes for the single-law readability experiments: force decomposition, U-Net share, bottleneck linear fits, top-k feature ablation.
Pre-registrations: docs/debugging/experiment-log.md entries two-number-message-bottleneck-single-law and sparse-code-unet-single-law."""
import numpy as np
import torch

from scripts import scatter_far_general as sfg
from scripts import scatter_multi as sm

PAIR_RANGE = (0.1, 1.9)
KERNEL_RANGE = (2.0, 32.0)
SLOPE_RANGE = (4.0, 16.0)
NPTS = 400
SITES = [f"enc{l}" for l in range(6)] + [f"dec{l}" for l in range(5)]
LAW = sfg.LAWS["pow2"]


def true_force(r):
    return sfg.force_t(r, LAW)


def true_potential(r):
    return -1.0 / torch.sqrt(r.double() ** 2 + LAW[5] ** 2)


@torch.no_grad()
def force_parts(model, pos, vel, mass, mask, field):
    B, G = pos.shape[0], model.grid
    x = model.scatter(pos, vel, mass, mask)
    m1 = torch.zeros_like(x[:, :1])
    ak = model.kernel_acc_multi(x[:, :1] - m1, m1)
    ak = torch.cat([ak[:, :2], ak[:, 2:] * 0], 1)
    cmap = model.cond.to(pos.device, pos.dtype).view(-1, sm.DC, 1, 1).expand(B, sm.DC, G, G)
    field = model.net(torch.cat([x, m1, ak, field, cmap], 1))
    mk = mask[..., None]
    parts = {"unet": model.gather(model.neg_grad(field[:, :1]), pos) * mk, "kernel": model.gather(ak[:, :2], pos) * mk,
             "pair": model.pp_acc(pos, mass, mask) * mk}
    return parts, field


@torch.no_grad()
def unet_share(model, data, dev, nf=40):
    S = data["P"].shape[0]
    num = {"unet": 0.0, "kernel": 0.0, "pair": 0.0}
    tot = 0.0
    for c0 in range(0, S, 16):
        sl = slice(c0, min(S, c0 + 16))
        Nb = int(data["n"][sl].max())
        P, V = data["P"][sl][:, :, :Nb], data["V"][sl][:, :, :Nb]
        mk = data["mask"][sl][:, :Nb]
        m = torch.ones_like(mk)
        f = model.init_field(P.shape[0], dev)[0]
        for t in range(nf + 1):
            parts, f = force_parts(model, P[:, t], V[:, t], m, mk, f)
            if t == 0:
                continue
            a = sum(parts.values()).double()
            tot += float((a ** 2).sum())
            for k, v in parts.items():
                num[k] += float((v.double() ** 2).sum())
    return {k: v / tot for k, v in num.items()}


def r2(X, y):
    X, y = X.double(), y.double()
    coef = torch.linalg.lstsq(X, y[:, None]).solution
    res = y - (X @ coef)[:, 0]
    return float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _code(mlp, f, cond):
    return mlp.code(f, cond) if mlp.head is not None else mlp(f, cond)[..., :1]


@torch.no_grad()
def bottleneck_fit(model, dev):
    cond = torch.zeros(1, sm.DC, device=dev)
    rp = torch.linspace(*PAIR_RANGE, NPTS, device=dev)
    win = (1 - (rp / model.pp).clamp(max=1.0) ** 2) ** 2
    mlp = model.ppmlp.mlp
    zp = _code(mlp, torch.stack([rp, torch.log(rp + 0.05)], -1), cond)
    yp = (mlp(torch.stack([rp, torch.log(rp + 0.05)], -1), cond)[..., 0] * win)
    ft = true_force(rp).to(dev)
    rk = torch.linspace(*KERNEL_RANGE, NPTS, device=dev)
    mlp = model.kmlp
    fk = torch.stack([rk, torch.log(rk + 0.5)], -1)
    zk = _code(mlp, fk, cond)
    phi = mlp(fk, cond)[..., 0]
    out = {"pair_code_dim": int(zp.shape[-1]), "pair_r2_code": r2(zp * win[:, None], ft), "pair_r2_output": r2((yp)[:, None], ft),
           "kernel_r2_code": r2(torch.cat([zk, torch.ones_like(rk)[:, None]], 1), true_potential(rk).to(dev)),
           "kernel_r2_output": r2(torch.stack([phi, torch.ones_like(rk)], 1), true_potential(rk).to(dev))}
    rs = torch.linspace(SLOPE_RANGE[0] - 0.5, SLOPE_RANGE[1] + 0.5, 4 * NPTS, device=dev)
    fs = torch.stack([rs, torch.log(rs + 0.5)], -1)
    ph = mlp(fs, cond)[..., 0].double()
    dr = rs[1] - rs[0]
    fm = -(ph[2:] - ph[:-2]) / (2 * dr)
    rm = rs[1:-1]
    sel = (rm >= SLOPE_RANGE[0]) & (rm <= SLOPE_RANGE[1])
    lr = torch.log(rm[sel]).double()
    out["kernel_slope_model"] = slope(lr, torch.log(fm[sel].abs().clamp_min(1e-30)))
    out["kernel_slope_true"] = slope(lr, torch.log(true_force(rm[sel]).to(dev).abs()))
    out["kernel_slope_rel_diff"] = abs(out["kernel_slope_model"] - out["kernel_slope_true"]) / abs(out["kernel_slope_true"])
    return out


def slope(x, y):
    X = torch.stack([x, torch.ones_like(x)], 1)
    return float(torch.linalg.lstsq(X, y[:, None]).solution[0, 0])


@torch.no_grad()
def scene_force_err(model, data, dev, nf=40):
    S = data["P"].shape[0]
    out = []
    for c0 in range(0, S, 16):
        sl = slice(c0, min(S, c0 + 16))
        Nb = int(data["n"][sl].max())
        P, V = data["P"][sl][:, :, :Nb], data["V"][sl][:, :, :Nb]
        mk = data["mask"][sl][:, :Nb]
        par = data["params"][sl]
        m = torch.ones_like(mk)
        w = mk[:, :, None].double()
        f = model.init_field(P.shape[0], dev)[0]
        e = torch.zeros(P.shape[0], dtype=torch.float64, device=dev)
        n = torch.zeros_like(e)
        for t in range(nf + 1):
            a, f = model.force(P[:, t], V[:, t], m, mk, f)
            if t == 0:
                continue
            ref = sm.exact_acc(P[:, t].double(), None, mk.double(), par)
            e += (((a.double() - ref) * w) ** 2).sum((1, 2))
            n += ((ref * w) ** 2).sum((1, 2))
        out.append((e / n).sqrt().cpu())
    return torch.cat(out).numpy()


def median_err(model, data, dev, ablate=()):
    model.net.ablate = tuple(ablate)
    try:
        return float(np.median(scene_force_err(model, data, dev)))
    finally:
        model.net.ablate = ()


@torch.no_grad()
def activation_frequency(model, data, dev, nf=10):
    net = model.net
    net.record = {}
    scene_force_err(model, {k: (v[:8] if torch.is_tensor(v) else v) for k, v in data.items()}, dev, nf)
    freq = {s: (net.record[s][0] / net.record[s][1]).cpu().numpy() for s in net.record}
    net.record = None
    return freq


def rank_features(model, data, dev, sites=SITES, width=32):
    base = median_err(model, data, dev)
    rows = []
    for s in sites:
        for c in range(width):
            rows.append((median_err(model, data, dev, ((s, c),)) - base, s, c))
    rows.sort(key=lambda r: (-r[0], SITES.index(r[1]), r[2]))
    return base, rows


def matched_random(top, freq, rng, pool=6):
    picked = []
    for _, s, c in top:
        order = np.argsort(np.abs(freq[s] - freq[s][c]), kind="stable")
        cand = [int(j) for j in order if int(j) != c and freq[s][int(j)] > 0 and (s, int(j)) not in [(a[1], a[2]) for a in top]][:pool]
        picked.append((s, int(rng.choice(cand))))
    return picked


def ablation_test(model, ranked, freq, data, dev, draws=30, ntop=5, seed=4738):
    top = ranked[:ntop]
    rng = np.random.default_rng(seed)
    base = median_err(model, data, dev)
    d_top = median_err(model, data, dev, [(s, c) for _, s, c in top]) - base
    d_rand = [median_err(model, data, dev, matched_random(top, freq, rng)) - base for _ in range(draws)]
    ratio = d_top / max(float(np.mean(d_rand)), 0.01 * base)
    return {"baseline_median": base, "delta_top": d_top, "delta_random": d_rand, "delta_random_mean": float(np.mean(d_rand)),
            "delta_random_max": float(np.max(d_rand)), "ratio": ratio, "above_random_max": bool(d_top > np.max(d_rand)),
            "top": [(s, c) for _, s, c in top]}
