"""Network utilization measurements for the scatter-field models (curiosity, inference only; docs/debugging/network-utilization-statistics.md).
Per-layer activation statistics of the U-Net (enc0-enc5 with enc5 the bottleneck, dec0-dec4) and the pair MLP on the sae_interp.py scene mix, the dec2/dec3 dead-layer
characterisation, U-Net path ablations, checkpoint scan, cross-checkpoint activation alignment, and re-encoded statistics for every saved SAE (*_sae.pt).
Usage: PYTHONPATH=. python scripts/network_utilization.py --out results/network_utilization
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from scripts import scatter_field as sf
from scripts import scatter_multi as sm
from scripts.sae_interp import LAWS, pair_acts

TAU = 0.01
ENC = [f"enc{i}" for i in range(6)]
DEC = [f"dec{i}" for i in range(5)]
LAYERS = ENC + DEC


def load_model(path, kind, dev):
    if kind == "M":
        m = sm.build().to(dev).eval()
    else:
        m = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    ck = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model"])
    return m, ck


def layer_module(net, name):
    return (net.dec if name.startswith("dec") else net.enc)[int(name[3:])]


@torch.no_grad()
def collect(model, kind, dev, scenes, cells, warm, seed, layers=LAYERS, only=None):
    """Same scenes, cell selection and hook point as sae_interp.mesh_acts, all layers in one pass. only: set of law indices to run forward for."""
    rng = np.random.default_rng(seed)
    store = {}
    hs = [layer_module(model.net, n).register_forward_hook(lambda m, i, o, n=n: store.__setitem__(n, o)) for n in layers]
    X = {n: [] for n in layers}
    lab, dens, rad, sid = [], [], [], []
    k = 0
    for li, law in enumerate(LAWS):
        for s in range(scenes):
            p, v = sm.init_scene(rng, "uniform" if s % 2 == 0 else "cluster")
            if only is not None and li not in only:
                k += 1
                continue
            n = p.shape[0]
            pad = lambda a: torch.tensor(np.concatenate([a, np.zeros((sm.N_PAD - n,) + a.shape[1:])]), dtype=torch.float32, device=dev)[None]
            pos, vel = pad(p), pad(v)
            mask = torch.zeros(1, sm.N_PAD, device=dev)
            mask[0, :n] = 1.0
            mass = mask.clone()
            if kind == "M":
                cond = torch.tensor([sm.cond_of(sm.slots_for("single", law, 1.0))], dtype=torch.float32, device=dev)
                model.set_law(cond, torch.zeros(1, sm.N_PAD, device=dev))
            field = model.init_field(1, dev)[0]
            for _ in range(warm + 1):
                _, field = model.force(pos, vel, mass, mask, field)
            rho = model.scatter(pos, vel, mass, mask)[0, 0]
            G = rho.shape[0]
            ij = (rho > 0.05).nonzero()
            ij = ij[torch.randperm(len(ij), generator=torch.Generator().manual_seed(seed + k))[:cells].to(dev)]
            c = (ij.float() + 0.5 - G / 2) * model.h
            for nm in layers:
                a = store[nm][0]
                stride = G // a.shape[-1]
                X[nm].append(a[:, ij[:, 0] // stride, ij[:, 1] // stride].T.cpu())
            lab.append(torch.full((len(ij),), li))
            dens.append(torch.log(rho[ij[:, 0], ij[:, 1]]).cpu())
            rad.append(c.norm(dim=-1).cpu())
            sid.append(torch.full((len(ij),), k))
            k += 1
    for h in hs:
        h.remove()
    return {n: torch.cat(v) for n, v in X.items()}, torch.cat(lab), torch.cat(sid), torch.cat(dens), torch.cat(rad)


def eig_stats(X, standardize):
    X = X.double()
    N = len(X)
    if standardize:
        sd = X.std(0)
        X = X[:, sd > 0]
        X = (X - X.mean(0)) / X.std(0) if X.shape[1] else X
    if X.shape[1] == 0 or N < 2:
        return {"pr": 0.0, "n90": 0, "n99": 0}
    Xc = X - X.mean(0)
    ev = torch.linalg.eigvalsh(Xc.T @ Xc / (N - 1)).clamp_min(0).flip(0)
    s = float(ev.sum())
    if s <= 0:
        return {"pr": 0.0, "n90": 0, "n99": 0}
    cum = torch.cumsum(ev, 0) / s
    return {"pr": float(s * s / (ev ** 2).sum()), "n90": int((cum < 0.90).sum()) + 1, "n99": int((cum < 0.99).sum()) + 1}


def layer_stats(X, full=True):
    N, C = X.shape
    on = X > TAU
    out = {"n_rows": N, "n_channels": C, "n_exactly_dead": int((X.abs().max(0).values == 0).sum()),
           "n_ever_fire_nonzero": int((X.abs().max(0).values > 0).sum()), "n_ever_fire_gt_tau": int(on.any(0).sum()),
           "frac_channels_ever_fire": float(on.any(0).float().mean()), "frac_firing_per_token": float(on.float().mean()),
           "n_constant_nonzero": int((((X.max(0).values - X.min(0).values) <= 1e-6 * X.abs().max(0).values) & (X.abs().max(0).values > 0)).sum())}
    if full:
        out.update({"mean": float(X.double().mean()), "mean_abs": float(X.abs().double().mean()), "max": float(X.max()), "min": float(X.min()),
                    "mean_token_l2": float(X.double().norm(dim=1).mean())})
    out["cov"] = eig_stats(X, False)
    out["corr"] = eig_stats(X, True)
    return out


def stats_with_laws(X, lab, laws):
    out = {"pooled": layer_stats(X)}
    per = {}
    for li in laws:
        s = layer_stats(X[lab == li], full=False)
        per[LAWS[li]] = {"frac_channels_ever_fire": s["frac_channels_ever_fire"], "frac_firing_per_token": s["frac_firing_per_token"], "n_exactly_dead": s["n_exactly_dead"],
                         "cov_pr": s["cov"]["pr"], "cov_n90": s["cov"]["n90"], "cov_n99": s["cov"]["n99"], "corr_pr": s["corr"]["pr"], "n_rows": s["n_rows"]}
    out["per_law"] = per
    return out


@torch.no_grad()
def pair_hidden(model, kind, dev, amps=(0.3, 1.0, 3.0), nr=1000):
    if kind == "M":
        out = {}
        for layer in ("l1", "l2"):
            X, lab, _, _ = pair_acts(model, dev, amps, nr, layer)
            out[layer] = (X, lab)
        return out
    r = torch.exp(torch.linspace(np.log(0.05), np.log(4.0), nr, device=dev))
    f = torch.stack([r, torch.log(r + 0.05)], -1)
    seq = model.ppmlp
    h1 = F.gelu(seq[0](f))
    h2 = F.gelu(seq[2](h1))
    lab = torch.full((3 * nr,), 2)
    return {"l1": (h1.repeat(3, 1).cpu(), lab), "l2": (h2.repeat(3, 1).cpu(), lab)}


def cka_linear(A, B):
    A = A.double() - A.double().mean(0)
    B = B.double() - B.double().mean(0)
    num = float((A.T @ B).norm() ** 2)
    den = float((A.T @ A).norm() * (B.T @ B).norm())
    return num / den if den > 0 else float("nan")


def std_cols(A):
    A = A.double()
    return A[:, A.std(0) > 0]


def corr_matrix(A, B):
    A = A.double() - A.double().mean(0)
    B = B.double() - B.double().mean(0)
    na, nb = A.norm(dim=0), B.norm(dim=0)
    C = (A.T @ B) / (na[:, None] * nb[None, :]).clamp_min(1e-300)
    C[na == 0] = 0
    C[:, nb == 0] = 0
    return C


def align_stats(A, B, seed):
    assert len(A) == len(B), (A.shape, B.shape)
    g = torch.Generator().manual_seed(seed)
    Bp = B[torch.randperm(len(B), generator=g)]
    C = corr_matrix(A, B)
    Cn = corr_matrix(A, Bp)
    out = {"cka_linear": cka_linear(A, B), "cka_linear_standardized": cka_linear(std_cols(A), std_cols(B)) if std_cols(A).shape[1] and std_cols(B).shape[1] else None,
           "cka_linear_rowshuffle_null": cka_linear(A, Bp)}
    if A.shape[1] == B.shape[1]:
        out["mean_abs_corr_identity_channels"] = float(C.diag().abs().mean())
        out["mean_abs_corr_identity_rowshuffle_null"] = float(Cn.diag().abs().mean())
    return out, C.numpy(), Cn.numpy()


@torch.no_grad()
def dead_layer(model, kind, dev, seed, warm=3):
    """full-resolution capture on scene 0 of each law: per-decoder-block output, both conv pre-activations, concat input halves, mass-cell correlation of dec3 magnitude."""
    rng = np.random.default_rng(seed)
    net = model.net
    store = {}
    hs = []
    for l in range(5):
        blk = net.dec[l]
        hs.append(blk.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(("out", l), o)))
        hs.append(blk[0].register_forward_hook(lambda m, i, o, l=l: store.__setitem__(("pre1", l), o)))
        hs.append(blk[2].register_forward_hook(lambda m, i, o, l=l: store.__setitem__(("pre2", l), o)))
        hs.append(blk.register_forward_pre_hook(lambda m, i, l=l: store.__setitem__(("in", l), i[0])))
    hs.append(net.register_forward_pre_hook(lambda m, i: store.__setitem__("netin", i[0])))
    hs.append(net.register_forward_hook(lambda m, i, o: store.__setitem__("netout", o)))
    per = {f"dec{l}": [] for l in range(5)}
    ablate = {}
    share = []
    corr3 = []
    scenes_done = 0
    for li, law in enumerate(LAWS):
        p, v = sm.init_scene(rng, "uniform")
        n = p.shape[0]
        pad = lambda a: torch.tensor(np.concatenate([a, np.zeros((sm.N_PAD - n,) + a.shape[1:])]), dtype=torch.float32, device=dev)[None]
        pos, vel = pad(p), pad(v)
        mask = torch.zeros(1, sm.N_PAD, device=dev)
        mask[0, :n] = 1.0
        mass = mask.clone()
        if kind == "M":
            cond = torch.tensor([sm.cond_of(sm.slots_for("single", law, 1.0))], dtype=torch.float32, device=dev)
            model.set_law(cond, torch.zeros(1, sm.N_PAD, device=dev))
        field = model.init_field(1, dev)[0]
        for _ in range(warm + 1):
            _, field = model.force(pos, vel, mass, mask, field)
        rho = model.scatter(pos, vel, mass, mask)[0, 0]
        G = rho.shape[0]
        for l in range(5):
            o = store[("out", l)][0]
            C, H, W = o.shape
            s = G // H
            m = F.avg_pool2d(rho[None, None], s)[0, 0] * s * s > 0.05 if s > 1 else rho > 0.05
            nz = (o != 0)
            cin = store[("in", l)][0]
            per[f"dec{l}"].append({
                "law": law, "frac_entries_nonzero": float(nz.float().mean()), "n_channels_all_zero": int((~nz.flatten(1).any(1)).sum()),
                "n_cells_any_nonzero": int(nz.any(0).sum()), "n_cells": H * W, "n_mass_cells": int(m.sum()), "n_mass_cells_any_nonzero": int((nz.any(0) & m).sum()),
                "mean_abs": float(o.abs().mean()), "max": float(o.max()),
                "channel_mean_abs": o.abs().flatten(1).mean(1).tolist(),
                "pre1_min": float(store[("pre1", l)].min()), "pre1_max": float(store[("pre1", l)].max()),
                "pre2_min": float(store[("pre2", l)].min()), "pre2_max": float(store[("pre2", l)].max()), "pre2_mean": float(store[("pre2", l)].mean()),
                "pre2_frac_below_neg10": float((store[("pre2", l)] < -10).float().mean()),
                "input_up_half_mean_abs": float(cin[:32].abs().mean()), "input_skip_half_mean_abs": float(cin[32:].abs().mean())})
            if l == 3:
                mag = o.abs().amax(0)
                mm = m
                if mm.sum() > 3:
                    lm = torch.log(F.avg_pool2d(rho[None, None], s)[0, 0] * s * s + 1e-9) if s > 1 else torch.log(rho + 1e-9)
                    a, b = mag[mm].flatten().double(), lm[mm].flatten().double()
                    ctr = ((torch.arange(H, device=dev).float() + 0.5 - H / 2) * model.h * s)
                    rr = torch.sqrt(ctr[:, None] ** 2 + ctr[None, :] ** 2)[mm].double()
                    def pc(x, y):
                        x, y = x - x.mean(), y - y.mean()
                        return float((x * y).sum() / (x.norm() * y.norm() + 1e-300))
                    corr3.append({"law": law, "corr_max_abs_vs_log_cell_mass": pc(a, b), "corr_max_abs_vs_radius": pc(a, rr), "frac_cells_max_abs_gt_100": float((mag > 100).float().mean()),
                                  "frac_mass_cells_max_abs_gt_100": float((a > 100).float().mean())})
        netin, netout = store["netin"], store["netout"]
        rhoM = (rho > 0.05)
        base = netout
        ref = unet_run(net, netin)
        chk = float((ref - base).abs().max())
        denom = float(base[:, :, rhoM].norm())
        if kind == "M":
            ak = netin[:, 4:8]
            nb = model.neg_grad(base[:, :1])
            share.append({"law": law, "rms_net_acc_over_rms_kernel_acc_mass_cells": float(nb[:, :, rhoM].pow(2).mean().sqrt() / (ak[:, :2][:, :, rhoM].pow(2).mean().sqrt() + 1e-30))})
        for l in range(5):
            for name, kw in (("drop_upsampled_path", dict(drop_up=l)), ("drop_skip", dict(drop_skip=l)), ("zero_block_output", dict(zero_dec_out=l))):
                d = unet_run(net, netin, **kw)
                ablate.setdefault(f"dec{l}", {}).setdefault(name, []).append(float((d - base)[:, :, rhoM].norm()) / (denom + 1e-30))
        ablate.setdefault("_baseline_replicate_max_abs_diff", []).append(chk)
        scenes_done += 1
    for h in hs:
        h.remove()
    summ = {}
    for l, d in ablate.items():
        if l.startswith("_"):
            summ[l] = {"max": float(np.max(d))}
        else:
            summ[l] = {k: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in d.items()}
    return {"per_scene": per, "dec3_cell_correlation": corr3, "unet_share_of_acceleration": share, "ablation_rel_change_of_unet_output_mass_cells": summ}


def unet_run(net, x, drop_up=None, drop_skip=None, zero_dec_out=None):
    h = net.enc[0](F.gelu(net.inp(x)))
    skips = [h]
    for l in range(1, net.levels + 1):
        h = net.enc[l](F.avg_pool2d(h, 2))
        skips.append(h)
    for l in reversed(range(net.levels)):
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        if drop_up == l:
            h = torch.zeros_like(h)
        s = torch.zeros_like(skips[l]) if drop_skip == l else skips[l]
        h = net.dec[l](torch.cat([h, s], 1))
        if zero_dec_out == l:
            h = torch.zeros_like(h)
    return net.out(h)


@torch.no_grad()
def quick_scan(model, dev, seed):
    X, lab, sid, _, _ = collect(model, "M", dev, 2, 200, 3, seed, layers=["dec0", "dec1", "dec2", "dec3", "dec4"], only={2})
    return {n: {"mean_abs": float(v.abs().mean()), "max": float(v.max()), "frac_exact_zero": float((v == 0).float().mean()), "n_channels_dead": int((v.abs().max(0).values == 0).sum())} for n, v in X.items()}


def encode(sae, x):
    pre = (x - sae["bd"]) @ sae["We"] + sae["be"]
    if sae["method"] == "topk":
        v, i = pre.topk(sae["k"], dim=-1)
        return torch.zeros_like(pre).scatter(-1, i, F.relu(v))
    return F.relu(pre)


def feature_stats(Z):
    on = Z > 0
    freq = on.float().mean(0)
    mass = Z.double().sum(0)
    srt = torch.sort(mass, descending=True).values
    cum = torch.cumsum(srt, 0) / srt.sum().clamp_min(1e-300)
    live = freq > 0
    edges = [0.0, 1e-3, 1e-2, 1e-1, 0.5, 1.0000001]
    hist = {"exactly_0": int((freq == 0).sum())}
    for a, b in zip(edges[:-1], edges[1:]):
        hist[f"({a:g},{min(b, 1):g}]"] = int(((freq > a) & (freq <= b)).sum())
    return {"l0": float(on.float().sum(-1).mean()), "dead_fraction": float(1 - live.float().mean()), "n_features": Z.shape[1],
            "freq_min": float(freq.min()), "freq_median": float(freq.median()), "freq_max": float(freq.max()),
            "freq_median_live": float(freq[live].median()) if live.any() else None, "freq_histogram": hist,
            "n_features_for_50pct_mass": int((cum < 0.5).sum()) + 1, "n_features_for_90pct_mass": int((cum < 0.9).sum()) + 1}


def nmse(sae, X):
    rec = encode(sae, X) @ sae["W"] + sae["bd"]
    return float(((rec - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())


def sae_part(pt_path, dev, seed, cache, scene_opts=(12, 24, 48), heldout_opts=(1, 6, 16), cells=400, warm=3):
    saved = torch.load(pt_path, map_location=dev, weights_only=False)
    js = json.load(open(pt_path.replace("_sae.pt", ".json")))
    js.setdefault("layer", "dec0")
    js.setdefault("sae", "relu_l1")
    js.setdefault("pair_layer", "l2")
    res = {"json": os.path.basename(pt_path.replace("_sae.pt", ".json")), "ckpt": js["ckpt"], "layer": js["layer"], "sae": js["sae"], "entries": {}}
    for key, s in saved.items():
        part, d = key.split("_")
        s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        jr = js[part][f"dict_{d}"]
        if part == "mesh":
            ck = js["ckpt"]
            found = None
            for S in scene_opts:
                for h in heldout_opts:
                    if h >= S:
                        continue
                    ck_key = (ck, S, js["layer"])
                    if ck_key not in cache:
                        if ck not in cache:
                            cache[ck] = load_model(ck, "M", dev)[0]
                        X, lab, sid, _, _ = collect(cache[ck], "M", dev, S, cells, warm, js["data_seed"], layers=[js["layer"]])
                        cache[ck_key] = (X[js["layer"]], sid % S)
                    X, grp = cache[ck_key]
                    te = grp > grp.max() - h
                    if int((~te).sum()) == js["mesh"]["n_train"] and int(te.sum()) == js["mesh"]["n_heldout"]:
                        found = (S, h)
                        break
                if found:
                    break
            if found is None:
                raise RuntimeError(f"cannot reconstruct scene split for {pt_path} {key}")
            X, grp = cache[(ck, found[0], js["layer"])]
            te = grp > grp.max() - found[1]
            meta = {"scenes": found[0], "heldout_scenes": found[1]}
        else:
            ck = js["ckpt"]
            if ck not in cache:
                cache[ck] = load_model(ck, "M", dev)[0]
            X, _, _, grp = pair_acts(cache[ck], dev, (0.3, 1.0, 3.0), 1000, js["pair_layer"])
            te = grp > grp.max() - 1
            meta = {}
        Xn = ((X - s["mu"].cpu()) / (s["sc"].cpu() if torch.is_tensor(s["sc"]) else s["sc"])).to(dev)
        tr, te = (~te).to(dev), te.to(dev)
        Ztr, Zte = encode(s, Xn[tr]), encode(s, Xn[te])
        e = {**meta, "n_train": int(tr.sum()), "n_heldout": int(te.sum()), "method": s["method"], "k": s["k"],
             "train": feature_stats(Ztr), "heldout": feature_stats(Zte),
             "nmse_train_recomputed": nmse(s, Xn[tr]), "nmse_heldout_recomputed": nmse(s, Xn[te]),
             "nmse_heldout_json": jr["recon_nmse_heldout"], "nmse_train_json": jr["recon_nmse_train"], "l0_heldout_json": jr["l0_heldout"],
             "dead_fraction_json_last_window": jr["dead_fraction"], "heldout_dead_fraction_json": jr.get("heldout_dead_fraction")}
        if abs(e["nmse_heldout_recomputed"] - e["nmse_heldout_json"]) > 0.02 * abs(e["nmse_heldout_json"]) + 1e-6:
            e["MISMATCH_vs_json"] = True
        res["entries"][key] = e
        print("sae", pt_path, key, "l0", round(e["heldout"]["l0"], 2), "nmse", e["nmse_heldout_recomputed"], e["nmse_heldout_json"], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--cells", type=int, default=400)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--models", default="mixed_s4738:ckpt/main_s4738.pt:M,mixed_s4739:ckpt/main_s4739.pt:M,pinned_gravity_E_frozen:kern/E_frozen.pt:E")
    ap.add_argument("--sae-glob", default="results/*_sae.pt")
    ap.add_argument("--scan-glob", default="ckpt/**/*.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sae-scene-options", default="12,24,48")
    ap.add_argument("--sae-heldout-options", default="1,6,16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    res = {"seed": a.seed, "scenes": a.scenes, "cells": a.cells, "warm": a.warm, "tau_on": TAU, "laws": LAWS, "models": {}}
    acts, pairs, labs = {}, {}, {}
    for spec in a.models.split(","):
        name, path, kind = spec.split(":")
        assert os.path.exists(path), f"missing {path}"
        model, ck = load_model(path, kind, dev)
        r = {"ckpt": path, "kind": kind, "ckpt_keys": [k for k in ck.keys()], "ckpt_scalars": {k: v for k, v in ck.items() if isinstance(v, (int, float, str))}}
        only = None if kind == "M" else {2}
        X, lab, sid, dens, rad = collect(model, kind, dev, a.scenes, a.cells, a.warm, a.seed, only=only)
        laws = sorted(set(lab.tolist()))
        r["mesh"] = {n: stats_with_laws(v, lab, laws) for n, v in X.items()}
        ph = pair_hidden(model, kind, dev)
        r["pair"] = {n: stats_with_laws(v[0], v[1], sorted(set(v[1].tolist()))) for n, v in ph.items()}
        r["pair"] = {("pair_" + n): v for n, v in r["pair"].items()}
        r["dead_layer"] = dead_layer(model, kind, dev, a.seed)
        res["models"][name] = r
        acts[name], pairs[name], labs[name] = X, ph, lab
        print("model", name, "done", flush=True)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    names = list(acts)
    res["alignment"] = {}
    npz = {}
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            key = f"{na}__vs__{nb}"
            res["alignment"][key] = {}
            sel = lambda lab: (lab == 2) if (na.startswith("pinned") or nb.startswith("pinned")) else torch.ones_like(lab, dtype=torch.bool)
            for n in LAYERS:
                A, B = acts[na][n], acts[nb][n]
                sa, sb = sel(labs[na]), sel(labs[nb])
                st, C, Cn = align_stats(A[sa], B[sb], a.seed)
                st["rows"] = int(sa.sum())
                res["alignment"][key][n] = st
                npz[f"{key}__{n}__corr"], npz[f"{key}__{n}__corr_null"] = C.astype(np.float32), Cn.astype(np.float32)
            for pl in ("l1", "l2"):
                A, la = pairs[na][pl]
                B, lb = pairs[nb][pl]
                sa, sb = sel(la), sel(lb)
                st, C, Cn = align_stats(A[sa], B[sb], a.seed)
                st["rows"] = int(sa.sum())
                res["alignment"][key]["pair_" + pl] = st
                npz[f"{key}__pair_{pl}__corr"], npz[f"{key}__pair_{pl}__corr_null"] = C.astype(np.float32), Cn.astype(np.float32)
    np.savez_compressed(os.path.join(a.out, "alignment_corr_matrices.npz"), **npz)
    scan = {}
    for p in sorted(set(glob.glob(a.scan_glob, recursive=True))):
        if not re.search(r"main|4738|4739|mixed", os.path.basename(p)):
            continue
        try:
            m, ck = load_model(p, "M", dev)
        except Exception as ex:
            scan[p] = {"loadable_as_multi_scatter_field": False, "error": str(ex)[:200]}
            continue
        scan[p] = {"loadable_as_multi_scatter_field": True, "mtime": os.path.getmtime(p), "ckpt_keys": list(ck.keys()), "ckpt_scalars": {k: v for k, v in ck.items() if isinstance(v, (int, float, str))},
                   "dec": quick_scan(m, dev, a.seed)}
        del m
    res["checkpoint_scan"] = scan
    res["sae"] = {}
    cache = {}
    res["sae_files_found"] = sorted(glob.glob(a.sae_glob))
    json.dump(res, open(os.path.join(a.out, "network_utilization.json"), "w"), indent=1)
    errors = []
    for p in res["sae_files_found"]:
        try:
            res["sae"][os.path.basename(p)] = sae_part(p, dev, a.seed, cache, tuple(int(v) for v in a.sae_scene_options.split(",")), tuple(int(v) for v in a.sae_heldout_options.split(",")), a.cells, a.warm)
        except Exception as ex:
            errors.append(p)
            res["sae"][os.path.basename(p)] = {"ERROR": repr(ex)[:300]}
            print("SAE FAILED", p, repr(ex)[:300], flush=True)
    res["sae_errors"] = errors
    json.dump(res, open(os.path.join(a.out, "network_utilization.json"), "w"), indent=1)
    if errors:
        raise SystemExit(f"{len(errors)} SAE files failed: {errors}")
    print("DONE")


if __name__ == "__main__":
    main()
