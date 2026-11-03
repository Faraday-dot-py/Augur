"""Sparse autoencoder (SAE) curiosity experiment on the mixed multi-law scatter-field model (no pass/kill; see supervisor-log 2026-11-03).
Mesh net: 32-channel output of a U-Net block (default UNet.dec[0], last decoder block, full resolution) at cells holding mass. Pair MLP: 128-unit hidden
layer of ppmlp.mlp (CondMLP), a pure function of (r, law, amplitude). SAE: ReLU encoder, unit-norm decoder; sparsity by L1 with a coefficient adapted only to
hold mean L0 at the target (--sae relu_l1) or by keeping the top k pre-activations (--sae topk, L0 exactly k). Dead features are resampled twice during training.
Report per dictionary size: normalized reconstruction error (train and held-out), L0, dead fraction, feature alignment with law identity (ANOVA eta^2) and with
distance / density (|Pearson|). Held out: last scene index (mesh), every 7th radius (pair).
Usage: PYTHONPATH=. python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --out results/sae_interp_topk_dec0_s4738.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from scripts import scatter_multi as sm

LAWS = ["pow1", "pow1.5", "pow2", "pow3", "yuk8", "yuk32", "yuk128", "lj"]
MESH_SIZES = (128, 512, 2048)
PAIR_SIZES = (256, 1024, 4096)


def layer_module(model, name):
    kind, i = name[:3], int(name[3:])
    return model.net.dec[i] if kind == "dec" else model.net.enc[i]


@torch.no_grad()
def mesh_acts(model, dev, scenes, cells, warm, seed, layer):
    rng = np.random.default_rng(seed)
    store = {}
    h = layer_module(model, layer).register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    X, lab, dens, rad, sid = [], [], [], [], []
    k = 0
    for li, law in enumerate(LAWS):
        for s in range(scenes):
            p, v = sm.init_scene(rng, "uniform" if s % 2 == 0 else "cluster")
            n = p.shape[0]
            pad = lambda a: torch.tensor(np.concatenate([a, np.zeros((sm.N_PAD - n,) + a.shape[1:])]), dtype=torch.float32, device=dev)[None]
            pos, vel = pad(p), pad(v)
            mask = torch.zeros(1, sm.N_PAD, device=dev)
            mask[0, :n] = 1.0
            mass = mask.clone()
            slots = sm.slots_for("single", law, 1.0)
            cond = torch.tensor([sm.cond_of(slots)], dtype=torch.float32, device=dev)
            model.set_law(cond, torch.zeros(1, sm.N_PAD, device=dev))
            field = model.init_field(1, dev)[0]
            for _ in range(warm + 1):
                _, field = model.force(pos, vel, mass, mask, field)
            a = store["a"][0]
            rho = model.scatter(pos, vel, mass, mask)[0, 0]
            G = rho.shape[0]
            stride = G // a.shape[-1]
            ij = (rho > 0.05).nonzero()
            ij = ij[torch.randperm(len(ij), generator=torch.Generator().manual_seed(seed + k))[:cells].to(dev)]
            c = (ij.float() + 0.5 - G / 2) * model.h
            X.append(a[:, ij[:, 0] // stride, ij[:, 1] // stride].T.cpu())
            lab.append(torch.full((len(ij),), li))
            dens.append(torch.log(rho[ij[:, 0], ij[:, 1]]).cpu())
            rad.append(c.norm(dim=-1).cpu())
            sid.append(torch.full((len(ij),), k))
            k += 1
    h.remove()
    return torch.cat(X), torch.cat(lab), {"log_cell_mass": torch.cat(dens), "radius_from_centre": torch.cat(rad)}, torch.cat(sid)


@torch.no_grad()
def pair_acts(model, dev, amps, nr, layer):
    X, lab, logr, grp = [], [], [], []
    r = torch.exp(torch.linspace(np.log(0.05), np.log(4.0), nr, device=dev))
    f = torch.stack([r, torch.log(r + 0.05)], -1)
    mlp = model.ppmlp.mlp
    for li, law in enumerate(LAWS):
        for amp in amps:
            cond = torch.tensor(sm.cond_of(sm.slots_for("single", law, amp)), dtype=torch.float32, device=dev)
            h1 = F.gelu(mlp.l1(torch.cat([f, cond.expand(nr, -1)], -1)))
            X.append((h1 if layer == "l1" else F.gelu(mlp.l2(h1))).cpu())
            lab.append(torch.full((nr,), li))
            logr.append(torch.log(r).cpu())
            grp.append((torch.arange(nr) % 7 == 0).long())
    return torch.cat(X), torch.cat(lab), {"log_r": torch.cat(logr)}, torch.cat(grp)


def train_sae(X, d_dict, method, target_l0, steps, seed, dev, bs=4096, lr=1e-3, clip=0.0):
    g = torch.Generator(device=dev).manual_seed(seed)
    d = X.shape[1]
    W = torch.randn(d_dict, d, device=dev, generator=g)
    W = (W / W.norm(dim=1, keepdim=True)).requires_grad_()
    We = (W.detach().clone().T.contiguous()).requires_grad_()
    be = torch.zeros(d_dict, device=dev, requires_grad=True)
    bd = torch.zeros(d, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, We, be, bd], lr=lr)
    lam = 1e-3
    k = int(target_l0)
    fired = torch.zeros(d_dict, dtype=torch.bool, device=dev)
    l0s = []

    def act(x):
        pre = (x - bd) @ We + be
        if method == "topk":
            v, i = pre.topk(k, dim=-1)
            return torch.zeros_like(pre).scatter(-1, i, F.relu(v))
        return F.relu(pre)

    for it in range(steps):
        x = X[torch.randint(0, len(X), (bs,), device=dev, generator=g)]
        Wn = W / W.norm(dim=1, keepdim=True)
        z = act(x)
        rec = z @ Wn + bd
        loss = F.mse_loss(rec, x) + (lam * z.sum(-1).mean() if method == "relu_l1" else 0.0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite SAE loss at step {it} (d_dict {d_dict}, lr {lr})")
        opt.zero_grad()
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_([W, We, be, bd], clip)
        opt.step()
        for gp in opt.param_groups:
            gp["lr"] = lr * (1 - it / steps) + 1e-5
        fired |= (z > 0).any(0)
        l0s.append((z > 0).float().sum(-1).mean().item())
        if method == "relu_l1" and it % 20 == 19:
            lam *= 1.02 if float(np.mean(l0s[-20:])) > target_l0 else 1 / 1.02
        if it in (steps // 4, steps // 2):
            dead = ~fired
            if dead.any():
                with torch.no_grad():
                    err = ((act(x) @ (W / W.norm(dim=1, keepdim=True)) + bd - x) ** 2).sum(-1)
                    pick = err.topk(min(int(dead.sum()), len(err))).indices
                    nd = dead.nonzero()[: len(pick), 0]
                    dirs = x[pick] - bd
                    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    W[nd] = dirs
                    We[:, nd] = dirs.T * 0.2
                    be[nd] = 0.0
            fired = torch.zeros_like(fired)
    with torch.no_grad():
        W = W / W.norm(dim=1, keepdim=True)
    return W.detach(), We.detach(), be.detach(), bd.detach(), fired, method, k


def encode(sae, x):
    W, We, be, bd, _, method, k = sae
    pre = (x - bd) @ We + be
    if method == "topk":
        v, i = pre.topk(k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, i, F.relu(v))
    return F.relu(pre)


def eta2(Z, lab, k):
    tot = Z.var(0, unbiased=False) * len(Z) + 1e-12
    mu = Z.mean(0)
    bet = sum((lab == j).sum() * (Z[lab == j].mean(0) - mu) ** 2 for j in range(k) if (lab == j).any())
    return (bet / tot).clamp(0, 1)


def abscorr(Z, y):
    Zc, yc = Z - Z.mean(0), (y - y.mean())[:, None]
    return ((Zc * yc).sum(0) / (Zc.norm(dim=0) * yc.norm() + 1e-12)).abs()


def nmse(sae, X):
    rec = encode(sae, X) @ sae[0] + sae[3]
    return float(((rec - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())


def evaluate(sae, Xtr, Xte, lab_te, cont_te):
    Zte = encode(sae, Xte)
    out = {"recon_nmse_train": nmse(sae, Xtr), "recon_nmse_heldout": nmse(sae, Xte), "l0_heldout": float((Zte > 0).float().sum(-1).mean()),
           "dead_fraction": float(1 - sae[4].float().mean())}
    live = (Zte > 0).any(0)
    e_law = eta2(Zte, lab_te, len(LAWS))
    out["n_live_heldout"] = int(live.sum())
    out["heldout_dead_fraction"] = float(1 - live.float().mean())
    out["law_eta2_live_mean"] = float(e_law[live].mean()) if live.any() else None
    out["law_eta2_live_max"] = float(e_law[live].max()) if live.any() else None
    out["n_live_law_eta2_gt_0.5"] = int((e_law[live] > 0.5).sum())
    cors = {}
    for name, y in cont_te.items():
        c = abscorr(Zte, y)
        cors[name] = c.cpu().numpy()
        out[f"{name}_abscorr_live_mean"] = float(c[live].mean()) if live.any() else None
        out[f"{name}_abscorr_live_max"] = float(c[live].max()) if live.any() else None
        out[f"n_live_{name}_abscorr_gt_0.5"] = int((c[live] > 0.5).sum())
    top = torch.argsort(e_law * live, descending=True)[:5]
    out["top_law_features"] = []
    for j in top.tolist():
        per_law = [float(Zte[lab_te == li, j].mean()) for li in range(len(LAWS))]
        out["top_law_features"].append({"feature": j, "eta2": float(e_law[j]), "mean_act_per_law": dict(zip(LAWS, per_law))})
    return out, e_law.cpu().numpy(), cors, Zte


def run(name, X, lab, cont, grp, sizes, a, dev, saved, heldout=1):
    te = grp > grp.max() - heldout
    mu, sc = X[~te].mean(0), X[~te].norm(dim=-1).mean() / np.sqrt(X.shape[1])
    print(name, "input stats: nonfinite", int((~torch.isfinite(X)).sum()), "channel std min/max", float(X.std(0).min()), float(X.std(0).max()), "mean |x|", float(X.abs().mean()), flush=True)
    if a.channel_norm:
        sc = X[~te].std(0).clamp_min(1e-6)
    Xn = ((X - mu) / sc).to(dev)
    tr = (~te).to(dev)
    res, arrs = {"n_train": int(tr.sum()), "n_heldout": int(te.sum()), "d_input": X.shape[1]}, {}
    for d_dict in sizes:
        sae = train_sae(Xn[tr], d_dict, a.sae, a.target_l0, a.steps, a.sae_seed, dev, lr=a.lr, clip=a.clip)
        r, e_law, cors, Zte = evaluate(sae, Xn[tr], Xn[~tr], lab[te].to(dev), {k: v[te].to(dev) for k, v in cont.items()})
        res[f"dict_{d_dict}"] = r
        arrs[f"{name}_{d_dict}_law_eta2"] = e_law
        for k, v in cors.items():
            arrs[f"{name}_{d_dict}_{k}_abscorr"] = v
        arrs[f"{name}_{d_dict}_decoder"] = sae[0].cpu().numpy()
        arrs[f"{name}_{d_dict}_heldout_codes"] = Zte[:4000].cpu().numpy().astype(np.float16)
        arrs[f"{name}_{d_dict}_heldout_law"] = lab[te][:4000].numpy()
        for k, v in cont.items():
            arrs[f"{name}_{d_dict}_heldout_{k}"] = v[te][:4000].numpy()
        saved[f"{name}_{d_dict}"] = {"W": sae[0].cpu(), "We": sae[1].cpu(), "be": sae[2].cpu(), "bd": sae[3].cpu(), "mu": mu, "sc": sc.cpu() if torch.is_tensor(sc) else float(sc), "method": sae[5], "k": sae[6]}
        print(name, d_dict, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "top_law_features"}, flush=True)
    return res, arrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--sae-seed", type=int, default=None)
    ap.add_argument("--sae", choices=("relu_l1", "topk"), default="topk")
    ap.add_argument("--layer", default="dec0")
    ap.add_argument("--pair-layer", choices=("l1", "l2"), default="l2")
    ap.add_argument("--parts", default="mesh,pair")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--target-l0", type=float, default=8.0)
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--cells", type=int, default=400)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--mesh-sizes", default=",".join(map(str, MESH_SIZES)))
    ap.add_argument("--pair-sizes", default=",".join(map(str, PAIR_SIZES)))
    ap.add_argument("--heldout-scenes", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--channel-norm", action="store_true")
    a = ap.parse_args()
    a.sae_seed = a.seed if a.sae_seed is None else a.sae_seed
    torch.manual_seed(a.sae_seed)
    dev = torch.device("cuda")
    model = sm.build().to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    out = {"ckpt": a.ckpt, "data_seed": a.seed, "sae_seed": a.sae_seed, "sae": a.sae, "layer": a.layer, "pair_layer": a.pair_layer, "target_l0": a.target_l0, "steps": a.steps,
           "laws": LAWS}
    arrs, saved = {}, {}
    if "mesh" in a.parts:
        X, lab, cont, sid = mesh_acts(model, dev, a.scenes, a.cells, a.warm, a.seed, a.layer)
        print("mesh acts", tuple(X.shape), flush=True)
        out["mesh"], ar = run("mesh", X, lab, cont, sid % a.scenes, [int(v) for v in a.mesh_sizes.split(",")], a, dev, saved, a.heldout_scenes)
        arrs.update(ar)
    if "pair" in a.parts:
        X, lab, cont, grp = pair_acts(model, dev, (0.3, 1.0, 3.0), 1000, a.pair_layer)
        print("pair acts", tuple(X.shape), flush=True)
        out["pair"], ar = run("pair", X, lab, cont, grp, [int(v) for v in a.pair_sizes.split(",")], a, dev, saved)
        arrs.update(ar)
    json.dump(out, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out.replace(".json", ".npz"), **arrs)
    torch.save(saved, a.out.replace(".json", "_sae.pt"))
    print("DONE")


if __name__ == "__main__":
    main()
