"""Sparse autoencoder (SAE) curiosity experiment on the mixed multi-law scatter-field model (no pass/kill; see supervisor-log 2026-11-03).
Mesh net: 32-channel output of UNet.dec[0] (last decoder block, full resolution) at cells holding mass. Pair MLP: 128-unit second hidden layer of
ppmlp.mlp (CondMLP), a pure function of (r, law). Report per dictionary size: normalized reconstruction error, L0, dead fraction, feature alignment with
law identity (ANOVA eta^2) and with distance / density (|Pearson|).
Usage: PYTHONPATH=. python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --out results/sae_interp_s4738.json
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


@torch.no_grad()
def mesh_acts(model, dev, scenes, cells, warm, seed):
    rng = np.random.default_rng(seed)
    store = {}
    h = model.net.dec[0].register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
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
            assert a.shape[-2:] == rho.shape, (a.shape, rho.shape)
            ij = (rho > 0.05).nonzero()
            ij = ij[torch.randperm(len(ij), generator=torch.Generator().manual_seed(seed + k))[:cells].to(dev)]
            G = rho.shape[0]
            c = (ij.float() + 0.5 - G / 2) * model.h
            X.append(a[:, ij[:, 0], ij[:, 1]].T.cpu())
            lab.append(torch.full((len(ij),), li))
            dens.append(torch.log(rho[ij[:, 0], ij[:, 1]]).cpu())
            rad.append(c.norm(dim=-1).cpu())
            sid.append(torch.full((len(ij),), k))
            k += 1
    h.remove()
    return torch.cat(X), torch.cat(lab), {"log_cell_mass": torch.cat(dens), "radius_from_centre": torch.cat(rad)}, torch.cat(sid)


@torch.no_grad()
def pair_acts(model, dev, amps, nr):
    X, lab, logr, amp_id = [], [], [], []
    r = torch.exp(torch.linspace(np.log(0.05), np.log(4.0), nr, device=dev))
    f = torch.stack([r, torch.log(r + 0.05)], -1)
    mlp = model.ppmlp.mlp
    for li, law in enumerate(LAWS):
        for ai, amp in enumerate(amps):
            cond = torch.tensor(sm.cond_of(sm.slots_for("single", law, amp)), dtype=torch.float32, device=dev)
            c = cond.expand(nr, -1)
            h1 = F.gelu(mlp.l1(torch.cat([f, c], -1)))
            X.append(F.gelu(mlp.l2(h1)).cpu())
            lab.append(torch.full((nr,), li))
            logr.append(torch.log(r).cpu())
            amp_id.append(torch.full((nr,), ai))
    return torch.cat(X), torch.cat(lab), {"log_r": torch.cat(logr)}, torch.cat(amp_id)


def train_sae(X, d_dict, target_l0, steps, seed, dev, bs=4096):
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    d = X.shape[1]
    W = torch.randn(d_dict, d, device=dev, generator=g)
    W = (W / W.norm(dim=1, keepdim=True)).requires_grad_()
    We = (W.detach().clone().T.contiguous()).requires_grad_()
    be = torch.zeros(d_dict, device=dev, requires_grad=True)
    bd = torch.zeros(d, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, We, be, bd], lr=1e-3)
    lam = 1e-3
    fired = torch.zeros(d_dict, dtype=torch.bool, device=dev)
    l0s = []
    for it in range(steps):
        x = X[torch.randint(0, len(X), (bs,), device=dev, generator=g)]
        Wn = W / W.norm(dim=1, keepdim=True)
        z = F.relu((x - bd) @ We + be)
        loss = F.mse_loss(z @ Wn + bd, x) + lam * z.sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        l0 = (z > 0).float().sum(-1).mean().item()
        l0s.append(l0)
        if it >= steps // 2:
            fired |= (z > 0).any(0)
        if it % 20 == 19:
            m = float(np.mean(l0s[-20:]))
            lam *= 1.02 if m > target_l0 else 1 / 1.02
        for gp in opt.param_groups:
            gp["lr"] = 1e-3 * (1 - it / steps) + 1e-5
    with torch.no_grad():
        W = W / W.norm(dim=1, keepdim=True)
    return W.detach(), We.detach(), be.detach(), bd.detach(), fired


def encode(sae, x):
    W, We, be, bd = sae[:4]
    return F.relu((x - bd) @ We + be)


def eta2(Z, lab, k):
    tot = Z.var(0, unbiased=False) * len(Z) + 1e-12
    mu = Z.mean(0)
    bet = sum((lab == j).sum() * (Z[lab == j].mean(0) - mu) ** 2 for j in range(k) if (lab == j).any())
    return (bet / tot).clamp(0, 1)


def abscorr(Z, y):
    Zc, yc = Z - Z.mean(0), (y - y.mean())[:, None]
    return ((Zc * yc).sum(0) / (Zc.norm(dim=0) * yc.norm() + 1e-12)).abs()


def evaluate(sae, Xtr, Xte, lab_te, cont_te, fired, norm):
    Zte = encode(sae, Xte)
    rec = Zte @ sae[0] + sae[3]
    out = {"recon_nmse_heldout": float(((rec - Xte) ** 2).sum() / ((Xte - Xte.mean(0)) ** 2).sum()), "l0_heldout": float((Zte > 0).float().sum(-1).mean()),
           "dead_fraction": float(1 - fired.float().mean())}
    live = (Zte > 0).any(0)
    e_law = eta2(Zte, lab_te, len(LAWS))
    out["n_live_heldout"] = int(live.sum())
    out["law_eta2_live_mean"] = float(e_law[live].mean()) if live.any() else None
    out["law_eta2_live_max"] = float(e_law[live].max()) if live.any() else None
    out["n_live_law_eta2_gt_0.5"] = int((e_law[live] > 0.5).sum())
    for name, y in cont_te.items():
        c = abscorr(Zte, y)
        out[f"{name}_abscorr_live_mean"] = float(c[live].mean()) if live.any() else None
        out[f"{name}_abscorr_live_max"] = float(c[live].max()) if live.any() else None
        out[f"n_live_{name}_abscorr_gt_0.5"] = int((c[live] > 0.5).sum())
    top = torch.argsort(e_law * live, descending=True)[:5]
    out["top_law_features"] = []
    for j in top.tolist():
        per_law = [float(Zte[lab_te == li, j].mean()) for li in range(len(LAWS))]
        out["top_law_features"].append({"feature": j, "eta2": float(e_law[j]), "mean_act_per_law": dict(zip(LAWS, per_law))})
    return out, e_law.cpu().numpy(), {k: abscorr(Zte, y).cpu().numpy() for k, y in cont_te.items()}


def run(name, X, lab, cont, grp, sizes, a, dev):
    te = grp == grp.max()
    mu, sc = X[~te].mean(0), X[~te].norm(dim=-1).mean() / np.sqrt(X.shape[1])
    Xn = ((X - mu) / sc).to(dev)
    tr = (~te).to(dev)
    res, arrs = {"n_train": int(tr.sum()), "n_heldout": int(te.sum()), "heldout_group": int(grp.max())}, {}
    for d_dict in sizes:
        sae = train_sae(Xn[tr], d_dict, a.target_l0, a.steps, a.seed, dev)
        r, e_law, ac = evaluate(sae, Xn[tr], Xn[~tr], lab[te.cpu()].to(dev), {k: v[te].to(dev) for k, v in cont.items()}, sae[4], sc)
        res[f"dict_{d_dict}"] = r
        arrs[f"{name}_{d_dict}_law_eta2"] = e_law
        for k, v in ac.items():
            arrs[f"{name}_{d_dict}_{k}_abscorr"] = v
        arrs[f"{name}_{d_dict}_decoder"] = sae[0].cpu().numpy()
        print(name, d_dict, {k: v for k, v in r.items() if k != "top_law_features"}, flush=True)
    return res, arrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--target-l0", type=float, default=8.0)
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--cells", type=int, default=400)
    ap.add_argument("--warm", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = sm.build().to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    out = {"ckpt": a.ckpt, "seed": a.seed, "target_l0": a.target_l0, "steps": a.steps, "laws": LAWS}
    arrs = {}
    X, lab, cont, sid = mesh_acts(model, dev, a.scenes, a.cells, a.warm, a.seed)
    print("mesh acts", tuple(X.shape), flush=True)
    grp = sid % a.scenes
    out["mesh"], ar = run("mesh", X, lab, cont, grp, MESH_SIZES, a, dev)
    arrs.update(ar)
    X, lab, cont, aid = pair_acts(model, dev, (0.3, 1.0, 3.0), 1000)
    print("pair acts", tuple(X.shape), flush=True)
    out["pair"], ar = run("pair", X, lab, cont, aid, PAIR_SIZES, a, dev)
    arrs.update(ar)
    json.dump(out, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out.replace(".json", ".npz"), **arrs)
    print("DONE")


if __name__ == "__main__":
    main()
