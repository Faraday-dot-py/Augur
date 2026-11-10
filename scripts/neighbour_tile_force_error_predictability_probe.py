"""Stage-1 probe for summary-token exchange between neighbour tiles: how much of the frozen replay checkpoint's per-body force error
e = a_model - a_true is predictable from the previous tick's state of the 8 neighbouring tiles, beyond what the body's own state and own tile predict.
Inference only; the model is never trained. A fixed small MLP is fitted on rollouts of fresh scenes (TRAIN_SEEDS) and scored on held-out scenes (EVAL_SEEDS).
f = 1 - sum|e - e_hat|^2 / sum|e|^2 (fraction of force-error energy explained). Delta = f(own + neighbour) - f(own); Delta_net subtracts the
shuffled-neighbour control's gain."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from scripts import exp12_hier as h
from scripts import replay_mechanism_force_error_breakdown as rb
from scripts import tier0_stage1_probe_common as pc
from scripts import tier0_tiled_closed_loop as tc
from scripts import train_tiered as ttr
from scripts.eval_tier0_tiled_closed_loop import load

TRAIN_SEEDS = (7100, 7200, 7300)
EVAL_SEEDS = pc.SEEDS
TICKS = tuple(range(20, 101, 5))
LINE = {"pass_pooled": 0.19, "pass_each_seed": 0.10, "kill_pooled": 0.06, "oracle_min": 0.90}
MLP = {"width": 64, "layers": 2, "lr": 1e-3, "epochs": 30, "wd": 1e-4, "batch": 1024}
OFFS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def neighbour_index():
    idx = torch.full((tc.NW, 8), -1, dtype=torch.long)
    for i in range(tc.NT):
        for j in range(tc.NT):
            for n, (di, dj) in enumerate(OFFS):
                if 0 <= i + di < tc.NT and 0 <= j + dj < tc.NT:
                    idx[i * tc.NT + j, n] = (i + di) * tc.NT + j + dj
    return idx


def tile_summary(pos, vel, a, m, mk, fld, cen, n):
    """(B, NW, 17): 8 raw body numbers (count, mass, centre-of-mass offset, mean velocity, velocity dispersion, position spread) and 9 learned-state
    numbers (mean acceleration, mean |acceleration|, recurrent-field core pooled to 2x2, field mean |.| and std)."""
    B, N = mk.shape
    own = h.owner(pos)
    oh = F.one_hot(own, tc.NW).float() * mk[..., None]
    cnt = oh.sum(1)
    cn = cnt.clamp_min(1)
    ms = torch.einsum("bnk,bn->bk", oh, m)
    com = torch.einsum("bnk,bn,bnd->bkd", oh, m, pos) / ms.clamp_min(1e-9)[..., None]
    com = torch.where(cnt[..., None] > 0, com, cen[None])
    vbar = torch.einsum("bnk,bnd->bkd", oh, vel) / cn[..., None]
    abar = torch.einsum("bnk,bnd->bkd", oh, a) / cn[..., None]
    g2 = lambda t: torch.gather(t, 1, own[..., None].expand(-1, -1, 2))
    disp = (torch.einsum("bnk,bn->bk", oh, ((vel - g2(vbar)) ** 2).sum(-1)) / cn).sqrt()
    spread = (torch.einsum("bnk,bn->bk", oh, ((pos - g2(com)) ** 2).sum(-1)) / cn).sqrt()
    amag = torch.einsum("bnk,bn->bk", oh, a.norm(dim=-1)) / cn
    G = fld.shape[-1]
    lo = (G - n) // 2
    fc = fld.reshape(B, tc.NW, *fld.shape[1:])[..., lo:lo + n, lo:lo + n]
    pooled = fc.mean(2).reshape(B, tc.NW, 2, n // 2, 2, n // 2).mean((3, 5)).reshape(B, tc.NW, 4)
    raw = torch.cat([cnt[..., None], ms[..., None], com - cen[None], vbar, disp[..., None], spread[..., None]], -1)
    learned = torch.cat([abar, amag[..., None], pooled, fc.abs().mean((2, 3, 4))[..., None], fc.flatten(2).std(-1)[..., None]], -1)
    return torch.cat([raw, learned], -1)


class Collector:
    def __init__(self, tt, d, nidx):
        self.tt, self.m, self.mk, self.nidx = tt, d["M"].float(), d["mask"].float(), nidx.to(tt.dev)
        self.prev = None
        self.rows = {k: [] for k in ("own", "n1", "n2", "e", "a_true", "scene", "tick")}

    def __call__(self, t, pos, vin, a_model, a_used, fld, fld_out):
        cen = self.tt.cen
        cur = tile_summary(pos, vin, a_model, self.m, self.mk, fld, cen, self.tt.n)
        if self.prev is not None and t in TICKS:
            self.emit(t, pos, vin, a_model, self.prev, cen)
        self.prev = cur

    def emit(self, t, pos, vin, a_model, prev, cen):
        B, N = self.mk.shape
        at = ttr.true_acc(pos, self.m, self.mk)
        own = h.owner(pos)
        pad = torch.cat([prev, torch.zeros_like(prev[:, :1])], 1)
        nb = pad[:, torch.where(self.nidx < 0, tc.NW, self.nidx)]
        valid = (self.nidx >= 0).float()[None].expand(B, -1, -1)[..., None]
        nb_body = torch.gather(nb.reshape(B, tc.NW, -1), 1, own[..., None].expand(-1, -1, 8 * 17)).reshape(B, N, 8, 17)
        v_body = torch.gather(valid.reshape(B, tc.NW, 8), 1, own[..., None].expand(-1, -1, 8))[..., None]
        own_sum = torch.gather(prev, 1, own[..., None].expand(-1, -1, 17))
        rel = pos - cen[own]
        feat = torch.cat([rel, vin, a_model, a_model.norm(dim=-1, keepdim=True), rb.density(pos, self.mk)[..., None].float(), self.m[..., None],
                          rb.edge_dist(pos)[..., None], torch.full_like(self.m, t / 100.0)[..., None], own_sum], -1)
        n1 = torch.cat([nb_body[..., :8], v_body], -1).reshape(B, N, -1)
        n2 = nb_body[..., 8:].reshape(B, N, -1)
        live = self.mk > 0
        sc = torch.arange(B, device=pos.device)[:, None].expand(B, N)
        for k, v in (("own", feat), ("n1", n1), ("n2", n2), ("e", a_model - at), ("a_true", at), ("scene", sc), ("tick", torch.full_like(sc, t))):
            self.rows[k].append(v[live].cpu())


def collect(tt, seed, scenes, steps, dev, nidx):
    d = ttr.make_scenes(scenes, steps, seed, dev)
    c = Collector(tt, d, nidx)
    pc.rollout(tt, d, observe=c)
    out = {k: torch.cat(v).float() if k not in ("scene", "tick") else torch.cat(v) for k, v in c.rows.items()}
    out["seed"] = torch.full_like(out["scene"], seed)
    return out


def cat(parts):
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def fit_predict(Xtr, ytr, Xte, dev, seed=4738):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev)
    ytr = ytr.to(dev)
    layers, w = [], Xtr.shape[1]
    for _ in range(MLP["layers"]):
        layers += [torch.nn.Linear(w, MLP["width"]), torch.nn.GELU()]
        w = MLP["width"]
    net = torch.nn.Sequential(*layers, torch.nn.Linear(w, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=MLP["lr"], weight_decay=MLP["wd"])
    g = torch.Generator().manual_seed(seed)
    for _ in range(MLP["epochs"]):
        perm = torch.randperm(Xtr.shape[0], generator=g).to(dev)
        for i in range(0, len(perm), MLP["batch"]):
            b = perm[i:i + MLP["batch"]]
            loss = ((net(Xtr[b]) - ytr[b]) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(Xte).cpu()


def explained(pred, e, sel):
    return float(1 - ((e[sel] - pred[sel]) ** 2).sum() / (e[sel] ** 2).sum())


def evaluate(tr, te, dev):
    gen = torch.Generator().manual_seed(4738)
    def shuf(x, r):
        idx = torch.arange(x.shape[0])
        for t in r["tick"].unique():
            rows = (r["tick"] == t).nonzero().squeeze(1)
            idx[rows] = rows[torch.randperm(len(rows), generator=gen)]
        return x[idx]

    nb = lambda r: torch.cat([r["n1"], r["n2"]], 1)
    variants = {
        "own": (tr["own"], te["own"]),
        "own+n1": (torch.cat([tr["own"], tr["n1"]], 1), torch.cat([te["own"], te["n1"]], 1)),
        "own+n2": (torch.cat([tr["own"], tr["n2"]], 1), torch.cat([te["own"], te["n2"]], 1)),
        "own+n1+n2": (torch.cat([tr["own"], nb(tr)], 1), torch.cat([te["own"], nb(te)], 1)),
        "own+shuffled-neighbour": (torch.cat([tr["own"], shuf(nb(tr), tr)], 1), torch.cat([te["own"], shuf(nb(te), te)], 1)),
        "own+oracle-true-acc": (torch.cat([tr["own"], tr["a_true"]], 1), torch.cat([te["own"], te["a_true"]], 1)),
    }
    f = {}
    for name, (xa, xb) in variants.items():
        pred = fit_predict(xa, tr["e"], xb, dev)
        f[name] = {"pooled": explained(pred, te["e"], torch.ones_like(te["seed"], dtype=torch.bool)),
                   "per_seed": {str(s): explained(pred, te["e"], te["seed"] == s) for s in EVAL_SEEDS},
                   "late_ticks_60_100": explained(pred, te["e"], te["tick"] >= 60)}
        print(name, json.dumps(f[name]), flush=True)
    return f


def summarise(f):
    gain = lambda a, b, k: f[a][k] - f[b][k]
    s = {"f": f}
    s["delta"] = {"pooled": gain("own+n1+n2", "own", "pooled"), "per_seed": {k: f["own+n1+n2"]["per_seed"][k] - f["own"]["per_seed"][k] for k in f["own"]["per_seed"]}}
    s["delta_shuffled"] = {"pooled": gain("own+shuffled-neighbour", "own", "pooled"), "per_seed": {k: f["own+shuffled-neighbour"]["per_seed"][k] - f["own"]["per_seed"][k] for k in f["own"]["per_seed"]}}
    s["delta_net"] = {"pooled": s["delta"]["pooled"] - max(s["delta_shuffled"]["pooled"], 0),
                      "per_seed": {k: s["delta"]["per_seed"][k] - max(s["delta_shuffled"]["per_seed"][k], 0) for k in s["delta"]["per_seed"]}}
    s["delta_n1_only"], s["delta_n2_only"] = gain("own+n1", "own", "pooled"), gain("own+n2", "own", "pooled")
    s["oracle_f"] = f["own+oracle-true-acc"]["pooled"]
    return s


def verdict(s):
    d = s["delta_net"]
    if d["pooled"] >= LINE["pass_pooled"] and min(d["per_seed"].values()) >= LINE["pass_each_seed"]:
        return "PASS"
    if d["pooled"] < LINE["kill_pooled"]:
        return "KILL" if s["oracle_f"] >= LINE["oracle_min"] else "INCONCLUSIVE"
    return "INCONCLUSIVE"


def main():
    global EVAL_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path; the first is the verdict checkpoint, the rest are reported replicates")
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(pc.SEEDS), help="held-out scene seeds; 12000 12100 12200 is the fresh set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pc.SEEDS = tuple(a.seeds)
    assert not set(a.seeds) & set(TRAIN_SEEDS), "eval seeds overlap the MLP training seeds"
    EVAL_SEEDS = pc.SEEDS
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, paths = zip(*(c.split("=", 1) for c in a.ckpt))
    for p in paths:
        assert os.path.exists(p), f"MISSING {p}"
    nidx = neighbour_index()
    out = {"args": vars(a), "line": LINE, "mlp": MLP, "train_seeds": TRAIN_SEEDS, "eval_seeds": EVAL_SEEDS, "ckpt": {}}
    for n, p in zip(names, paths):
        tt = tc.TiledTier0(load(p, dev), dev, chunk=a.chunk, exchange="halo", ckpt=False)
        tr = cat([collect(tt, s, a.scenes, a.steps, dev, nidx) for s in TRAIN_SEEDS])
        te = cat([collect(tt, s, a.scenes, a.steps, dev, nidx) for s in EVAL_SEEDS])
        print(n, "rows train", tr["e"].shape[0], "eval", te["e"].shape[0], "own/n1/n2 dims", tr["own"].shape[1], tr["n1"].shape[1], tr["n2"].shape[1], flush=True)
        s = summarise(evaluate(tr, te, dev))
        s["rows"] = {"train": int(tr["e"].shape[0]), "eval": int(te["e"].shape[0])}
        out["ckpt"][n] = {"summary": s, "verdict_rule_applied": verdict(s)}
        print(n, "delta_net", json.dumps(s["delta_net"]), "oracle", s["oracle_f"], "verdict-rule", out["ckpt"][n]["verdict_rule_applied"], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
