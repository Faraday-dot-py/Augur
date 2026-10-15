"""Few-shot isolated adapter on a fully frozen Exp E (E_ms_kp_pot_v_g128). See docs/debugging/few-shot-adaptation.md (Part 3).

Adapter = MassGainAdapter: per-body learned gain on the frozen model's total acceleration, from [log m_i, log max_j m_j, log |a_i|, log local grid density];
output exp(2 tanh(mlp)) with a zero-initialised last layer, so ON at init is exactly 1 and OFF skips the hook (bitwise Exp E). Applied before momfix, so momentum
stays conserved. O(N), pp stays 2, no E weight is trainable.

  PYTHONPATH=. python scripts/fewshot_adapter.py run --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/fewshot_adapter [--smoke]
"""
import argparse
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import scatter_regress as rg
from scripts import scatter_scenes as ss
from scripts import scatter_solar as so

SEED = 4738
DT, EPS = 0.1, 0.5
SOLAR_K = 4.0
N_TRAIN = 16
N_HELD = 12
SOLAR_HELD_SEED = 9100
STAR_HELD_SEED = 9200
WIN = {"100": (81, 100), "300": (281, 300), "1000": (981, 1000)}
STAR_WIN = {"20": (11, 20), "100": (81, 100)}


class MassGainAdapter(nn.Module):
    def __init__(self, width=32, gated=False):
        super().__init__()
        self.on, self.gated = True, gated
        self.mlp = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, a, mass, mask, rho):
        mm = mass * mask
        mmax = mm.amax(1, keepdim=True)
        mmean = mm.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1.0)
        amag = torch.sqrt((a ** 2).sum(-1) + 1e-12)
        f = torch.stack([torch.log(mass + 1e-4) / 3, torch.log(mmax + 1e-4).expand_as(mass) / 3, torch.log(amag + 1e-2) / 2,
                         torch.log(rho[..., 0].clamp_min(0) + 1e-2) / 2], -1)
        out = 2 * torch.tanh(self.mlp(f)[..., 0])
        if self.gated:
            out = out * torch.tanh(torch.log(mmax / mmean.clamp_min(1e-9)))
        return a * torch.exp(out)[..., None]


def n_params(mod):
    return sum(p.numel() for p in mod.parameters())


class Rig:
    """Two copies of frozen E: m (eager; training and scene evals) and mh (default speed opts; harness rows, as scatter_regress). Same adapter object on both."""

    def __init__(self, ckpt, dev):
        self.dev = dev
        self.m = rg.build_model(SimpleNamespace(variant="ms_kp_pot_v_g128", exp="E", dt=DT, ckpt=ckpt, opt="none"), dev)
        self.mh = rg.build_model(SimpleNamespace(variant="ms_kp_pot_v_g128", exp="E", dt=DT, ckpt=ckpt, opt="default"), dev)
        for mod in (self.m, self.mh):
            for p in mod.parameters():
                p.requires_grad_(False)
        self.ref = {k: v.detach().clone().cpu() for k, v in self.m.state_dict().items()}
        self.adapter = None

    def set(self, adapter, on=True):
        if adapter is not None:
            adapter.on = on
        self.adapter = adapter
        for mod in (self.m, self.mh):
            mod.adapter = adapter
            mod._graphs.clear()
            mod._kf = None

    def e_unchanged(self):
        for mod in (self.m, self.mh):
            sd = {k: v for k, v in mod.state_dict().items() if not k.startswith("adapter.")}
            if set(sd) != set(self.ref) or not all(torch.equal(sd[k].cpu(), self.ref[k]) for k in sd):
                return False
        return True


def gen_solar(seeds, T, dev):
    scenes = [so.solar_scene(s, 200, SOLAR_K) for s in seeds]
    tr = [ss.nbody_truth(s["pos"], s["vel"], s["mass"], T, dev) for s in scenes]
    return scenes, np.stack([t[0] for t in tr]), np.stack([t[1] for t in tr])


def solar_weights(scenes):
    W = []
    for s in scenes:
        w = np.zeros(len(s["kind"]))
        for kd in (1, 2, 3):
            w[s["kind"] == kd] = 1.0 / (s["kind"] == kd).sum()
        W.append(w / w.sum())
    return np.stack(W)


def tensors_solar(scenes, P, V, dev):
    n = len(scenes[0]["mass"])
    M = torch.tensor(np.stack([s["mass"] for s in scenes]), dtype=torch.float32, device=dev)
    return dict(P=torch.tensor(P, dtype=torch.float32, device=dev), V=torch.tensor(V, dtype=torch.float32, device=dev), M=M, mask=torch.ones(len(scenes), n, device=dev),
                W=torch.tensor(solar_weights(scenes), dtype=torch.float32, device=dev))


def tensors_data(data, dev, star):
    P, V, M, mask = [t.float() for t in sf.to_tensors(data, dev)]
    W = mask.clone()
    if star:
        W[:, 0] = 0.0
    return dict(P=P, V=V, M=M, mask=mask, W=W / W.sum(1, keepdim=True))


def train_adapter(rig, data, its, batch, k0, k1, lr0, gated, log, tag, ckpt_path):
    dev = rig.dev
    torch.manual_seed(SEED)
    ad = MassGainAdapter(gated=gated).to(dev)
    rig.set(ad, True)
    opt = torch.optim.Adam(ad.parameters(), lr=lr0)
    P, V, M, mask, W = data["P"], data["V"], data["M"], data["mask"], data["W"]
    S, T = P.shape[0], P.shape[1]
    g = torch.Generator(device="cpu").manual_seed(SEED)
    skipped, t_start, hist = 0, time.time(), []
    for it in range(its):
        frac = it / max(1, its - 1)
        k = int(round(k0 + (k1 - k0) * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr0 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        idx = torch.randint(S, (batch,), generator=g).to(dev)
        t0 = torch.randint(0, T - k, (batch,), generator=g).to(dev)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
        Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
        m, mk, w = M[idx], mask[idx], W[idx][:, :, None]
        pos, vel = Pw[:, 0], Vw[:, 0]
        field = rig.m.init_field(batch, dev)
        loss = 0.0
        for s in range(k):
            pos, vel, m, field, _ = sf._step_m(rig.m, pos, vel, m, mk, field)
            loss = loss + ((((pos - Pw[:, s + 1]) ** 2) + 0.1 * ((vel - Vw[:, s + 1]) ** 2)) * w).sum() / (batch * k)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn) and loss.item() < 100:
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
        hist.append(float(loss))
        if it % 50 == 0 or it == its - 1:
            log(f"[{tag}] it {it} k {k} loss {np.mean(hist[-50:]):.5f} skipped {skipped} t {time.time() - t_start:.0f}s")
            torch.save({"adapter": ad.state_dict(), "it": it + 1, "gated": gated}, ckpt_path)
    ad.eval()
    return ad, {"its": its, "seconds": time.time() - t_start, "skipped": skipped, "loss_first50": float(np.mean(hist[:50])), "loss_last50": float(np.mean(hist[-50:]))}


def clip_err(e):
    return np.minimum(np.nan_to_num(e, nan=1e3, posinf=1e3), 1e3)


def eval_solar(model, scenes, Pt, Vt, steps, dev, keep=0):
    Pm, Vm, _ = ss.model_rollout(model, scenes, steps, dev)
    recs = []
    for i, sc in enumerate(scenes):
        e = clip_err(np.linalg.norm(Pm[i] - Pt[i], axis=-1))
        pl = np.where(sc["kind"] == 1)[0]
        r = {"finite": bool(np.isfinite(Pm[i]).all())}
        for k, (lo, hi) in WIN.items():
            if hi > steps:
                continue
            r["err_" + k] = float(e[lo:hi + 1].mean())
            r["planet_err_" + k] = float(e[lo:hi + 1][:, pl].mean())
        ticks = [t for t in (100, 300, 1000) if t <= steps]
        sm = so.solar_metrics(np.nan_to_num(Pm[i], nan=1e3), np.nan_to_num(Vm[i], nan=0.0), Pt[i], Vt[i], sc, ticks)
        for t in [str(x) for x in ticks]:
            r["planets_retained_" + t] = 1.0 - sm["planets"][t]["lost_frac"]
            r["belt_retained_" + t] = 1.0 - sm["belt"][t]["lost_frac"]
        tl = str(ticks[-1])
        r["planet_period_err_last"] = sm["planets"][tl]["period_err_median"]
        r["dE_model_last"] = sm["conservation"][tl]["dE_model"]
        recs.append(r)
    keep_arr = (Pm[:keep].astype(np.float16), Vm[:keep].astype(np.float16)) if keep else None
    return recs, keep_arr


def eval_star(model, data, dev):
    P, V, M, mask = sf.to_tensors(data, dev)
    Pm, Vm = sf.rollout(model, P[:, 0].float(), V[:, 0].float(), M.float(), mask.float(), P.shape[1] - 1)
    e = ((Pm.double() - P) ** 2).sum(-1).sqrt()
    e = e.nan_to_num(nan=1e3).clamp(max=1e3)
    e = (e * mask[:, None]).sum(2) / mask.sum(1)[:, None]
    recs = []
    for i in range(P.shape[0]):
        r = {"finite": bool(torch.isfinite(Pm[i]).all())}
        for k, (lo, hi) in STAR_WIN.items():
            r["err_" + k] = float(e[i, lo:hi + 1].mean())
        recs.append(r)
    return recs


def summarize(recs):
    out = {"n": len(recs)}
    for k in recs[0]:
        v = np.array([r[k] for r in recs], dtype=float)
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "per_scene": v.tolist()}
    return out


def harness(rig, dev, expb):
    args = SimpleNamespace(eps=EPS, dt=DT)
    cons = rg.conservation_rows(rig.mh, dev, args)
    eb, _ = rg.metrics_ab(rig.mh, expb, dev, EPS)
    return {"nr100_dE": cons["nr100"]["model"]["dE"], "nr100_dL": cons["nr100"]["model"]["dL"], "bh300_dE": cons["bh300"]["model"]["dE"],
            "bh300_dL": cons["bh300"]["model"]["dL"], "nr100_finite": cons["nr100"]["model"]["finite"], "expB_err20": eb["err"]["20"], "expB_err100": eb["err"]["100"],
            "expB_dE100": eb["dE_rel_model"]}


def bitwise_checks(rig, dev, log):
    torch.use_deterministic_algorithms(True, warn_only=True)
    scenes, P, V = gen_solar([SEED], 200, dev)
    out = {}
    rig.set(None)
    ref = ss.model_rollout(rig.m, scenes, 200, dev)
    again = ss.model_rollout(rig.m, scenes, 200, dev)
    out["E_repeat_equal"] = bool(np.array_equal(ref[0], again[0]) and np.array_equal(ref[1], again[1]))
    log(f"E repeat == E (deterministic mode, solar 200 ticks): {out['E_repeat_equal']}")
    ad = MassGainAdapter().to(dev)
    for name, adapter, on in (("off", ad, False), ("zero_init_on", ad, True), ("gated_zero_on", MassGainAdapter(gated=True).to(dev), True)):
        rig.set(adapter, on)
        cand = ss.model_rollout(rig.m, scenes, 200, dev)
        out[name] = bool(np.array_equal(ref[0], cand[0]) and np.array_equal(ref[1], cand[1]))
        log(f"bitwise {name} == E (solar 200 ticks, eager): {out[name]}")
    rig.set(None)
    torch.use_deterministic_algorithms(False)
    return out


ARMS = [("S1", "solar", 1, False), ("S4", "solar", 4, False), ("S16", "solar", 16, False), ("M1", "star", 1, False), ("M4", "star", 4, False), ("M16", "star", 16, False),
        ("C16", "ctrl", 16, False), ("S16g", "solar", 16, True), ("M16g", "star", 16, True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--its", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    log = lambda s: print(s, flush=True)
    res_path = f"{args.out}/results.json"
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    save = lambda: json.dump(res, open(res_path, "w"), indent=1)
    torch.manual_seed(SEED)
    smoke = args.smoke
    T_SOLAR = 100 if smoke else 1000
    T_TRAIN_SOLAR = 60 if smoke else 300
    n_tr, n_held = (2, 2) if smoke else (N_TRAIN, N_HELD)
    its = 6 if smoke else args.its
    rig = Rig(args.ckpt, dev)
    log(f"adapter params {n_params(MassGainAdapter())}")
    res["meta"] = {"ckpt": args.ckpt, "its": its, "batch": args.batch, "lr": args.lr, "k": [4, 20], "adapter_params": n_params(MassGainAdapter()), "seed": SEED, "smoke": smoke,
                   "solar_k": SOLAR_K, "n_train": n_tr, "n_held": n_held, "T_solar": T_SOLAR, "T_train_solar": T_TRAIN_SOLAR, "gpu": torch.cuda.get_device_name(dev)}
    res["bitwise"] = bitwise_checks(rig, dev, log)
    save()

    sol_tr = gen_solar([SEED + i for i in range(n_tr)], T_SOLAR, dev)
    sol_he = gen_solar([SOLAR_HELD_SEED + i for i in range(n_held)], T_SOLAR, dev)
    star_tr = sf.make_star_dataset(n_tr, 100 if not smoke else 40, SEED, dev, dt=DT, eps=EPS)
    star_he = sf.make_star_dataset(n_held, 150 if not smoke else 40, STAR_HELD_SEED, dev, dt=DT, eps=EPS)
    ctrl_tr = gs.make_dataset(n_tr, (100, 100), 100 if not smoke else 40, SEED, scale=True, device=dev, dt=DT, eps=EPS)
    expb = gs.make_dataset(48 if not smoke else 4, (10, 100), 100, 9200, scale=True, device=dev, dt=DT, eps=EPS)
    train_sets = {"solar": tensors_solar(sol_tr[0], sol_tr[1][:, :T_TRAIN_SOLAR + 1], sol_tr[2][:, :T_TRAIN_SOLAR + 1], dev), "star": tensors_data(star_tr, dev, True),
                  "ctrl": tensors_data(ctrl_tr, dev, False)}

    def evals(tag, own_fam, n):
        out = {}
        sl = slice(0, n)
        out["solar_held"] = summarize(eval_solar(rig.m, sol_he[0], sol_he[1], sol_he[2], T_SOLAR, dev)[0])
        out["star_held"] = summarize(eval_star(rig.m, star_he, dev))
        if own_fam == "solar" or tag == "free":
            nn_ = n if tag != "free" else n_tr
            out["solar_train"] = summarize(eval_solar(rig.m, sol_tr[0][:nn_], sol_tr[1][:nn_], sol_tr[2][:nn_], T_SOLAR, dev)[0])
        if own_fam == "star" or tag == "free":
            nn_ = n if tag != "free" else n_tr
            out["star_train"] = summarize(eval_star(rig.m, star_tr[:nn_], dev))
        return out

    def report(tag, ev):
        g = lambda d, k: f"{d[k]['mean']:.4g}" if k in d else "-"
        s, st = ev["solar_held"], ev["star_held"]
        log(f"[{tag}] solar held: err@100 {g(s, 'err_100')} err@300 {g(s, 'err_300')} err@1000 {g(s, 'err_1000')} planets kept@300 {g(s, 'planets_retained_300')} "
            f"@1000 {g(s, 'planets_retained_1000')} | star held err@20 {g(st, 'err_20')} err@100 {g(st, 'err_100')}")

    if "free" not in res:
        rig.set(None)
        res["free"] = {"eval": evals("free", None, n_tr), "harness": []}
        report("free E", res["free"]["eval"])
        save()
    while len(res["free"]["harness"]) < (1 if smoke else 3):
        rig.set(None)
        res["free"]["harness"].append(harness(rig, dev, expb))
        log(f"[free E] harness {json.dumps(res['free']['harness'][-1])}")
        save()

    want = set(args.arms.split(",")) if args.arms else None
    keep = {}
    res.setdefault("arms", {})
    for tag, fam, n, gated in ARMS:
        if (want and tag not in want) or tag in res["arms"]:
            continue
        sub = {k: v[:n] for k, v in train_sets[fam].items()}
        ad, tr = train_adapter(rig, sub, its, args.batch, 4, 20, args.lr, gated, log, tag, f"{args.out}/adapter_{tag}.pt")
        rig.set(ad, True)
        ev = evals(tag, fam, n)
        report(tag, ev)
        if tag in ("S16", "S4", "C16"):
            recs, kp = eval_solar(rig.m, sol_he[0][:3], sol_he[1][:3], sol_he[2][:3], T_SOLAR, dev, keep=3)
            keep[tag] = kp
        hs = [harness(rig, dev, expb)]
        log(f"[{tag}] harness ON {json.dumps(hs[0])}")
        res["arms"][tag] = {"family": fam, "n": n, "gated": gated, "train": tr, "eval": ev, "harness_on": hs, "e_unchanged": rig.e_unchanged()}
        save()
    for tag in ("S16", "M16"):
        if tag in res["arms"] and len(res["arms"][tag]["harness_on"]) < (1 if smoke else 3):
            st = torch.load(f"{args.out}/adapter_{tag}.pt", map_location=dev)
            ad = MassGainAdapter(gated=st["gated"]).to(dev)
            ad.load_state_dict(st["adapter"])
            rig.set(ad, True)
            while len(res["arms"][tag]["harness_on"]) < (1 if smoke else 3):
                res["arms"][tag]["harness_on"].append(harness(rig, dev, expb))
                log(f"[{tag}] harness ON {json.dumps(res['arms'][tag]['harness_on'][-1])}")
                save()
    if any(a in res["arms"] for a in ("S16", "M16")):
        ad = MassGainAdapter().to(dev)
        rig.set(ad, False)
        res["off_harness"] = res.get("off_harness", [])
        while len(res["off_harness"]) < (1 if smoke else 3):
            res["off_harness"].append(harness(rig, dev, expb))
            log(f"[adapter attached, OFF] harness {json.dumps(res['off_harness'][-1])}")
            save()
    rig.set(None)
    res["e_weights_unchanged_at_end"] = rig.e_unchanged()
    if keep:
        rig.set(None)
        recs, kp = eval_solar(rig.m, sol_he[0][:3], sol_he[1][:3], sol_he[2][:3], T_SOLAR, dev, keep=3)
        keep["free"] = kp
        np.savez_compressed(f"{args.out}/solar_held_rollouts.npz", truth=sol_he[1][:3].astype(np.float16), mass=np.stack([s["mass"] for s in sol_he[0][:3]]), kind=sol_he[0][0]["kind"],
                            **{f"{k}_P": v[0] for k, v in keep.items()})
    save()
    log("done")


if __name__ == "__main__":
    main()
