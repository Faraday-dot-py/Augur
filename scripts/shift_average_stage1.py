"""Stage 1 of staggered-double-grid-with-quadratic-deposit: inference-time shift averaging of frozen Exp E (E_ms_kp_pot_v_g128).
1 pass (control), 2 passes (grid offset by half a cell on the diagonal), 4 passes (offsets 0, x, y, diagonal by half a cell), each pass with its own recurrent field.
Per seed (scene seeds 9100/9200/9300): expB 8 scenes + cluster 4 scenes; free rollout 100 ticks vs fp64 truth; err@k = mean body position error, dE@k = |E_model - E_truth| / |E_truth| at the same tick.
Shift probe: median over bodies of |a_shifted - a| / |a| (bodies above the scene's 10th |a| percentile), tick-0 force from a zero field, rigid shift of the scene.
  half-cell x, y, diagonal ("probe_half": the offsets of the 4-pass average are closed under these shifts, so 4-pass is ~0 by construction); quarter-cell diagonal and (0.37, 0.11) cells ("probe_off": not in the group).
Usage: PYTHONPATH=. python scripts/shift_average_stage1.py --ckpt E_ms_kp_pot_v_g128.pt --out results/shift_average_stage1/stage1.json [--smoke]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_error_growth as seg
from scripts import scatter_scenes as ss

SEEDS = (9100, 9200, 9300)
FAMS = {"expB": 8, "cluster": 4}
TR = 100
KS = (5, 20, 100)
H = 64.0 / 128
PASSES = {1: [(0.0, 0.0)], 2: [(0.0, 0.0), (0.5, 0.5)], 4: [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]}
PROBE_HALF = ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
PROBE_OFF = ((0.25, 0.25), (0.37, 0.11))


class ShiftAvg:
    def __init__(self, model, offsets, dev):
        self.model, self.dev = model, dev
        self.off = [torch.tensor([ox * H, oy * H], dtype=torch.float64, device=dev) for ox, oy in offsets]

    def init(self):
        return [self.model.init_field(1, self.dev)[0] for _ in self.off]

    def force(self, sc, p, v, fields):
        m = sc["m"][None].float()
        ones = torch.ones(1, len(p), device=self.dev)
        outs, nf = [], []
        for d, f in zip(self.off, fields):
            a, f2 = self.model.force((p + d)[None].float(), v[None].float(), m, ones, f)
            outs.append(a[0].double())
            nf.append(f2)
        return torch.stack(outs).mean(0), nf


def energy(p, v, m):
    d = p[None] - p[:, None]
    r = (d.pow(2).sum(-1) + seg.EPS ** 2).sqrt()
    mm = m[None] * m[:, None]
    pe = -(torch.triu(mm / r, diagonal=1)).sum()
    return 0.5 * (m * v.pow(2).sum(-1)).sum() + pe


def curves(sc, Pm, Vm):
    e = (Pm - sc["P"][: TR + 1]).norm(dim=-1).mean(1)
    Et = torch.stack([energy(sc["P"][t], sc["V"][t], sc["m"]) for t in range(TR + 1)])
    Em = torch.stack([energy(Pm[t], Vm[t], sc["m"]) for t in range(TR + 1)])
    de = (Em - Et).abs() / Et.abs()
    return e.cpu().numpy(), de.cpu().numpy()


def probe(model, sc, shifts, dev):
    m = sc["m"][None].float()
    ones = torch.ones(1, len(sc["m"]), device=dev)

    def force(offs, pos):
        acc = 0.0
        for ox, oy in offs:
            d = torch.tensor([ox * H, oy * H], dtype=torch.float64, device=dev)
            f = model.init_field(1, dev)[0]
            a, _ = model.force((pos + d)[None].float(), torch.zeros_like(pos)[None].float(), m, ones, f)
            acc = acc + a[0].double()
        return acc / len(offs)

    out = {}
    for k, offs in PASSES.items():
        a0 = force(offs, sc["P"][0])
        mag = a0.norm(dim=-1)
        keep = mag > torch.quantile(mag, 0.1)
        vals = []
        for sx, sy in shifts:
            s = torch.tensor([sx * H, sy * H], dtype=torch.float64, device=dev)
            a1 = force(offs, sc["P"][0] + s)
            vals.append(float(((a1 - a0).norm(dim=-1) / mag)[keep].median()))
        out[k] = float(np.mean(vals))
    return out


def agg(rows):
    return {str(k): {n: float(np.mean([r[k][n] for r in rows])) for n in rows[0][k]} for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global TR, KS
    dev = torch.device("cpu" if args.smoke else "cuda")
    if args.smoke:
        TR, KS, seg.T = 10, (5, 10, 10), 10
        torch.set_num_threads(2)
    seg.DEV = dev
    torch.manual_seed(4738)
    model = ss.build_model("E", seg.DT, dev, args.ckpt)
    out = {"ckpt": args.ckpt, "passes": {str(k): v for k, v in PASSES.items()}, "seeds": {}}
    t0 = time.time()
    with torch.no_grad():
        for seed in SEEDS[:1] if args.smoke else SEEDS:
            per = {k: [] for k in PASSES}
            probes = {"half": {k: [] for k in PASSES}, "off": {k: [] for k in PASSES}}
            cost = {k: [] for k in PASSES}
            for fam, n in FAMS.items():
                scenes = seg.make_scenes(fam, 1 if args.smoke else n, seed)
                for sc in scenes:
                    for k, offs in PASSES.items():
                        net = ShiftAvg(model, offs, dev)
                        ta = time.time()
                        Pm, Vm = seg.roll(net, sc, steps=TR)
                        if dev.type == "cuda":
                            torch.cuda.synchronize()
                        cost[k].append((time.time() - ta) / TR)
                        e, de = curves(sc, Pm, Vm)
                        per[k].append({**{f"err@{t}": float(e[t]) for t in KS}, **{f"dE@{t}": float(de[t]) for t in KS}, "finite": float(np.isfinite(e).all() and np.isfinite(de).all())})
                    for nm, sh in (("half", PROBE_HALF), ("off", PROBE_OFF)):
                        for k, v in probe(model, sc, sh, dev).items():
                            probes[nm][k].append(v)
                print(f"seed {seed} {fam} {len(scenes)} scenes {time.time() - t0:.0f}s", flush=True)
            res = {str(k): {n: float(np.mean([r[n] for r in per[k]])) for n in per[k][0]} for k in per}
            res["probe_half"] = {str(k): float(np.mean(v)) for k, v in probes["half"].items()}
            res["probe_off"] = {str(k): float(np.mean(v)) for k, v in probes["off"].items()}
            res["sec_per_tick"] = {str(k): float(np.mean(v)) for k, v in cost.items()}
            res["n_scenes"] = len(per[1])
            out["seeds"][str(seed)] = res
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(out, open(args.out, "w"), indent=1)
    for k in ("2", "4"):
        for m_ in [f"err@{t}" for t in KS] + [f"dE@{KS[-1]}"]:
            r = [out["seeds"][s][k][m_] / out["seeds"][s]["1"][m_] for s in out["seeds"]]
            print(f"{k}-pass {m_} / 1-pass: " + " ".join(f"{x:.3f}" for x in r) + f" pooled {np.mean([out['seeds'][s][k][m_] for s in out['seeds']]) / np.mean([out['seeds'][s]['1'][m_] for s in out['seeds']]):.3f}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
