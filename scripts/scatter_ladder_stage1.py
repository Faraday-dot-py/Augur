"""Stage 1 of gaussian-ladder-far-kernel-replacing-fft (docs/debugging/architecture-push-prereg/gaussian-ladder-far-kernel-replacing-fft.md).

Cells, each written to --out as it finishes (a rerun skips finished cells; SIGTERM exits 143 after the current cell):
  fit:<model>:<law>     ladder fitted to the frozen learned kernel; fit error, amplitudes, exponent read from the amplitudes
  tiled                 kernel-acceleration difference, tile-by-tile (tile 64 cells, halo 32 cells, no global term) against the single grid
  rollout:<seeds>       scatter-gravity-clustered-baseline with its kernel replaced by the ladder (U-Net frozen at E's weights): err@5/20 against scatter-gravity-clustered-baseline on the same scenes
  timing                kernel acceleration time, FFT path against ladder, at grid 128
Usage: PYTHONPATH=. python scripts/scatter_ladder_stage1.py --e-ckpt kern/E_frozen.pt --main-ckpt ckpt/main_s4739.pt --out results/x.json
"""
import argparse
import json
import os
import signal
import sys
import time

import torch

from scripts import scatter_field as sf
from scripts import scatter_ladder as sl
from scripts import scatter_multi as sm
from scripts import scatter_multi_eval as sme

EXPECTED_FORCE_EXPONENT = {"pow1": 1.0, "pow2": 2.0, "pow3": 3.0}
STOP = {"flag": False}


def on_term(*_):
    STOP["flag"] = True


def build_e(dev):
    return sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=sm.DT, momfix=True, in_scale=1.0, potential=True,
                           kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()


def build_ladder(dev, sd, amps):
    m = sl.LadderScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=sm.DT, momfix=True, in_scale=1.0, potential=True,
                              kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    m.load_state_dict(sd)
    m.set_amps(amps)
    return m


@torch.no_grad()
def multi_kernel(m, shape):
    G = m.grid
    idx = torch.arange(2 * G, dtype=torch.float64)
    d = torch.minimum(idx, 2 * G - idx) * m.h
    r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
    cond = torch.tensor(sm.cond_of(sm.slots_for("single", shape, 1.0)), dtype=torch.float32)
    f = torch.stack([r.float(), torch.log(r.float() + 0.5)], -1)
    K = m.kmlp(f, cond[None, None, :])[..., 0].double()
    return r, K * (1 - (1 - (r / m.pp).clamp(max=1.0) ** 2) ** 2)


def fit_cell(r, K, expected):
    sig = sl.sigmas()
    amps = sl.fit_amps(r, K, sig)
    out = {"amps": [float(a) for a in amps], "sigmas": sig, "fit_rel_err": sl.fit_error(r, K, amps, sig),
           "exponent_read": sl.read_exponent(amps, sig)}
    if expected is not None:
        out["expected_force_exponent"] = expected
        out["exponent_rel_err"] = abs(out["exponent_read"] - expected) / expected
    return out


@torch.no_grad()
def tiled_cell(e, amps, dev, n):
    d = sm.gen_cases([("pow2", "single", 1.0)] * n, 9100, dev, 1, tag=2)
    pos, vel, mask = d["P"][:, 0], d["V"][:, 0], d["mask"]
    rho = e.scatter(pos, vel, torch.ones_like(mask), mask)[:, :1].double()
    one = e.neg_grad(sl.ladder_potential(rho, amps, e.h))
    tiled = e.neg_grad(sl.ladder_potential_tiled(rho, amps, e.h, 64, 32))
    return {"rel_l1_diff": float((one - tiled).abs().sum() / one.abs().sum()), "tile_cells": 64, "halo_cells": 32, "scenes": int(pos.shape[0])}


def rollout_cell(e, lad, dev, seeds, n):
    out = {}
    for s in seeds:
        d = sm.gen_cases([("pow2", "single", 1.0)] * n, s, dev, 20, tag=2)
        out[str(s)] = {"E": sme.eval_case(e, d, dev, True), "ladder": sme.eval_case(lad, d, dev, True)}
    for k in ("err5", "err20", "force_rel"):
        ratios = [out[s]["ladder"][k] / out[s]["E"][k] for s in out]
        e_vals = [out[s]["E"][k] for s in out]
        out[f"{k}_ratio_mean"] = sum(ratios) / len(ratios)
        out[f"{k}_ratio_per_seed"] = ratios
        out[f"{k}_E_noise_floor"] = (max(e_vals) - min(e_vals)) / (sum(e_vals) / len(e_vals))
    return out


@torch.no_grad()
def timing_cell(e, lad, dev, n, reps=20):
    rho = torch.rand(n, 1, 128, 128, device=dev)
    res = {}
    for name, m in (("fft", e), ("ladder", lad)):
        m.kernel_acc(rho)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            m.kernel_acc(rho)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        res[name + "_ms"] = (time.perf_counter() - t) / reps * 1000
    res["ladder_over_fft"] = res["ladder_ms"] / res["fft_ms"]
    res["batch"] = n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-ckpt", required=True)
    ap.add_argument("--main-ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seeds", default="9100,9200,9300")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-rollout", action="store_true")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_term)
    dev = torch.device(a.device)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}

    def put(key, fn):
        if key in res:
            return
        res[key] = fn()
        json.dump(res, open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)
        print(f"[{time.strftime('%H:%M:%S')}] cell {key} written", flush=True)
        if STOP["flag"]:
            print("SIGTERM: exiting 143 after cell", flush=True)
            sys.exit(143)

    sd = torch.load(a.e_ckpt, map_location=dev, weights_only=False)["model"]
    e = build_e(dev)
    e.load_state_dict(sd)
    r, K = sl.learned_kernel(e.cpu())
    e.to(dev)
    put("fit:E:pow2", lambda: fit_cell(r, K, EXPECTED_FORCE_EXPONENT["pow2"]))
    amps = res["fit:E:pow2"]["amps"]
    if a.main_ckpt:
        mm = sm.build().eval()
        mm.load_state_dict(torch.load(a.main_ckpt, map_location="cpu", weights_only=False)["model"])
        for shape in sm.TRAIN_SHAPES:
            rr, KK = multi_kernel(mm, shape)
            put(f"fit:main:{shape}", lambda: fit_cell(rr, KK, EXPECTED_FORCE_EXPONENT.get(shape)))
    lad = build_ladder(dev, sd, amps)
    put("tiled", lambda: tiled_cell(e, amps, dev, min(a.n, 8)))
    put("timing", lambda: timing_cell(e, lad, dev, 8))
    if not a.skip_rollout:
        put("rollout:" + a.seeds, lambda: rollout_cell(e, lad, dev, [int(s) for s in a.seeds.split(",")], a.n))
    open(a.out + ".done", "w").write(json.dumps({"cells": sorted(res)}))


if __name__ == "__main__":
    main()
