import json
import os
import time
from types import SimpleNamespace

import numpy as np

from scripts import train_scatter_field as tsf

PLATEAU_REL = 0.03
VAL_POINTS = 50


def train_with_curve(model, tensors, val_fn, args, ckpt, out, stop, log, eps=0.5):
    """Resumable tsf.train with a validation curve (val_fn(model) -> float) logged VAL_POINTS times; curve is saved in out/val_curve.json and survives restarts."""
    curve_path = f"{out}/val_curve.json"
    curve = {int(k): v for k, v in json.load(open(curve_path)).items()} if os.path.exists(curve_path) else {}
    every = max(1, args.iters // VAL_POINTS)

    rate = {}
    stop_at = getattr(args, "stop_at", 0)

    def it_hook(it):
        n = it + 1
        rate["completed"] = n
        if "n0" not in rate:
            rate["n0"], rate["t0"] = n, time.time()
        elif "done" not in rate and n - rate["n0"] >= 100:
            rate["done"] = True
            log(f"s_per_it_first100={(time.time() - rate['t0']) / (n - rate['n0']):.4f}")
        if n % every == 0 and n not in curve:
            curve[n] = val_fn(model)
            json.dump({str(k): v for k, v in sorted(curve.items())}, open(curve_path, "w"))
            log(f"val err@20 at it {n}: {curve[n]:.5f}")

    ns = SimpleNamespace(device=args.device, seed=args.seed, batch=args.batch, steps=args.steps, k_start=args.k_start, k_end=args.k_end, iters=args.iters,
                         time_budget=0.0, log_every=args.log_every, long_windows=0, lr=args.lr, eps=eps, stop=lambda: stop() or rate.get("completed") == stop_at, it_hook=it_hook, exact_resume=getattr(args, "exact_resume", stop_at > 0))
    log(f"params {sum(p.numel() for p in model.parameters())}")
    return tsf.train(model, tensors, ns, ckpt, log)


def plateau(out, rel=PLATEAU_REL):
    """INCONCLUSIVE-undertrained guard: mean validation value over the last 10% of the curve more than rel below the mean over the preceding 10%."""
    curve = {int(k): v for k, v in json.load(open(f"{out}/val_curve.json")).items()}
    its = sorted(curve)
    w = max(1, len(its) // 10)
    if len(its) < 2 * w:
        raise SystemExit(f"val curve has {len(its)} points, need {2 * w}")
    last, prev = float(np.mean([curve[i] for i in its[-w:]])), float(np.mean([curve[i] for i in its[-2 * w:-w]]))
    r = (prev - last) / prev
    return {"curve": curve, "window_points": w, "mean_prev_window": prev, "mean_last_window": last, "rel_improvement": r,
            "undertrained": bool(r > rel), "threshold": rel}
