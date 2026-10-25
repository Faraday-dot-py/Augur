"""Hole 8 cost profile: tiled window model, dense vs sparse tile activation, over domain size and occupancy at constant particle density.
Reports ms per force call, active tiles, peak MB, and a torch.profiler breakdown (CUDA time vs CPU time, top ops) for the largest domain.
  python scripts/tiered_prof_sparse.py --model scatter:<E.pt> --out results/tiered/prof_sparse.json"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import tiered_eval as te
from scripts import tiered_iface as ti


MODES = dict(dense=dict(), sparse=dict(sparse=True), dense_batched=dict(batched=True), sparse_batched=dict(sparse=True, batched=True),
             sparse_bucketed=dict(sparse=True, batched=True, bucket=True))


def scene(span, occ, dev, rng, dens=0.05):
    w = span / 2 * np.sqrt(occ)
    n = int(dens * (2 * w) ** 2)
    pos = torch.as_tensor(rng.uniform(-w, w, (1, n, 2)), device=dev)
    return pos, torch.zeros_like(pos), torch.ones(1, n, device=dev, dtype=torch.float64), torch.ones(1, n, device=dev, dtype=torch.float64), n


@torch.no_grad()
def bench(tl, sc, reps=5):
    pos, vel, mass, mask, _ = sc
    st = tl.init_state(1, pos.device)
    for _ in range(2):
        tl.force(pos, vel, mass, mask, st)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(reps):
        tl.force(pos, vel, mass, mask, st)
    torch.cuda.synchronize()
    return (time.time() - t) / reps * 1e3, torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spans", type=float, nargs="+", default=[128.0, 256.0, 512.0, 1024.0])
    ap.add_argument("--tile", type=float, default=32.0)
    ap.add_argument("--out", default="results/tiered/prof_sparse.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    base = te.load_model(a.model, dev, 0.1)
    rng = np.random.default_rng(4738)
    out = {}
    for span in a.spans:
        for occ in (1.0, 0.5, 0.25, 0.125):
            sc = scene(span, occ, dev, rng)
            row = {}
            for mode, kw in MODES.items():
                tl = ti.Tiled(base, span, a.tile, **kw)
                ms, mb = bench(tl, sc)
                row[mode] = dict(ms=ms, mb=mb, active=tl.last_active, tiles=tl.last_tiles)
            row["n"] = sc[4]
            out[f"span{int(span)}_occ{occ}"] = row
            print(f"span {span:.0f} occ {occ} n {sc[4]} {' '.join(f'{k} {v["ms"]:.1f}' for k, v in row.items() if k != 'n')} ms, {row['sparse']['active']:.0f}/{row['dense']['tiles']} active", flush=True)
    span = a.spans[-1]
    sc = scene(span, 0.125, dev, rng)
    tl = ti.Tiled(base, span, a.tile, sparse=True)
    st = tl.init_state(1, dev)
    tl.force(sc[0], sc[1], sc[2], sc[3], st)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        with torch.no_grad():
            tl.force(sc[0], sc[1], sc[2], sc[3], st)
        torch.cuda.synchronize()
    ka = pr.key_averages()
    out["profile_sparse_largest_occ0125"] = dict(
        cuda_ms=sum(k.self_device_time_total for k in ka) / 1e3, cpu_ms=sum(k.self_cpu_time_total for k in ka) / 1e3,
        top=[dict(name=k.key, cpu_ms=k.self_cpu_time_total / 1e3, cuda_ms=k.self_device_time_total / 1e3, calls=k.count)
             for k in sorted(ka, key=lambda k: -(k.self_cpu_time_total + k.self_device_time_total))[:12]])
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["profile_sparse_largest_occ0125"], indent=1))


if __name__ == "__main__":
    main()
