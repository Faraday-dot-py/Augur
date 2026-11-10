"""Stage 1 of law-inferred-from-motion-on-random-law-zoo: generate the zoo truth data on GPU and check the counts reconcile.

Writes one chunk file per 64 scenes (skipped on rerun), then data_summary.json with the numbers of requested, accepted and rejected scenes, the zoo-draw rejections by the held-out
exclusion, and the encoder forward check. Usage: PYTHONPATH=. python scripts/law_zoo_data_check.py --out-dir results/law_zoo_stage1
"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import law_zoo as lz

CHUNK = 64


def build_cases(args):
    zoo, excluded = lz.sample_zoo(args.laws, args.seed)
    held = lz.sample_heldout(args.heldout, args.seed)
    rng = np.random.default_rng([args.seed, 3])
    cases = [("train", i, law, lz.sample_scale(rng)) for i, law in enumerate(zoo) for _ in range(args.scenes_per_law)]
    cases += [("heldout", i, law, lz.sample_scale(rng)) for i, law in enumerate(held)]
    return cases, excluded


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--laws", type=int, default=300)
    ap.add_argument("--scenes-per-law", type=int, default=2)
    ap.add_argument("--heldout", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    cases, excluded = build_cases(args)
    accepted = 0
    enc = lz.LawEncoder()
    for c0 in range(0, len(cases), CHUNK):
        path = f"{args.out_dir}/chunk{c0 // CHUNK:03d}.pt"
        if os.path.exists(path):
            accepted += int(torch.load(path, weights_only=False)["ok"].sum())
            continue
        sub = cases[c0:c0 + CHUNK]
        rng = np.random.default_rng([args.seed, 4, c0])
        d = lz.gen_zoo_batch([(c[2], c[3]) for c in sub], rng, dev, args.steps)
        d = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        d["split"] = [c[0] for c in sub]
        d["law_index"] = [c[1] for c in sub]
        d["laws"] = [c[2] for c in sub]
        d["scale"] = [c[3] for c in sub]
        d["p_eff"] = [lz.effective_exponent(c[2]) for c in sub]
        torch.save(d, path + ".tmp")
        os.replace(path + ".tmp", path)
        accepted += int(d["ok"].sum())
        print(f"chunk {c0 // CHUNK} scenes {len(sub)} accepted {int(d['ok'].sum())}", flush=True)
    first = torch.load(f"{args.out_dir}/chunk000.pt", weights_only=False)
    with torch.no_grad():
        code = enc(first["P"][:4].double(), first["V"][:4].double(), first["mask"][:4].double())
    files = sorted(f for f in os.listdir(args.out_dir) if f.startswith("chunk") and f.endswith(".pt"))
    total = sum(len(torch.load(f"{args.out_dir}/{f}", weights_only=False)["ok"]) for f in files)
    held_d = [lz.held_out_distance(c[2]) for c in cases if c[0] == "heldout"]
    train_d = [lz.held_out_distance(c[2]) for c in cases if c[0] == "train"]
    summary = dict(requested=len(cases), generated=total, accepted=accepted, rejected_by_truth_check=total - accepted, zoo_draws_excluded_by_heldout_rule=excluded,
                   train_min_heldout_distance=float(min(train_d)), heldout_max_distance=float(max(held_d)), shape_tol=lz.SHAPE_TOL,
                   encoder_code_shape=list(code.shape), encoder_code_finite=bool(torch.isfinite(code).all()), encoder_neighbour_search="dense-cdist-O(N^2)-stage-1-only", seed=args.seed, steps=args.steps)
    assert summary["generated"] == summary["requested"], summary
    assert summary["train_min_heldout_distance"] >= lz.SHAPE_TOL and summary["heldout_max_distance"] < lz.SHAPE_TOL, summary
    assert summary["encoder_code_finite"], summary
    json.dump(summary, open(f"{args.out_dir}/data_summary.json", "w"), indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
