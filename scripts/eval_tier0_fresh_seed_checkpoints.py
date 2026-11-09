"""Fresh held-out evaluation of frozen tiled tier-0 checkpoints on scenes no tiered result has used (default seeds 12000/12100/12200, 24 scenes x 100 ticks,
scripts/train_tiered.make_scenes), together with the tiled baseline (tiled Exp E + far kernel, BL) on the same scenes: BL has to be computed here because the stored
baseline file only covers seeds 9100/9200/9300. Checkpoints are scored as stored (halo exchange as trained); nothing is selected on these seeds.
Usage: PYTHONPATH=. python scripts/eval_tier0_fresh_seed_checkpoints.py --tier0 $HOME/E_ms_kp_pot_v_g128.pt --ckpt seed4738=<pt> seed4739=<pt> --out <json>"""
import argparse
import json
import os

import torch

from scripts import exp12_hier as h
from scripts import tiered_adapters as ta
from scripts import train_tiered as ttr
from scripts import tier0_tiled_closed_loop as tc
from scripts.eval_tier0_tiled_closed_loop import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier0", required=True, help="Exp E checkpoint of the tiled baseline")
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[12000, 12100, 12200])
    ap.add_argument("--eval-scenes", type=int, default=24)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--exchange", choices=["halo", "zero", "none"], default="halo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda s: print(s, flush=True)
    pairs = [c.split("=", 1) for c in a.ckpt]
    assert all(len(p) == 2 for p in pairs), "--ckpt takes name=path"
    for _, path in pairs:
        assert os.path.exists(path), f"missing {path}"
    bl = ta.load_tiled(f"{a.tier0},nt=4,tile=32,halo=16,far=1", dev)
    res = {"seeds": a.seeds, "BL": {}}
    scenes = {sd: ttr.make_scenes(a.eval_scenes, a.eval_steps, sd, dev) for sd in a.seeds}
    for sd in a.seeds:
        res["BL"][str(sd)] = h.eval_generic(lambda p, v, m, k, st: bl.force(p, v, m, k, st), lambda S: bl.init_state(S, dev), scenes[sd])
        log(f"BL {sd} {json.dumps({k: round(v, 5) for k, v in res['BL'][str(sd)].items()})}")
    del bl
    for name, path in pairs:
        model = load(path, dev)
        tt = tc.TiledTier0(model, dev, chunk=a.chunk, exchange=a.exchange)
        res[name] = {}
        for sd in a.seeds:
            res[name][str(sd)] = h.eval_generic(lambda p, v, m, k, st: tt.force(p, v, m, k, st), tt.init_state, scenes[sd])
            log(f"{name} {sd} {json.dumps({k: round(v, 5) for k, v in res[name][str(sd)].items()})}")
        del tt, model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
