"""Re-score gap-investigation checkpoints (trained on Exp B) and pp8-ablation checkpoints (trained on Exp E) under both eval protocols.
Protocol B: Exp B config (10-100 bodies scale-init), 48 scenes x 100 steps, seeds 9000/12000 (gap_probe.make_sets).
Protocol E: Exp E config, seeds 9000/12000, `ab` = first 24 scenes (10-200 bodies), `cl` = last 24 (clusters), as scatter_prior_pp8.evaluate.
Usage: PYTHONPATH=. python scripts/scatter_protocol_rescore.py --out results/scatter_pp8/protocol_rescore.json"""
import argparse
import json
import os

import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as ts
from scripts import gap_probe as gp
from scripts import scatter_prior_pp8 as pp

HOME = os.path.expanduser("~")
GAP = f"{HOME}/augur/checkpoints/gap"
ABL = f"{HOME}/augur_pp8/checkpoints/scatter_pp8"
FROZEN = f"{HOME}/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
LANDED_PAIR = "ms_kp_pot_v_g128_prior_pp8"

ENTRIES = [
    ("gap_cf_4000 (trained B)", lambda: gp.build("cf"), f"{GAP}/cf_4000.pt"),
    ("gap_sf_prior_pp8_4000 (trained B)", lambda: gp.build("sf_prior_pp8"), f"{GAP}/sf_prior_pp8_4000.pt"),
    ("gap_sf_prior_pp8_4000 as landed pair class", lambda: pp.build("pair"), f"{GAP}/sf_prior_pp8_4000.pt"),
    ("gap_sf_full_4000 (trained B)", lambda: gp.build("sf_full"), f"{GAP}/sf_full_4000.pt"),
    ("abl_cf (trained E)", lambda: pp.build("cf"), f"{ABL}/cf.pt"),
    ("abl_ctl (trained E)", lambda: pp.build("ctl"), f"{ABL}/ctl.pt"),
    ("abl_pair (trained E)", lambda: pp.build("pair"), f"{ABL}/pair.pt"),
    ("abl_prior (trained E)", lambda: pp.build("prior"), f"{ABL}/prior.pt"),
    ("abl_pp8 (trained E)", lambda: pp.build("pp8"), f"{ABL}/pp8.pt"),
    ("frozen_E (60k it)", lambda: pp.build("ctl"), FROZEN),
]


def score(model, data, dev):
    m, _ = ts.evaluate_model(model, data, dev, pp.DT, pp.EPS)
    return [m["err"][i] for i in (4, 9, 19, 49, 99)]


def mean(rows):
    return [sum(r[i] for r in rows) / len(rows) for i in range(len(rows[0]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scatter_pp8/protocol_rescore.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    sets = {}
    for seed in ts.EVAL_SEEDS:
        dB = ts.make_data(ts.EXPS["B"], 48, 100, seed, dev, pp.DT, pp.EPS)
        dE = ts.make_data(ts.EXPS["E"], 48, 100, seed, dev, pp.DT, pp.EPS)
        sets[f"B_{seed}"] = dB
        sets[f"Eab_{seed}"] = dE[:pp.N_AB]
        sets[f"Ecl_{seed}"] = dE[pp.N_AB:]
        for k in (f"B_{seed}", f"Eab_{seed}", f"Ecl_{seed}"):
            print(k, "scenes", len(sets[k]), "mean n", float(sf.to_tensors(sets[k], dev)[3].sum(1).mean()), flush=True)
    res = {}
    for name, builder, ck in ENTRIES:
        if not os.path.exists(ck):
            print("missing", ck, flush=True)
            continue
        torch.manual_seed(4738)
        model = builder().to(dev)
        try:
            model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
        except Exception as e:
            print("load fail", name, str(e)[:200], flush=True)
            continue
        model.eval()
        r = {}
        for tag in ("B", "Eab", "Ecl"):
            r[tag] = mean([score(model, sets[f"{tag}_{s}"], dev) for s in ts.EVAL_SEEDS])
        res[name] = r
        print(name, {t: [round(v, 4) for v in (r[t][2], r[t][4])] for t in r}, "(err@20, err@100)", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
