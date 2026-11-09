"""Applies the pre-stated read-out rule to results/sae_random_pool_dose_response_*.json, recomputing ranks from the raw per-arm errors.
Usage: python scripts/summarise_sae_random_pool_dose_response.py > results/summary_sae_random_pool_dose_response.txt
"""
import glob
import json

import numpy as np


def main():
    for f in sorted(glob.glob("results/sae_random_pool_dose_response_*.json")):
        d = json.load(open(f))
        assert d["complete"]
        laws = list(d["errors"])
        print(f"\n== {f.split('sae_random_pool_dose_response_')[1][:-5]} live={d['n_live']} draws={d['draws']} pool_size={d['pool_size']}")
        print("k  pool_avail  laws_top>p95  laws_inside(<95)  mean_rank  reading  top/recon  rand_med/recon  rand_p95/recon  outlier_draws(all laws)")
        for k in d["ks"]:
            ranks, above, tr, rr, rp, out = [], 0, [], [], [], 0
            for law in laws:
                E = d["errors"][law]
                r = np.array([E[f"rand{k}_{i}"] for i in range(d["draws"])])
                t, rec = E[f"top{k}"], E["sae_reconstruction"]
                ranks.append(((r < t).sum() + 0.5 * (r == t).sum()) / len(r) * 100)
                above += t > np.percentile(r, 95)
                tr.append(t / rec)
                rr.append(np.median(r) / rec)
                rp.append(np.percentile(r, 95) / rec)
                out += len(d["summary"][law][str(k)]["outlier_draws"])
            special = sum(x >= 95 for x in ranks)
            inside = sum(x < 95 for x in ranks)
            m = float(np.mean(ranks))
            reading = "special" if special >= 6 else "any-removal-hurts" if inside >= 6 else ("special(tie-break)" if m >= 95 else "not-special(tie-break)")
            print(f"{k:<3}{d['pool_available_by_k'][str(k)]:<12}{special:<14}{inside:<18}{m:<11.1f}{reading:<24}{np.mean(tr):<11.2f}{np.mean(rr):<16.2f}{np.mean(rp):<16.2f}{out}")
        k1 = {}
        for law in laws:
            for w in d["summary"][law]["1"]["worst_random_features"][:1]:
                k1.setdefault(w["feature"], []).append((law, round(w["mean_draw_error"], 3)))
        print("worst random single features (k=1, law, mean error):", {j: v for j, v in sorted(k1.items(), key=lambda kv: -len(kv[1]))[:4]})
        print("their activation frequencies:", {j: d["feature_freq"][j] for j in k1})
        print("top features k=1:", d["top_features"]["1"])


if __name__ == "__main__":
    main()
