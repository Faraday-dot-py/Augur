import json, re, statistics as st, sys

D = "results/width32_loss_spike_instrumented_rerun"
REF = "results/width32_loss_spike_validation/seed4740/run.log"
pat = re.compile(r"width 32 it (\d+)/8000 k (\d+) lr \S+ loss (\S+) dropped (\d+)")


def trace(seed):
    last = {}
    for line in open(f"{D}/seed{seed}/trace.jsonl"):
        r = json.loads(line)
        last[r["it"]] = r
    its = sorted(last)
    assert its == list(range(len(its))), (seed, "gap in trace")
    return [last[i] for i in its]


def epoch_scores(tr, per=160):
    out = {}
    for e in range(1, len(tr) // per + 2):
        lo, hi = per * (e - 1), min(per * e, len(tr))
        if lo >= len(tr):
            break
        n = hi - lo
        out[e] = {"its": [lo, hi - 1], "n": n, "mean_per_160": sum(r["loss"] for r in tr[lo:hi] if r["applied"]) / per,
                  "mean_per_n": sum(r["loss"] for r in tr[lo:hi] if r["applied"]) / n}
    return out


def logged(path):
    return {int(m.group(1)): (float(m.group(3)), int(m.group(4)), int(m.group(2))) for m in map(pat.search, open(path)) if m}


res = {}
for seed in (4740, 4738):
    log = open(f"{D}/seed{seed}/run.log").read()
    tr = trace(seed)
    skips = [r["it"] for r in tr if not r["applied"]]
    ep = epoch_scores(tr)
    score = {}
    for e, v in ep.items():
        if e >= 6:
            base = st.median(ep[j]["mean_per_160"] for j in range(e - 5, e))
            full = v["n"] == 160
            val = v["mean_per_160"] if full else v["mean_per_n"]
            score[e] = {"its": v["its"], "n": v["n"], "loss": val, "preceding5_median": base, "ratio": val / base}
    res[seed] = {"resumed_in_run_log": "resumed" in log, "trace_len": len(tr), "first_skipped_it": skips[0] if skips else None,
                 "skipped": len(skips), "dropped_total": tr[-1]["dropped_total"], "epochs_14_to_22": {e: score[e] for e in score if e >= 14},
                 "max_iter_loss": max((r["loss"], r["it"], r["k"]) for r in tr),
                 "max_iter_vel_share_kept": max(((r["vel_kept"][0] + r["vel_kept"][1]) * 0.1 / max(r["loss"], 1e-12), r["it"]) for r in tr if r["applied"] and r["loss"] > 0.05)
                 if any(r["applied"] and r["loss"] > 0.05 for r in tr) else None}

a, b = logged(f"{D}/seed4740/run.log"), logged(REF)
rows = []
for it in range(100, 3301, 100):
    rows.append({"it": it, "rerun": a[it][0], "ref7442": b[it][0], "ratio": a[it][0] / b[it][0] if b[it][0] else None, "ref_dropped": b[it][1], "rerun_dropped": a[it][1]})
res["reproducibility_4740_vs_7442"] = {"rows": rows, "bitwise_equal_its": [r["it"] for r in rows if r["rerun"] == r["ref7442"]],
                                       "max_abs_log_ratio_before_3000": max(abs(r["ratio"] - 1) for r in rows if r["it"] < 3000 and r["ratio"])}

e = res[4740]["epochs_14_to_22"]
spike = [x for x in (19, 20, 21) if x in e and e[x]["ratio"] > 10]
first = res[4740]["first_skipped_it"]
pas = (first is not None and 3021 <= first <= 3121) or bool(spike)
res["verdict_4740"] = {"first_skip_in_3021_3121": first is not None and 3021 <= first <= 3121, "spike_epochs_19_21": spike, "label": "PASS" if pas else "KILL"}
json.dump(res, open(f"{D}/score.json", "w"), indent=1)
print(json.dumps(res, indent=1))
