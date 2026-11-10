import json, math, statistics as st

D = "results/width32_loss_spike_eight_repeats_fixed_seed_4740"
SPIKE_N = 10.0


def trace(r):
    last = {}
    for line in open(f"{D}/repeat{r}/trace.jsonl"):
        x = json.loads(line)
        last[x["it"]] = x
    its = sorted(last)
    assert its == list(range(3300)), (r, "trace not 0..3299")
    return [last[i] for i in its]


def epochs(tr):
    out = []
    for e in range(1, 22):
        lo, hi = 160 * (e - 1), min(160 * e, 3300)
        s = sum(x["loss"] for x in tr[lo:hi] if x["applied"])
        out.append(s / (100 if e == 21 else 160))
    return out


res = {"repeats": {}}
traces = {}
for r in range(1, 9):
    tr = trace(r)
    traces[r] = tr
    ep = epochs(tr)
    skips = [x["it"] for x in tr if not x["applied"]]
    spikes = []
    for e in range(6, 22):
        base = st.median(ep[e - 6:e - 1])
        v = ep[e - 1]
        if not math.isfinite(v) or v > SPIKE_N * base:
            spikes.append({"epoch": e, "loss": v, "preceding5_median": base, "ratio": v / base if base else None})
    maxratio = max(ep[e - 1] / st.median(ep[e - 6:e - 1]) for e in range(6, 22))
    resumed = "resumed" in open(f"{D}/repeat{r}/run.log").read()
    res["repeats"][r] = {"first_skipped_it": skips[0] if skips else None, "skipped": len(skips), "dropped_total": tr[-1]["dropped_total"],
                         "spikes": spikes, "max_epoch_ratio": maxratio, "resumed": resumed,
                         "event": bool(skips) or bool(spikes)}
first_diff = {}
for r in range(2, 9):
    d = next((i for i in range(3300) if traces[r][i]["loss"] != traces[1][i]["loss"]), None)
    first_diff[r] = d
res["first_iteration_loss_differs_from_repeat1"] = first_diff
res["all_bitwise_identical"] = all(v is None for v in first_diff.values())
n = sum(v["event"] for v in res["repeats"].values())
res["repeats_with_event"] = n
res["label"] = "PASS" if n >= 1 else "KILL"
json.dump(res, open(f"{D}/score.json", "w"), indent=1)
print(json.dumps(res, indent=1))
