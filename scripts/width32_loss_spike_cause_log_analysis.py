import json, re, sys, statistics as st

RUNS = {
    "width4-seed4738": ("results/single_law_network_sizing/run.log", 4),
    "width8-seed4738": ("results/single_law_network_sizing/run.log", 8),
    "width16-seed4738": ("results/single_law_network_sizing/run.log", 16),
    "width32-seed4738": ("results/single_law_network_sizing/run.log", 32),
    "width64-seed4738": ("results/single_law_network_sizing/run.log", 64),
}
for s in (4739, 4740, 4741):
    RUNS[f"width32-seed{s}"] = (f"results/width32_loss_spike_validation/seed{s}/run.log", 32)

pat = re.compile(r"width (\d+) it (\d+)/8000 k (\d+) lr (\S+) loss (\S+) dropped (\d+)")
out = {}
for name, (path, w) in RUNS.items():
    rows = []
    for line in open(path):
        m = pat.search(line)
        if m and int(m.group(1)) == w:
            rows.append((int(m.group(2)), int(m.group(3)), float(m.group(5)), int(m.group(6))))
    assert len(rows) == 80, (name, len(rows))
    ratios = []
    for i, (it, k, loss, dr) in enumerate(rows):
        if i < 8:
            continue
        base = st.median(r[2] for r in rows[i - 8:i])
        ratios.append((loss / base, it, k, loss))
    ratios.sort(reverse=True)
    window = [r[2] for r in rows if 2700 <= r[0] <= 3200]
    out[name] = {
        "n_logged": len(rows),
        "top3_logged_iter_ratio_vs_prev8_median": [{"ratio": round(a, 2), "it": b, "k": c, "loss": d} for a, b, c, d in ratios[:3]],
        "median_logged_loss_its_2700_to_3200": st.median(window),
    }
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
