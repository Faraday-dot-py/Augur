"""Core drift per tick from escape_roll.py outputs: |centre(t1) - centre(t0)| / (t1 - t0) of the shrinking-sphere core centre, default t0=100, t1=300, plus the
U-Net-term / raw core bias at the probe ticks. Usage: python scripts/core_drift_summary.py label=dir[,label=dir...] [arm=S1] [out.txt]"""
import json
import sys

import numpy as np


def centre_at(rows, t):
    for r in rows:
        if r["t"] == t and "centre" in r:
            return np.array(r["centre"])
    return None


def main():
    pairs = [x.split("=", 1) for x in sys.argv[1].split(",")]
    arm = sys.argv[2] if len(sys.argv) > 2 else "S1"
    lines = ["| run | centre@100 | centre@300 | drift/tick 100-300 | core15 raw bias@100 | core15 U-Net mean@100 | in-box net model/exact@100 |", "|---|---|---|---|---|---|---|"]
    for label, d in pairs:
        j = json.load(open(f"{d}/{arm}.json"))
        c1, c3 = centre_at(j["rows"], 100), centre_at(j["rows"], 300)
        drift = float(np.linalg.norm(c3 - c1) / 200) if c1 is not None and c3 is not None else float("nan")
        bias = un = net = ""
        for p in j["probes"]:
            if p["t"] == 100:
                bias = f"{p['cls']['core15']['bias_norm']:.2f}"
                un = str([round(x, 2) for x in p["parts"]["core15"]["unet_mean"]]) if "parts" in p else ""
                net = f"{np.linalg.norm(p['net_raw_model_inbox']):.3g} / {np.linalg.norm(p['net_exact_inbox']):.3g}"
        lines.append(f"| {label} | {np.round(c1, 1).tolist()} | {np.round(c3, 1).tolist()} | {drift:.4f} | {bias} | {un} | {net} |")
    out = "\n".join(lines)
    print(out)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(out + "\n")


if __name__ == "__main__":
    main()
