import json
from collections import defaultdict

d = json.load(open("results/tiling_accuracy.json"))
rows = d["rows"]
frames = d["frames"]
tiled = [r for r in rows if r["kind"] == "tiled"]
glob = {(r["frame"], r["mode"]): r for r in rows if r["kind"] == "global"}
halo = [r for r in rows if r["kind"] == "halo"]

print("=== P1: design C matches global (fp64 round-off), A/B within 1.5x, momentum ~1e-16 ===")
for design in ("A", "B", "C"):
    dr = [r for r in tiled if r["design"] == design]
    diffs = [r["diff_vs_global_rel"] for r in dr]
    moms = [r["net_ratio"] for r in dr]
    costr = [r["cost"] / glob[(r["frame"], r["mode"])]["cost"] for r in dr]
    print(f"{design}: diff_vs_global_rel max={max(diffs):.3e} mean={sum(diffs)/len(diffs):.3e} | "
          f"momentum max={max(moms):.3e} | cost/global max={max(costr):.3f} mean={sum(costr)/len(costr):.3f}")

print("\n=== P1 detail: C exactly at round-off? ===")
c_diffs = [r["diff_vs_global_rel"] for r in tiled if r["design"] == "C"]
print("C diff_vs_global_rel: min", min(c_diffs), "max", max(c_diffs))

print("\n=== P2: imports fall with tile size; bytes/tile vs all-gather at T<=32; estimator overhead vs geo ===")
for design in ("A", "B", "C"):
    for T in (2, 4, 8, 16, 32):
        dr = [r for r in tiled if r["design"] == design and r["T"] == T and r["mode"] == "geo"]
        if not dr:
            continue
        ratio = [r["imp_bytes_mean"] / r["allgather_bytes"] for r in dr]
        print(f"{design} T={T}: imp_bytes_mean/allgather mean={sum(ratio)/len(ratio):.4f} max={max(ratio):.4f}")

print("\n--- estimator vs geometric import overhead (design C) ---")
for T in (2, 4, 8, 16, 32):
    geo = [r for r in tiled if r["design"] == "C" and r["T"] == T and r["mode"] == "geo"]
    est = [r for r in tiled if r["design"] == "C" and r["T"] == T and r["mode"] == "est"]
    if not geo or not est:
        continue
    gb = sum(r["imp_bytes_mean"] for r in geo) / len(geo)
    eb = sum(r["imp_bytes_mean"] for r in est) / len(est)
    print(f"T={T}: est/geo imp_bytes_mean ratio = {eb/gb:.3f}")

print("\n=== P3: extent-independence (accuracy, rel imports) across frames t0..t10000 ===")
for f, meta in frames.items():
    print(f, "extent", meta["extent"])
for design in ("C",):
    for f in ("t0", "t1000", "t5000", "t10000"):
        dr = [r for r in tiled if r["design"] == design and r["frame"] == f and r["mode"] == "geo" and r["T"] == 8]
        if not dr:
            continue
        rl2 = sum(r["rel_l2"] for r in dr) / len(dr)
        ib = sum(r["imp_bytes_mean"] / r["allgather_bytes"] for r in dr) / len(dr)
        print(f"{f}: rel_l2={rl2:.4f} imp/allgather={ib:.4f}")

print("\n=== P4: work imbalance, strips vs morton, count-balanced ===")
for part in ("strips", "morton"):
    dr = [r for r in tiled if r["part"] == part and r["mode"] == "geo"]
    if not dr:
        continue
    wi = [r["work_imb"] for r in dr]
    ci = [r["count_imb"] for r in dr]
    print(f"{part}: work_imb mean={sum(wi)/len(wi):.3f} max={max(wi):.3f} | count_imb mean={sum(ci)/len(ci):.3f} max={max(ci):.3f}")

print("\n--- flyby-only (clumped) imbalance by partition ---")
for part in ("strips", "morton"):
    dr = [r for r in tiled if r["part"] == part and r["mode"] == "geo" and r["frame"] in frames]
    wi = [r["work_imb"] for r in dr]
    if wi:
        print(f"{part}: n={len(wi)} work_imb mean={sum(wi)/len(wi):.3f} max={max(wi):.3f}")

print("\n=== P5: strips vs morton import counts (design C) ===")
for part in ("strips", "morton"):
    dr = [r for r in tiled if r["design"] == "C" and r["part"] == part and r["mode"] == "geo"]
    if not dr:
        continue
    n = [r["imp_nodes_mean"] for r in dr]
    print(f"{part}: imp_nodes_mean mean={sum(n)/len(n):.1f} max={max(n):.1f}")

print("\n=== Halo / cutoff truncation ===")
for r in halo[:6]:
    print(r)
print("\ncutoff_rel_l2 by frame/radius (from frames meta):")
for f, meta in frames.items():
    print(f, meta["cutoff_rel_l2"])
