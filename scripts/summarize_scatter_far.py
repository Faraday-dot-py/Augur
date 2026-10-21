"""Condense scatter_far_eval.py logs (JSON lines) into tables: held-out (9000, 12000) means, dev 4738 separately.
Usage: python scripts/summarize_scatter_far.py log1 [log2 ...]
"""
import json
import sys
from collections import defaultdict

HELD = (9000, 12000)


def rows_of(paths):
    wide, far, table = [], [], {}
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if "variant" in d:
                    wide.append(d)
                elif "far_rel_err" in d:
                    far.append(d)
            elif ' {"ratio_at"' in line:
                name, js = line.split(" ", 1)
                table[name] = json.loads(js)
    return wide, far, table


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    wide, far, table = rows_of(sys.argv[1:])
    if table:
        print("## kernel table")
        for k, v in table.items():
            ra = v["ratio_at"]
            print(f"{k}: ratio r=4 {ra['4']:.2f} 16 {ra['16']:.2f} 32 {ra['32']:.2f} 40 {ra['40']:.2f} 48 {ra['48']:.2f} 64 {ra['64']:.2f} 80 {ra['80']:.2f} 100 {ra['100']:.2f} "
                  f"200 {ra['200']:.2f} 300 {ra['300']:.2f} 512 {ra['512']:.2f} | min/max[4,512] {v['ratio_min_4_512']:.3f}/{v['ratio_max_4_512']:.3f} minF {v['min_force_ge_4']:.2e} "
                  f"mono {v['monotone_decay_ge32']} dev-vs-old[4,32] {v['max_rel_dev_vs_old_4_32']:.4f} [1,32] {v['max_rel_dev_vs_old_1_32']:.4f}")
    if wide:
        print("## wide scene (held-out mean of 9000/12000; dev 4738)  err@5/10/20, diff_single20, force_vs_exact, dP")
        g = defaultdict(list)
        for d in wide:
            g[(d["spec"], d["variant"], d.get("far_c"))].append(d)
        for k, v in g.items():
            h = [d for d in v if d["seed"] in HELD]
            dv = [d for d in v if d["seed"] == 4738]

            def f(ds):
                if not ds:
                    return "-"
                e = [mean([d["err"][t] for d in ds]) for t in ("5", "10", "20")]
                s = f"err {e[0]:.3f}/{e[1]:.3f}/{e[2]:.3f}"
                if "diff_single20" in ds[0]:
                    s += f" diff1g {mean([d['diff_single20'] for d in ds]):.3f} dP {mean([d['dP'] for d in ds]):.1e}"
                return s + f" force {mean([d['force_vs_exact'] for d in ds]):.3f}"
            print(f"{k[0]:14s} {k[1]:9s} c={k[2]}: held {f(h)} | dev {f(dv)}")
    if far:
        print("## far term vs exact outside-box acceleration (tick 0; mean over seeds)")
        g = defaultdict(list)
        for d in far:
            g[(d["spec"], d["far_c"])].append(d)
        for k, v in g.items():
            b = {e: mean([d["by_edge_dist"][e] for d in v]) for e in v[0]["by_edge_dist"]}
            print(f"{k[0]:14s} c={k[1]}: far rel err {mean([d['far_rel_err'] for d in v]):.4f}  err/total acc {mean([d['far_err_over_total_acc'] for d in v]):.4f}  "
                  f"far/total {mean([d['far_over_total_acc'] for d in v]):.3f}  by edge dist " + " ".join(f"{e}:{x:.3f}" for e, x in b.items()))


if __name__ == "__main__":
    main()
