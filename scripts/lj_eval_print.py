import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/lj_eval.json"))
f = d["force"]
print(f"iter {d['iter']}  force rel_l2 {f['rel_l2']:.4f}  max|df| core(<1.2) {f['max_abs_core_0.75_1.2']:.3f}  tail {f['max_abs_tail_1.2_2.5']:.4f}")
for name, r in d.items():
    if name in ("iter", "force"):
        continue
    t, m = r["truth"], r["model"]
    print(f"{name:8s} err model {' '.join(f'{k}:{v:.3f}' for k, v in r['err_model'].items())}")
    print(f"{'':8s} err floor {' '.join(f'{k}:{v:.3f}' for k, v in r['err_rerun_floor'].items())}")
    print(f"{'':8s} last200 T {t['T']:.3f}/{m['T']:.3f}  P {t['P']:.3f}/{m['P']:.3f}  E {t['E']:.4f}/{m['E']:.4f}  psi6 {t['psi6']:.3f}/{m['psi6']:.3f}  g L2 {(sum((a - b) ** 2 for a, b in zip(t['g'], m['g']))) ** 0.5:.3f}")
