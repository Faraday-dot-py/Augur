"""Pre-registered gate for the heads-only full-schedule arm. Usage: python scripts/heads_full_gate.py TAG... (frozenE rows must exist in results/el_eval).
PASS per tag = ab err@20 <= .0076 and nr100 dE@1000 (median of 3) <= 1.15 x frozenE's; exit 0 only if every tag passes."""
import sys

from scripts.ladder_summary import row

ref = row("frozenE")
ok = True
for t in sys.argv[1:]:
    r = row(t)
    p = r["ab20"] <= 0.0076 and r["nr100_dE1000"] <= 1.15 * ref["nr100_dE1000"]
    ok &= p
    print(f"{t}: ab20 {r['ab20']:.4f} nr100_dE1000 {r['nr100_dE1000']:.4f} {r['nr100_dE1000_all']} (frozenE {ref['nr100_dE1000']:.4f}, gate {1.15 * ref['nr100_dE1000']:.4f}) bh300_dE1000 {r['bh300_dE1000']:.3f} (E {ref['bh300_dE1000']:.3f}) {'PASS' if p else 'FAIL'}")
sys.exit(0 if ok else 1)
