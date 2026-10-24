import json
import sys

for path in sys.argv[1:]:
    print("==", path)
    for k, r in json.load(open(path)).items():
        print(f"{k:24s} fin={r['finite']} dE/N={r['dE_per_N']:+.4f} T={r['T_end']:.3f} T_truth={r['T_truth_end']:.3f}")
