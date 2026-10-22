"""Print the own-centre of truth and arms at every snapshot tick >= 100 (shrinking sphere), to see whether a residual drift is steady or a jump.
Usage: python scripts/armA_series.py results/armA/n2e4_s12000 S1d4 S1c4"""
import sys

from scripts.armA_drift import series


def main():
    d = sys.argv[1]
    tr = {t: c for t, c, _ in series(f"{d}/truth.npz")}
    for n in sys.argv[2:]:
        print(n, " ".join(f"{t}:({c[0]:.1f},{c[1]:.1f})" for t, c, k in series(f"{d}/{n}.npz") if t >= 100 and t in tr))
    print("truth", " ".join(f"{t}:({c[0]:.1f},{c[1]:.1f})" for t, c in tr.items() if t >= 100))


if __name__ == "__main__":
    main()
