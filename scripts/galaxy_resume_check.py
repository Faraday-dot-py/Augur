"""Compare diag.json of a straight run and a stop/resume run (tags as args); prints max relative difference over the float fields at common ticks."""
import json
import sys


def main():
    a = {d["tick"]: d for d in json.load(open(f"results/galaxy/{sys.argv[1]}/diag.json"))}
    b = {d["tick"]: d for d in json.load(open(f"results/galaxy/{sys.argv[2]}/diag.json"))}
    worst = 0.0
    for t in sorted(set(a) & set(b)):
        for k, v in a[t].items():
            if k in ("wall", "pnet", "dE") or not isinstance(v, float):
                continue
            worst = max(worst, abs(v - b[t][k]) / (abs(v) + 1e-9))
    print(f"ticks compared {sorted(set(a) & set(b))} max rel diff {worst:.2e}")


if __name__ == "__main__":
    main()
