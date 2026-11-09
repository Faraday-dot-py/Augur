import argparse
import glob
import json
import os
import sys

PASS_SLOPE = 0.3
MIN_WORKING = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    files = sorted(glob.glob(os.path.join(args.root, "rep*", "slow-patch", "seed*", "results.json")))
    if not files:
        print("no results.json found")
        sys.exit(1)
    working = 0
    for f in files:
        r = json.load(open(f))
        parts = f.split(os.sep)
        tag = f"{parts[-4]}/{parts[-2]}"
        cells = {}
        for n in ("10", "100"):
            for arm, v in r["eval"][n].items():
                cells[(n, arm.split("-")[0])] = v["rate_statistics"]["300"]["slope_pred_on_true"]
        var10, var100 = cells[("10", "variant")], cells[("100", "variant")]
        ok = var10 >= PASS_SLOPE and var100 >= PASS_SLOPE
        working += ok
        print(f"{tag} verdict {r['verdict']} slope300 variant N=10 {var10:.2f} N=100 {var100:.2f} | baseline {cells[('10', 'baseline')]:.2f}/{cells[('100', 'baseline')]:.2f} | control {cells[('10', 'control')]:.2f}/{cells[('100', 'control')]:.2f} | rate head works {ok}")
    n = len(files)
    print(f"runs {n} rate head works (slope300 >= {PASS_SLOPE} at N=10 and N=100) {working}")
    if n < 6:
        print("INCOMPLETE")
        sys.exit(1)
    print("REPRODUCTION", "PASS" if working >= MIN_WORKING else "FAIL", f"(pass line: >= {MIN_WORKING} of {n})")


if __name__ == "__main__":
    main()
