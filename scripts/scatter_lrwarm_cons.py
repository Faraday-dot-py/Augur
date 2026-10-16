"""Median-of-3 final-tick (index 1000 of 1000 steps) dE/dL for lr-warmup arms (docs/debugging/lr-warmup.md); dE as in scatter_conservation_summarize.derived."""
import sys

import numpy as np

from scripts.scatter_conservation_summarize import derived

d = sys.argv[1]
for reg, n in (("nr100", 100), ("bh300", 300)):
    for a in ("ctl", "w1", "w2"):
        de, dl = [], []
        for r in (1, 2, 3):
            z = np.load(f"{d}/{a}_r{r}/{reg}.npz")
            x = derived(z, "m", n)
            i = int(np.argmin(np.abs(z["ticks"] - 1000)))
            de.append(float(np.median(np.abs(x["dE"][i])))); dl.append(float(np.median(np.abs(x["dL"][i]))))
        print(reg, a, "dE@1000 per eval", [round(v, 4) for v in de], "median", round(float(np.median(de)), 4), "| dL median", round(float(np.median(dl)), 4))
