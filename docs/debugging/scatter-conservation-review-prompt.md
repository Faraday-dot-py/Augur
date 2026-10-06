# Scatter-conservation diagnostic review prompt (hypothesis-free)

Render: `python3 scripts/scatter_conservation_summarize.py` writes `videos/scatter_conservation_<regime>_ts.png` and `_snap.png`. Dispatch a fresh agent per pair with:

---
Read the two images `<ts.png>` and `<snap.png>`. The first has 10 small plots, each versus time (log x axis, ticks+1) for one simulated N-body scene set. Each plot has up to four curves: a learned model (red) and reference simulations with differing integration accuracy (black, blue, grey); shaded bands are the spread across scenes. Panel titles name the quantity. The second image shows body positions at several times, top row reference, bottom row model; the red box marks the region the model was designed to cover.

Describe plainly what you see: which quantities the model's curves track the reference on and for how long, where and how they diverge (gradual or sudden, which quantity first), whether the model differs from the reference in kind rather than degree, and what the position snapshots show at late times. Do not assume a cause. Short paragraph per image pair.
---
