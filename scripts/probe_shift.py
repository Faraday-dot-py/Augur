import numpy as np

d = np.load("results/gravity_long_central_far.npz")
t, m = d["truth"], d["model"]
print("frame step | truth median(x,y) | model median | truth COM | model COM | truth 1-99 box centre")
for f in range(380, 441, 5):
    lo, hi = np.percentile(t[f], 1, axis=0), np.percentile(t[f], 99, axis=0)
    print(f, f * 20, np.median(t[f], 0).round(2), np.median(m[f], 0).round(2), t[f].mean(0).round(1), m[f].mean(0).round(1), ((lo + hi) / 2).round(1))
