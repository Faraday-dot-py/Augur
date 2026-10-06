# Collapse threshold prediction (made before the sweep)

Setup: compact cold start (sigma 0.5, v=0), softened gravity eps 0.5, dt 0.1, 4 substeps, relativistic c=10, G=m=1. Script: scripts/predict_collapse_n.py.

Model: conserved c^2(gamma-1)+Phi, so peak gamma = 1 + D/c^2 with D = k(N-1). k = 1.03 calibrated on N=24/25/26 (job 3335, peak v/c 0.573/0.618/0.598). No sharp threshold; N depends on the collapse definition.

| definition | predicted N |
|---|---|
| hoop: core radius <= R_s=2N/c^2 (core floor ~0.3) | ~15 |
| peak v/c >= 0.7 | 40 |
| peak v/c >= 0.8 | 66 |
| peak v/c >= 0.9 | 127 |
| peak v/c >= 0.95 | 215 |
| peak v/c >= 0.99 | 594 |

Predicted peak v/c: N=5 .28, 10 .40, 15 .49, 20 .55, 25 .60, 50 .75, 100 .87, 130 .90, 200 .94, 300 .97.

Expected breakdown: free-fall time ~ 1.57 sqrt(R^3/(2N)) falls below the 0.025 substep for N >~ 100 (R=0.5), and the model grid cell (g128 over 60 units, 0.47) is ~eps, so model/truth agreement and truth energy conservation are doubtful at large N.
