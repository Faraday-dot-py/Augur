"""Predicted peak v/c and collapse N for the compact cold start (sigma 0.5, zero velocity), softened gravity eps 0.5, relativistic c=10.

Energy conservation: c^2 (gamma - 1) + Phi = const, so a body that falls through potential drop D reaches gamma = 1 + D/c^2, v/c = sqrt(1 - gamma^-2).
D = k (N - 1), k calibrated on the N=24/25/26 runs (job 3335, peak v/c 0.573/0.618/0.598).
Hoop criterion: core radius <= R_s = 2N/c^2, core radius floor r_c from the same runs.
"""
import numpy as np

C, EPS = 10.0, 0.5
OBS = {24: 0.573, 25: 0.618, 26: 0.598}
R_C = 0.3


def vmax(d):
    g = 1 + d / C ** 2
    return np.sqrt(1 - g ** -2)


def d_of_v(v):
    return C ** 2 * (1 / np.sqrt(1 - v ** 2) - 1)


ks = []
for n, v in OBS.items():
    g = 1 / np.sqrt(1 - v ** 2)
    ks.append(C ** 2 * (g - 1) / (n - 1))
k = float(np.mean(ks))
print(f"calibrated k = {k:.3f} (per-N: {[round(x, 3) for x in ks]})")
print(f"hoop N (r_c {R_C}): {R_C * C ** 2 / 2:.1f}")
for vt in (0.7, 0.8, 0.9, 0.95, 0.99):
    print(f"v/c >= {vt}: N = {1 + d_of_v(vt) / k:.0f}")
print("N, R_s, predicted peak v/c")
for n in (5, 10, 15, 20, 25, 50, 100, 130, 200, 300):
    print(n, round(2 * n / C ** 2, 2), round(float(vmax(k * (n - 1))), 3))
