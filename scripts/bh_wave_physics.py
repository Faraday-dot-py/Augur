"""Energy, virial, angular momentum, spin and shell-radius tables from bh_wave_run npz files. Usage: python scripts/bh_wave_physics.py <data_dir> <scene> [arms comma]"""
import sys

import numpy as np

RLIN = np.arange(0, 48.01, 0.25)
TICKS = [0, 10, 25, 50, 100, 150]


def main():
    d, scene = sys.argv[1:3]
    arms = (sys.argv[3] if len(sys.argv) > 3 else "truth4,truth1,pm128,pm256,pm128s4,model_old,model_new").split(",")
    for a in arms:
        try:
            z = np.load(f"{d}/{scene}_{a}.npz")
        except FileNotFoundError:
            continue
        n = z["rad_lin"][0].sum()
        L, absl, KE, W, pv = z["L"], z["absl"], z["KE"], z["W"], z["pv"]
        print(f"--- {scene} {a} N={int(n)}")
        print("tick  L/L_scale  |l|/N  KE/N  W/N  E/N  2K_nr/|W|(pv)  omega_net(r50)  omega_abs(r50)  r50  p_med  frac_v>.9c  centre_off")
        scale = absl[0] if absl[0] else 1
        for t in TICKS:
            w = W[t]
            e = KE[t] + w if np.isfinite(w) else np.nan
            vir = pv[t] / abs(w) if np.isfinite(w) else np.nan
            print(f"{t:4d} {L[t] / scale:9.2e} {absl[t] / n:9.3g} {KE[t] / n:9.4g} {w / n if np.isfinite(w) else np.nan:9.4g} {e / n if np.isfinite(e) else np.nan:9.4g} {vir:6.2f} "
                  f"{z['omega_net'][t][2]:9.2e} {z['omega_abs'][t][2]:9.2e} {z['rq'][t][2]:6.2f} {z['p_median'][t]:7.1f} {z['frac_v_gt_0.9c'][t]:.3f} {np.linalg.norm(z['centre'][t]):.3f}")
        lo = np.nanmax(np.abs(L - L[0])) / scale
        print(f"max|L-L0|/sum|l|0 = {lo:.3e}")


if __name__ == "__main__":
    main()
