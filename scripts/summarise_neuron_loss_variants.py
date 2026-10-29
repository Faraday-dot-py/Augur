import json
import sys


def main(paths):
    for p in paths:
        r = json.load(open(p))
        print("==", r["variant"], "params", r["parameters"])
        for n in ("10", "100"):
            for name, v in r["eval"][n].items():
                rs = v["rate_statistics"]
                print(f"  N={n} {name}: err@10 {v['err@10'][0]:.2f} err@20 {v['err@20'][0]:.2f} err@100 {v['err@100'][0]:.2f} F20 {v['spike_f1@20'][0]:.2f} F100 {v['spike_f1@100'][0]:.2f} "
                      f"div {v['divergence_horizon'][0]:.1f} rate-err {v['cell_rate_error_hz'][0]:.1f} | rate100 ratio {rs['100']['ratio']:.2f} slope {rs['100']['slope_pred_on_true']:.2f} corr {rs['100']['correlation']:.2f}"
                      f" | rate300 ratio {rs['300']['ratio']:.2f} slope {rs['300']['slope_pred_on_true']:.2f} corr {rs['300']['correlation']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
