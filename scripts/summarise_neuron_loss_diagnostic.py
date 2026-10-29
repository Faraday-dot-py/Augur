import json
import sys


def r(x, d=3):
    return None if x is None else round(x, d)


def main(path):
    d = json.load(open(path))
    for arm, t in d.get("training", {}).items():
        for stage in ("teacher-forced", "rollout-in-loop"):
            for lab in ("first third", "last third"):
                s = t[stage][lab]
                print(f"TRAIN {arm} {stage} {lab}: loss share {({k: r(v, 2) for k, v in s['loss_share'].items()})} grad-norm share "
                      f"{({k: r(v, 2) for k, v in s['grad_norm_share'].items()})} cos(v,spike) mean {r(s['cos_voltage_spike_mean'])} frac<0 {r(s['cos_voltage_spike_frac_negative'], 2)}")
    for n, e in d["eval"].items():
        print(f"\n=== N={n} regime fractions {({k: r(v, 3) for k, v in e['regime_fractions_all_ticks'].items()})}")
        print("channel errors vs step (voltage mV / adaptation / synaptic / spike mismatch frac / pred spike frac / true spike frac)")
        for arm, ce in e["channel_errors_vs_step"].items():
            print(" ", arm)
            for k in ("1", "5", "10", "20", "50", "100", "300"):
                c = ce[k]
                print(f"    k={k}: {r(c['voltage_rmse_mv'], 2)} / {r(c['adaptation_rmse'], 2)} / {r(c['synaptic_rmse'], 2)} / {r(c['spike_mismatch_frac'], 4)} / {r(c['pred_spike_frac'], 4)} / {r(c['true_spike_frac'], 4)}")
        print("teacher-forced one-step by regime")
        for arm, reg in e["teacher_forced_one_step_by_regime"].items():
            print(" ", arm)
            for nm, v in reg.items():
                print(f"    {nm}: frac {r(v['cell_step_fraction'], 3)} v-rmse {r(v['voltage_rmse_mv_one_step'], 2)} share v-sq {r(v['share_of_voltage_sq_error'], 2)} bce {r(v['bce_per_cell_step'], 3)} share bce {r(v['share_of_bce'], 2)} p_pred {r(v['mean_predicted_spike_prob'], 4)} true {r(v['true_spike_rate'], 4)}")
        print("rollout voltage error by regime (k: regime rmse / share of sq err)")
        for arm in ("full-synaptic-messages", "no-message-ablation"):
            for k in ("1", "20", "100"):
                reg = e["rollout_error_by_regime"][arm][k]
                print(f"  {arm} k={k}: " + "; ".join(f"{nm} {r(v['voltage_rmse_mv'], 1)}/{r(v['share_of_voltage_sq_error'], 2)}" for nm, v in reg.items()))
        print("gradient conflict at trained checkpoint (cosine voltage-MSE vs spike-BCE)")
        for arm, g in e["gradient_conflict_trained_checkpoint"].items():
            print(" ", arm, {k: (r(v["cosine"], 3), r(v["grad_norm_voltage"], 2), r(v["grad_norm_spike"], 2)) for k, v in g.items()})
        print("rate statistics (per cell, horizon 100 / 300): mean true Hz, mean pred Hz, ratio, slope, corr, silent true, silent pred")
        for arm, rs in e["rate_statistics"].items():
            for h in ("100", "300"):
                v = rs[h]
                print(f"  {arm} {h}: {r(v['mean_true_hz'], 1)} {r(v['mean_pred_hz'], 1)} {r(v['ratio'], 2)} {r(v['slope_pred_on_true'], 2)} {r(v['correlation'], 2)} {r(v['silent_cells_true'], 2)} {r(v['silent_cells_pred'], 2)}")


if __name__ == "__main__":
    main(sys.argv[1])
