# Scatter integration retrain (cu + eps + mass norm + all OOD families, one run)

Branch `scatter-integration-retrain` (master 78f6044 + ported 7407b50, 708c235, 7633d4d scripts from `scatter-review-4-6`). User chose a bundled retrain after job 3378 (only ctrl passed the 15% gate when each fix ran alone). Baseline: Exp E `E_ms_kp_pot_v_g128.pt`; gate baseline `results/baseline_v2_regress.json`.

## Design
Variant `ms_kp_pot_v_g128_cu_eps_mnorm` (architectural handling: cell-unit pair/kernel inputs + pp window scaling with h, log(eps/0.5) input to kernel MLP/pair MLP/U-Net, scatter density and kernel acceleration divided by mean mass). Warm start from E via `load_compat` (exact at h=0.5, eps=0.5, unit mass; new inputs zero). `train_scatter_ft.py --fix all`: replay fraction `--replay` (default 0.4) = E-style scenes (0.7 at h 0.5, 0.15 each at G256 and G64 i.e. random-h); remaining 0.6: masses 0.1-10 (25%), 0.01-100 (12%), eps in {0.1,0.2,0.35,1,2} (6% each), dense/small-core + Plummer (20%), large-N 1000-6000 (13%). lr 2e-4 cosine to 5%, warmup 200 it, loss cap 3 (eps<=0.2 groups k<=10, large-N uncapped), unroll k 4->20, seed 4738, full-state checkpoint (model+Adam+iter) every 50 it, resumable. Held-out seeds: harness seeds 9100/9200/9300/4738 differ from training seed offsets (4738+1..+8, +3+eps*100).

## Predictions (before running)
- Gate: P(pass) ~0.4. In job 3378 even cu (40% replay at h=0.5) failed +30-83%, so some cost is from the changed inputs/capacity rather than data share; 40% replay + low lr should land worst-ratio in 1.1-1.4. Most likely failing rows: bh300/chaotic regimes err@10/20 and timing (extra feature columns).
- Per-axis (relative to single fixes, expect 50-80% of the gain): cu G256 err@20 1.93 -> ~0.3, G64 1.12 -> ~0.3; mass q100 7.0 -> ~3, q10 .35 -> ~.15; eps2 .73 -> ~.1, eps1 .39 -> ~.1; dense plummer a.25 3.3 -> ~2; largen cold_1e4 err@5 1.77 -> ~1.5; N=1e5 err@5 stays near E (35) or worse (50-80).
- If the gate fails: diagnose by family via per-group losses and regress rows, one principled adjustment (replay ratio or per-family weights), at most two retries.
