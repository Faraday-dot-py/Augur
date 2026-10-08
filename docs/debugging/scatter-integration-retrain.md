# Scatter integration retrain (cu + eps + mass norm + all OOD families, one run)

Branch `scatter-integration-retrain` (master 78f6044 + ported 7407b50, 708c235, 7633d4d scripts from `scatter-review-4-6`). User chose a bundled retrain after job 3378 (only ctrl passed the 15% gate when each fix ran alone). Baseline: Exp E `E_ms_kp_pot_v_g128.pt`; gate baseline `results/baseline_v2_regress.json`.

## Design
Variant `ms_kp_pot_v_g128_cu_eps_mnorm` (architectural handling: cell-unit pair/kernel inputs + pp window scaling with h, log(eps/0.5) input to kernel MLP/pair MLP/U-Net, scatter density and kernel acceleration divided by mean mass). Warm start from E via `load_compat` (exact at h=0.5, eps=0.5, unit mass; new inputs zero). `train_scatter_ft.py --fix all`: replay fraction `--replay` (default 0.4) = E-style scenes (0.7 at h 0.5, 0.15 each at G256 and G64 i.e. random-h); remaining 0.6: masses 0.1-10 (25%), 0.01-100 (12%), eps in {0.1,0.2,0.35,1,2} (6% each), dense/small-core + Plummer (20%), large-N 1000-6000 (13%). lr 2e-4 cosine to 5%, warmup 200 it, loss cap 3 (eps<=0.2 groups k<=10, large-N uncapped), unroll k 4->20, seed 4738, full-state checkpoint (model+Adam+iter) every 50 it, resumable. Held-out seeds: harness seeds 9100/9200/9300/4738 differ from training seed offsets (4738+1..+8, +3+eps*100).

## Predictions (before running)
- Gate: P(pass) ~0.4. In job 3378 even cu (40% replay at h=0.5) failed +30-83%, so some cost is from the changed inputs/capacity rather than data share; 40% replay + low lr should land worst-ratio in 1.1-1.4. Most likely failing rows: bh300/chaotic regimes err@10/20 and timing (extra feature columns).
- Per-axis (relative to single fixes, expect 50-80% of the gain): cu G256 err@20 1.93 -> ~0.3, G64 1.12 -> ~0.3; mass q100 7.0 -> ~3, q10 .35 -> ~.15; eps2 .73 -> ~.1, eps1 .39 -> ~.1; dense plummer a.25 3.3 -> ~2; largen cold_1e4 err@5 1.77 -> ~1.5; N=1e5 err@5 stays near E (35) or worse (50-80).
- If the gate fails: diagnose by family via per-group losses and regress rows, one principled adjustment (replay ratio or per-family weights), at most two retries.

## Results (2026-10-07)
Jobs: 3410 pilot 1 (600 s, no loss norm), 3412 pilot 2 (600 s, `--loss-norm`), 3413 full (`--loss-norm`, 9000 s = 75515 iters, 6019 skipped spikes), 3416 ctrl diagnostic. Checkpoint (Polaris): `~/augur_int/checkpoints/scatter_s46/int_full1.pt` (full state: model+Adam+iter). Results: `results/scatter_int/int_full1_*.json`. pytest 230 pass and equivalence smoke OK on Polaris (3410).

Pilot 1 failed the gate (+67..+235%): groups mass0.01-100 and N3000-6000 had losses ~10^3 x base and dominated Adam. Adjustment (one, principled): per-group EMA loss normalisation + spike skip (`--loss-norm`). Pilot 2: worst err@20 +42%. Full run: accuracy rows all PASS (worst +13.8%, bh300 err@5).

Gate (compare vs baseline_v2_regress.json): err@5/10/20 PASS (two_body -8.5..+3.0%, expB +5.7..+12.6%, bh300 +11..+13.8%); timing PASS (+1.7..+3.0%, bh300 peak alloc 33->88 MB info); FAIL rows: cons nr100@1000 dE .017->.287, cons bh300@1000 dE +35%, bh300 dE_rel@100 .039->.090, cons bh300 vir_gap@1000 .037->.271. Diagnostic job 3416: plain ctrl (E-style data only, same optimiser, 3378 checkpoint) fails the same cons rows (nr100 dE .280, bh300 dE +31%), so the dE failures come from any fine-tune, not from the integration; vir_gap bh300 is the only integration-specific miss.

Predictions vs result: gate pass 0.4 (accuracy rows yes, cons rows no); cu G256 err@20 1.93->~0.3 MISS (0.97; single fix .105), G64 1.12->.117 HIT; mass q100 7.0->~3 HIT (0.48 at err@2; single fix 2.4); q10 .35->.07 HIT; eps2 .73->.016 HIT (single .043); eps1 .39->.021; plummer a.25 3.3->5.2 WORSE (single 1.22); largen cold_1e4 err@5 1.77->2.62 WORSE (single 1.42), cold_1e5 err@5 35->32.5 (not worse than E; single fix 80); warm 1e4 .34->.23, warm 31623 1.67->.87 improved.
