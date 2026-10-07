# Scatter review fixes #4 (scaling) and #6 (OOD): per-fix runs vs frozen Exp E

Branch `scatter-review-4-6` (base master 79833dc). Decisions: `scatter-review-decisions-2026-10-06.md` #4 and #6. Baseline (frozen, never retrained): Exp E `ms_kp_pot_v_g128`, Polaris `~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt` (job 3272); harness numbers `results/scatter_baseline.json` (job 3296); stress rows from the existing scatter-scaling / scatter-ood results (`results/scatter_scaling_*.json`, `results/scatter_ood.json`).

Rule: each fix is its own run (one change vs E), judged by (a) the regression harness (`scatter_regress.py`, gate unchanged) and (b) the stress rows relevant to that fix; a control run (same optimiser/budget, E-style data only) separates "more fine-tuning" from the fix. Combined run only afterwards as an integration check. #6 fix 2 (far-field monopole) is NOT implemented here (built on another branch from #5 fix 2); deferred.

## Implementation (all opt-in, existing checkpoints load unchanged)
- `scatter_field.py`: `ScatterField(cu=, epsin=, norm=)`. `cu`: kernel/pair MLP inputs in cell units (r/h, log(r/h+1), log(h0/h), h0 = 0.5) and the pair/kernel-split window pp scales with h (4 cells). `epsin`: log(eps/0.5) input to kernel MLP, pair MLP and (constant channel) the U-Net; `model.eps_cur` set per scene/batch. `norm="mass"`: scattered density and kernel acceleration divided by the scene mean mass, gathered acceleration multiplied back (linear in source mass, identity for unit masses). `norm="dens"`: same with the scale = max(1, mean of the 64 densest cells / d0), d0 = 8 (guess; trained cold clusters peak near 4-5). `load_compat` warm-starts variants from E (cu is an exact reparametrisation at h = 0.5; new columns/channels zero).
- `train_scatter_ft.py` (+ `scatter_ft_data.py`): fine-tune from E on per-fix scene groups (`ctrl`, `cu` random h in {0.25, 0.5, 1.0} i.e. grid 256/128/64 at extent 64, `largen`, `mass`, `masswide`, `eps`, `dense`); full-state checkpoint (model + Adam + iteration) every 50 iterations, resumable.
- `scatter_regress.py run --battery` adds the compact 15-point OOD battery (decision #6.5) as a non-gating `ood_battery` row (err@2 time units, 2 seeds); `compare` prints it as info.
- `scatter_scaling.py` phases `r46` (resolution sweep G 32..512 at extent 64, N=300) and `n46`/`n46big` (cold+warm N 316..31623, +1e5 at 20 ticks) take `--variant`; `scatter_ood.py --variant`, sets `model.eps_cur` to the scene eps.
- Fine-tune settings (all runs): lr 2e-4 cosine to 5%, 1800 s time budget, batch 16 (large-N groups 2/1), unroll k 4->20 (cap 12/8 for large N), 1500 scenes x 50 ticks per group, seed 4738.

## Predictions (written before the runs)
| run | prediction |
|---|---|
| C0 control | harness inside the 15% gate (extra fine-tuning at low lr, same data), OOD rows unchanged within seed noise (~10-15%). Gives the noise floor for the other runs. |
| 4.2 cu + random h | h=0.5 harness within the gate (maybe +5-10% err). Resolution sweep (N=300, err@20): G256 (h .25) 1.93 -> < 0.3, G64 (h 1) 1.12 -> < 0.3; G512 (h .125, outside training range) and G32 (h 2) stay poor (> 1). N sweep unchanged. |
| 4.3a large-N fine-tune | N=1e4 cold err@5 1.8 -> < 0.6, err@20 42 -> < 20; N<=1e3 and harness unchanged within gate; N=1e5 improves less (core density 25x beyond training). |
| 4.3b N-aware density norm (zero-shot) | harness bh300 within gate (the collapse core may cross d0 and rescale slightly); N=1e4 cold err@5 improves modestly (1.8 -> ~1.0) or not at all; zero-shot normalisation alone is unlikely to beat 4.3a. |
| 6.1a masses 0.1-10 fine-tune | OOD err@2: binary q=10 .35 -> < .1, q=100 7.0 -> < 1.5, heavy30 1.47 -> < .3, spectrum 0.1-10 .21 -> < .06, equal m=10 1.9 -> < .5; harness within gate. |
| 6.1b mass norm | zero-shot equal-mass scale m=3/10 improves (acceleration scaling fixed, dt mismatch remains); spectrum/heavy rows unchanged; fine-tuned variant >= 6.1a on heavy/spectrum rows. |
| 6.1c masses 0.01-100 (init 6.1a) | spectrum .01-100 20.6 -> < 3; heavy100 10.6 -> < 3; q=1000 stays > 1; harness within gate. |
| 6.3 eps input + mixed eps | cluster eps .1 1.42 -> < .5, eps 2 .73 -> < .15, binary d1 eps .1/.25/1 -> < .3; eps .5 rows and harness unchanged. |
| 6.4 dense/small-core fine-tune | sigma 2 .40 -> < .1, sigma 1 1.15 -> < .4, Plummer a=.25 3.3 -> < 1.0 (grid h .5 still limits), a=.5 .60 -> < .2; harness within gate. |

## Results
(filled in per run below)
