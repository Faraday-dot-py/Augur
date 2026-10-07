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
Job 3378 (H200, 3h47m, 21:11-00:59), results `results/scatter_s46/`, log `results/scatter_s46/job3378.log`. Harness ratios = run/E_base err@5/10/20 (gate 15%). Full tables: `python scripts/scatter_s46_summary.py NAME`.

Anomalies: (1) the E_base row was skipped by the wrapper (`E_base_regress.json` existed from job 3373), so E_base OOD rows (`ood:ref,mass_ratio_2body`) were never run; baselines come from the old `results/scatter_ood.json`. (2) `largen` `n46big` crashed at cold_100000 (scatter_scaling.py:159, TICKS index 50 with a 20-tick run); its n46big file only holds N<=31623 duplicates, no N=1e5 row. (3) The baseline is a different number from E_base in battery (see E_base row below: ref 0.028, ctrl 0.033). No divergence or non-finite rows in any run.

| run | harness ratio (worst of 9) | gate | key stress / battery vs prediction |
|---|---|---|---|
| E_base | 1.00 | PASS | battery ref .028, geomean ratio 11.4 |
| ctrl | 1.03-1.15 | PASS | noise floor: OOD rows within ~10-25% (binary_q100 7.0->24.8, shift24 .43->.51, plummer a.25 3.4->3.75) |
| cu (4.2) | 1.30-1.83 | FAIL | G64 err@20 1.12->0.145, G256 1.93->0.105 (pred <0.3, HIT), G512 7.9->1.48 (pred >1, hit), G32 1.79->1.2 (pred >1, hit); h=.5 harness +30-83% (pred +5-10%, MISS); filament battery .31->.97, q10 .37->1.16 worse |
| largen (4.3a) | 1.68-1.94 | FAIL | cold_1e4 err@5 1.77->1.42 (pred <0.6, MISS), err@20 42->33 (pred <20, MISS); cold_31623 12.7->19.4 worse; warm_1e4 .34->.23, warm_31623 1.67->.70 improved; N=1e5 not run (crash) |
| mass (6.1a) | 1.15-1.86 | FAIL | q10 .35->.13 (pred <.1, near miss), q100 7.0->2.4 (pred <1.5, MISS), heavy30 1.47->.39 (pred <.3, miss), spectrum .1-10 .21->.156 (pred <.06, MISS), equal m10 1.9->.92 (pred <.5, MISS); in-dist equal_m1 .031->.058 |
| massnorm (6.1b) | 1.21-1.93 | FAIL | zero-shot-style gains: q100 7.0->.78, q1000 216->3.4, heavy1000 694->46, m100 737->17; m3 .119->.154 (pred improve: MISS), spectrum/heavy30 .56 vs mass .39 (pred >= 6.1a: MISS on heavy30/spectrum, better on q>=100 and heavy>=300) |
| masswide (6.1c) | 2.5-3.9 | FAIL (worst) | spectrum .01-100 20.6->2.64 (pred <3, HIT); heavy100 10.6->1.14 (HIT); q1000 216->98 (pred >1, hit); spectrum 0.1-10 .21->.28 and heavy1 .03->.115 degrade |
| eps (6.3) | 1.32-1.88 | FAIL | cluster eps2 .73->.043 (pred <.15, HIT), eps1 .39->.044, binary eps.25 .37->.20 (pred <.3 HIT), eps.1 .51->.23 (HIT); cluster eps.1 .45->.42 (pred <.5 HIT, no real change); eps .5 rows NOT unchanged: .031->.055 (MISS) |
| dense (6.4) | 2.2-2.8 | FAIL | plummer a.25 3.3->1.22 (pred <1.0, near miss), sigma1 1.15->.90 (pred <.4, MISS), sigma2 .40->.41 (pred <.1, MISS), a.5 .60->.51 (pred <.2, MISS); large-sigma/diffuse rows regress 3-10x (sigma32 .088->.85) |

Conclusions: only ctrl passes the gate; every fix regresses the in-distribution harness by 30-280% (ctrl 3-15%), so the harness cost is caused by the fixes, not by extra fine-tuning. Targeted OOD rows mostly improve in the predicted direction (cu resolution sweep, eps, masswide extremes, massnorm large ratios) but magnitudes are often short of the predictions, and mixed-distribution fine-tunes trade away the training regime. Not bundled; no combined run launched.

## Jobs
- 3360 seed copy (CPU-ish), 3361 smoke: equivalence vs E at the operating point OK (cu rel 3.4e-4, eps/mnorm/dnorm exact, mnorm linear in mass 2.99999), backward OK, cell/adapt pair term OK with cu. Smoke exposed eps-group divergence (eps .1/.2, k 20) -> added warmup 200 it, loss cap 3 (skipped steps), k cap 10 for eps<=0.2 groups.
- 3373 E_base harness only (queued behind other agents). 3378 = ONE serial wrapper `scripts/polaris_scatter_s46_all.sh` (12 h limit): E_base (battery + 2 OOD rows), ctrl, cu, largen, mass, massnorm, masswide (init mass), eps, dense; each = 1500 s fine-tune (full-state ckpt every 50 it, resumable) + harness + battery + its stress rows; skips runs whose `results/<name>_regress.json` exists. Estimated 1-1.3 h per run, ~8-10 h total.
- Headline for N scaling is err@5 (decision #4.4); `scatter_s46_summary.py NAME` prints baseline -> run tables.
