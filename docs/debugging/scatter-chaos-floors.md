# Per-group chaos / reference-noise floors (review decision #2, job 3371)

`scripts/chaos_floors.py` -> `results/chaos_floors.json` (Polaris H200, ~10 min). Checkpoint-independent: depends only on the scene distribution. Reference = float64 exact sim at base substeps (4; bounce 8). Floors: `f32` (float32), `sub_x2` (2x substeps) = **reference-noise floor** (a model within it is indistinguishable from the reference), `pert_1e-05` = **chaos floor** (1e-5 x rms-radius initial position perturbation). Also stored per group: pert 1e-6/1e-4, `sub_d2`, `f32abs`, and dE/E, dP, dL/L, bound-fraction and virial gaps for each variant (see JSON). Re-implementation checked against gs.rollout_torch / sf.rollout_mass_torch / st.bounce_truth (max diff <= 7e-13).

Policy: BH err@100 is not quoted (bh300 sub_x2 floor 1.07 vs model 4.8, chaos ceiling); headline BH at err@50. err@100 only where floor << model (Exp B-like: floor .013 vs model .4-1.4). The 8-substep retrain is dropped.

err floors, sub_x2 (reference-noise) / pert_1e-5 (chaos), at ticks 5, 20, 50, 100:

| group | @5 | @20 | @50 | @100 |
|---|---|---|---|---|
| two_body (harness) | 4.1e-6 / 5.5e-5 | 1.3e-5 / 5.7e-5 | 3.5e-5 / 6.5e-5 | 8.3e-5 / 9.3e-5 |
| expB (harness) | 2.9e-5 / 1.6e-4 | 1.1e-4 / 2.3e-4 | 5.8e-4 / 8.8e-4 | .0132 / .0188 |
| bh300 (harness) | 9.2e-5 / 1.6e-4 | 5.4e-4 / 3.9e-4 | .0151 / .0059 | 1.07 / .539 |
| Exp A | 3.0e-6 / 5.5e-5 | 8.3e-6 / 5.9e-5 | 2.3e-5 / 6.7e-5 | 6.3e-5 / 9.0e-5 |
| Exp B | 3.0e-5 / 1.6e-4 | 1.1e-4 / 2.3e-4 | 6.0e-4 / 8.5e-4 | .0133 / .0168 |
| Exp C (star+planets) | 2.3e-6 / 2.5e-4 | 9.7e-6 / 2.6e-4 | 6.6e-5 / 3.4e-4 | 1.7e-4 / 5.2e-4 |
| Exp D (orbit) | 3.4e-5 / 4.7e-5 | 6.2e-5 / 5.6e-5 | 1.4e-4 / 8.2e-5 | 2.8e-4 / 1.4e-4 |
| Exp E plain | 3.3e-5 / 2.1e-4 | 1.2e-4 / 3.1e-4 | 6.0e-4 / .0012 | .0131 / .0253 |
| Exp E cluster | 7.4e-5 / 1.5e-4 | 6.0e-4 / 3.7e-4 | .0136 / .0053 | .679 / .443 |
| bh n100 | 2.6e-5 / 1.7e-4 | 9.3e-5 / 2.7e-4 | 7.5e-4 / .0013 | .048 / .070 |
| bh n1000 | 2.9e-4 / 1.6e-4 | .0053 / 4.8e-4 | .221 / .031 | 2.98 / 1.7 |
| bh sigma5 | 2.0e-4 / 1.1e-4 | .0038 / 3.4e-4 | .089 / .0124 | 2.05 / 1.09 |
| bh sigma12 | 4.7e-5 / 2.3e-4 | 1.9e-4 / 4.3e-4 | .0019 / .0027 | .201 / .283 |
| bh c5 | 8.5e-5 / 1.6e-4 | 4.1e-4 / 3.5e-4 | .0064 / .0036 | .578 / .352 |
| bh c20 | 9.4e-5 / 1.6e-4 | 6.1e-4 / 4.1e-4 | .0209 / .0086 | 1.12 / .665 |
| binary (1000 ticks) | 6.3e-7 / 4.9e-5 | 2.3e-6 / 4.9e-5 | 5.3e-6 / 5.2e-5 | 1.2e-5 / 6.9e-5 (@1000: 1.6e-4 / 7.6e-4) |
| planetary (1000 ticks) | 1.5e-5 / 1.2e-4 | 5.1e-5 / 1.5e-4 | 2.3e-4 / 3.5e-4 | 5.1e-4 / 6.9e-4 (@1000: .82 / .86) |
| plummer (1000 ticks) | 2.0e-3 / 9.5e-5 | .0193 / 3.6e-4 | .318 / .0073 | 2.47 / .74 (@300+ saturated, ~8) |
| bounce (box physics) | .13 / 9.9e-4 | 4.9 / 2.55 | 5.9 / 5.8 | 6.2 / 6.0 |

Notes
- Bounce floors saturate (~6, decorrelated) past ~20 ticks; horizon kept as is, so bounce err beyond ~tick 10-20 is not informative (the reference-noise floor alone is .13 at tick 5).
- Plummer: sub_x2 floor exceeds the 1e-5 chaos floor 40x at tick 50 (dense cores need finer substeps than the 4 used), so its late-horizon numbers depend on the reference definition; saturated by tick 300.
- bh n1000 and Exp E cluster are also reference-limited by tick 50 (floor .22 / .014).
- Exp A/two_body/binary/D are near-integrable: floors stay <= 1e-3 even at tick 1000 (binary) so model error there is fully model-dominated.
- `compare` prints candidate vs these floors for the three harness regimes (err, dE/E, dP, dL/L, bound/virial gaps); WITHIN = candidate <= reference-noise floor.
