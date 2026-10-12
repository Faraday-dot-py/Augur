# Staged closed-loop relabel curriculum (scatter field, Exp E line)

Branch `worktree-agent-a9510055ea4ddb35d` (master d1607f9 + this). Code: `scripts/train_scatter_cl.py` (trainer, exact batched fp64 sim, relabel, gate), `scripts/eval_scatter_cl.py` (past-horizon eval), `scripts/polaris_scatter_cl.sh`, `tests/test_scatter_cl.py`. Note: `docs/debugging/scatter-correction-research.md` (candidate #3) does not exist in any branch/worktree; the design is taken from the task statement only.

## Design (as implemented)
- Frozen base: Exp E `E_ms_kp_pot_v_g128.pt` (variant ms_kp_pot_v_g128). One change vs control: the source of the training windows.
- Stage offsets 0, 5, 10, 20. Stage 0 = ordinary truth window (random t0 in the 30-tick truth data; identical RNG use and loss to `tsf.train`, test-verified bit-equal params after 3 iterations). Stage s>0: truth IC (frame 0 of a training scene) -> current model rolled s ticks (no grad, detached; recurrent field and Verlet a_prev carried) -> exact fp64 sim (KDK, 4 substeps, G 1, eps .5, masses/mask supported; matches gs.rollout_torch/sf.rollout_mass_torch <1e-8, tested) k ticks from that state = targets. 25% of batches at stages>0 are stage-0 truth batches.
- Relabel per iteration on the current model, no cache (the roll-in state moves with every weight update; a cache is stale after a few hundred its). Cost: roll-in (<= 20 no-grad ticks) + exact sim (k x 4 substeps, B 32, N<=300) per batch, all on GPU.
- Perturbation: one value, gaussian sigma 1e-3 on pos and vel of the roll-in state (zero-mean per scene), a_prev reset (recomputed). Chosen 1e-3: far under the cell/softening scale (0.5) and below the model's own err@5 (3e-3), far above fp32 noise, so it regularises without competing with the drift being corrected; no sweep. Relabel is taken from the perturbed state so targets stay consistent.
- Ablation (if cheap): `--reset-field` carries zero field instead of the roll-in state.
- Loss/hparams (same for control and relabel): pos MSE + 0.1 vel MSE + mass term, momfix, grad clip 1.0, loss cap 3 / skip non-finite, Adam lr 2e-4 cosine to 5% over all iterations, batch 32, k 4 -> 20 over the global fraction, 16000 its, seed 4738, default train opt stack (cl, graphtrain), 2000 train scenes (Exp E mix incl. clusters), eval sets 48 scenes x seeds 9000/12000.
- Gate: eval every 500 its on relabelled err@20 at the current offset (mean over seeds 9000/12000; truth-IC err@20 and the other offsets logged alongside). Advance when it improved < 2% (vs the best so far in the stage) for 2 consecutive evals after >= 1000 its in the stage, or at 4000 its in the stage. Decisions saved in the ckpt and the result JSON. Full-state checkpoint (model, Adam, it, stage, gate state) at every eval; resumable.
- Control: `--mode truth` = same loop, every batch a truth window.

## Evaluation
`scatter_regress.py run/compare` vs `baseline_v2_regress.json` (err@5/10/20, bound/virial, cons rows nr100/bh300 at 1000 ticks incl. dE, timing) and relabel-vs-control compare; `eval_scatter_cl.py` (ticks 20-100: err, per-channel pos/vel rms, field-channel rms/max, model vs truth |a| rms/max (a0), first non-finite tick, relabelled err@20 per offset).

## Predictions (before any result)
1. Relabelled err@20 at offset 20 (closed-loop one-window error) falls >= 30% vs base and vs control; control improves it < 15%.
2. Truth-IC err@20 (expE seeds 9000/12000): relabel within +-15% of control (P .6); worse than control by > 15% (P .25).
3. Past the horizon (ticks 20-100): relabel err@50/100 lower than control by 10-30% (P .45), field rms and |a| rms stay within 2x truth to tick 100 for both; no non-finite tick up to 100 in either (P .9).
4. Harness gate (err@5/10/20 <= 1.15x baseline): control passes (P .7), relabel passes (P .5; bh300 and two_body err@5 the likely misses).
5. Conservation: both fail the nr100@1000 dE row (earlier finding: any short-unroll fine-tune breaks it, ctrl dE .017 -> .28); P(relabel passes cons rows) .2. The relabel does not touch conservation and trains on longer drifted trajectories, so I expect no improvement in dE.
6. Gate: stage 0 plateaus within 1000-3000 its (the base is converged on truth windows); later stages hit the 4000 cap or plateau at ~2000.

## Results (job 3501, H200, 16000 its each, seed 4738; control 2447 s, relabel 3279 s; 0 skipped; results/scatter_cl/)
Gate decisions (relabel): it 1000 offset 0->5 (plateau; note best was the pre-fine-tune it-0 eval .0383, the first lr-2e-4 evals rose to .047), it 3500 5->10 (plateau), it 5500 10->20 (plateau), then stage 20 to the end. 11193 relabel batches. Gate is noisy: err@20 on the 96 eval scenes jumps +-10% between 500-it evals at lr 2e-4.

err and past-horizon, Exp E eval (seeds 9000/12000 mean, 48 scenes each; chaos/ref floors for E plain/cluster at tick 100 .013/.68; err@100 is model-dominated only for the plain half):

| model | err@5 | err@20 | err@50 | err@100 | field rms@100 | a rms / truth @100 | a max / truth @100 | pos x/y rms@100 | first non-finite | relabelled err@20 offset 0/5/10/20 |
|---|---|---|---|---|---|---|---|---|---|---|
| base E | .0046 | .0383 | .441 | 2.815 | 5.85 | 1.03 | .86 | 3.30/3.39 | none | .0383/.0479/.0632/.1057 |
| control (truth windows) | .0034 | .0298 | .407 | 2.678 | 6.53 | 1.08 | .98 | 3.24/3.33 | none | .0298/.0398/.0572/.1062 |
| relabel curriculum | .0055 | .0461 | .489 | 2.927 | 4.06 | 1.10 | 1.07 | 3.42/3.45 | none | .0461/.0562/.0715/.1158 |

Harness (4 seeds, err@5/10/20, gate x1.15 vs baseline_v2):

| model | two_body @20 | expB @20 | bh300 @20 | expB @100 | bh300 @100 | nr100@1000 dE | bh300@1000 dE | bh300 bound m/t | bh300 vir_gap | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| base | .00910 | .02395 | .0622 | 1.42 | 4.73 | .0170 | .481 | .753/.880 | .037 | ref |
| control | .00571 | .01754 | .0487 | 1.22 | 4.87 | .182 | .434 | .797/.880 | .056 | FAIL (only 2 rows: nr100@1000 dE, bh300 dE_rel@100; all accuracy rows pass) |
| relabel | .00873 | .02841 | .0761 | 1.55 | 5.05 | .0229 | 1.59 | .363/.880 | 1.32 | FAIL (expB@10/20, bh300 err@5/10/20, 12 cons rows) |

Prediction check: (1) MISS - relabel did not reduce its own relabelled err@20 at offset 20 (.1057 -> .1158, control .1062). (2) MISS - relabel truth-IC err@20 +55% vs control (predicted +-15%). (3) MISS - err@50/100 worse than control (+20%/+9%); field/|a| bounded (<=1.1x truth rms), no non-finite to tick 100 (HIT). (4) control passes gate: MISS (cons rows fail, accuracy rows pass); relabel fails accuracy rows too (MISS). (5) HIT for control (nr100 dE .017->.18); relabel kept nr100 dE low (.023) but bh300 conservation collapsed (dE 1.6, bound .36 vs .88). (6) Gate timing HIT-ish (stage 0 plateau at the 1000 floor).
Finding: same-schedule truth-window fine-tuning alone improves accuracy 15-35% (err@20 .0383->.0298, expB@20 -27%, bh300@20 -22%) at the price of 1000-tick energy; the relabel curriculum is strictly worse than that control on accuracy and on bh300 conservation, better only on nr100@1000 dE. Not validated by video review (numeric rollout only, no render made).
