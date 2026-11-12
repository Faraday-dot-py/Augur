# bit-output-cell-model-at-64-grid-with-long-range-law: pre-registration (stage 1)

Written before any launch. Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md section 7 item 19, section 10a. Not yet copied to experiment-log.md (supervisor does that before launch).

## Definitions
- Cell model: one shared small MLP (multilayer perceptron) per grid cell; it outputs the cell's mean momentum change per tick (dp, per axis). Every particle in the cell gets dp divided by DT (time step, 0.1).
- Bits: dp is quantised to an 8-bit two's-complement integer per axis (step = scale, scale = 0.999-quantile of |dp| over calibration scenes divided by 127) and the MLP predicts the 16 bits with BCE (binary cross-entropy). Float twin: same MLP, 2 outputs, MSE (mean squared error) on dp/scale.
- F: particle-level relative force error = sum of |model force - exact force| over particles, divided by sum of |exact force|. R: rollout error at tick 20 = mean position error after 20 self-fed ticks against the exact rollout. Zero-force: F 1.0. Oracle: exact cell-mean dp applied to particles (the floor for any cell-mean model).
- Ratio = bits value / float-twin value, each averaged over the three eval seeds first.
- GPU-h: GPU-hours on one Polaris H200.

## Change
Control: the float twin, trained in the same job. The only difference is the output encoding (bits plus soft decode against float regression).
- Grid 64x64 (cell size 1, domain [0,64)^2, reflecting walls as in scripts/bit_force_smoke.py), N = 1024 particles (density 0.25 per cell), 8x8 toy was G = 8, N = 48.
- Long-range law: softened inverse-square, attractive, a_i = -K sum_j d_ij / (|d_ij|^2 + eps^2)^1.5, K = 0.05, eps = 1.0, exact O(N^2) truth. Why: range spans the whole grid (the 8x8 toy force was zero beyond 1.5 cells), and it is the spec's "long-range law" (the gravity of the main model family).
- Coarser copies as extra inputs: deposits on 32x32, 16x16 and 8x8 grids (coarse cell side 2, 4, 8 fine cells). Each deposit has 3 channels: density (count / side^2), and mean offset of particles from the coarse cell centre in coarse-cell units. Each level contributes an 8x8 window of coarse cells centred on the coarse cell that contains the fine cell (zero padded), so the 8x8 level sees the whole grid. The fine level contributes the 8x8 fine-cell window centred on the cell (3 channels: count, mean offset; as the toy "centered" mode). Input size 4 x 192 = 768. The "global" embedding mode is not used at 64x64 (input would be 12288 per cell).
- Encoding: SOFT DECODE (not Gray code). Plain-bit weights are trained, the decoded value is the expectation of the bit probabilities (decode-only, no extra training). Reason: on the 11-08 toy, soft decode had the lower global ratio (1.031 against 1.076 for Gray, three seeds) and costs no extra run; Gray plus soft is untested and not mixed in. Hard-decoded plain bits are reported as a secondary column, not scored.
- Training: 3000 steps, batch 128, Adam lr 1e-3, same for both models (the 8x8 control's step count; same batch stream via generator seed+100, same first-layer init). Calibration 128 scenes; validation set 16 scenes seed 9000 (not an eval seed).

## Lines (copied from spec item 19; no loosening)
PASS: bits-to-float ratio at most 1.5 on force error and on rollout error at tick 20. KILL: either above 3. Seed 4738; seeds 4739 and 4740 only after a pass. Fails pass: 2.0.
Between: INCONCLUSIVE. Concrete value that fails PASS: a ratio of 1.51 on either metric.
Tie-break: a ratio exactly at a line counts as passing PASS (at most) and not as KILL (above). Verdict uses ratios of the three-eval-seed means; per-seed ratios are reported. The scorer applies this mechanically (score() in scripts/bit_output_64_grid_long_range.py).

## Evaluation
Held-out eval seeds 9100/9200/9300 (128 scenes force, 64 scenes x 20 ticks rollout each). Fresh seeds 12000/12100/12200 are evaluated by the same job on the same trained models (second invocation with --fresh) before the result is called real; a PASS counts only if both sets PASS. Train seed 4738 first, 4739 and 4740 only after a pass (full script).
Noise floor: the twin's own spread across the three eval seeds (relative standard deviation of F and R20) is reported in the verdict; a ratio within one such spread of a line is flagged "line-adjacent" in the write-up, the label is unchanged. No extra twin seeds are trained.

## Plateau guard
Validation loss (BCE for bits, MSE for float) every 30 steps. If the mean over the last 10% of iterations (steps 2700-3000) is more than 3% below the mean over the preceding 10% (steps 2400-2700), that model is undertrained; if either bits or twin is undertrained and the label is not PASS, the label becomes INCONCLUSIVE-undertrained (not KILL). One extension at double length for both is then allowed. Control iteration count kept (3000).

## Prediction
Soft-decode bits will stay within the line on force error (F ratio about 1.15, range 1.0-1.5) and on rollout error at tick 20 (R20 ratio about 1.15). Risk: the long-range force is smooth, so most cells hold small dp against a scale set by the edge cells, and the low bits are noise; soft decode should absorb that. Stated chance of PASS: about 65%. KILL (above 3) predicted at under 5%.

## Constraints check
pp=2 knn=16 does not apply: this is the standalone bit-output toy family, no pair term, no scatter field (flagged as a different model family, like experiment 4). No hard-wired conservation: no antisymmetry, no potential output; the net force sum is not enforced (exact truth is momentum conserving, the model is not). Energy is not scored here. O(N) per cell: deposit and window gather are linear in N and in cell count; only the truth is O(N^2), used for data and scoring, not in the model. Coarse inputs are fixed-size windows, cost per cell constant. Requirements: Gen, Til (windows are local; the 8x8 level is global in reach, tiling not tested).

## Already tried (rechecked 2026-11-10)
bit-output encoding fixes (job 7192, 8x8, 48 particles, short-range, seeds 4738-4740): plain bits PASS on three seeds; soft decode centered ratio 0.911/0.900/0.936, global 1.031/1.020/1.030; Gray 0.911/0.908/0.929; Gray plus soft untested. No 64x64 or long-range run exists in experiment-log.md or docs.

## Cost
Spec: 0.3 GPU-h stage 1, 1.5 full. Logged rate: job 7192 trained each 8x8 model (3000 steps, batch 128) in about 4 s on an H200 (results/bit_output_encoding_fixes/results.json, train_seconds). Scaling to 64x64 (4096 cells per scene against 64, 768 inputs against 192, 1024 particles against 48 in an O(N^2) truth): untested estimate 50-100 ms per step, 2.5-5 min per model. Stage 1 (2 models, 2 eval sets, rollouts): estimated 0.25 GPU-h, wall limit 1.5 h. Full (seeds 4739 and 4740, both eval sets): estimated 0.5 GPU-h, limit 2.5 h; spec cap 1.5. All estimates untested; no timing job was run.

## Operations
Code at $HOME/augur_bit_output_64 on Polaris (scripts/bit_force_smoke.py, scripts/bit_output_encoding_fixes.py, scripts/bit_output_64_grid_long_range.py). Job scripts: scripts/polaris_bit-output-cell-model-at-64-grid-with-long-range-law.sh (stage 1) and ...-full.sh (NOT YET AUTHORISED; refuses to run unless the stage 1 labels are all PASS). Both carry `--dependency=afterany:SLOT_TAIL_PLACEHOLDER`: the supervisor replaces the placeholder with the slot tail before sbatch. Checkpoints every 250 steps (model, optimiser, step, batch-stream state, validation history), --resume, SIGTERM saves and exits 143. Results: results/bit_output_64_grid/results.json (runs, val, scale, refs, verdicts). After the job: independent reviewer with the raw file and this pre-registration, rollout-grid diagnostic is not applicable (no rendered output); log RESULT in experiment-log.md.
Toy sanity path (--grid 8 --n 48 --law short --coarse "") reproduces the oracle force error 0.596 (tests/test_bit_output_64_grid.py).
