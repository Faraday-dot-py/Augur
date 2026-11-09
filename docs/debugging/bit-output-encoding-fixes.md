# Bit-output encoding and loss fixes (2026-11-08)

Question: the bit-output cell MLP (scripts/bit_force_smoke.py) learns only the top 2-3 bits of the cell's mean delta momentum; can a different loss or encoding teach proper binary, given low bits are near-random noise? Pre-registration: docs/debugging/experiment-log.md, section "2026-11-08 bit-output-encoding-fixes". Job 7192 (slot 2, one GPU, 8x8 grid, 48 particles, 3000 steps, batch 128), raw results results/bit_output_encoding_fixes/results.json, code scripts/bit_output_encoding_fixes.py (uncommitted), summary scripts/summarise_bit_output_encoding_fixes.py.

Job note: Slurm reported exit 3 after 2 min 9 s. All 31 runs had finished and the log ends "ALL DONE"; the failure was the job script's own completion check looking for the log in the wrong directory (fixed in scripts/polaris_bit-output-encoding-fixes.sh). Results are complete; no resubmission was needed.

## Headline (ratio to the float-regression twin, 1.00 = as good as plain regression; force = particle-level relative force error, rollout = position error at tick 20)

Centered input (primary mode), seeds 4738 / 4739 / 4740:

| arm | force ratio | rollout ratio | improves on plain bits (needs more than 0.02) |
| --- | --- | --- | --- |
| plain bits (baseline) | 1.060 / 1.058 / 1.093 | 0.997 / 0.989 / 1.038 | n/a |
| soft decode | 0.911 / 0.900 / 0.936 | 0.898 / 0.886 / 0.922 | yes, all seeds |
| Gray code | 0.911 / 0.908 / 0.929 | 0.887 / 0.881 / 0.902 | yes, all seeds |
| drop low 5 bits | 1.034 / 1.022 / 1.034 | 0.955 / 0.945 / 0.962 | yes, all seeds (small) |
| drop low 4 bits | 1.054 (seed 4738 only) | 0.979 | no (tie goes to baseline) |
| significance-weighted loss | 1.107 (seed 4738 only) | 1.050 | no (worse) |
| mu-law companding | 1.131 (seed 4738 only) | 1.035 | no (worse) |
| coarse-to-fine | not run (trigger not met) | | |

Global-embedding input: plain bits 1.243 / 1.221 / 1.233; soft decode 1.031 / 1.020 / 1.030; Gray 1.076 / 1.057 / 1.081; drop 5 bits 1.188 / 1.182 / 1.182; mu-law 2.109 (PARTIAL, seed 4738 only); drop 4 bits 1.233, significance-weighted 1.278 (no improvement, seed 4738 only).

Absolute, seed 4738 centered: float twin 0.808, plain bits 0.857, soft decode 0.736, Gray 0.736. The oracle cell-mean reference is 0.590 and zero force is 1.0 (rollout: float 0.411, plain bits 0.410, soft 0.369, Gray 0.364, oracle 0.318, zero force 0.563). Both best arms beat the float twin on seed 4738 (0.736 vs 0.808), and sit about 0.15 above the oracle. Baseline numbers reproduce the 2026-10-28 smoke test (0.857 and 0.808 centered, 1.103 and 0.887 global).

Verdicts (pre-registered rule, worse of force and rollout ratio): every arm PASS except mu-law global on seed 4738, PARTIAL (2.1x). No KILLs.

## Bit accuracy per position (x axis, least to most significant, seed 4738 centered)

- plain bits: 0.53 0.53 0.54 0.56 0.65 0.79 0.89 0.92
- Gray code: 0.53 0.54 0.57 0.66 0.80 0.91 0.97 0.92
- drop low 5 bits (3 bits kept): 0.80 0.91 0.94 (these are plain bits 5-7)
- significance-weighted: 0.52 0.52 0.54 0.56 0.64 0.78 0.89 0.92
- mu-law: 0.52 0.52 0.52 0.53 0.55 0.69 0.82 0.91
Soft decode shares weights with plain bits, so its bit accuracy is identical.

## Reading (interpretation marked untested where applicable)

1. Low bits stay at chance under every encoding that keeps them: the noise is in the target, not the loss weighting. Weighting by significance did nothing useful (predicted).
2. Soft decode is a free fix: same weights, expected value from the bit probabilities instead of hard thresholds. It removes the damage hard thresholds do when low bits are coin flips (predicted, and the largest gain in global mode: 1.24 to 1.03).
3. Gray code improved clearly, against my prediction (I predicted no gain). Plausible mechanism, untested: in two's complement, values just below and just above zero (the most common cell values) differ in every bit, whereas in Gray code they differ in one, so the sign-crossing is smooth and the bits above the noise floor become more predictable (sixth bit 0.89 to 0.97). Gray was not tested together with soft decode; combination is an obvious next step.
4. Dropping low bits helps only modestly (5 bits dropped: 0.03 in ratio centered, 0.06 global); dropping 4 did not clear the 0.02 margin.
5. Mu-law companding hurt, especially global (2.1x).
6. Coarse-to-fine was not run: soft decode and Gray both reached a centered ratio below the 1.03 trigger line, so the pre-registered condition for running it was not met. Untested.

## Caveats

- 3000 steps, one small synthetic problem (8x8 grid, 48 particles, static force); not shown to carry to the real model.
- Arms beyond 4738 ran only for arms that met the seed-extension rule; drop-4, weighted and mu-law have a single seed. Seed spread of plain bits alone is about 0.035 in ratio.
- Soft decode and Gray both put the bit model at or better than the float twin on force error; that the float twin is the better-or-equal reference is not guaranteed in general.
- CUDA scatter-add is non-deterministic; reruns will differ slightly.

## Independent review

Fresh reviewer ran (no hypothesis primed; raw json, pre-reg, code). Verdict: every headline ratio and verdict recomputed from results.json matches; soft decode and Gray improve on plain bits on all three seeds in both modes; drop-low-5 also improves (centered gaps 0.026-0.059); coarse-to-fine correctly did not trigger; 36 runs complete, log ends ALL DONE; same data stream, init and eval per arm; Gray round-trips exactly; soft decode exact on hard bits; seed-extension and trigger logic match the pre-reg; no bugs found. Reviewer caveats: (1) soft decode outputs continuous values, so part of its gain may be skipping quantisation rather than better bit handling; (2) drop-bit arms clip the positive tail (max +112 or +96 step units vs +127 for full range), a slight handicap; (3) seed 4738 selection and reporting share the eval set, so seeds 4739 and 4740 are the true held-out check and agree; (4) single training run per cell, plain-bits ratio varies about 0.035 across seeds, so the 0.02 margin is safe only because comparisons are paired within a seed. Missed predictions: Gray (predicted no gain), drop-5 global 1.18 vs predicted 1.12 or better, drop-4 (predicted improves, did not), mu-law global PARTIAL (predicted PASS).
