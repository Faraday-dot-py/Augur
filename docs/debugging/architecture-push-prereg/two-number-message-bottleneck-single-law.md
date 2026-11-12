# two-number-message-bottleneck-single-law: pre-registration (stage 1)

Status: not run. Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md (experiment 17, section 10a overrides).

## Setup
- Candidate: single-law width-8 model whose k-MLP and pair-MLP pass through a 2-number bottleneck (BOTTLENECK = 2) before the output. Hidden layers l1, l2 and U-Net initialised identically to the control (copied from control-shaped MLPs, same seed).
- Control: job 7256 width 8, `~/augur-single-law-network-sizing/ckpt/width8.pt` on Polaris, training seed 4738.
- Same iteration count as the control (8000 iterations, 50 epochs, 160 iterations per epoch). Candidate and control are never given different budgets.
- pp=2, knn=16, O(N). No hard-wired conservation term; pp=2 is unchanged.
- Eval seeds 9100/9200/9300 (held-out sets). Fresh seeds 12000/12100/12200 before any result is called real. Second training seed 4739 only after a stage-1 pass.

## Lines (copied from spec)
- PASS: linear-fit R^2 >= 0.95 AND exponent within 10% AND err@20 (rollout mean position error at tick 20) <= 1.15x control AND U-Net share of force reported.
- KILL: R^2 < 0.80 or err@20 > 1.30x control.
- Example that fails PASS: R^2 0.90.
- Between PASS and KILL: reported as neither (inconclusive, no verdict).

## Operationalisation (assumptions, flagged)
- R^2 is the pair-force code fit (pair_r2_code, r 0.1-1.9), per the literal spec text. Kernel R^2 and slope over r 4-16 are reported; exponent check is kernel slope relative difference <= 0.10 from the true law.
- err@20 is `rollout_mean_pos_err["20"]`; ratio is candidate over the control's value on the same seeds. The control's own noise floor: ratios are quoted next to the control's spread across 9100/9200/9300; the width-8 control is 0.894x the clean width-32 seeds (job 7442 follow-on).
- Tie-break for any argmin or selection: lowest seed index, then lowest loss; no selection on the eval seeds.

## Plateau guard
INCONCLUSIVE-undertrained if mean validation loss over the last 10% of iterations (epochs 46-50) is more than 3% below the mean over the preceding 10% window (epochs 41-45). One double-length extension allowed, applied to candidate and control together.

## Prediction
A 2-number bottleneck at width 8 is expected to keep err@20 within 1.15x (the pair term is a 1-D radial law) and give pair R^2 around 0.97. Stated before running; expected PASS, so the KILL risk is the err@20 line.

## Cost
Stage 1 est. 0.7-0.8 GPU-h (8000 iterations at about 0.276 s/it from job 7256, plus evals). Spec says 0.5-0.7.

## Needs-user
- Stage-1 line for exponent uses kernel slope; if the user wants the pair slope instead, say so before running.
