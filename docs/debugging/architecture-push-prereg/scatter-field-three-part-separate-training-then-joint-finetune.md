# scatter-field-three-part-separate-training-then-joint-finetune: pre-registration

Written 2026-11-13 by the main session at the Pilot's request, before any code or run. Rating 3. GPU slot: slot 2 only (slot 1 is the voxel extension). Builds on branch scatter-field-three-part-exact-swap-drift-attribution (7863e45): job 9819 FAILED because the three learned parts cancel each other (teacher-forced relative rms vs exact: kernel 0.43-0.57, U-Net 0.4-0.6, pair 0.7-0.97; total only 6-10%), so single exact swaps raise error 12-19x.

## Question
Does training the three parts to match their own exact parts (error correction), then fine-tuning them jointly, give a model whose parts are individually accurate and whose closed-loop error is no worse than the jointly trained control, and does that make exact-swap attribution well posed?

## Checked first
experiment-log.md has no per-part staged training. Nearest: sf_floor, sf_nonet, sf_prior_pp8 (parts removed, zeroed or fixed by retraining, not matched to exact parts); R1 step 1 (the U-Net compensates its own mesh response, 6.5% to 63%); the unsquared-norm fine-tune series (every U-Net fine-tune lost Exp E energy behaviour, gate nr100 dE@1000 <= 1.15x frozen). grep-based, not exhaustive.

## Definitions (reused from the drift-attribution prereg and its scripts, unchanged)
Exact parts per body, fp64, unit masses, eps 0.5, G 1: F exact softened force; K_exact the model's own mesh pipeline with the analytic split kernel; P_exact all pairs with r < pp minus the analytic mesh share; U_exact = F - K_exact - P_exact (per-body vector, oracle at the current state). Code: scripts/scatter_three_part_exact_swap_drift_attribution.py (kernel_exact, exact_parts, learned_parts) on the drift-attribution branch. Parts of the learned model: kernel MLP (kmlp) through the mesh, U-Net potential residual gathered to bodies, pair MLP (ppmlp) over the 16 nearest neighbours (knn truncation counts as pair error). Architecture, parameter count, pp=2, knn=16, O(N) inference cost all unchanged; the labels are used at training time only.

## Arms (all one seed, 4738; same architecture as exp-e-frozen, model `ms_kp_pot_v_g128`)
- joint-continuation-control: fine-tune exp-e-frozen (`~/augur_nfr/kern/E_frozen.pt`) for the stage-2 length with the standard rollout loss only. Controls for extra training.
- separate-parts-only: stage 1 only (no joint stage). Stage 1a: kernel MLP against K_exact and pair MLP against P_exact (U-Net off or frozen at zero residual). Stage 1b: U-Net against U_exact, with the stage-1a learned kernel acceleration as its input (not the exact kernel, so no input shift), kernel and pair frozen. Per-part loss = squared force error divided by the squared rms exact force, along teacher-forced truth trajectories with the recurrent field carried, inside |x| < 32.
- separate-then-joint-with-part-losses: stage 1, then stage 2 joint fine-tune with the standard rollout loss plus the three per-part losses, weight 1.0 on their sum (fixed, not tuned).
- separate-then-joint-rollout-loss-only: stage 1, then stage 2 joint fine-tune with the standard rollout loss only. Shows whether cancellation returns.
Reference for every ratio: exp-e-frozen (E_frozen.pt, 60k-iteration recipe, lr 5e-4), also reported next to the continuation control. Stage-1 and stage-2 iteration counts: set by the agent from logged s/it so that stage 1 + stage 2 does not exceed the 60k-iteration recipe; the same stage-2 length in all three joint arms; recorded in the log before launch.

## Data, evaluation and units
Held-out seeds 9100/9200/9300 and fresh seeds 12000/12100/12200, same scenes and 150-tick truth as the drift-attribution prereg (expB and cluster families, 48 scenes per set); training scenes from seeds disjoint from every eval seed. Per arm: err@20/50/100, dE@100 against truth E(t) at the same tick (never E(0)), teacher-forced relative rms by part, and the 8-arm exact-swap eval re-run on each new checkpoint with the existing scripts (single-swap gains at err@20). Unit check as in the drift prereg must pass (no TF32).

## Pre-registered lines (applied to separate-then-joint-with-part-losses and to separate-then-joint-rollout-loss-only; the other arms are reported, never given a verdict)
Ratio = arm over exp-e-frozen, held-out seeds pooled over both families; fresh seeds must give the same outcome before a verdict is called real.
- PASS: err@20 ratio <= 0.90 AND every part's teacher-forced relative rms <= 0.20 AND all three single-swap gains at err@20 > 0 AND dE@100 ratio <= 1.15 (the heads-full precedent gate).
- KILL: err@20 ratio >= 0.97.
- Else INCONCLUSIVE.
- A value that would still fail PASS: err@20 ratio 0.95; or pair rms 0.50 with everything else passing; or dE@100 ratio 1.30.
- Tie-break and selection: no checkpoint selection, last iteration only; no argmin. Thresholds inclusive for PASS, strict for KILL.
- Floor clause: err@20 is a ratio against the control's own error, not an absolute; the part rms lines are against the analytic exact force in fp64 (no numerical-reference floor); dE is against truth E(t).
- Reported with no line: separate-parts-only and joint-continuation-control values, err@50/100, per-family rows, per-part rms at ticks 1/5/10/20/50/100.

## Prediction (stated before running)
- separate-parts-only: parts accurate (kernel and pair rms <= 0.2, U-Net 0.2-0.4) but err@20 worse than control (1.2-2x), because the mesh discretisation the U-Net used to absorb is now fitted per part; single-swap gains positive (70%).
- separate-then-joint-with-part-losses: err@20 ratio 0.85-1.05, part rms 0.2-0.3. P(PASS) about 20%, P(KILL) about 40%.
- separate-then-joint-rollout-loss-only: part rms climbs back above 0.4 within the joint stage (cancellation returns); err@20 ratio near the continuation control (0.95-1.00).
- joint-continuation-control: err@20 ratio 0.97-1.00 of frozen.

## Constraints check
pp=2, knn=16, O(N) inference: unchanged. No hard-coded conservative forces: labels are training-time supervision, inference is the learned model (soft prior). Core requirement of generalisation: the exact-part labels are gravity-specific, so a PASS here does not show law-agnostic training; flagged as a needs-user item if PASS. Tileable and interpretable: unchanged. Seed 4738 (4739 replicate only after a PASS on 4738, a separate follow-up).

## Compute
Estimate from the logged s/it of the nearest Exp E-recipe run (no timing job) before launch; write it in the log. If the total estimate over all four trainings exceeds 10 GPU-h (proposed limit), stop and return to the main session with the estimate. Slot 2 only, one GPU at a time, chained. Jobs over 1 h need --requeue, full-state checkpoints and the SIGTERM save (--save-on-sigterm); confirm the runner does this before launch. Every dispatched agent log entry goes in docs/debugging/experiment-log.md.
