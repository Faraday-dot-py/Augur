# Far-kernel generalisation across interaction laws (scatter-field, Exp E)

Branch `far-kernel-generalization` (from `far-kernel-fix` 2d6d8f7). Code: `scripts/scatter_far_general.py` (laws, training of arms a/b/c, eval, bake), `scripts/scatter_far.py` (`CondKernel`), summarizers `scripts/scatter_far_general_summarize.py`, `..._wide_summarize.py`, job scripts `scripts/polaris_far_gen/`. Results: `results/far_general_eval.json` (+ `_summary.txt`), `results/far_general_wide.json` (+ `_summary.txt`). Pre-registration is in `docs/debugging/experiment-log.md` (2026-10-21 entry, written before any run).

Definitions. F(r) = signed inward pair force at separation r (positive = attraction); ratio = F_kernel/F_true. Zone = r in [4,512] where |F_true| >= 2% of amp x Newtonian force (all of [4,512] for non-decaying laws); tail = |F| / (amp x Newtonian) outside the zone for r <= 1500 (Yukawa/LJ, must be <= 0.05); interior = max relative deviation from arm (a) for r in [4,32]. Held out of all training: p=1.5, lam=32 (so pow1.5, yuk32, scr32 = screened repulsive lam32, nr2b = prey->pred attractive p1.5). Arm (a) = per-interaction retrained plain MLP (4,417 params, R1b recipe, 30k steps); arm (b) = MLP (3x128) with theta = (a1,p1,a2,p2,1/lam) as input; arm (c) = same net, no theta, input = 8 exact force samples at r in [16,32]. (b),(c) trained 40k steps on shapes {p 1,2,3; Yukawa 8,128; LJ} with random signed amplitude 10^U(-1,1) and gravity slots anchored to the R1b targets; 3 seeds each (4738-4740). Loss is asinh-relative force error plus a sign hinge (an earlier clamp(-5,5) variant stalled at loss ~25 and an unclipped asinh version hit NaN; gradient clipping 1.0 added). GPU: ~12 jobs-min, smoke 3831, train 3842/3843, eval 3849, gravity 3850.

## Kernel-level results (3 seeds per cell; cell = zone ratio min..max; full table in `results/far_general_eval_summary.txt`)
| law | (a) per-interaction | (b) theta input | (c) inferred from r 16-32 |
|---|---|---|---|
| p=1 | .998..1.002 | .50-.61..1.07-1.18 (1 KILL) | KILL 2/3 (ratio -.05, 0) |
| **p=1.5 held** | .998..1.002 | .86-.62-.88 .. 1.22-1.27-1.37 | KILL 3/3 (neg. force, sign 85-100%) |
| p=2 gravity | .933..1.059 | .93..1.04 (PASS 3/3) | .41-.86 .. 1.01-1.05 (1 KILL) |
| p=3 | .64..1.16 (baseline itself fails) | .89-.92-.89 .. 1.23-1.12-1.06 | KILL 3/3 (.06-.16 low) |
| Yukawa 8 | 1.000 | .90-.97 .. 1.02-1.09, tail .02 | .89-.98 .. 1.09-1.15, tail .02 |
| **Yukawa 32 held** | .997..1.003, tail .02 | KILL (.97..3.37), .63..1.77, .55..1.14; interior .18-.34 | KILL 3/3 (up to 12.7x), tail .16-.26 |
| Yukawa 128 | .995..1.004 | .70-.82 .. 1.36-1.07 | KILL 3/3 (up to 27x), tail .26-.58 |
| LJ | .994..1.004 | .98..1.07, .93..1.02, seed 3 collapsed to zero force (KILL) | .87..1.17, tail .02 |
| repulsive p=2 (like charges / pred->prey) | .995..1.004 | .83-.93 .. 1.03-1.21, sign ok | KILL 3/3 (ratio .01-.04 low) |
| prey->pred attractive p3 | .79..1.09 | .82-.94 .. 1.00-1.24 | KILL 3/3 |
| **screened repulsive lam32 held** | .998..1.002 | KILL .14..1.19; 1.00..1.79; KILL .34..1.22 | KILL 3/3 (up to 7.2x) |
| **prey->pred attractive p1.5 held** | .996..1.003 | .65-.81 .. 1.29-1.44 | KILL 3/3 (.04-.20 low) |

Signs were correct on every point in every zone except p=1 / p=1.5 / LJ failures of (c) and (b)-LJ seed 3 (collapse). Repulsive and non-reciprocal channels get the right sign from (b) in all seeds. Interior (r<=32) within 5% of arm (a): (b) fails everywhere except gravity (1%): typical 5-30%.

Mixed-sign direct-sum scene (1500 charges +-1 in a 512 box, relative L1 force error, like repels, r in [4,1500]): arm (a) 0.0003-0.04 (gravity 3.7e-2 is the known R1b +-6%); (b) pow2 .04-.06, pow1.5 held .18-.36, yuk32 held .11-.14, pow3 .02-.06, yuk8 .014-.034; (c) pow2 .13-.17, pow1.5 .17-.69, yuk32 .27-.40. Cancellation between like/unlike pairs amplifies kernel error on held-out laws.

## Verdicts against the pre-registered criteria
- PASS needed every held-out law in >= 2 of 3 seeds inside ratio .9-1.1, tail, sign, interior (5%): neither (b) nor (c) passes any held-out law (best (b): pow1.5 at .86-1.37).
- KILL (ratio outside .5-2 on a held-out law in >= 2/3 seeds): (b) KILL (screened-repulsive lam32: 2/3 seeds; Yukawa 32 1/3 seeds, others partial); (c) KILL (every held-out law, all seeds).
- (b) PARTIAL-KILL reading: interpolation in p works to ~+-30% (pow1.5, nr2b, p=3), is not reliable in lam (screening length, a nonlinear scale in a 5-D theta that the 6 training shapes sample sparsely), and is not stable per seed (LJ seed 3 collapsed; 2/3 seeds are fine). Roughly 3x worse than arm (a) on every law and 10-100x worse than (a) in the tail/interior metrics.
- Prediction check: (b) PASS ~35% -> not met; (c) KILL ~70% -> met. Prediction that (b) interpolates p=1.5 ratio .8-1.2: seed 1 .86-1.22, seeds 2-3 worse (.62-1.37).
- Arm (a) itself: fine for all but p=3 (.64..1.16 near r~1-8; the unsoftened 1/r^3 dynamic range at 30k steps, not retried) and nr1b (.79..1.09, same cause); gravity R1b .933..1.059.

## Gravity end-to-end (gravity-bound kernels; tiled + far, 16 tiles of 64, halo 16, held-out 9000/12000, `results/far_general_wide_summary.txt`)
err@5/10/20: R1b .018/.053/.169; (b) theta gravity s4738/4739/4740 .018/.053/.168, .018/.053/.168, .019/.054/.169; (c) gravity-context (exact context) .019/.054/.170, .018/.054/.172, .018/.053/.169; (c) with 2% noise on the context samples .019/.056/.179, .018/.053/.169, .019/.056/.186. All <= 1.1x R1b (.186) at the limit; noisy-context seeds degrade up to +10%. The 4x4-of-32-cores halo 16 (G=128) rolls (jobs 3850, `results/far_general_t32_{r1b,theta,ctx}.json` on Polaris, logs in the job log) equal the single grid within the usual noise. Conditioned kernels therefore do not hurt gravity when theta is the gravity theta, but they do not extend to other laws.

## Interpretation and limits
- The kernel-level result is that one small MLP with ~6 training law shapes does not generalise to unseen laws even when told the law (b), and does not identify laws from [16,32] samples (c). A scale-free power law such as 1/r^p is nearly unidentifiable from a one-octave window of force values (p and amplitude trade off), and a screened force beyond r~100 is invisible there; (c) needs more than the near-field the model already computes, unless the model also feeds back far-field samples (not permitted by the O(N) near-field-only constraint).
- Soft training only (no exp/positive structure). Caveats: single architecture (3x128 MLP, 40k steps) per arm, six training shapes, no hyperparameter search (rule), 3 seeds; p=3 baseline underfits; LJ treated as a radial law with unsoftened singular core clamped at r>=1. Non-reciprocal laws were treated as two ordered-pair kernels (no coupling of the two near-field channels); the harder non-reciprocity problem lives in the near-field model, not tested here. Interior check against arm (a) is stricter than needed for conditioned kernels because the near field has its own learned force.
- Exp E near-field is gravity-trained; only gravity is run through the harness.

## Both (b) and (c) KILL: near-field retraining NOT launched. Plan for the near-field attempt, for the user's go-ahead
Goal: a model that learns the interaction from data (repulsive, non-reciprocal) so that the far kernel can be fitted per interaction by the cheap R1b recipe (decision (a), 90 s, 4,417 params) or by a conditioned kernel as a prior.
1. Model/data: Exp E variant `E_ms_kp_pot_v_g128` architecture unchanged (G=128 grid, pp=2, pot+velocity heads); train on mixed-interaction scenes: per scene sample one law from the training families (gravity p in {1,2,3}, Yukawa lam in {8,32,128}, LJ, like/unlike charge pairs with random +-1 charges as an extra per-body channel (the model already takes per-body features; charge enters as a second mass-like channel), predator-prey ordered channels with two species), amplitude 10^U(-1,1), truth by fp64 direct sum on N <= 4800 (existing generator + law module `scatter_far_general.force_t`). Hold out p=1.5 and lam=32 (and one non-reciprocal pair) for evaluation.
2. Scenes: reuse Exp E's scene mix (Gaussian warm/cold, two-cluster, uniform) at extent 128 and tile copies, sigma 8-32; 12k scenes per law family, 6 families.
3. Cost estimate: Exp E training took roughly 3-4 H200-hours for 50 epochs on its single-law set (log `augur-trainloss-E-3606.log`); the mixed set is 6x wider, so 50 epochs ~ 12-16 GPU-h split over 2 GPUs (~7 h wall) with --resume, full-state checkpoints each epoch, SIGTERM trap. Memory: same as Exp E (~540 MB single grid, batch 8 scenes -> ~20 GB), fits one 80-GB H200 slice with 2 concurrent runs. Generation of fp64 labels: ~1.5 GPU-h (direct sums N=4800, fp64).
4. Pass/kill (to be recorded before launch): PASS = on held-out p=1.5, lam=32 and the held-out non-reciprocal pair, tick-1 force error vs exact <= 1.5x the same law trained alone (matching the Exp E gravity value .24 within 1.5x) AND err@20 on the 16-tile wide scene <= 2x the single-law-trained model, AND gravity err@20 (.168) not regressed by > 10%; KILL = held-out force error > 3x single-law model or gravity regressed > 25%. Fallback if KILL: per-interaction retraining of the whole model (decision (a) extended) which is already known to work for gravity.
5. Kernel after retrain: refit R1b-recipe per law on the new model's force (2 min each), or re-run (b) with the new near-field as target; not before.
Decision requested: go-ahead for 1-5 (about 12-16 GPU-h), or stay with decision (a) (per-interaction retraining of kernel and, if needed, model).
