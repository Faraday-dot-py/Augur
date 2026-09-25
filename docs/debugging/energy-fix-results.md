# Energy-conservation fixes: results (2026-09-24)

Branch `worktree-agent-a8140833d78c9d71e`. Follows `energy-conservation-investigation.md`. All compute on Polaris GPU
(jobs 2951/2952 fix 1, 2953 training, 2955 eval + ball_1k). Gravity 9 kept; nothing done on gravity direction.

## Fix 1: larger / adaptive pair radius (opt-in `adaptive_radius={off,pair,all}`; fixed r=16/32 via `neighbor_radius`)

Prediction (stated first): removes tunnelling for s>13, does not fix the energy law.
Result (`results/fix1_probe.json`, `scripts/summarize_fix1.py`, soup B, head-on pair, 30 steps): prediction held, and the
mechanism was weaker than thought.
- Head-on offset 0, s=3..20: adaptive/fixed radii give the same numbers as base r=4 for adaptive_pair/all (relKE 0.45/0.19/0.10/0.04/0.08/0.01
  vs truth 1.0; relvy -0.67..+0.27 vs truth -1.0). So at s<=20 the pair edge already existed; the pair head is wrong regardless.
- Momentum error unchanged (4-10 velocity units, truth 0). Fixed r=16/32 made it worse (up to 60) because pair_head sees out-of-range d.
- s>=30: truth itself passes through (relvy +0.8..+1.0): the penalty contact (k=400, 8 substeps) is too soft at relative speed >=60. So "missed
  fast collisions" at s>=30 are partly correct behaviour; base_r4 vs adaptive differ there (adaptive relKE 0.07-0.31 vs base 0.47-0.84, truth 0.65-1.05), not closer to truth.
Verdict: fix 1 does not help soup B. It is unnecessary for fix 2 (edges are rebuilt at contact radius every sub-step).

## Fix 2: conservative contact (`TokenFreeDynamics(conservative_contact=True)`)

Learned distance-only pair force `pen * MLP(pen) * 100` (pen = (2r-d)/2r, applied +/- along the pair line: exact pair momentum, potential exists),
wall force `pen * MLP(pen)` on the four walls (same MLP), learned uniform acceleration (init 0, learned 8.90-9.03 during training), velocity
Verlet with 8 sub-steps per dt, graph rebuilt each sub-step at radius 2r. No velocity/hidden inputs. `contact_residual=True` adds the old
attention/GRU/delta path (not trained here). Training: `scripts/train_conservative.py`, from scratch, 50 epochs (1200 scenes, batch 8; K=1/4/8 unroll for
epochs 0-9/10-29/30-49), scenes on GPU truth (wall, pair, wide, 20x20), speeds capped at 40, seed 4738, ~39 min, checkpoint every epoch
(`results/cons_pure.pt`, `results/cons_pure_train_2953.log`).
Prediction: exact pair momentum, wall restitution ~1 in range, bounded energy; beyond 40 uncertain.

Eval (`scripts/eval_energy_fix.py`, `results/eval_energy_fix.json`, log `eval_energy_fix_2955.log`), held-out seeds 9000 and 12000, state-space rollouts from the
true start (24 scenes n=20 x 4 balls; 6 scenes n=100 x 100 balls), same seeds for both models. Note these numbers start from truth state, not the grid init used for
the earlier err@5/10/20 headline, so they are not comparable with 0.15/0.45/2.5.

| metric | soup B | conservative (pure) | truth |
|---|---|---|---|
| err@5/10/20 (4 balls, n=20) | 0.28 / 0.67 / 3.19 | 0.12 / 0.31 / 2.39 | |
| err@5/10/20 (100 balls, n=100) | 0.24 / 0.49 / 3.27 | 0.11 / 0.30 / 1.90 | |
| E/ball @100 (4 balls) | -113.5 | -81.8 | -77.7 |
| E/ball @100 (100 balls) | -778 | -387.7 | -363.7 |
| y-wall restitution s=3..80 | 1.65, 0.95, 0.81, 0.64, 0.43, 0.41, 0.54, 0.60, 0.59 | 1.02, 1.04, 1.03, 1.02, 1.02, 1.01, 1.01, 1.01, 1.00 | 1.0 |
| x-floor dE s=20/40/60/80 | -67 / -472 / +1296 / -1524 | -0.2 / +8.8 / +38.8 / +26.2 | 3.2 / 0.1 / -9.0 / 21.5 |
| pair head-on relKE s=3/10/20/40/80 | 0.43 / 0.10 / 0.01 / 0.67 / 0.84 | 1.00 / 0.99 / 1.00 / 0.86 / 1.05 | 1.0 / 1.0 / 1.0 / 0.86 / 1.05 |
| pair momentum err (offset 0) | 3.3-26.7 | 0.88 (constant) | 0 |
| ball_1k first non-finite step | 75 | none (500 steps) | |
| ball_1k KE/ball @100 | (stopped at 75; @50 659 vs truth 1171) | 523 | 513 |
| ball_1k KE/ball @500 | | 2237 | 2344 |
| ball_1k err@5/10/20 | 0.19 / 0.39 / 2.14 | 0.11 / 0.28 / 1.38 | |

Comparison with soup B held: better on every listed metric, including speeds 40/60/80 (never trained beyond 40). The 0.88 pair momentum "error" is the
gravity mismatch (learned g_x 8.90 vs 9), not a pair term; pair momentum is exact by construction (`tests/test_token_free.py`). Total-momentum error vs truth
in the rollouts is large for both models (chaos + wall impulses) and is not a conservation metric. Positions decorrelate from truth after ~step 30-50 in
both (err ~7-8 in n=20, ~45 in n=100, 150-165 at 1000 balls): chaotic, expected.
Wall dE at s=40-80 (+9..+39) is small vs soup B (-472..-1524) but nonzero; gravity error contributes.
Training loss plateaued (K=8 ~1.3-1.5, val8 0.59-0.82) because chaotic wide scenes dominate; the contact curves are nonetheless close to truth.

CentralForceDynamics: not comparable as a rollout model (trained for 1/r^2 gravity, no walls, dt 0.1, no contact). It is the design source for this fix
(distance-only force, Verlet); no bounce-domain baseline was run.

## Video review (unbiased subagent, grid `videos/ball_1k_energy_fix_grid.png`, video `videos/ball_1k_energy_fix.mp4`, gitignored)

Reviewer, without hypothesis: all three rows match to step ~20; by step 50 all show a dense bottom band and empty top; by 100+ truth and cons_pure re-expand into
a vertical gradient with a slightly more diffuse cons_pure; soup B is sparser, packed on the floor, does not re-expand. Caveat: soup B panels after step 75 show
its last finite frame (it went non-finite), so the "sparser" late look is partly that hold. No lattice/regular artifacts seen. (Truth itself has 7-30 balls past the wall
at any time: soft walls.)

## Decision

Fix 2 supersedes fix 1 (fix 1 gave no benefit on soup B; conservative contact has no radius problem). Not tested: contact_residual variant, combination with the
attention path, grid-init (observation-derived) start, mass channel.

## Decisions for the user
- Adopt the pure conservative model as the token-model dynamics for free rollout? It replaces attention/GRU (no learned non-contact dynamics), so anything beyond
  gravity + contact + walls (e.g. other forces) needs `contact_residual` and retraining.
- The 8-substep Verlet and 100x force scale mirror the truth integrator/stiffness; say if you consider that too much prior for the paper.
- Gravity is a learned constant (8.90-9.03): a short fine-tune with lower lr would remove the 0.88 offset.
