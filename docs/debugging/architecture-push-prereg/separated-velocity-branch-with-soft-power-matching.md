# separated-velocity-branch-with-soft-power-matching: stage 1 split only (pre-registration)

Written 2026-11-10 by the stage-1 worker, before any run. Per spec section 10a item 3 the power-matching term is NOT run. Stage 1 is an inference-only measurement of the work done by each input group on the frozen replay checkpoint. The term may be proposed later only if this measurement shows the velocity-and-history inputs do the leaking work.

Definitions. err@k: mean position error per body after k self-fed ticks against the exact simulation. dE@k: relative energy drift at tick k against truth E(t) at the same tick, never E(0). v: post-kick velocity of a body. P_err = sum over bodies of m v.(a_model - a_true): the rate at which the model's energy departs from the true energy. dP_g: the part of the model's power sum m v.a_model carried by input group g, estimated by re-evaluating the net at the visited state with group g scaled by lambda (0 = removed, 0.9 = first-order finite difference, divided by 1 - lambda). Groups: velocity, history (recurrent field), velocity+history, kernel-acc, ext+boundary. S_vh = sum dP_vh / sum P_err, signed, cumulative to tick 100, pooled over scenes. Control: the frozen replay checkpoint (tier 0 tiled, halo exchange, job 4089, seed 4738).

## Spec lines (copied)
- Section 7 item 13 (split plus term; does not apply to a split-only run): PASS dE@100 at most 0.75x control AND err@20 at most 1.05x. KILL dE@100 at or above 0.95x or err@20 above 1.15x. Fails pass: dE@100 0.85x. Because this bundles a split and a loss term, the split alone (no term) is run as the ablation arm in the same job.
- Section 10a item 3: the split-only pass/kill line must be written before launch.
- Default lines (section 6): PASS err@20 at most 1.00x control AND target metric meets its line AND cost at most 1.5x. KILL err@20 above 1.15x OR no gain. Between: INCONCLUSIVE. No loosening after launch.

## Split-only lines (PROPOSED-BY-WORKER; the supervisor or user must approve before launch)
Stage 1 (this job, measurement on the frozen model; verdict-rule output of scripts/input_group_work_split_probe.py):
- Harness VOID: the identity sum P_err over sum measured model drift (E_model(t) - E_model(0), same rollout) at tick 100 outside 0.75-1.25.
- PASS (velocity-and-history inputs carry the leaking work; the power-matching term may be proposed): S_vh at least 0.50 at tick 100 for both estimators (zero, fd), AND at least 0.35 on each held-out seed for both.
- KILL (the split cannot isolate the leak; velocity-and-history do not carry it): S_vh below 0.20 for both estimators.
- Else INCONCLUSIVE. A value that would still fail PASS: S_vh 0.45 (pooled), or 0.50 pooled with one seed at 0.30.
- Also reported, no line: all five groups, ticks 20/60/100, non-additivity residual, per-tile-occupancy terciles.
Full split-only run (architecture split, no term, if stage 1 does not close it; derived from the section 7 numbers, with the term removed so no drift reduction is expected from the split alone): PASS err@20 at most 1.05x control AND dE@100 at most 1.00x control (the split costs nothing and the branch work shares are measurable). KILL err@20 above 1.15x OR dE@100 at or above 1.10x. Fails pass: err@20 1.10x. The 0.75x dE@100 line belongs to split plus term and is not claimed by split alone.

## Run
- Control and checkpoint (Polaris, verified by polaris_ls 2026-11-10): `~/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt`. Seed-4739 replicate (not run, optional): `~/augur_tiered_replay_second_base_fresh_seed_eval/results/tiered_replay_on_second_training_base/tier0-visited-state-label-replay-on-second-base-seed4739.pt`.
- Eval seeds 9100/9200/9300; fresh set 12000/12100/12200 (`--seeds 12000 12100 12200`) before any result is called real.
- Tie-break rule: no argmin or selection. PASS thresholds inclusive (at least), KILL strict (below). Ratios use pooled sums over the three seeds.
- Numerical-reference noise floor: the identity check above is the harness's own noise floor (ratio of two independent computations of the same drift, tolerance 25%); the zero and fd estimators must agree in sign for a PASS.
- Script: scripts/input_group_work_split_probe.py; job: scripts/polaris_separated-velocity-branch-with-soft-power-matching.sh; output `results/input_group_work_split_probe/input_group_work_split_probe.json`.
- Plateau guard: not applicable to this probe (no training). For a later full split-only run: INCONCLUSIVE-undertrained if mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window; one extension at double length for candidate and control together.
- Cost: estimate 1.2 GPU-h (untimed: 11 forward passes per tick instead of 1); time limit 1.5 h; spec figure 0.3 GPU-h is probably low.

## Prediction
Velocity and history inputs leak energy (molecular-dynamics mesh drift -0.26/-0.29 with them, at most 0.013 with them zeroed) but removal on gravity was KILL for err@20. S_vh 0.4-0.8 for the zero estimator; fd close to zero. P(PASS) about 40%.

## Constraints check
pp=2 knn=16 untouched (frozen model, inference only). No hard-wired conservation: nothing is imposed; the power term is NOT run here (it is a soft similarity-to-truth term, opt-in, flagged in the spec, and would need the user's call). O(N) and tileable: frozen tiled model with halo exchange as trained. Already tried: input removal (KILL), total-force loss terms (KILL); no split (grep 2026-11-10 found no input-group power measurement).

## Experiment-log entry text
separated-velocity-branch-with-soft-power-matching, stage 1 split only (job <id>, script scripts/input_group_work_split_probe.py, commit <hash>). Tried: input-group power attribution on the frozen replay checkpoint (job 4089); the power-matching term was not run. Numbers: <S_vh zero/fd pooled and per seed at tick 100, identity ratio>. Verdict: <verdict-rule output> against the PROPOSED-BY-WORKER stage-1 line.
