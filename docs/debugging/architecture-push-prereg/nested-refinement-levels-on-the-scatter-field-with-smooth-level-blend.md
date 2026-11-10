# nested-refinement-levels-on-the-scatter-field-with-smooth-level-blend: stage 1 ceiling probe (pre-registration)

Written 2026-11-10 by the stage-1 worker, before any run. Per spec section 10a item 8 only the ceiling probe runs; the full run is a user decision afterward.

Definitions. err@k: mean position error per body after k self-fed ticks against the exact simulation. dE@k: relative energy drift at tick k against the truth's E(t) at the same tick, never E(0). Dense body: a body with at least 20 other bodies within radius 2.0 (R_DENS in scripts/replay_mechanism_force_error_breakdown.py). Dense-bin error: late-tick (60-100) squared force error on dense bodies. Control: the frozen replay checkpoint (tier 0 tiled, halo exchange, visited-state label replay, job 4089, seed 4738). Ceiling: an upper bound on what any refinement tier could remove, obtained by giving dense bodies the exact direct-sum force.

## Spec lines (copied, section 7 item 1; these govern the later full run, not this probe)
- Line: PASS dense-bin error at most 0.70x control AND dE@100 at most 0.75x AND err@20 at most 1.00x AND cost at most 1.5x. KILL dense-bin above 0.90x or err@20 above 1.15x. Fails pass: dense-bin 0.80x.
- Default lines (section 6): PASS err@20 at most 1.00x control AND target metric meets its line AND cost at most 1.5x. KILL err@20 above 1.15x OR target shows no gain. Between: INCONCLUSIVE. No loosening after launch.

## Probe line (PROPOSED-BY-WORKER; the spec has no ceiling line; supervisor to confirm)
Arms (each a full 100-tick closed-loop rollout, 24 scenes per seed): base; dense-exact (exact force for dense bodies, decided every tick on the model's own state); random-exact (exact force for the same number of random bodies per scene and tick); all-exact (harness check). Ratios are dense-exact / base, pooled over scenes and seeds.
- Harness VOID: all-exact err@20 above 0.25x base (the exact-force harness must land near the exact trajectory; its residual is the float32 numerical noise floor of the reference, reported as a ratio to base).
- PASS (ceiling clears the full-run lines with headroom, full run worth proposing): dE@100 at most 0.60x base AND err@20 at most 0.95x base AND dE@100 at least 0.10 below the random-exact ratio (the gain is specific to dense bodies).
- KILL (a perfect refinement could not reach the full-run line, so refinement is not worth building): dE@100 above 0.75x base OR err@20 above 1.00x base.
- Else INCONCLUSIVE. A value that would still fail PASS: dE@100 0.70x. Also reported, no line: late-tick dense-bin and non-dense-bin squared force error of the unreplaced model on the dense-exact arm's visited states (knock-on), per-seed dE@100 ratios, early/late split.
- Reading: the dense-bin 0.70x full-run line is a trained-model line; the probe cannot test it, only bound the drift side. The stage-1 verdict is the verdict-rule output of the script; the supervisor reports it as a ceiling result, never as the experiment's pass/kill.

## Run
- Control and checkpoint (Polaris, verified by polaris_ls 2026-11-10): `~/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt` (3,246,381 bytes). Seed-4739 replicate (not run, optional): `~/augur_tiered_replay_second_base_fresh_seed_eval/results/tiered_replay_on_second_training_base/tier0-visited-state-label-replay-on-second-base-seed4739.pt`.
- Eval seeds 9100/9200/9300 (default). Fresh set 12000/12100/12200 (`--seeds 12000 12100 12200`) before any result is called real.
- Tie-break rule: no argmin or selection over candidates. Dense means count at least 20 (20 is dense). Random-exact uses a fixed generator seeded 4738 plus the scene seed. Line values are inclusive for PASS (at most) and strict for KILL (above).
- Script: scripts/dense_region_exact_force_ceiling_probe.py; job: scripts/polaris_nested-refinement-levels-on-the-scatter-field-with-smooth-level-blend.sh; output `results/dense_region_exact_force_ceiling_probe/dense_region_exact_force_ceiling_probe.json`.
- Plateau guard: not applicable to this probe (no training). For the later full run: INCONCLUSIVE-undertrained if mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window; one extension at double length for candidate and control together.
- Cost: estimate 1.0 GPU-h (untimed, from breakdown jobs at about 0.3 GPU-h per label rollout set); limit 1.5 h.

## Prediction
Densest 10% of bodies hold 12-26% of squared force error (loss analysis 2026-10-26) and drift is signed and dense-region (experiment-log 2026-10-26). dense-exact dE@100 0.55-0.85x base, err@20 0.90-1.00x base. P(PASS) about 35%, P(KILL) about 25%.

## Constraints check
pp=2 knn=16: untouched (the frozen model keeps its pair term; exact force is only a probe substitution, not a model change). No hard-wired conservation: the exact force is a measurement device, nothing is imposed on a model; energy is compared with truth E(t). O(N): not exercised (probe uses O(N^2) exact labels at small N; the full design's refined-cell count per level must be stated before the full run). Tileable: the frozen tiled model runs with halo exchange as trained. Already tried: MultiLevelScatterField (2026-10-01) and far-term-nested-coarse-levels (2026-10-20, far field only); no exact-force dense-region ceiling is in the log (grep 2026-11-10).

## Experiment-log entry text (for the supervisor to paste when the run finishes)
nested-refinement-levels-on-the-scatter-field-with-smooth-level-blend, stage 1 ceiling probe (job <id>, script scripts/dense_region_exact_force_ceiling_probe.py, commit <hash>). Tried: exact direct-sum force for dense bodies (at least 20 neighbours within 2.0) substituted at every tick of closed-loop rollouts of the replay checkpoint (job 4089), plus random-exact and all-exact controls. Numbers: <dE@100, err@20, per-seed ratios vs base; all-exact harness ratio>. Verdict: <verdict-rule output>, ceiling result only; the full run is a user decision.
