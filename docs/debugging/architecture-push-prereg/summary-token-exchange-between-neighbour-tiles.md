# summary-token-exchange-between-neighbour-tiles: stage 1 predictability measurement (pre-registration)

Written 2026-11-10 by the stage-1 worker, before any run. Stage 1 is the spec's inference-only measurement of how much of the control's force error is predictable from neighbour-tile state. No model is trained except a fixed small MLP fitted on top of the frozen model's rollouts.

Definitions. e = a_model - a_true: per-body force error of the frozen replay checkpoint (tier 0 tiled, halo exchange, job 4089, seed 4738) along its own closed-loop rollout. f: fraction of force-error energy explained, 1 - sum|e - e_hat|^2 / sum|e|^2, by an MLP (2 layers of 64, 30 epochs, Adam lr 1e-3) fitted on rollouts of training scenes (seeds 7100/7200/7300) and scored on held-out scenes (9100/9200/9300), ticks 20-100 every 5. "own" features: the body's own state, its own tile's previous-tick summary. "neighbour" features: previous-tick 17-number summaries of the 8 neighbouring tiles. Delta = f(own+neighbour) - f(own). Delta_net = Delta minus the gain of a shuffled-neighbour control (floored at 0). Oracle: own + true acceleration, must explain at least 0.90 for a KILL to count.

## Spec lines (copied; they govern the later full run)
- Line (audit row 29, with a drift guard added): PASS tiled err@20 at most 0.90x control AND dE@100 at most 1.00x control AND cost at most 1.1x. KILL err@20 at or above 0.97x. Fails pass: 0.94x.
- Default lines (section 6): between PASS and KILL is INCONCLUSIVE. No loosening after launch.

## Stage-1 line (PROPOSED-BY-WORKER; the spec gives no stage-1 line; not calibrated against a trained exchange, supervisor or user to confirm)
- PASS (worth the 3 GPU-h full run): Delta_net at least 0.19 pooled AND at least 0.10 on each held-out seed.
- KILL: Delta_net below 0.06 pooled AND oracle f at least 0.90 (the probe could detect predictable error and found none).
- Else INCONCLUSIVE (including Delta_net below 0.06 with oracle f below 0.90: the probe itself is too weak).
- A value that would still fail PASS: Delta_net 0.15 pooled, or 0.19 pooled with one seed at 0.08.
- Rationale: the full-run PASS needs err@20 down 10%; Delta_net 0.19 is the force-error-energy fraction judged enough to give a 10% rollout error cut under a rough square-root relation (error amplitude scales with the square root of explained energy, 1 - sqrt(1 - 0.19) = 0.10). The relation is an assumption, untested.
- Also reported, no line: n1-only and n2-only gains (raw body numbers versus learned-state numbers), late ticks 60-100, shuffled control gain.

## Run
- Control and checkpoint (Polaris, verified by polaris_ls 2026-11-10): `~/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt`. Seed-4739 replicate (not run, optional): `~/augur_tiered_replay_second_base_fresh_seed_eval/results/tiered_replay_on_second_training_base/tier0-visited-state-label-replay-on-second-base-seed4739.pt`.
- Eval seeds 9100/9200/9300; fresh set 12000/12100/12200 (`--seeds 12000 12100 12200`) before any result is called real. MLP training seeds 7100/7200/7300 (disjoint from every eval seed); MLP initialisation and shuffles seeded 4738.
- Tie-break rule: no argmin or selection; one MLP per variant, fixed hyperparameters, no tuning on eval seeds. PASS thresholds inclusive, KILL strict.
- Numerical-reference noise floor: the shuffled-neighbour control gain and the oracle f measure the probe's own floor and ceiling; Delta_net is already net of the shuffled gain.
- Script: scripts/neighbour_tile_force_error_predictability_probe.py; job: scripts/polaris_summary-token-exchange-between-neighbour-tiles.sh; output `results/neighbour_tile_force_error_predictability_probe/neighbour_tile_force_error_predictability_probe.json`.
- Plateau guard: not applicable to stage 1 (the frozen model is not trained; the MLP runs a fixed 30 epochs). For the full run: INCONCLUSIVE-undertrained if mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window; one extension at double length for candidate and control together.
- Cost: estimate 0.5 GPU-h (spec figure; untimed); time limit 1.5 h.

## Prediction
Halo exchange of the recurrent field already moved tiled err@20 from 1.66-1.82x to 0.75-0.82x BL, so much of the neighbour information is already present in the halo. Delta_net 0.02-0.10; most likely KILL or INCONCLUSIVE. P(PASS) about 20%.

## Constraints check
pp=2 knn=16 untouched (frozen model). No hard-wired conservation: pure error regression, no physics imposed. O(N): constant cost per tile in the later design (17 numbers per tile, 8 neighbours); the probe itself is not a cost test. Tileable: summaries are per tile with tile-local neighbour reads. Already tried: halo overlap and halo exchange of the recurrent field only; no summary-token exchange (grep 2026-11-10).

## Experiment-log entry text
summary-token-exchange-between-neighbour-tiles, stage 1 predictability probe (job <id>, script scripts/neighbour_tile_force_error_predictability_probe.py, commit <hash>). Tried: fixed MLP regression of the frozen replay checkpoint's force error on own-state versus own-plus-8-neighbour-tile previous-tick summaries, shuffled-neighbour control and true-acceleration oracle. Numbers: <f for each variant, Delta_net pooled and per seed, oracle f>. Verdict: <verdict-rule output> against the PROPOSED-BY-WORKER stage-1 line.
