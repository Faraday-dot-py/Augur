# neighbour-context-pair-branch-train-seed-replication: pre-registration (written 2026-11-12, before any run)

Follows neighbour-context-pair-branch-replacing-central-pair-term (stage 1 and fresh-seed eval, train seed 4739 only: err@20 0.803x control on fresh 12000-12200, tangential/radial 0.078). Definitions (base checkpoint ms_kp_pot_v_g128, pair frame, err@k, momentum fix, tangential-to-radial) are those of that prereg; the model and recipe are unchanged.

## Change
None to the model. Same recipe as the 4739 run (`scripts/train_scatter_ft.py`: `--fix ctrl` and `--fix pairctx`, 1500 its, 1000 scenes x 50 ticks, batch 16, k 4->20, lr 2e-4 cosine, loss cap 3, warm start from the base checkpoint (ms_kp_pot_v_g128)), new train seeds 4738 and 4740. Each seed has its own matched control fine-tune (same data and batch stream, stock pair net). Question: is the 4739 gain a property of the candidate or of that seed.

## Lines (user, 2026-11-12; fixed before launch)
- PASS: err@20 ratio (candidate / matched control) at most 0.90 on BOTH new train seeds, on held-out 9100/9200/9300 AND on fresh 12000/12100/12200.
- KILL: err@20 ratio above 0.95 on EITHER new train seed (on either eval set).
- Otherwise INCONCLUSIVE.
- A value that would still fail PASS: ratio 0.93 on one seed (and 0.86 on the other).
- Equality: exactly 0.90 passes; exactly 0.95 is not KILL (above 0.95 kills), so it is INCONCLUSIVE.
- Tie-break: wherever one seed must be shown or reported alone (examples, single-seed tangential figure), the lowest seed index (4738). Seeds are never selected on outcome.
- Plateau guard (as in the stage 1 prereg): a non-PASS label becomes INCONCLUSIVE-undertrained if the candidate's mean validation loss over iterations 1350-1499 is more than 3% below its mean over 1200-1349 (validation: 32 scenes, seed 9000, 20-tick loss, every 37 its); the control's guard is reported too. One extension at double length for candidate and control together is then allowed.
- Reported only, not part of the lines: tangential/radial per seed (stage 1 line was below 0.05), momentum ratio per seed, control err@20 spread across eval seeds (coefficient of variation), frozen base checkpoint reference.
- Mechanical reading: scorer `scripts/pair_context_eval.py --lines seed-replication` gives one label per eval set. Overall label: PASS only if both eval sets are PASS; KILL if either is KILL; else INCONCLUSIVE. The main session applies this combination.

## Prediction (stated before running)
The 4739 result (0.79-0.80x) is probably partly real: the candidate adds velocity and density inputs and a free tangential output, and the control sees the same data. Predicted ratios 0.80-0.92 per seed, about 55% that both land at or below 0.90 on both eval sets, 10% that one exceeds 0.95 (KILL), remainder INCONCLUSIVE. Tangential/radial predicted 0.05-0.10 (as 4739: 0.064-0.078), not part of these lines.

## Constraints check
pp=2 knn=16 kept; dense path only (N <= 512), O(N) claims are as in the stage 1 prereg; no hard-wired conservation (momfix identical in both arms); one change vs matched control per seed (here none beyond the seed); no sweep; train seeds 4738 (start seed) and 4740, eval seeds as above. Seeds: 4738, 4740 train; 9100-9300 and 12000-12200 eval.

## Already tried
Stage 1 train seed 4739: err@20 0.792x held-out, 0.803x fresh, tangential/radial 0.064 / 0.078, INCONCLUSIVE (docs/debugging/architecture-push-results.md row 9). Frozen-base residual with the same inputs: no rollout gain. Checked experiment-log 2026-11-12: no seed replication exists.

## Cost
4 trainings (about 355 s each per job 9277: 711 s per candidate-plus-control pair on H200) = about 1420 s, plus two scoring passes (about 5 min each, 5 arms x 3 seeds), about 0.5 GPU-h total, 1 GPU. No timing job. Limit 1.5 h; every cell checkpoints (full state every 50 its and on SIGTERM), `--resume`, per-cell skip of finished cells, SIGTERM trap, `--requeue` (does not requeue on TIMEOUT; resubmit by hand, finished cells are skipped). Script scripts/polaris_neighbour-context-pair-branch-train-seed-replication.sh; work dir `$HOME/augur_pair_context_replication` (separate from the 4739 results in `$HOME/augur_pair_context`); outputs results/pair_context_replication/results.json and results_fresh.json.
