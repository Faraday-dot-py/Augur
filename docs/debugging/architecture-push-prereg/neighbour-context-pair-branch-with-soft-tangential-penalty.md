# neighbour-context-pair-branch-with-soft-tangential-penalty: pre-registration (written 2026-11-12, before any run)

Follows neighbour-context-pair-branch-replacing-central-pair-term (train seed 4739: err@20 0.803x control on fresh seeds, tangential/radial 0.078 against a pass line below 0.05). Question: does a soft penalty on the tangential pair output cut it below 0.05 without losing the err@20 gain.

## Change
Opt-in `--tan-penalty W` in `scripts/train_scatter_ft.py` (default 0 = off; requires `--fix pairctx`; existing checkpoints and scripts unchanged; model state dict unchanged). When on, each training step adds W times the mean over the k rollout ticks of (sum of squared tangential pair-force components) / (sum of squared radial components, detached), pooled over all pair entries of the batch, to the training loss. This is the square of the pass-line metric (tangential/radial rms), measured on the training rollout states. Soft penalty only: no projection, no zeroing, no hard-wired central or conservative force; the net can still use the tangential output if the error loss pays for it. The radial term is detached so the penalty cannot be lowered by inflating the radial output. Validation loss logged without the penalty (comparable to the control).
Weight W = 0.3, chosen so that at the pass line (ratio 0.05, squared 0.0025) the penalty is about 0.0008 against a validation error loss of about 0.002 (job 9277 candidate, 0.0019-0.0020 at the end), and at the 4739 value (0.078 squared 0.006) it is about 0.0018: comparable to the error loss, neither ignored nor dominant. No sweep; if this fails the weight is not tuned, the result is reported.

## Design
Train seed 4738 (start seed). Candidate: `--fix pairctx --variant ms_kp_pot_v_g128_pairctx --tan-penalty 0.3`. Matched control: `--fix ctrl` stock pair net, same seed (same data and batch stream), same recipe (1500 its, 1000 scenes x 50 ticks, batch 16, k 4->20, lr 2e-4 cosine, loss cap 3, warm start from the base checkpoint (ms_kp_pot_v_g128)). Eval: held-out 9100/9200/9300 and fresh 12000/12100/12200 (32 scenes x 30 ticks per seed); err@20 per arm = mean over the 3 eval seeds; ratio = candidate / control; tangential/radial = rms pooled over pair entries on truth states at ticks 0/10/20.

## Lines (user, 2026-11-12; fixed before launch)
- PASS: tangential/radial below 0.05 AND err@20 ratio at most 0.90.
- KILL: err@20 ratio above 0.95.
- Otherwise INCONCLUSIVE.
- A value that would still fail PASS: tangential/radial 0.06 with err@20 ratio 0.85.
- Equality: ratio exactly 0.90 passes the ratio clause; exactly 0.95 is not KILL; tangential exactly 0.05 fails (must be below).
- Both eval sets are scored; overall label PASS only if both are PASS, KILL if either is KILL, else INCONCLUSIVE. Single seed 4738, so no seed tie-break is needed; if a second seed is later run, lowest seed index is reported first.
- Plateau guard as in the stage 1 prereg (candidate validation last 10% window more than 3% below the preceding window turns a non-PASS into INCONCLUSIVE-undertrained).
- Reported only: momentum ratio, control spread, frozen base checkpoint reference, and the unpenalised 4738 candidate from the replication job (job id in the registry) for the penalty-versus-no-penalty comparison; that comparison is not a line.

## Prediction (stated before running)
Tangential/radial falls to 0.02-0.05 (the penalty acts directly on it, the tangential head started at 0); err@20 ratio 0.82-0.95 (it may lose part of the gain if the tangential output helped). PASS probability about 35%, KILL about 10%, INCONCLUSIVE about 55% (most likely a tangential near 0.05 or a ratio between 0.90 and 0.95).

## Constraints check
pp=2 knn=16 kept; dense path only; no hard projection and no hard-wired conservative force (soft penalty, user rule); one change (penalty) against a matched control; no sweep (one weight); start seed 4738; Polaris job, not local; O(N) unchanged (penalty is a training-time term over the same top-16 pair entries).

## Already tried
Stage 1 and fresh-seed eval of the unpenalised candidate (seed 4739). Replication seeds 4738/4740 queued separately (neighbour-context-pair-branch-train-seed-replication). No tangential penalty run exists (experiment-log checked 2026-11-12).

## Cost
2 trainings (about 355 s each, job 9277 rate; penalty adds no extra forward pass) plus two scoring passes, about 0.3 GPU-h, 1 GPU; limit 1 h (under the 1 h checkpoint rule threshold, but full-state checkpoints every 50 its, `--resume`, SIGTERM trap and `--requeue` are in the script anyway). Script scripts/polaris_neighbour-context-pair-branch-with-soft-tangential-penalty.sh (reads the same code tarball as the replication job); work dir `$HOME/augur_pair_context_tangential_penalty`; outputs results/pair_context_tangential_penalty/results.json and results_fresh.json. Staged, NOT submitted (user-slot cap).
