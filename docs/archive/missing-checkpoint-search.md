# Missing-checkpoint search: neighbour-context pair branch (job 9277) and staggered double grid (job 9248)

Date 2026-11-23. Read-only search and download on Polaris (no deletes, moves, writes, job submissions or GPU). Local copies: ~/polaris-mcp-files/archive-checkpoints-2026-11-23/.

## Candidates and outcome (10 files looked for in total; rows 1-3 are checkpoints)

| # | Polaris path (relative to home) | Job | Result |
|---|---|---|---|
| 1 | augur_pair_context/ckpt/pairctx-cand-s4739.pt | 9277 | found |
| 2 | augur_pair_context/ckpt/pairctx-ctrl-s4739.pt | 9277 | found |
| 3 | augur_staggered_double_grid/ckpt/single-law-gravity-checkpoint-g128_ms_kp_pot_v_g128_quad_dual.pt | 9248 | found |
| 4 | augur_pair_context/results/pair_context/results.json | 9277 | found |
| 5 | augur_pair_context/results/pair_context/train/pairctx-{cand,ctrl}-s4739_train.json (2 files) | 9277 | found |
| 6 | augur_staggered_double_grid/results/verdict_primary.json, verdict_fresh.json, eval_primary.json, plateau_guard.json (4 files) | 9248 | found |

Paths come from scripts/polaris_neighbour-context-pair-branch-replacing-central-pair-term.sh (W=$HOME/augur_pair_context, CK=ckpt, TRAIN=results/pair_context/train) and scripts/polaris_staggered-double-grid-with-quadratic-deposit-full.sh (W=$HOME/augur_staggered_double_grid, CK=$W/ckpt/${N}_$V.pt). Candidates: 6 checkpoint/result groups, 6 found, 0 not found (10 files downloaded: 3 checkpoints, 7 json).

## Not downloaded

eval_fresh.json (9248) and the 9248 .val.jsonl were not fetched; the fresh verdict json was. results_fresh.json (job 9621, pair context fresh seeds) was not fetched. The staggered TRAIN_DONE marker and trainer log dirs were not checked.

## Files (size in bytes, md5)

| file | bytes | md5 |
|---|---|---|
| pair_context_9277_results.json | 7714 | 3c5d9fbb622e5cfede1ae3c04417e52b |
| pairctx-cand-s4739.pt | 3209530 | 7b68bdea46e26fe5235053613676e70a |
| pairctx-cand-s4739_train.json | 2403 | e22796626be06bcb0c71691a7ef28863 |
| pairctx-ctrl-s4739.pt | 3206458 | 8e4f021a1dce903b4e2812b1fa76680a |
| pairctx-ctrl-s4739_train.json | 2384 | 18715b726e793f25cee157ef1898632b |
| single-law-gravity-checkpoint-g128_ms_kp_pot_v_g128_quad_dual.pt | 3211877 | 5c9dc27858d13fae92c9b97e8b5d586b |
| staggered_9248_eval_primary.json | 2669 | c8aa8948d7f0b7a84cb68687847acfc6 |
| staggered_9248_plateau_guard.json | 153 | c8ace4ce0df1f6d83cb8843024f04f47 |
| staggered_9248_verdict_fresh.json | 389 | 5a3205a8bad33b63c2e6209516eb2157 |
| staggered_9248_verdict_primary.json | 383 | 056741071316b84099fe23ef4041d2b9 |

## Consistency with the registered numbers

- Staggered: verdict_primary.json err@20 0.6893, dE@100 1.0403, probe_half 0.1976, cost 2.578; verdict_fresh.json 0.6901, 1.1802, 0.2084, 2.593; plateau PLATEAUED. Equal to docs/debugging/architecture-push-results.md row for 9248 and the 2026-11-13 review. Both verdicts INCONCLUSIVE.
- Pair context: results.json err@20 ratio 0.7924, momentum ratio 0.8764, tangential/radial 0.0643, verdict INCONCLUSIVE, train seed 4739 only. Equal to the registered row for 9277. Each train json records 1500 iters.

## What could not be checked

- Whether each .pt is the exact file the job wrote: remote timestamps were not listed (the listing tool is unreliable for these paths). The staggered checkpoint directory could have been rewritten by a later job; the match of the registered eval numbers (and 11098 reusing the same checkpoints, per the experiment log) is indirect support only.
- Checkpoint contents: no torch locally. Each .pt is a torch zip whose pickle starts with an OrderedDict under the key "model" (weights only, about 3.2 MB each). Iteration count, optimiser state, and that the staggered file is at iteration 60769 were not verified. Later pair-context jobs 9688 (train seeds 4738, 4740) used other seeds so should not have overwritten the 4739 files; not verified on the remote.
- Checkpoints are evaluation weights, not full optimiser state, as far as the zip pickle header shows.
