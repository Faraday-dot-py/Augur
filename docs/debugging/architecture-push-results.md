# Architecture push: results table (19 experiments)

Spec: `docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md` (section 10a overrides). Supervisor: architecture-push-supervisor. Working branch: architecture-push-ported-trainers (prep commit fde8467). Updated after every result. Experiment 7 dropped (user). Numbers in parentheses after names are the spec cross-reference only.

Columns: stage reached, job ids, control, deciding numbers with file citations, verdict (PASS / KILL / INCONCLUSIVE / INCONCLUSIVE-undertrained / curiosity), reviewer verdict (fresh reviewer ran: yes/no), commit. GPU-h = H200 GPU-hours.

| experiment | stage | jobs | control | deciding numbers (citation) | verdict | reviewer | commit | GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nested-refinement-levels-on-the-scatter-field-with-smooth-level-blend (1) | stage 1 ceiling probe only, then user | - | replay checkpoint | - | pending | - | - | 0 |
| summary-token-exchange-between-neighbour-tiles (2) | stage 1 prepared | - | replay checkpoint | - | pending | - | - | 0 |
| staggered-double-grid-with-quadratic-deposit (3) | stage 1 prepared | - | single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| ball-tree-particle-attention-tier-0 (4) | stage 1 only, branch ball-tree-particle-attention-stage-1 | - | single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| three-dimensional-sparse-voxel-scatter-field (5) | stage 1, then needs-user before full | - | 2D single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| gaussian-ladder-far-kernel-replacing-fft (6) | stage 1 done | 7838 (46 s, no training) | single-law-gravity-checkpoint-g128 (E_frozen, FFT far kernel) | err@20 ratio 16.57 (kill line 1.30); tiled 16.7% (pass 4%); pow2 exponent error 50.8% (pass 10%); results/gaussian_ladder_far_kernel_replacing_fft_stage_one/stage-one.json | KILL | PARTLY CONFIRMED | 2286506 | 0.01 |
| one-network-for-all-tiers-with-scale-input (8) | stage 1 prepared | - | per-tier weights | - | pending | - | - | 0 |
| neighbour-context-pair-branch-replacing-central-pair-term (9) | stage 1 prepared | - | single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| learned-per-body-property-channels (10) | stage 1 prepared | - | single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| law-blind-unet-trunk (11) | stage 1 done | 7839 (cancelled it 71995), 9247 (resume to 80000, 3608 s) | E_frozen (scatter-gravity-clustered-baseline), main_s4739 | gravity err@20 ratio vs E 395.9 (per seed 429, 405, 353; kill line 1.15, pass 1.05); held-out pow1.5 37.8x, yuk32 22.5x vs comparators; validation loss min 0.120 at it 29000 then 3.17 at it 80000 (diverged); law-blind-unet-trunk-stage-one-score.json, scorer scripts/score_architecture_push_stage_one.py:75 | KILL (heads diverged; mechanism untested) | CONFIRMED (fresh reviewer: not an eval, load or resume bug; plateau guard "flat" mislabels a rising loss) | 8fe82a3+ (this commit) | about 5.0 (4:25 + 1:00 H200) |
| law-inferred-from-motion-on-random-law-zoo (12) | stage 1 prepared | - | trunk model or main_s4739 | - | pending | - | - | 0 |
| separated-velocity-branch-with-soft-power-matching (13, split only) | stage 1 prepared | - | replay checkpoint | - | pending | - | - | 0 |
| sampled-force-tier-0-scored-on-distributions-past-the-chaos-horizon (14) | stage 1 prepared | - | single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| rotation-steerable-unet-weights (15, diagnostic only) | stage 1 diagnostic only, then user | - | frozen four-pass symmetrised model | - | pending | - | - | 0 |
| timestep-as-input-model (16) | stage 1 prepared | - | single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| two-number-message-bottleneck-single-law (17) | stage 1 prepared | - | width-8 single-law | - | pending | - | - | 0 |
| sparse-code-unet-single-law (18) | stage 1 prepared | - | width-8 and width-32 single-law | - | pending | - | - | 0 |
| bit-output-cell-model-at-64-grid-with-long-range-law (19) | stage 1 prepared | - | float twin | - | pending | - | - | 0 |
| finite-speed-local-field-medium (20) | curiosity, 1 GPU-h cap | - | none | - | curiosity (no verdict) | - | - | 0 |

## Prep log

- 2026-11-09 prep commit fde8467: tiered eval/arms/summarize scripts and tests/test_scatter_multi.py, test_tiered.py, test_tiered_harness.py ported from origin/near-field-retrain and origin/tier0-tiled-closed-loop; 49 CPU tests pass. scripts/train_scatter_multi.py, scatter_multi.py, scatter_multi_eval.py, train_tiered.py, tiered_iface.py, tiered_adapters.py, tiered_gen.py, tiered_scenes.py, tiered_truth.py, model/tiered/ were already on HEAD; HEAD's train_scatter_multi.py is newer than the near-field-retrain branch version (adds pow1 amp cap, two-body fraction, warm start) and was kept.
- Polaris control checkpoints seen by polaris_ls 2026-11-09: ~/augur_nfr/ckpt/main_s4738.pt and main_s4739.pt (present); ~/augur-single-law-network-sizing/ckpt/width8.pt (present). single-law-gravity-checkpoint-g128, replay (4089) and four-pass symmetrised arm-A checkpoints: inventory worker running.
