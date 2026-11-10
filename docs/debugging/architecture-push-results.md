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
| gaussian-ladder-far-kernel-replacing-fft (6) | stage 1 done, job 7838 (46 s, no training) | err@20 ratio 16.57 vs control (pass <= 1.15, kill > 1.30); tiled 16.7% (pass <= 4%); pow2 exponent error 50.8% (pass <= 10%) | single-law-gravity-checkpoint-g128 (E_frozen, FFT far kernel) | seeds 9100/9200/9300, 48 scenes each, control spread 0.18 | KILL | results/gaussian_ladder_far_kernel_replacing_fft_stage_one/; reviewer PARTLY CONFIRMED | - | 0.01 |
| one-network-for-all-tiers-with-scale-input (8) | stage 1 prepared | - | per-tier weights | - | pending | - | - | 0 |
| neighbour-context-pair-branch-replacing-central-pair-term (9) | stage 1 prepared | - | single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| learned-per-body-property-channels (10) | stage 1 prepared | - | single-law-gravity-checkpoint-g128 | - | pending | - | - | 0 |
| law-blind-unet-trunk (11) | stage 1 prepared | - | main_s4739 | - | pending | - | - | 0 |
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
