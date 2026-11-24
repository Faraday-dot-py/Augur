# Archive inventory

Read-only inventory taken 2026-11-23. Nothing was moved, deleted or modified. Tool: `os.walk` over /home/awebb/Research/Augur excluding `.claude` and the git directory, `git ls-tree -r -l` for tracked state, `git check-ignore` for ignored state, `du` for directory totals. The main checkout is on branch stage-1-mass-conditioned-property-channels-and-distribution-control-launch; tracked state below is that branch's tree (1,982 tracked files). Sizes are MiB (1 MiB = 1,048,576 bytes).

## 1. Where the ~33 GB is

`du -sh /home/awebb/Research/Augur` reports 33 GB. It is not all project files:

| part | size | note |
| --- | --- | --- |
| project files outside `.claude` and git (os.walk, 3,376 files) | 19,116 MiB | results, videos, checkpoints, docs, code |
| `.claude/worktrees/` (34 agent worktree directories) | 12 GB | duplicate checkouts plus untracked agent outputs; not project content |
| git directory (`.git`) | 2.1 GB | all 86 branches; see section 5 |

Outside the 33 GB: the sibling worktrees (/home/awebb/Research/Augur-<name>, /home/awebb/wt-nested-harness; roughly 3 GB, derived as du of all non-main worktree directories, 15,478 MiB, minus the 12 GB under .claude/worktrees) and /home/awebb/Research/Augur-backups (3,479 MiB, not examined). Largest single worktree: .claude/worktrees/agent-abd021892286ed5f9 at 1,087 MiB (untracked 1-billion-body clumpy outputs).

## 2. By top-level directory (main checkout, excluding `.claude`, git)

| dir | total MiB | tracked MiB | tracked files | untracked MiB | untracked files |
| --- | --- | --- | --- | --- | --- |
| results | 16,702 | 220 | 733 | 16,482 | 106 |
| videos | 2,330 | 71 | 134 | 2,259 | 234 |
| checkpoints | 48 | 0 | 0 | 48 | 62 |
| docs | 22 | 14 | 209 | 9 | 64 |
| scripts | 6 | 3 | 722 | 3 | 205 |
| web | 5 | 3 | 85 | 2 | 2 |
| tests | 1 | 0 | 51 | 1 | 56 |
| .remember | 1 | 0 | 0 | 1 | 603 |
| model | 1 | 0 | 36 | 0 | 44 |
| (root files) | 0 | 0 | 12 | 0 | 1 |
| __pycache__ | 0 | 0 | 0 | 0 | 3 |
| .pytest_cache | 0 | 0 | 0 | 0 | 6 |
| .superpowers | 0 | 0 | 0 | 0 | 8 |

Totals: tracked 311 MiB in 1982 files on disk; untracked 18,805 MiB in 1394 files. Of the untracked bytes, 12,609 MiB are gitignored (`*.mp4`, `checkpoints/`, listed `results/*.npz` patterns, `.superpowers/`, `__pycache__/`) and 6,196 MiB in 677 files are untracked and not ignored. Tracked files absent from disk: 0.

Untracked and not ignored, by top-level dir: results 6,195 MiB, .remember 1 MiB, videos 0 MiB, .pytest_cache 0 MiB, scripts 0 MiB, docs 0 MiB, (root files) 0 MiB. `.remember/` (596 files) holds local session notes and is not in `.gitignore`; see the secrets report.

### results/ by subdirectory (largest)

| path | MiB |
| --- | --- |
| results/ top-level files (624 files) | 16,564 |
| results/loss_analysis/ | 79 |
| results/conservation/ | 37 |
| results/solar/ | 9 |
| results/scenes/ | 5 |
| results/square_wall/ | 2 |
| results/notebook_assets/ | 1 |
| results/voxel_divergence_diagnosis/ | 1 |
| results/replay_mechanism/ | 1 |
| results/errgrowth/ | 1 |
| results/neuron_seed_robust/ | 1 |

### videos/ and checkpoints/

videos/: 368 files, 2,330 MiB; 233 are .mp4 totalling 2,259 MiB, all gitignored by `*.mp4`; the tracked videos files are images and small items (134 files, 71 MiB). Largest: videos/bh_learned_r5_lensed_dyn.mp4 84 MiB, videos/bh_learned_r5_lensed.mp4 83 MiB, videos/bh_learned_r5.mp4 62 MiB.

checkpoints/: 62 files, 48 MiB, all gitignored (`checkpoints/` in `.gitignore`), all dated 2026-09-21 to 2026-09-30 (early stage and token-model versions). The checkpoints behind the later headline results are not here; they are in /home/awebb/polaris-mcp-files (section 6).

### Root-level logs and json

13 files at the root, 0 MiB total. Logs and json: .mcp.json (220 bytes, untracked), bounce-contact-eval-symlog-3139.log (1,221 bytes, tracked), bounce-contact-eval-v2-3136.log (1,215 bytes, tracked), bounce-contact-gen-symlog-3138.log (1,815 bytes, tracked), bounce-contact-gen-v2-3135.log (1,842 bytes, tracked), decomp_result_4738.json (9,567 bytes, tracked), decomp_result_9000.json (9,600 bytes, tracked).

## 3. By file type (main checkout, excluding `.claude`, git)

| extension | files | MiB |
| --- | --- | --- |
| .npz | 166 | 14,407 |
| .mp4 | 238 | 2,268 |
| .npy | 6 | 2,259 |
| .png | 154 | 76 |
| .pt | 67 | 48 |
| .json | 509 | 35 |
| .gif | 4 | 9 |
| .pyc | 287 | 5 |
| .md | 275 | 3 |
| .py | 563 | 3 |
| .bin | 13 | 2 |
| .log | 281 | 1 |
| .pdf | 2 | 0 |
| .sh | 259 | 0 |

## 4. Files over 50 MB

47 files exceed 50 MiB in the main checkout; 0 of them are tracked. Total 15,700 MiB, which is 82% of the project bytes. These are the raw snapshots and rollouts that the decision keeps private.

| MiB | path | tracked |
| --- | --- | --- |
| 1,737 | results/rollout_clumpy_1m_5k_snaps.npz | no |
| 1,737 | results/rollout_clumpy_1m_5k_mesh_snaps.npz | no |
| 1,339 | results/showcase_bh_1e9_300ticks_kpadext_lens.npy | no |
| 916 | results/orbit_bh_learned_r5_snaps.npz | no |
| 894 | results/showcase_galaxy_collapse_1e9_clumpy_kpadext_lens.npy | no |
| 778 | results/scatter_bh_1b_snaps.npz | no |
| 778 | results/scatter_bh_10b_snaps.npz | no |
| 764 | results/orbit_bh_100k_collapse_1000steps_snaps.npz | no |
| 539 | results/scatter_bh_10b_hist.npz | no |
| 536 | results/showcase_bh_1e10_kpadext_hist.npz | no |
| 383 | results/scatter_bh_1b_hist.npz | no |
| 383 | results/scatter_bh_1b_panels_hist.npz | no |
| 347 | results/flyby_100k.npz | no |
| 320 | results/orbit_bh_100k_collapse_slowmo_snaps.npz | no |
| 275 | results/orbit_bh_learned_interp3_snaps.npz | no |
| 253 | results/orbit_binary_v2_mesh_snaps.npz | no |
| 231 | results/showcase_bh_100k_kpadext_snaps.npz | no |
| 231 | results/scatter_bh_100000_snaps.npz | no |
| 224 | results/orbit_binary_v4_d1200_snaps.npz | no |
| 223 | results/orbit_binary_v5_box_snaps.npz | no |
| 184 | results/tiled_video_10m_3000.npz | no |
| 177 | results/orbit_bh_100k_250steps_snaps.npz | no |
| 176 | results/orbit_bh_100k_250snaps_relativistic_snaps.npz | no |
| 174 | results/flyby_100k_modelonly.npz | no |
| 154 | results/ball_50k_rollout.npz | no |
| 153 | results/ball_rect_5000x1000.npz | no |
| 144 | results/orbit_bh_uncapped_snaps.npz | no |
| 144 | results/tiled_video_10m.npz | no |
| 138 | results/orbit_bh_capped_snaps.npz | no |
| 115 | results/orbit_bh_100k_first20_8substeps_snaps.npz | no |
| 109 | results/flyby_10k.npz | no |
| 109 | results/flyby_cic_10k.npz | no |
| 84 | results/orbit_bh_learned_snaps.npz | no |
| 84 | videos/bh_learned_r5_lensed_dyn.mp4 | no |
| 83 | videos/bh_learned_r5_lensed.mp4 | no |
| 80 | results/collision_cic_10k.npz | no |
| 80 | results/collision_10k.npz | no |
| 71 | results/orbit_bh_100k_100steps_snaps.npz | no |
| 67 | results/flyby_100k_100steps.npz | no |
| 66 | results/gravity_1m_far_cic_10k.npz | no |
| 66 | results/gravity_1m_far_10k.npz | no |
| 62 | results/orbit_binary3d_xz_snaps.npz | no |
| 62 | videos/bh_learned_r5.mp4 | no |
| 58 | results/rollout_flyby_mesh1024_snaps.npz | no |
| 58 | results/rollout_flyby_exact_snaps.npz | no |
| 58 | results/rollout_flyby_adaptive_snaps.npz | no |
| 57 | results/orbit_binary3d_snaps.npz | no |

The same file names recur in the worktrees (not counted here), so the 12 GB under `.claude/worktrees/` includes copies and untracked outputs not present in the main checkout.

## 5. Tracked files over 10 MB and git history bloat

Tracked files over 10 MiB on branch stage-1-mass-conditioned-property-channels-and-distribution-control-launch: 7.

| MiB | path |
| --- | --- |
| 25.3 | results/scatter_bh_1b_panels_grids.npy |
| 16.4 | results/ball_1k_fix.npz |
| 15.3 | results/ball_1k_rollout_guard.npz |
| 14.4 | results/conservation/bh1000.npz |
| 11.0 | results/gravity_long_central.npz |
| 10.9 | results/gravity_long_central_far.npz |
| 10.8 | results/gravity_long.npz |

History (all refs, `git rev-list --objects --all`): 12 distinct blobs over 10 MiB, 484 MiB combined. Two are old versions of a file that is now gitignored and untracked (results/ball_50k_rollout.npz, 154 MiB each); the other ten are still tracked or were tracked on some branch.

| MiB | blob | path |
| --- | --- | --- |
| 154.1 | fe18500efc | results/ball_50k_rollout.npz |
| 154.1 | 0692d3a23d | results/ball_50k_rollout.npz |
| 38.2 | 9b4f96c0f6 | results/gdir_central_1k.npz |
| 25.3 | a58deb0848 | results/scatter_bh_1b_panels_grids.npy |
| 22.9 | 3fe06076de | results/gdir_dir45_1k.npz |
| 16.4 | 0dcd68ee55 | results/ball_1k_fix.npz |
| 15.3 | e6201f083e | results/ball_1k_rollout_guard.npz |
| 14.4 | 5267be268c | results/conservation/bh1000.npz |
| 11.0 | 9ac43c434d | results/gravity_long_central.npz |
| 10.9 | 5774ab3005 | results/gravity_long_central_far.npz |
| 10.8 | fdd76b8a2b | results/gravity_long.npz |
| 10.3 | 0f8e1ccd12 | results/fewshot_adapter/solar_held_rollouts.npz |

`git count-objects -vH`:

    count: 6323
    size: 521.07 MiB
    in-pack: 7665
    packs: 1
    size-pack: 524.49 MiB
    prune-packable: 0
    garbage: 1
    size-garbage: 1011.40 MiB

The 2.1 GB git directory is: pack 524 MiB, loose objects 521 MiB, and 1,011 MiB of garbage, which is a single leftover file objects/pack/tmp_pack_Vr1Bbs (1,060,526,731 bytes) from an interrupted pack write. It is not part of any ref; `git gc` would clear it but nothing was run here. Real repository content is about 1 GB (pack plus loose). The blobs over 10 MiB sum to 484 MiB uncompressed, so they are a large share of that. Checked with `git rev-list --objects --remotes`: the two 154 MiB blobs (results/ball_50k_rollout.npz, commits 3dc1bb1 and 3f1e628, 2026-09-27) and two other blobs (results/gdir_central_1k.npz 38.2 MiB, results/gdir_dir45_1k.npz 22.9 MiB) are NOT on origin; the other eight are. The two 154 MiB blobs exceed GitHub's 100 MB per-file limit, so any push of a branch that contains them will be rejected. Ten local branches contain them (interp-architecture, scatter-conservation, scatter-ood, scatter-opt, scatter-regress-harness, scatter-scaling, scatter-scenes, worktree-agent-a2a6d39d354ae38ca, worktree-agent-a3afdebef86797bbc, worktree-cfd-contact-gen); master and the 2026-11 work branches do not. No history rewrite was done or proposed here; see the needs-user list in the triage report.

## 6. Final checkpoints that back headline results

Checkpoints are small (46 distinct .pt files, 97 MiB; 69 files, 148 MiB with duplicates, all under /home/awebb/polaris-mcp-files, none tracked). Byte-identity was checked by md5: for example E_ms_kp_pot_v_g128.pt, law_blind/E_frozen.pt, three_part_swap/E_frozen.pt, stage_one_inputs/E_ms_kp_pot_v_g128.pt, showcase_smoke/E_ms_kp_pot_v_g128.pt and render_ckpts/scatter-gravity-clustered-baseline.pt are the same bytes. They fit one GitHub release asset (limit 2 GB per file).

Mapping basis: file names cited in docs/ and scripts/ (grep) and the evidence-table row text. Rows are in docs/paper/paper-claims-and-evidence-table.md, section letters A to F. Items marked inferred are matched by name or job id only and were not traced to a script argument. The load-bearing files are the 8 groups below; anything not listed is an intermediate or smoke file.

| checkpoint (under /home/awebb/polaris-mcp-files) | size | evidence-table rows it backs (section) | basis |
| --- | --- | --- | --- |
| E_ms_kp_pot_v_g128.pt (Exp E, scatter field; identical bytes as law_blind/E_frozen.pt, three_part_swap/E_frozen.pt, render_ckpts/scatter-gravity-clustered-baseline.pt) | 3.1 MiB | scatter-field-linear-cost-to-ten-billion-bodies (A); scatter-field-force-accuracy-at-one-to-three-million-bodies (A); scatter-field-small-n-gravity-accuracy-at-100k (B); chaos-floor-err20-is-model-dominated (B); grid-edge-artifact-fix-without-retrain (B); learned-vs-plain-particle-mesh-force-at-n1000 (D); the control for rows in E (e.g. three-part-separate-then-joint-training-kill, law-blind-unet-trunk) | docs name it as Exp E and as the baseline checkpoint (architecture-push-results.md, scatter-ood.md, experiment-log.md); md5 identity with E_frozen. Linking each A and B row to this exact file is inferred; the 1e9 and 1e10 runs' checkpoint argument was not opened |
| main_s4739.pt (also stage_one_inputs/main_s4739.pt) | 3.5 MiB | mixed-six-law-retrain-fails-gravity (D; cited as main_s4739 0.1064) | named in the row text |
| cmp_pow1.5_single.pt, cmp_yuk32_single.pt | 3.5 MiB each | law-blind-unet-trunk (E, held-out pow1.5 and yuk32) | named in docs/debugging/architecture-push-prereg/law-blind-unet-trunk.md |
| S1.pt (also e12stage/results/tier0_ext/S1.pt), nv/seed4738.pt, nv/seed4739.pt, augur_tiered_c2_tier0_matched_60k.pt, replay4738.pt (also stage_one_inputs/replay_seed4738.pt) | 3.1 MiB each | tiled-closed-loop-tier-zero-accuracy (C); tiled-closed-loop-tier-zero-energy-drift-fails (C); visited-state-label-replay-cuts-drift (C); upper-tier-hierarchy-adds-nothing-end-to-end (C) | S1, matched_60k and replay4738 are named in scripts and docs; nv/seed4738.pt and seed4739.pt are inferred (the nv directory is not cited in docs) |
| three_part_swap/width8.pt, three_part_swap/E_frozen.pt | 0.7 and 3.1 MiB | three-part-separate-then-joint-training-kill (E); two-number-message-bottleneck-readability (E, width-8 control) | width8 is named in architecture-push-results.md and the two prereg files |
| voxel-it10000.pt, -it15000, -it20000, -it25000, voxel-at-30000-its.pt | 6.5 MiB each | three-dimensional-sparse-voxel-scatter-field (E) | named in scripts/voxel_divergence_diagnosis.py and its prereg |
| job4843/sae_interp_lowdead_dec0_s4738_heldout16_large_dictionaries_sae.pt, job4843/..._dec2_s4739checkpoint_large_dictionaries_sae.pt, job4850/..._512_dictionary_sae.pt (also in results/) | 0.1 MiB each | sae-random-pool-dose-response (D) | file names match the script names scripts/sae_ablation_random_pool_dose_response.py |
| cons_pure_100.pt (also checkpoints/ and e12stage/results/; results/cons_pure.pt) | 0.2 MiB | token-model-contact-energy-conserved (D); token-model-beats-gns-style-baseline (D, row text names cons-pure-100); token-model-linear-cost-to-one-billion-balls (A, inferred) | docs/architectures.md and the class harness row name it; use in the 1e9-ball benchmark is inferred |
| muon5073_ckpt.pt | 2.1 MiB | muon-optimizer-has-no-effect-on-drift (E) | inferred from the file name and job id 5073; not found in docs |

Not mapped to a row: gravity_central_3d, lj_force_v2/v3, the stage1/stage2 and token_model_h*_v* files in /home/awebb/Research/Augur/checkpoints, substep/pairfull0.pt and warm_euclid_4000.pt (the early error-correction work, EL only), far/r1_mlp.pt, r1b_mlp.pt, r2b_tail.pt, render_ckpts (video renders), smoke files. Evidence-table rows with no checkpoint file because the claim is a speed benchmark or an analysis of saved result files: scatter-2d-all-pairs-cost-blowup-diagnosis, shift-equivariance-probe, summary-token-exchange (probe on saved forces), pair-term-neighbour-search-at-three-million-bodies.

Checkpoints for the neighbour-context-pair-branch and staggered-double-grid rows (E) were not found as .pt files under polaris-mcp-files (the jobs' result json files are there: r11101/results.json, arch-push-2026-11-12/staggered_verdict_primary.json); they may remain on Polaris only. Needs-user: confirm whether any final checkpoint exists only on Polaris.

Note: the Pilot's 2026-11-17 release policy said no checkpoints are released; the 2026-11-23 decision releases final checkpoints as a GitHub release asset. This inventory follows the newer decision.
