# Architecture-push control inventory (Polaris, read-only, checked 2026-11-10)

Method: `polaris_ls` on each exact path (size in bytes). Nothing downloaded except the 4089 job log (797 bytes). "Confidence" is HIGH when the file was listed at the stated size, MEDIUM when the path comes from a job script only.

| Control | Remote path | Size (bytes) | Producing log / job | Confidence |
|---|---|---|---|---|
| the frozen single-law gravity checkpoint (ms_kp_pot_v_g128, 60769 it) | ~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt | 3200305 | experiment-log.md the frozen single-law gravity checkpoint; scatter_bh training | HIGH (listed) |
| the frozen single-law gravity checkpoint, copy used by far-field and core-drift jobs | ~/augur_far/kern/E_frozen.pt | 3200305 | polaris_scatter-core-drift-arm-a-fresh-held-out-seeds.sh GATE_CK | HIGH (same size as above; identity by size only, not hashed) |
| the frozen single-law gravity checkpoint init (warm start for retrains) | ~/augur/checkpoints/scatter_bh/init_B_ms_kp_pot_v_g128.pt | 3200305 | same dir | HIGH (listed) |
| Replay checkpoint (job 4089) | ~/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt | 3246381 | job 4089 log (TRAIN_SECONDS 5146, TOTAL_SECONDS 5501, PASS; downloaded to ~/polaris-mcp-files/inv_replay4089_job.log); .score.json, _eval.json, _drift_diagnoses.json alongside | HIGH (listed) |
| Tiled closed-loop base (job 4060 lineage) | ~/augur_tiled_closed_loop/results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo.pt | 3242799 | polaris_replay-mechanism-arms-and-force-error-breakdown.sh CK4060 | MEDIUM (path from script, listed; job id not re-verified) |
| Second-seed base (4739) | ~/augur_tier0_second_training_seed/results/tier0_second_training_seed/tier0-tiled-closed-loop-halo-base-seed4739.pt | 3246563 | same script | MEDIUM |
| Replay on second base (4739) | ~/augur_tiered_replay_second_base_fresh_seed_eval/results/tiered_replay_on_second_training_base/tier0-visited-state-label-replay-on-second-base-seed4739.pt | 3250327 | same script, fresh_seed_eval.json alongside | MEDIUM |
| Four-pass symmetrised arm (symmetrised-potential arm) model | ~/edge_ck/ghost_lsq.pt (DRIFT_CK) | 1063602 | polaris_scatter-core-drift-arm-a-fresh-held-out-seeds.sh; escape_roll.py arms S1 (stock) and S1c4 (four-pass symmetrised) run on this one checkpoint | HIGH for path; the claim that this file IS the arm-A model rests on the two job scripts using it as DRIFT_CK |
| Seed-12300 scene/eval inputs (n=2e4) | ~/augur_armA_fresh/results/armA_fresh/n2e4_s12300/ : truth.npz 38883498, truth.json 380342, S1.npz 38883498, S1c4.npz 38883498, S1.json 363287, S1c4.json 364754, core_ids.npy 16128, drift.txt 144 | as listed | job 6169 (scatter-core-drift-arm-a-fresh-held-out-seeds) | HIGH (listed) |
| Far kernel used by the symmetrised-potential arm | ~/augur_armA_fresh/results/far_kernels/r1b_mlp.pt (also ~/augur_far/kern/r1b_mlp.pt) | 20629 | same | HIGH (listed) |
| Code tree used by arm-A jobs | ~/augur_armA_fresh/{scripts,model,tests}, scripts/escape_roll.py 19295, armA_drift.py 1653, scatter_field.py 56141 | as listed | same | HIGH for presence; SYM_OPS inside scatter_field.py NOT checked (the job script greps for it and aborts loudly) |
| Neuron-graph / nfr controls | ~/augur_nfr/ckpt/main_s4738.pt 3659299; main_s4739.pt 3659299 (also .done markers and .it10000-.it80000 snapshots) | as listed | augur_nfr main runs | HIGH (listed) |
| Single-law sizing | ~/augur-single-law-network-sizing/ckpt/width8.pt 767563; width32.pt 3659073 (also width4 613579, width16 1359809, width64 12736449) | as listed | sizing job 7256 | HIGH (listed) |

## Found, accepted, excluded

Requested: 9 items (the frozen single-law gravity checkpoint, replay 4089, four-pass symmetrised arm checkpoints, seed-12300 inputs, main_s4738, main_s4739, width8, width32, plus arm-A far kernel as a dependency). All located: 10 requested files resolved at exact paths, 0 missing. Extra listed but not used: 4060/4739-base/second-base checkpoints (context only), width4/16/64, edge_ck variants.

## Defects found in the supervisor-facing diagnostics job script (fixed in this branch)

1. scripts/polaris_shift-average-and-core-force-diagnostics.sh set `E_CK=$HOME/E_ms_kp_pot_v_g128.pt`; polaris_ls says that path does not exist (the 4089-era script used it; the file is gone). Fixed to ~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt.
2. The same script untarred `$HOME/armA_fresh_code.tgz`; that tarball does not exist on Polaris. Fixed to symlink code_c to the existing ~/augur_armA_fresh tree (which holds escape_roll.py, armA_drift.py, scatter_field.py, far kernel, seed-12300 inputs). The script copies core_force_component_verdict.py into that tree's scripts/ (one added file).

## Missing / unverified

- No ~/E_ms_kp_pot_v_g128.pt at the home root.
- Home listing truncates at 1000 entries; absence claims are from direct path listings only.
- Hashes not compared (E_frozen.pt vs E_ms_kp_pot_v_g128.pt: same size only).
- Which tarball names exist for new code: none uploaded by this worker. ~/shift_core_diag_code.tgz is not verified present (it must be built and uploaded by the supervisor).
- The four-pass symmetrised arm has no separate checkpoint file: per the two job scripts it is the four-pass symmetrised arm (S1c4) run on ghost_lsq.pt (not verified in scatter_field.py code).
