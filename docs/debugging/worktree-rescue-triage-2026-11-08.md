# Worktree rescue triage, 2026-11-08

Nothing was deleted, renamed, merged, tagged, pushed or committed. Backups are outside the repo.

Locations:
- Files: `/home/awebb/Research/Augur-backups/worktree-rescue-2026-11-08/<worktree>/<relative path>` (63 files, about 979 MB on disk)
- Bundle of all refs: `/home/awebb/Research/Augur-backups/all-refs-2026-11-08.bundle` (647 MB; `git bundle verify` passed: complete history)
- Ref manifest (refname, sha; 223 lines, includes worktree HEAD refs): `/home/awebb/Research/Augur-backups/ref-manifest-2026-11-08.txt`

## Counts

- 22 worktrees have uncommitted or untracked files (the cleanup proposal said 26), 161 files in total, 1,026 MB if all were copied.
- JUNK 98, KEEP 39, UNSURE 24 (the stray cfd log was first marked JUNK, then moved to UNSURE and copied after the reviewer noted it fails the hash rule).
- JUNK rule used: file content hash (git hash-object) is present among all objects reachable from any ref (98), The 98 duplicates are scripts (including the modified tracked ones), one json, one pt and three pngs. No cache or pyc files appear in the dirty lists (they are gitignored).
- Correction to the cleanup proposal: wa-a48419 (80 dirty) holds only 7 files that exist nowhere else. The other 73 are already on some branch. The D1, D2, D5 and D6 scripts and the error-growth scripts are also already on branches; only the D1 to D6 docs and a few scripts are unique (listed below).
- Total size copied 1,026 MB (under the 2 GB limit).

## KEEP (39)

Error-correction docs (each the only copy):
- error-correction-D1-bounded-field.md (wa-ae75e)
- error-correction-D2-contraction.md (wa-abbe1)
- error-correction-D3-recurrent-noise.md (wa-a48419)
- error-correction-D4-nearfield.md (wa-a650)
- error-correction-D5-kend50.md (wa-a528)
- error-correction-D6-shift-jitter.md (wa-a5a0)

Other docs: loss-term-weighting.md (loss-weights), substep-router.md (substep-router), time-estimate-accuracy-review.md (solar-scale-generalized-systems-retrain), scatter-page-update-2026-10-03.md (web-scatter).

Scripts on no ref:
- wa-abbe1: scatter_d2.py, scatter_d2_dbg.py, scatter_d2_eval.py, scatter_d2_regress.py, scatter_d2_test.py, train_scatter_d2.py (D2 contraction), plus a modified scatter_field.py (differs from every committed version)
- wa-a48419: scatter_d3_eval.py, scatter_d3b_eval.py, scatter_noise_probe.py (D3), plus modified scatter_regress.py, train_scatter_field.py, scatter_contact_scenes.py
- wa-a5a0: scatter_d6_shift_jitter.py (D6)
- wa-a650: scatter_nearfield.py (D4)
- wa-a8b09: scatter_tier0.py (the tiered script the 17 branch commits depend on)
- wa-a96914: hetero_ext_checks.py
- wa-a3e1: polaris_nearfield/compare.sh
- wa-a39c: modified polaris_scatter_pp8.sh
- loss-force: modified train_scatter_field.py, scatter_forceaux.py, scatter_forceaux_summarize.py
- rootcause-pairfull: modified scatter_prior_pp8.py
- substep-router: substep_router.py, substep_router_summarize.py

Small result json: d1_bounded_field.json (D1 results, wa-ae75e); substep_router analyze.json, rollout.json, rollout2.json.

## UNSURE (24), with recommendation

Recommendation for all: keep the backup copy; do not delete the worktree until the owner decides.

1. Large result arrays from the 1e9 clumpy black-hole run (wa-abd02): scatter_bh_1b_clumpy_hist.npz (375 MB), scatter_bh_1b_clumpy_lens.npy (471 MB), lens_windows.npy (2.5 KB). Reason: regenerable by rerunning scatter_bh_1b.py, but that is a 1e9-particle GPU job. Recommendation: keep unless the video from it is already delivered, then they can be dropped.
2. solar_k4_traj.npz (21 MB, wa-abd02). Same reason; small enough to just keep.
3. Solar trajectory arrays: results/solar_stability/traj_seed4738.npz and traj_seed9300.npz (18.6 MB each, in both solar-orbit-error-source-diagnosis and solar-scale-orbit-stability-retrain; identical hashes across the two), results/solar_generalized/traj_seed4738.npz (66 MB). Recommendation: keep one copy of each; the duplicates between worktrees can be dropped later.
4. Hetero tiered generation arrays (4 npz, 10 MB total, hetero-tiered-integration). Reason: no confirmation the generator script is committed. Recommendation: keep.
5. Figures on no ref (7 png): multi_rollout_grid_s4738/s4739 (wa-a3e1), contact_box15_C_grid (wa-a98bd), 4 clumpy/solar frame images (wa-abd02, 1 to 4 MB). Recommendation: keep; the two multi-rollout grids may be the near-field retrain review images.
6. Job logs possibly holding the only copy of numbers (plus a 1.4 KB stray log in cfd-contact-gen): d6_job3526.log (wa-a5a0), substep_router log3613.txt and log3621.txt. Recommendation: keep; cheap.

## Not in scope, flagged: gitignored files

209 gitignored non-cache files, 1.19 GB, sit in 34 worktrees (largest: cfd-contact-gen 57 files incl. a 400 MB orbit_bh_100k npz and several mp4; solar-orbit-error-source-diagnosis has five 68 MB diag npz; galaxy and proposed-1e8 mp4s). They were not backed up because adding them would exceed the 2 GB cap and they are outside the "uncommitted or untracked" scope. List regenerates with `git ls-files --others --ignored --exclude-standard` per worktree. Decision needed from the user before removing any worktree.

## Review

Ran (fresh reviewer, no hypothesis primed). It rehashed all 99 original JUNK rows against 11,149 reachable objects: 98 confirmed byte-identical to ref content, 1 (the cfd log) was not and was reclassified UNSURE and copied. All 62 originally non-JUNK files confirmed absent from every ref and present in the backup with matching sha256. Bundle verify passed and all 132 local branch tips match the bundle heads.
