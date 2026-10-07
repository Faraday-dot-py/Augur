# Overnight plan 2026-10-07

Source: 2026-10-06 review decisions (scatter-review-decisions-2026-10-06.md), memory index, branch/worktree state, scatter-review-4-6 results (73fa983).
Concurrent agents (not touched here): scaling/OOD (scatter-review-4-6, job 3378), optimization (worktree-agent-a8d2b3989212e2233, jobs 3375/3376), prior-pp8 (scatter-prior-pp8, job 3382), obstacle contact (scatter-contact, job 3380).

## Ranked items

| # | What | Why | Owner | GPU | Risk | Status |
|---|------|-----|-------|-----|------|--------|
| 1 | Merge harness gate edits + chaos floors (3b7ff5d) to master | Decisions 1, 2; every later fix is gated on it | this agent | no | low | DONE a2b4140 |
| 2 | Merge conservation branch (fixes 2-5, opt-in) to master; resolve scatter_regress.py conflict | Decision 5; all new behaviour is opt-in flags | this agent | no | low-med (conflict resolved by keeping both gate row sets) | DONE 0afea95; pytest 230 pass, `compare` run on CPU. scatter_fix_tests.py and the new run path were validated by the branch's Polaris jobs 3348-3370 but NOT re-run post-merge: needs one GPU smoke run (item 4) |
| 3 | Remove finished/merged worktrees (harness, truth.js, banner, notice-banner) | Housekeeping. The richer-temporal worktrees listed in memory were already gone | this agent | no | none | DONE |
| 4 | Post-merge GPU smoke: `scatter_regress.py run` (default stack, cons rows) on master and compare vs baseline_v2 | Merge touched the harness both branches edited | next session / any agent after job 3382 | 1 short job | low | NOT RUN (queue busy; not worth contending) |
| 5 | scatter-review-4-6 anomaly A: E_base OOD rows (ood:ref, mass_ratio_2body) never ran because the wrapper skipped E_base | The OOD per-fix deltas have no reference row | scaling/OOD owner | one serial queue-aware wrapper | low | OPEN, not mine to touch |
| 6 | Anomaly B: largen n46big crashed at N=1e5 (IndexError scatter_scaling.py:159, index 50 on a 20-tick run); clamp the tick index to the run length, rerun only that row | No N=1e5 row for the largen fix | scaling/OOD owner | same wrapper as 5 | low | OPEN |
| 7 | Review finding: all 7 scaling/OOD fixes fail the gate (only ctrl passes). Decide per-fix whether to keep any as opt-in or fold into a final integration retrain | Individual-testing rule says each fix is judged vs frozen baseline | **Needs user** | no | - | OPEN |
| 8 | Optimization: adopt default stack after 3375/3376 return; `adapt` only for N>~5e4 | Decision 3 | optimization agent | running | low | pending jobs |
| 9 | Remaining unmerged branches triage (16): fix-experiment branches (scatter-ood/scaling/scenes/opt/regress-harness) are report branches stacked on the older harness; merging them brings 100k+ line result dumps | Do not merge blindly; harness-v1 commits are superseded by merged v2 | **Needs user** (which result docs to land on master) | no | med (repo bloat) | OPEN |
| 10 | Decision 7 follow-up: wait for obstacle-contact job 3380 | Not mine | contact agent | running | - | - |

## Needs user
- Item 7: fate of the 7 failed scaling/OOD fixes.
- Item 9: which report branches (and which result JSON/PNG sizes) to merge.
- Merged branches (scatter-conservation et al. already ancestors of master) and locked worktree agent-afa043b23751b5aec: delete when ready; left in place.
- Pushing / PRs: none done.
