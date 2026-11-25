# Branch and worktree triage

Snapshot of 2026-11-23, about 16:30 PST; active agent worktrees move (a reviewer saw the closing-report branch advance by one commit afterwards). Generated from git refs in the shared repository, read-only. Nothing was deleted, pushed, merged or removed. Inputs: `git for-each-ref`, `git worktree list --porcelain`, `git rev-list --left-right --count master...<branch>`, `git branch --merged master`, `git branch --contains`, `git branch -r --contains`, `git status --porcelain` in each worktree.

Definitions. Ahead/behind: commits on the branch not on master / commits on master not on the branch. master tip is b2278aa dated 2026-11-09, 848 commits; every later work branch forked before part of master, so each shows 141 behind. Pushed: `yes` means origin/<name> exists at the same commit; `no` means no origin/<name> ref; `behind-local` means origin/<name> exists at an older commit. On origin: the tip is reachable from some origin ref (only this counts for safety). Action: push = unique commits not on origin; tag = unique commits that must survive: either already on origin (tag, then the name can go) or private-only because the history holds a blob over GitHub's 100 MB limit; merge = candidate for the final public head; delete-candidate = nothing unique, or out of scope (reason says which); keep = leave as is. The recommended deletions are proposals only and none may happen until the Pilot confirms the private full copy exists.

## Counts

Local branches found: 86 (the brief said 82; four more exist now, including archive-prep-license-readme-banner created for this task and branches created by agents since). Classified: 86. Excluded: 0. Sum check: delete-candidate 43 + keep 1 + merge 4 + push 22 + tag 16 = 86.

Of the 86 branches, 21 are ancestors of master (includes master itself), 11 have origin/<name> at the same commit, 1 have it behind, 74 have no origin ref. Remote-tracking refs present: 76 (origin only).

Worktrees found: 48 (the brief said 46; the list now includes the main checkout, this task's worktree and the /tmp reviewer worktree). Classified: 48. Excluded: 0. Sum check: keep 28 + remove-candidate 20 = 48.

Out-of-project-scope list (Polaris hardware stress tests, per the Pilot's brief): cross-group-pairwise-nccl-two-group-scripts, exact-all-pairs-force-at-1e7-bodies-two-gpu-shards, nvlink-peer-bandwidth-and-allreduce-stress-on-granted-gpus, pcie-bus-host-device-and-cross-group-peer-stress-on-granted-gpus, barnes-hut-scatter-field-1e10-bodies-one-rollout-per-gpu. Each is 3 to 6 commits ahead of master and none is on origin. Their commit messages are the only record of the stress-test outcomes; the Pilot said to ignore them.

## Branch table

| branch | last commit | ahead/behind master | pushed | worktree | action | reason |
| --- | --- | --- | --- | --- | --- | --- |
| hetero-tiered-integration | 2026-10-26 | +0/-236 | yes | yes | delete-candidate | tip is an ancestor of master; nothing unique |
| solar-moon-failure-cause-training-arms | 2026-11-02 | +25/-249 | no | yes | delete-candidate | tip is an ancestor of unpushed solar-n-k-m-curriculum-regimes; delete only after that branch is pushed |
| solar-scale-orbit-stability-retrain | 2026-11-02 | +6/-249 | no | yes | delete-candidate | tip is an ancestor of unpushed solar-moon-failure-cause-diagnostics; delete only after that branch is pushed |
| solar-scale-translation-augmentation-more-moon-situations | 2026-11-02 | +20/-249 | no | yes | delete-candidate | tip is an ancestor of unpushed solar-moon-failure-cause-diagnostics; delete only after that branch is pushed |
| solar-scale-translation-augmentation-retrain | 2026-11-02 | +13/-249 | no | yes | delete-candidate | tip is an ancestor of unpushed solar-translation-second-seed-and-near-star-arms; delete only after that branch is pushed |
| adaptive-depth-3e6-argmax-body-brute-force | 2026-11-09 | +6/-141 | yes | yes | delete-candidate | tip is an ancestor of adaptive-depth-tie-and-single-query-diff-investigation (+18 more); also on origin |
| class-ablation-trainings | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| unexplored-directions-audit | 2026-11-09 | +9/-141 | behind-local | no | delete-candidate | tip is an ancestor of adaptive-depth-tie-and-single-query-diff-investigation (+28 more); also on origin |
| worktree-agent-a0a3d9fa4c30d2eeb | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a0d5bd5b0fe7dabfa | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a17625ffbcb39a354 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a1a238882644f00ff | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a2147e49bf4f07bb5 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a38a1cf0c6efaeb69 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a580823a6f5fd794a | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a5b8c1fe9ac978266 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a64df97314f41c020 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a77bfabf4456689ad | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a92e8f9d5429d3ec6 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a9323df7c8d536b64 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a97e5169b9ed1aa8c | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-a9fb0b7a244a85848 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-aa56f5fe196152a31 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-ac4858b775ea95107 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-adbe1945df376f1e8 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| worktree-agent-af1677d48cfe15da9 | 2026-11-09 | +0/-0 | no | no | delete-candidate | tip is an ancestor of master; nothing unique |
| ball-tree-particle-attention-stage-1 | 2026-11-10 | +19/-141 | no | no | delete-candidate | tip is an ancestor of voxel-30k-score-bincount-diagnosis (+2 more); also on origin |
| body-properties-timestep-sampling-law-zoo-stage1 | 2026-11-10 | +15/-141 | no | yes | delete-candidate | tip is an ancestor of architecture-push-ported-trainers (+16 more); also on origin |
| dense-ceiling-velocity-split-tile-predictability-stage1 | 2026-11-10 | +11/-141 | no | yes | delete-candidate | tip is an ancestor of architecture-push-ported-trainers (+16 more); also on origin |
| ladder-far-kernel-law-blind-trunk-scale-input-stage-one | 2026-11-10 | +14/-141 | no | yes | delete-candidate | tip is an ancestor of architecture-push-ported-trainers (+16 more); also on origin |
| single-law-readability-and-neighbour-pair-branch | 2026-11-11 | +11/-141 | no | no | delete-candidate | tip is an ancestor of architecture-push-ported-trainers (+17 more); also on origin |
| architecture-push-ported-trainers | 2026-11-12 | +62/-141 | no | no | delete-candidate | tip is an ancestor of neighbour-context-pair-branch-seed-replication-and-tangential-penalty (+15 more); also on origin |
| neighbour-context-pair-branch-seed-replication-and-tangential-penalty | 2026-11-13 | +70/-141 | yes | no | delete-candidate | tip is an ancestor of summary-token-exchange-probe-with-early-stopping-and-planted-signal (+11 more); also on origin |
| scatter-field-three-part-exact-swap-drift-attribution | 2026-11-13 | +71/-141 | yes | yes | delete-candidate | tip is an ancestor of scatter-field-three-part-separate-training-then-joint-finetune; also on origin |
| voxel-extension-stop-at-and-slice-scoring | 2026-11-13 | +21/-141 | no | yes | delete-candidate | tip is an ancestor of voxel-30k-score-bincount-diagnosis (+1 more); also on origin |
| class-evaluation-harness | 2026-11-14 | +1/-0 | no | yes | delete-candidate | tip is an ancestor of unpushed class-gns-vs-token-integration; delete only after that branch is pushed |
| class-gns-style-baseline | 2026-11-14 | +2/-0 | no | yes | delete-candidate | tip is an ancestor of unpushed class-gns-vs-token-integration; delete only after that branch is pushed |
| summary-token-exchange-stage-1-probe-launch | 2026-11-16 | +86/-141 | no | no | delete-candidate | tip is an ancestor of unpushed accuracy-at-1e6-bodies-and-barnes-hut-baseline; delete only after that branch is pushed |
| barnes-hut-scatter-field-1e10-bodies-one-rollout-per-gpu | 2026-11-20 | +3/-0 | no | yes | delete-candidate | out of project scope (Polaris hardware stress test) |
| cross-group-pairwise-nccl-two-group-scripts | 2026-11-20 | +3/-0 | no | yes | delete-candidate | out of project scope (Polaris hardware stress test) |
| exact-all-pairs-force-at-1e7-bodies-two-gpu-shards | 2026-11-20 | +6/-0 | no | yes | delete-candidate | out of project scope (Polaris hardware stress test) |
| nvlink-peer-bandwidth-and-allreduce-stress-on-granted-gpus | 2026-11-20 | +3/-0 | no | yes | delete-candidate | out of project scope (Polaris hardware stress test) |
| pcie-bus-host-device-and-cross-group-peer-stress-on-granted-gpus | 2026-11-20 | +5/-0 | no | yes | delete-candidate | out of project scope (Polaris hardware stress test) |
| master | 2026-11-09 | +0/-0 | yes | no | keep | default branch; 848 commits; behind every 2026-10-10 to 2026-11-23 work branch (141 commits not in them) |
| archive-prep-license-readme-banner | 2026-11-23 | +111/-141 | no | yes | merge | paper and archive-prep line, 111-113 commits ahead of master and not on origin; the stage-1 branch is a superset by 2 commits and is the newest tip; candidate public head |
| negative-results-closing-report-draft | 2026-11-23 | +111/-141 | no | yes | merge | paper and archive-prep line, 111-113 commits ahead of master and not on origin; the stage-1 branch is a superset by 2 commits and is the newest tip; candidate public head |
| paper-outline-and-evidence-table | 2026-11-23 | +111/-141 | no | no | merge | paper and archive-prep line, 111-113 commits ahead of master and not on origin; the stage-1 branch is a superset by 2 commits and is the newest tip; candidate public head |
| stage-1-mass-conditioned-property-channels-and-distribution-control-launch | 2026-11-23 | +113/-141 | no | yes | merge | paper and archive-prep line, 111-113 commits ahead of master and not on origin; the stage-1 branch is a superset by 2 commits and is the newest tip; candidate public head |
| conversation-linked-log | 2026-10-26 | +5/-327 | no | yes | push | unique commits, not on origin |
| solar-moon-failure-cause-diagnostics | 2026-11-02 | +22/-249 | no | yes | push | unique commits, not on origin |
| solar-orbit-error-source-diagnosis | 2026-11-02 | +11/-249 | no | yes | push | unique commits, not on origin |
| solar-scale-generalized-systems-retrain | 2026-11-02 | +12/-249 | no | yes | push | unique commits, not on origin |
| solar-translation-second-seed-and-near-star-arms | 2026-11-02 | +17/-249 | no | yes | push | unique commits, not on origin |
| solar-n-k-m-curriculum-regimes | 2026-11-03 | +30/-207 | no | yes | push | unique commits, not on origin |
| timesfm-neuron-ideas-slow-patch-token | 2026-11-03 | +23/-270 | no | yes | push | unique commits, not on origin |
| token-notebook-page | 2026-11-03 | +5/-249 | no | yes | push | unique commits, not on origin |
| worktree-rescue-keep-files | 2026-11-09 | +24/-0 | no | yes | push | unique commits, not on origin |
| scatter-shift-average-and-core-force-diagnostics | 2026-11-10 | +15/-141 | no | yes | push | unique commits, not on origin |
| chat-history-search-index | 2026-11-12 | +12/-141 | no | yes | push | unique commits, not on origin |
| worktree-agent-a3081b5ab7a6ab20c | 2026-11-12 | +1/-0 | no | yes | push | unique commits, not on origin |
| staggered-grid-cost-fix-same-outputs | 2026-11-13 | +64/-141 | no | yes | push | unique commits, not on origin |
| class-gns-vs-token-integration | 2026-11-14 | +7/-0 | no | yes | push | unique commits, not on origin |
| scatter-field-three-part-separate-training-then-joint-finetune | 2026-11-15 | +76/-141 | no | yes | push | unique commits, not on origin |
| accuracy-at-1e6-bodies-and-barnes-hut-baseline | 2026-11-17 | +109/-141 | no | yes | push | unique commits, not on origin |
| headline-seed-error-bars-and-chaos-floor-widening | 2026-11-17 | +4/-0 | no | yes | push | unique commits, not on origin |
| interior-force-offset-edge-check-subsample-reference | 2026-11-17 | +103/-141 | no | yes | push | unique commits, not on origin |
| nested-refinement-levels-ceiling-probe-harness-fix | 2026-11-17 | +92/-141 | no | yes | push | unique commits, not on origin |
| stage-1-probe-chain-cluster-two | 2026-11-17 | +89/-141 | no | yes | push | unique commits, not on origin |
| timestep-as-input-model-stage-1 | 2026-11-17 | +88/-141 | no | no | push | unique commits, not on origin |
| barnes-hut-baseline-item-independent-review | 2026-11-23 | +109/-141 | no | yes | push | unique commits, not on origin |
| worktree-cfd-contact-gen | 2026-09-30 | +49/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| scatter-conservation | 2026-10-04 | +47/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| scatter-ood | 2026-10-04 | +48/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| scatter-opt | 2026-10-04 | +47/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| scatter-regress-harness | 2026-10-04 | +44/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| scatter-scaling | 2026-10-04 | +46/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| scatter-scenes | 2026-10-04 | +49/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| worktree-agent-a2a6d39d354ae38ca | 2026-10-04 | +41/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| worktree-agent-a3afdebef86797bbc | 2026-10-04 | +43/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| interp-architecture | 2026-10-06 | +47/-527 | no | no | tag | history holds a 154 MiB blob (results/ball_50k_rollout.npz) over the 100 MB GitHub limit; keep in the private copy, do not push without a history decision |
| worktree-agent-abd021892286ed5f9 | 2026-10-12 | +4/-352 | yes | yes | tag | unique commits already on origin; tag to keep after the branch is dropped |
| adaptive-depth-tie-and-single-query-diff-investigation | 2026-11-10 | +25/-141 | yes | yes | tag | unique commits already on origin; tag to keep after the branch is dropped |
| width-32-loss-spike-cause-investigation | 2026-11-10 | +21/-141 | yes | yes | tag | unique commits already on origin; tag to keep after the branch is dropped |
| summary-token-exchange-probe-with-early-stopping-and-planted-signal | 2026-11-13 | +77/-141 | yes | yes | tag | unique commits already on origin; tag to keep after the branch is dropped |
| voxel-30k-score-bincount-diagnosis | 2026-11-14 | +26/-141 | yes | yes | tag | unique commits already on origin; tag to keep after the branch is dropped |
| voxel-score-diverged-rollout-records-all-finite-false | 2026-11-14 | +24/-141 | yes | yes | tag | unique commits already on origin; tag to keep after the branch is dropped |

## Worktree table

| worktree | branch | head | last commit | ahead | behind | dirty paths | action | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| .claude/worktrees/agent-abd021892286ed5f9 | worktree-agent-abd021892286ed5f9 | f247dc9 | 2026-10-12 | 4 | 352 | 8 | keep | 8 uncommitted or untracked path(s); Pilot decides whether to save or discard |
| .claude/worktrees/conversation-linked-log | conversation-linked-log | 1546062 | 2026-10-26 | 5 | 327 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/hetero-tiered-integration | hetero-tiered-integration | d2dc95d | 2026-10-26 | 0 | 236 | 1 | keep | 1 uncommitted or untracked path(s); Pilot decides whether to save or discard |
| .claude/worktrees/solar-moon-failure-cause-diagnostics | solar-moon-failure-cause-diagnostics | a516492 | 2026-11-02 | 22 | 249 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/solar-orbit-error-source-diagnosis | solar-orbit-error-source-diagnosis | 45688ea | 2026-11-02 | 11 | 249 | 2 | keep | 2 uncommitted or untracked path(s); Pilot decides whether to save or discard |
| .claude/worktrees/solar-scale-generalized-systems-retrain | solar-scale-generalized-systems-retrain | a0f5ebf | 2026-11-02 | 12 | 249 | 2 | keep | 2 uncommitted or untracked path(s); Pilot decides whether to save or discard |
| .claude/worktrees/solar-scale-orbit-stability-retrain | solar-scale-orbit-stability-retrain | f30c0ec | 2026-11-02 | 6 | 249 | 2 | keep | 2 uncommitted or untracked path(s); Pilot decides whether to save or discard |
| .claude/worktrees/solar-translation-second-seed-and-near-star-arms | solar-translation-second-seed-and-near-star-arms | 9ad592f | 2026-11-02 | 17 | 249 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/solar-n-k-m-curriculum-regimes | solar-n-k-m-curriculum-regimes | ef43c3d | 2026-11-03 | 30 | 207 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/timesfm-neuron-ideas | timesfm-neuron-ideas-slow-patch-token | 6d9d79c | 2026-11-03 | 23 | 270 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/token-notebook-page | token-notebook-page | 3289ecd | 2026-11-03 | 5 | 249 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| /home/awebb/Research/Augur-keep-rescue | worktree-rescue-keep-files | de2be58 | 2026-11-09 | 24 | 0 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/agent-a92e8f9d5429d3ec6 | scatter-shift-average-and-core-force-diagnostics | 7ad816b | 2026-11-10 | 15 | 141 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/agent-a3081b5ab7a6ab20c | worktree-agent-a3081b5ab7a6ab20c | 7a69f3e | 2026-11-12 | 1 | 0 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/agent-a38a1cf0c6efaeb69 | chat-history-search-index | 6903c0b | 2026-11-12 | 12 | 141 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/agent-a0a3d9fa4c30d2eeb | staggered-grid-cost-fix-same-outputs | 8aeb20a | 2026-11-13 | 64 | 141 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/agent-a9fb0b7a244a85848 | class-gns-style-baseline | 46b1381 | 2026-11-14 | 2 | 0 | 1 | keep | 1 uncommitted or untracked path(s); Pilot decides whether to save or discard |
| .claude/worktrees/agent-ac4858b775ea95107 | class-evaluation-harness | 9154438 | 2026-11-14 | 1 | 0 | 1 | keep | 1 uncommitted or untracked path(s); Pilot decides whether to save or discard |
| .claude/worktrees/class-gns-vs-token-integration | class-gns-vs-token-integration | 71ce14e | 2026-11-14 | 7 | 0 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/agent-aa56f5fe196152a31 | scatter-field-three-part-separate-training-then-joint-finetune | be8a52e | 2026-11-15 | 76 | 141 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| /home/awebb/Research/Augur-accuracy-at-1e6-bodies-and-barnes-hut-baseline | accuracy-at-1e6-bodies-and-barnes-hut-baseline | 12dea4b | 2026-11-17 | 109 | 141 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| /home/awebb/Research/Augur-headline-seed-error-bars-and-chaos-floor-widening | headline-seed-error-bars-and-chaos-floor-widening | 14f0df8 | 2026-11-17 | 4 | 0 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| /home/awebb/Research/Augur-interior-force-offset-edge-check-subsample-reference | interior-force-offset-edge-check-subsample-reference | 4220135 | 2026-11-17 | 103 | 141 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| /home/awebb/Research/Augur-stage-1-probe-chain-cluster-two | stage-1-probe-chain-cluster-two | ec156a2 | 2026-11-17 | 89 | 141 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| /home/awebb/wt-nested-harness | nested-refinement-levels-ceiling-probe-harness-fix | 286e1fd | 2026-11-17 | 92 | 141 | 0 | keep | clean, but branch action is push; remove only after the branch is on origin |
| .claude/worktrees/agent-a2147e49bf4f07bb5 | archive-prep-license-readme-banner | 5f272b8 | 2026-11-23 | 111 | 141 | 2 | keep | active agent worktree (locked) for this archive task or the closing report |
| .claude/worktrees/agent-a64df97314f41c020 | negative-results-closing-report-draft | 5f272b8 | 2026-11-23 | 111 | 141 | 0 | keep | active agent worktree (locked) for this archive task or the closing report |
| Augur (main checkout) | stage-1-mass-conditioned-property-channels-and-distribution-control-launch | f2b1773 | 2026-11-23 | 113 | 141 | 24 | keep | the Pilot's main checkout; 24 untracked paths (including .mcp.json and results directories); branch has a superset of the paper line |
| .claude/worktrees/solar-moon-failure-cause-training-arms | solar-moon-failure-cause-training-arms | bc217da | 2026-11-02 | 25 | 249 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/solar-scale-translation-augmentation-more-moon-situations | solar-scale-translation-augmentation-more-moon-situations | 070d465 | 2026-11-02 | 20 | 249 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/solar-scale-translation-augmentation-retrain | solar-scale-translation-augmentation-retrain | b37be7a | 2026-11-02 | 13 | 249 | 0 | remove-candidate | clean; branch is a delete-candidate |
| /home/awebb/Research/Augur-adaptive-depth-3e6-argmax-body-brute-force | adaptive-depth-3e6-argmax-body-brute-force | aeca91f | 2026-11-09 | 6 | 141 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/agent-a17625ffbcb39a354 | dense-ceiling-velocity-split-tile-predictability-stage1 | 78cb099 | 2026-11-10 | 11 | 141 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/agent-a1a238882644f00ff | body-properties-timestep-sampling-law-zoo-stage1 | 30f04e2 | 2026-11-10 | 15 | 141 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/agent-a580823a6f5fd794a | ladder-far-kernel-law-blind-trunk-scale-input-stage-one | 453778c | 2026-11-10 | 14 | 141 | 0 | remove-candidate | clean; branch is a delete-candidate |
| /home/awebb/Research/Augur-adaptive-depth-tie-and-single-query-diff-investigation | adaptive-depth-tie-and-single-query-diff-investigation | e32569e | 2026-11-10 | 25 | 141 | 0 | remove-candidate | clean; branch content is on origin (tag first) |
| /home/awebb/Research/Augur-width-32-loss-spike-cause-investigation | width-32-loss-spike-cause-investigation | bd2a682 | 2026-11-10 | 21 | 141 | 0 | remove-candidate | clean; branch content is on origin (tag first) |
| .claude/worktrees/agent-a5b8c1fe9ac978266 | summary-token-exchange-probe-with-early-stopping-and-planted-signal | 9fd5368 | 2026-11-13 | 77 | 141 | 0 | remove-candidate | clean; branch content is on origin (tag first) |
| .claude/worktrees/agent-a77bfabf4456689ad | voxel-extension-stop-at-and-slice-scoring | 9bdaded | 2026-11-13 | 21 | 141 | 0 | remove-candidate | clean; branch is a delete-candidate |
| /home/awebb/Research/Augur-three-part-swap | scatter-field-three-part-exact-swap-drift-attribution | 7863e45 | 2026-11-13 | 71 | 141 | 0 | remove-candidate | clean; branch is a delete-candidate |
| /home/awebb/Research/Augur-voxel-30k-score-bincount-diagnosis | voxel-30k-score-bincount-diagnosis | 7d5a795 | 2026-11-14 | 26 | 141 | 0 | remove-candidate | clean; branch content is on origin (tag first) |
| /home/awebb/Research/Augur-voxel-score-diverged-rollout-records-all-finite-false | voxel-score-diverged-rollout-records-all-finite-false | 0eab994 | 2026-11-14 | 24 | 141 | 0 | remove-candidate | clean; branch content is on origin (tag first) |
| .claude/worktrees/agent-a9323df7c8d536b64 | barnes-hut-scatter-field-1e10-bodies-one-rollout-per-gpu | dc36abf | 2026-11-20 | 3 | 0 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/agent-a97e5169b9ed1aa8c | exact-all-pairs-force-at-1e7-bodies-two-gpu-shards | 360b4ad | 2026-11-20 | 6 | 0 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/agent-adbe1945df376f1e8 | cross-group-pairwise-nccl-two-group-scripts | 0ae9d8d | 2026-11-20 | 3 | 0 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/agent-af1677d48cfe15da9 | nvlink-peer-bandwidth-and-allreduce-stress-on-granted-gpus | 0c9ad25 | 2026-11-20 | 3 | 0 | 0 | remove-candidate | clean; branch is a delete-candidate |
| .claude/worktrees/pcie-bus-host-device-and-cross-group-peer-stress-on-granted-gpus | pcie-bus-host-device-and-cross-group-peer-stress-on-granted-gpus | a87953d | 2026-11-20 | 5 | 0 | 0 | remove-candidate | clean; branch is a delete-candidate |
| /tmp/claude-1000/-home-awebb-Research-Augur/0114887a-441b-45f8-978e-404d4bf76553/scratchpad/wt | barnes-hut-baseline-item-independent-review | 9f607e2 | 2026-11-23 | 109 | 141 | 0 | remove-candidate | temporary reviewer worktree under /tmp; its branch must be pushed first (branch action: push) |

Sibling worktrees live at /home/awebb/Research/Augur-<name>. Also on disk but not worktrees: /home/awebb/Research/Augur-backups (3.5 GB, not examined).
