# Branch push audit 2026-10-29

Scope: 119 local branches, 7 on origin (Faraday-dot-py/Augur, public). master was already pushed (0f2b95d). "Unique commits" = `git rev-list master..branch`. Checks run per branch over its unique history: blobs over 100 MB, secrets/credential regex scan of added lines (AWS, GitHub, OpenAI/Anthropic, Slack, Hugging Face tokens; private keys; password/secret/api-key assignments), sensitive filenames (.env, *.pem, *.key, id_rsa, credentials), and claude.ai links other than the standard `Claude-Session:` commit trailer. No branch failed the secrets scan.

## 1. Pushed (58 branches, non-force, one `git push` per branch; tips verified against `git ls-remote --heads origin`)

| Branch | Unique commits | Tip | Why included (newest commit subject) |
|---|---|---|---|
| bh-wave-structures | 7 | 2f773de | BH wave structures: shell-crossing caustics present in truth/PM/reference; model core drif |
| close-pair-design | 1 | e9c256a | docs: close-pair design proposals (R1 exact discrete mesh-pair complement ranked first); r |
| energy-drift-seeds | 5 | 65ff422 | dE many-seed test (512 scenes/regime): ghost vs E median dE@1e3 ratio bh300 1.246 [1.220,1 |
| euclid-energy-ladder | 7 | 1bcb24d | euclid-energy ladder vs frozen E: no arm passes (ab err@20 .0233->.0216 all arms; nr100 dE |
| euclid-heads-full | 10 | 02b55be | heads-only Euclid fine-tune at full 60k-it lr 5e-4: KILL (ab err@20 .0215/.0216 vs gate .0 |
| exp-e-proposed-1e8 | 5 | 7cc7ec2 | proposed model 1e8: BH 1e8 132 ms/tick 9.1 GB, 0 non-finite, /v/<c, KE/N 1e8 vs 1e6 within |
| far-kernel-fix | 17 | 2d6d8f7 | Add R1b and R2b kernel files (results/far_kernels) |
| far-kernel-generalization | 20 | b2a0522 | Far-kernel generalisation: (b) theta-conditioned KILL (yuk32 3.4x, screened-rep 2/3 seeds) |
| fewshot-adapter | 4 | cbe7f3b | fewshot-adapter results (job 3663): solar held-out err@100 39.6 -> 6.5 (n=1..16), planets  |
| fewshot-assim | 3 | dd77ee1 | few-shot/assimilation probe results (job 3643): K=20 reset 86x solar err@100, planets 8/8  |
| fix-finetune-dE | 13 | 3708301 | ft-dE: option-C frozen-U-Net FAILS in-domain (err@20 7.92 vs C 5.10, tunnel@20 59%, off-gr |
| gravity-side-by-side | 3 | a6243d2 | Gravity page: sim view and 3D model view side by side (stacked on narrow screens), Plots t |
| hetero-tiered-integration | 107 | d2dc95d | scripts: comparison-video renderers (gravity, LJ, contact generations) and seam error spli |
| hole1-potential-field | 57 | 50f94a1 | contact-C ablation: no-recurrent err@100 258 vs same-recipe rerun 250 (no effect); rerun d |
| lj-scatter-field | 23 | 8ad26cf | Scatter-LJ diagnosis: mesh branch (velocity+recurrent inputs) injects ~1% non-conservative |
| loss-energy | 4 | 29dc799 | loss-energy arm B replicate (job 3636): nr100 dE@1000 below ctl 4/4 but .12-.81 across see |
| loss-euclid-fullscale | 3 | 7ba2f2e | Euclid rollout loss at full Exp E scale: ab err@20 .0217->.0063 (-71%), but nr100 dE@1000  |
| loss-local-work-bias-term | 112 | 6336ef3 | loss terms: local-work-bias and clump-centre fine-tunes of matched-60k tier 0 both KILL (d |
| loss-weights | 1 | 7858860 | loss-weights: Euclid velocity, step-weighted, unit-slope Huber arms; per-seed override in  |
| lr-warmup | 4 | 2b2ebe1 | lr-warmup (job 3664): NEGATIVE; warmup removes first-step jump but force error returns to  |
| pages-publish | 7 | bb8a284 | web: revert signed activation heights (cubes rise by /value/ again); keep weight-click par |
| r1-step0 | 2 | fcc7c13 | R1 step 0: exact mesh-pair complement cuts analytic P3M nn<1 force error 13.6/13.8/12.0 -> |
| regress-baseline-r3 | 6 | d5a5dfb | harness: widen cons dL tolerance to 150% (E vs E: nr100 dL .058 vs .023, bh300 2.79 vs 1.9 |
| regress-contact-energy | 4 | 02efda8 | harness: replicates median-of-3 (E nr100 dE spread .011-.039), report-only multi-ball gain |
| research-nbody-comparison | 1 | fb6394f | docs: N-body comparison options (research + proposal): 5 ranked options, baselines, benchm |
| scatter-core-drift-armA | 29 | 0cb0471 | ARM A: C4 symmetrised U-Net potential cuts core drift .087 to .0015-.0067/tick (3 seeds),  |
| scatter-core-drift-armB | 32 | d41785e | ARM B net-force penalty fine-tune: KILL. core drift .033-.038/tick vs E .064 (control .027 |
| scatter-escape-core-drift | 27 | 62d6714 | escape/core-drift: tiling recovers outside-box force (.03 to .95) not escapee trajectories |
| scatter-integration-retrain | 7 | 024dd0a | scatter integration retrain int_full1 (job 3413): accuracy gate PASS (worst +13.8%), cons  |
| scatter-joint-scratch | 10 | 1261155 | scatter-joint-scratch: unbiased image review appended |
| scatter-review-4-6 | 6 | 7633d4d | s46: fix tick index at N=1e5 (record ticks <= steps), run E_base OOD rows and largen cold_ |
| substep-exp | 3 | d22f1a5 | substep-exp: ckpt 'init' builds untrained seed-4738 model (pairfull0); Polaris stage-1 job |
| test-audit-2026-10-13 | 35 | 2dcbcf9 | audit decisions: keep Stage-2 line (live importers listed), legacy baseline copy, log P1-P |
| tier0-residual-corrector | 112 | a57452c | tier0 residual corrector KILL: held-out err@20 1.60-1.92x BL (line 1.25x), dE@100 4.7-5.3x |
| tier0-tiled-closed-loop | 118 | c18900d | tier0 tiled: pair-potential-by-separation similarity term (--energy-kind pebin), localisat |
| token-contact-default | 2 | 0db8a42 | Token page: stay on one selected ball by default, auto-hop off |
| web-conservative | 3 | 166443a | web: fix phone resize (fog range, panel/insets refit), collision-free arch labels, contact |
| web-nav-fix | 4 | dfc0928 | web: back-to-landing links on token and scatter pages; drop scatter link from gravity page |
| web-scatter-tiled | 7 | e40e682 | web: tiled page label/legend/mobile fixes after unbiased screenshot review; screenshots |
| worktree-agent-a226cff35f933b29f | 8 | d1b0e03 | D2b negative: contraction fine-tune diverges (D2b2 err@20 ~100, 1804 rollbacks) vs stable  |
| worktree-agent-a4059904ca1ecb03c | 45 | 0125127 | docs: tier0-matched-60k-no-velocity-no-recurrent KILL both seeds: oracle err@20 1.56-1.85x |
| worktree-agent-a51e0b89567ace982 | 8 | c1de2ae | softcap s=100 on fed-back field as inference default for contact-C ckpts: C contact runawa |
| worktree-agent-a61e7dc4af33d7032 | 16 | 5d53638 | B-frozen: note pending noise-check job 3596 |
| worktree-agent-a6dc0f9444b821098 | 11 | 4c58332 | Tiling round 2 complete (far-kernel check, adapt pair term, load balancing); see docs/debu |
| worktree-agent-a70045404fe811e26 | 8 | 04461b3 | galaxy: g2 frame review note |
| worktree-agent-a7c9f2330e19613c6 | 1 | aec42ea | Learned integrator research: KILL (exact-force+8% noise, KDK-2 err@20 -1.8% expB / -11% cl |
| worktree-agent-a851652ea9f780ebf | 3 | 5168703 | recurrent ablation: resume script for rec control (3593 diverged at it 22720, non-finite;  |
| worktree-agent-a8b09cd8f67d9fd44 | 21 | 936af00 | Tiled: per-particle hidden state keyed by particle id (owner-tile write, needs scatter_tie |
| worktree-agent-a8d2b3989212e2233 | 6 | a7adc67 | scatter-opt d3 follow-up 3398: static cap not cause of 1e-3 mismatch (cap parity 0..1e-6); |
| worktree-agent-a9510055ea4ddb35d | 5 | 6878d1a | scatter cl reset-field ablation (job 3504): err@20 .0440 vs carried .0461 vs control .0298 |
| worktree-agent-a96914a7f2c210071 | 29 | 3d23e74 | Sync log |
| worktree-agent-a98bdf87385c8eb37 | 5 | 2146183 | scatter contact C follow-up C40 (k_end 40): training diverged at it ~950, 7046/8000 steps  |
| worktree-agent-aaba3986bba03cec0 | 4 | 12a309c | R1 step 1 (job 3754): V3 vs frozen E learned kernel raises nn<1 force err 6.5 -> 63.2% exp |
| worktree-agent-abc314d273210207b | 2 | be1a836 | Scatter tiling results: ghosts+far PASS (err@20 0.95-1.00x single, seam/interior .54-1.06) |
| worktree-agent-abd021892286ed5f9 | 4 | f247dc9 | scatter solar system with SF (job 3507): orbit_ft2 best (benchmark 25% planets lost); sola |
| worktree-agent-ac8d0c1d33e4ac2c7 | 24 | b1898a5 | Hetero baselines: frame review, theory-doc measured baselines |
| worktree-agent-afd468dd2f6e11b2f | 1 | 88844e7 | Dark matter assessment: zero-shot two-component probe (live halo m_h 1/3/10/30, static hal |
| worktree-agent-aff78e45d3110a2fd | 29 | c5d8f3a | galaxy tiled: frame review note |

Note: worktree-agent-a98bdf87385c8eb37 has the same tip as worktree-agent-abbe1333d607536ad (2146183); only abbe1333 was pushed. Several pushed branches are stacked on each other (shared history), so unique-commit counts overlap.

## 2. Not pushed

| Branch | Unique commits | Tip | Reason |
|---|---|---|---|
| conversation-linked-log | 5 | 1546062 | withheld (claude.ai artifact link in diff) |
| descriptive-experiment-names | 0 | ed036be | empty (no commits beyond master) |
| interp-architecture | 47 | eeb907e | large-blob |
| loss-force | 0 | f0e6238 | empty (no commits beyond master) |
| loss-tests | 0 | f0e6238 | empty (no commits beyond master) |
| near-field-retrain | 24 | 173ed13 | already on origin at same tip |
| neuron-graph-smoke-test | 15 | 609fbc3 | active-agent |
| rootcause-pairfull | 0 | f0e6238 | empty (no commits beyond master) |
| scatter-conservation | 47 | a781a7e | large-blob |
| scatter-contact | 0 | 875da31 | empty (no commits beyond master) |
| scatter-edge-fix | 3 | 64a2b4d | already on origin at same tip |
| scatter-ood | 48 | 1d8dd96 | large-blob |
| scatter-opt | 47 | 8edc8b7 | large-blob |
| scatter-prior-pp8 | 0 | 7fe8979 | empty (no commits beyond master) |
| scatter-regress-harness | 44 | 103be23 | large-blob |
| scatter-scaling | 46 | 1dc0693 | large-blob |
| scatter-scenes | 49 | ba62307 | large-blob |
| substep-router | 0 | f0e6238 | empty (no commits beyond master) |
| tiered-drift-mechanism-diagnoses | 126 | 52e5d68 | active-agent |
| timesfm-architecture-ideas | 3 | a4acd4a | active-agent |
| unexplored-directions-audit | 4 | e209d75 | active-agent |
| viz-web | 1 | e6444cf | already on origin; origin/viz-web (e64be66) is ahead of local e6444cf (local is an ancestor), push was rejected non-fast-forward and nothing needed pushing |
| viz-web-local | 0 | 4385d8f | empty (no commits beyond master) |
| web-scatter | 0 | c19581c | empty (no commits beyond master) |
| web-scatter-tiled-precomputed | 14 | eb53d3c | already on origin at same tip |
| web-scatter-tiled-webgpu | 16 | 2dcb4cc | already on origin at same tip |
| web-token-and-central | 0 | fbbe239 | empty (no commits beyond master) |
| worktree-agent-a0eadcf91f9155905 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a2a6d39d354ae38ca | 41 | 4f071d4 | large-blob |
| worktree-agent-a34d2c3a12f8ae61c | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a35cca4b6a6ae0cb2 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a3afdebef86797bbc | 43 | a81ce96 | large-blob |
| worktree-agent-a3e1a2b25c5c98953 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a48419fdfc8826ae6 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a5092afccc4959286 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a521dac4f5efa083e | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a528407b2b0731e75 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a5a0b1cb10b222c88 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a5c465256f0158453 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a650284e79b045ce8 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a656778017dfcdfa0 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a720bcd134727fde5 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a861e4207c64b868e | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a8c48d63f02057ab5 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a8e4efcf883af6c73 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-a919f03efe00c0360 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-aa189c64974ee4737 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-aa9ba20f090d1ddc7 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-ab6ca13505c80874e | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-ab8fe98687d0a3ca6 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-ab9323bfdf900d014 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-abbe1333d607536ad | 5 | 2146183 | duplicate of worktree-agent-a98bdf87385c8eb37 |
| worktree-agent-ace28f2f5196cc718 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-ad5f0f354d427b4d4 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-adfbf776853ca7eea | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-ae1ce96228da053d3 | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-ae75e548a61fda7cf | 0 | a860d9f | empty (no commits beyond master) |
| worktree-agent-af217cf6e01e9bbf4 | 0 | e449363 | empty (no commits beyond master) |
| worktree-agent-afa043b23751b5aec | 0 | 9c55aac | empty (no commits beyond master) |
| worktree-agent-afe29fd2817396cfd | 0 | a860d9f | empty (no commits beyond master) |
| worktree-cfd-contact-gen | 49 | 307f576 | large-blob |

### Large-blob branches (decision needed)

10 branches contain `results/ball_50k_rollout.npz` (161,601,264 bytes; GitHub limit 100 MB): interp-architecture, scatter-conservation, scatter-ood, scatter-opt, scatter-regress-harness, scatter-scaling, scatter-scenes, worktree-agent-a2a6d39d354ae38ca, worktree-agent-a3afdebef86797bbc, worktree-cfd-contact-gen.

The blob enters history in two early commits, both from 2026-09-27: 3dc1bb1 (added the file, 50k-ball scaling test, job 3114/3116; blob fe18500efc) and 3f1e628 (modified it, job 3117; blob 0692d3a23d). Both are in the unique history of all 10 branches (41-49 unique commits each, forked from master at a1a5642 or earlier). Neither commit is in master. No other branch contains a blob over 50 MB.

Options (nothing was done; no LFS, no rewrite):

1. Git LFS migrate: `git lfs migrate import --include=results/ball_50k_rollout.npz --include-ref=<each branch>` rewrites those 10 branches (new hashes) and needs LFS storage/bandwidth on the repo. Do it on a throwaway clone, not here.
2. filter-repo on a throwaway clone: `git filter-repo --path results/ball_50k_rollout.npz --invert-paths --refs <the 10 branches>`, then push the filtered branches. The file is dropped from history; the local originals and the bundle stay intact. The cheapest option if the 50k-ball rollout npz is not needed on the remote.
3. Leave local only (already backed up).

## 3. Backup manifest summary

Location: ~/backups/augur-2026-10-29/ (about 5.3 GB; disk had 695 GB free).

| Item | Detail |
|---|---|
| Git bundle | augur-all.bundle, 554 MB, `git bundle create --all`; `git bundle verify` reports a complete history; 119 heads + 22 tags + 7 remote refs listed; heads/tags diff against `git for-each-ref` is empty |
| Bundle sha256 | 450181a3549a238c4fda269c14ad5b83d34444ae085638984d6c26402c274d7c |
| Working-tree state | 81 trees (main checkout + 80 worktrees): `git status --porcelain` saved per tree; untracked + modified files archived as tar.gz (gzip -3; zstd not installed), .git excluded, nested .claude/worktrees excluded |
| Trees with files archived | 23 (177 files, 5.06 GB compressed); 58 trees clean |
| Main checkout | 24 files, 4.47 GB compressed (about 6.5 GB raw: results/*.npz, *.npy, videos/); the 2 GB cap was waived because free disk is ample |
| Verification | `gzip -t` on every archive and `tar -t` entry count equals the file list for every archive; zero failures. Per-archive sha256 in trees.sha256 |
| Files | manifest.tsv, trees.txt, trees/*.status.txt, trees/*.files.txt, heads.txt, augur-all.bundle.sha256 |

## Anomalies

- The main checkout is shared with active agents: its branch changed during this audit (timesfm-architecture-ideas, then unexplored-directions-audit). The MAIN row of the manifest records timesfm-architecture-ideas; the 24 archived files are untracked state, which is branch-independent. unexplored-directions-audit did not exist when the first audit pass ran; a second pass picked it up (timesfm-architecture-ideas is an ancestor of it).
- master is not checked out anywhere, so this file was committed to master through a temporary worktree.
- refs/original/refs/heads/master (0627f7a) exists: a leftover from an earlier history rewrite (filter-branch). It is in the bundle; it was not pushed.
- Origin has a branch web-scatter-walkthrough that does not exist locally; it was not touched.
- viz-web: local is stale (behind origin by 14 commits); no push was needed.
