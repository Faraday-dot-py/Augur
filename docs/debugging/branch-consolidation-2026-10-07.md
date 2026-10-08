# Branch consolidation 2026-10-07

Decisions 2-3 of the 2026-10-07 session review. Method: `git cherry master <branch>` (patch-id) to find commits not already on master, then land code/tests only (no results/, videos/, data dumps). Every branch below also has an annotated tag `archive/<branch>` at its tip (results, videos and full history live there). Tests were not run (pytest runs on Polaris); landed .py/.sh files passed `py_compile` / `bash -n`.

Most of the stacked chain from a1a5642 is already on master by patch-id (via the a2b4140/0afea95 merges and earlier merges). The only commits unique to the chain branches were 3dc1bb1/3f1e628 (cons_pure_100 retrain + 50k-ball test); their code is identical to master a10c950, the rest is results/videos, so nothing landed from them.

| branch | tag | landed on master | skipped |
|---|---|---|---|
| scatter-pp8-protocol (+ scatter-prior-pp8) | archive/scatter-pp8-protocol, archive/scatter-prior-pp8 | merge 7d1088d: pp8 variant, opt-in prior kernel, knn, CFHarness, protocol rescore + wrapper, wrapper comma fix, doc note | results/scatter_pp8/ |
| scatter-review-4-6 | archive/scatter-review-4-6 | cherry-picks of 7407b50, 5b135b4; 708c235 and 7633d4d as code/docs only; doc and s46_all wrapper at tip | results/scatter_s46/, results table md, videos png |
| scatter-contact | archive/scatter-contact | merge bec2eb0: contact option A/B modules, scenes, tests, render, wrapper, doc | results/contact/, videos/contact_* (both options failed) |
| worktree-agent-a8d2b3989212e2233 | archive/worktree-agent-a8d2b3989212e2233 | cherry-picks a126c6a..a7adc67 (default opt stack resolver, auto pair-term mode, static-cap cell, compile probe, d3 follow-ups). Note: regress `--opt` default is now `default` | none |
| pages-publish | archive/pages-publish | nothing | based on pre-split `web/js/` layout; master has `web/token/` (fbbe239 port + weight editing) and viz-web is an ancestor of master; landing would regress the live layout. `stamp_web_version.py` and `test_web_ui.py` hard-code `web/js` |
| web-conservative | archive/web-conservative | nothing | same as pages-publish |
| interp-architecture | archive/interp-architecture | nothing | all code already on master; only 3dc1bb1/3f1e628 (results) unique |
| scatter-conservation | archive/scatter-conservation | nothing | already on master |
| scatter-ood | archive/scatter-ood | nothing | already on master |
| scatter-opt | archive/scatter-opt | nothing | already on master |
| scatter-regress-harness | archive/scatter-regress-harness | nothing | already on master |
| scatter-scaling | archive/scatter-scaling | nothing | already on master |
| scatter-scenes | archive/scatter-scenes | nothing | already on master |
| worktree-agent-a2a6d39d354ae38ca | archive/worktree-agent-a2a6d39d354ae38ca | nothing | already on master |
| worktree-agent-a3afdebef86797bbc | archive/worktree-agent-a3afdebef86797bbc | nothing | already on master |
| worktree-cfd-contact-gen | archive/worktree-cfd-contact-gen | nothing | already on master |

Excluded (in use): scatter-integration-retrain and its worktree. Its commits ec5257c/f9b6b86/9ffd968 duplicate the s46 content landed here with different hashes; rebase with `--skip` on any that become empty or conflict.
