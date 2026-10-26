# Experiment naming convention and legacy glossary

Standing instruction (user, 2026-10-26): experiment names must never be single letters or letter-number pairs. They carry no information about what happens inside a process, only that it happened, and they transfer poorly between layers of command (supervisor to subagent).

## Convention
- Descriptive kebab-case: what changed and against what. Example: `tier0-from-scratch-density-stratified-harvest`, `lj-mesh-energy-gradient-no-residual`.
- Never a bare letter, letter-number, or letter+letter (A, A2, S+J, H3, JE). Never "Exp 7".
- Job id, seed, date, commit and checkpoint path go in metadata (result JSON, experiment-log entry header), not in the name.
- Stable once chosen. Rename only by adding the new name and recording the old one here.
- Use the exact string everywhere: docs, commit subjects, branch/worktree names, script flag choices, result filenames, Polaris job names, subagent prompts and reports. Subagents must return exact names plus artifact paths.
- In rewritten docs, keep the legacy label once per entry as `(legacy: X)`.
- Exempt: math symbols (E0 initial energy, L0), plot colors (C0-C3), inline code spans, anything in a `(legacy: ...)` parenthesis. Allowlist: scripts/experiment_label_allowlist.txt.
- Legacy checkpoints on Polaris are NOT renamed or deleted; this file is the mapping.
- Enforcement: `python scripts/check_experiment_names.py [--base master] [--commits]` fails on added doc/script lines (and optionally commit subjects since base) with such labels or on short `choices=` values. `--all` prints a census of the whole tree.

## Audit (2026-10-26, master + worktrees read-only)
- Master census (`--all`): 476 lines, 731 label occurrences, 61 distinct tokens. Largest: supervisor-log.md 125, scatter-field-vs-cf-gap-investigation.md 42, error-correction-plan.md 40, scatter-contact-c.md 37, scatter-contact-c-table.md 30, substep-theory.md 27, scatter-correction-research.md 25.
- Worst worktree: hetero-tiered-integration, 704 lines / 1242 occurrences / 67 tokens (new arms S1, S2, B1g, Bh1-Bh6, Fa, Fb, FbL, FbH3, JL, JE, JT, JM, ENM, ENMT, JA, gv_R, gv_NG, gv_EN, Exp 1-12). Other worktrees: 320-550 lines each, mostly the same inherited master labels.
- Script surface: `scripts/train_scatter_field.py` `--exp` choices A, B, C, D, E (EXPS dict keys, also in ckpt names such as `E_ms_kp_pot_v_g128.pt`); other flag choices are descriptive (linear/symlog, token/central, ...). `scripts/train_scatter_ft.py --fix` choices are descriptive. Result-file labels in scripts: `scatter_substep_theory.py`, `rollout_analyze.py` (E0 is initial energy: allowlisted).
- Not renamed (legacy debt, glossary only): `--exp A..E` keys and the ckpt filename; older docs in docs/debugging/ (B2/A2 contact rows, D1-D6 error-correction fixes, S1/S2 substep schemes, E0 state numbers). Renaming `EXPS` keys would break checkpoint metadata and Polaris job scripts; do it only with a compat alias.
- Rewritten in this branch: docs/theory/supervisor-log.md (all agent, arm and run labels, 43 entries carry `(legacy: ...)`), docs/theory/heterogeneous-tiered.md (hole-9 arms, sequential/joint arms), docs/debugging/experiment-log.md (strip-tiling schemes). Prompt templates got a naming-rule section.

## Glossary: legacy label to descriptive name
Agents (supervisor log 2026-10-24/25). Branches/worktrees keep their git names; this maps the nickname only.

| Legacy | Descriptive name | Where / id |
|---|---|---|
| A | lj-potential-gradient-agent | branch hole1-potential-field, commits 48efe5a, 56f5111 |
| A2 | lj-potential-gradient-agent-resumed | 9f08b27, 291f82f |
| B | cutout-bias-and-seam-check-agent | branch worktree-agent-ac8d0c1d33e4ac2c7, b1898a5 |
| C | upper-tier-module-agent | branch worktree-agent-a4059904ca1ecb03c, ccf18d6, b311700, f0d4f0a, ec79ae3, 4853f0c |
| C2 | upper-tier-module-agent-resumed | jobs 3963, 3976, 3979, 3985; ckpts ~/augur_tiered_c2/ on Polaris |
| D | tiered-data-and-eval-harness-agent | bdde9b3, 437b4ea, fb34e1a, 46480ac |
| D2 | tiered-data-and-eval-harness-agent-resumed | 392b7d3 (scripts/tiered_holes.py) |
| E | tier0-extensions-agent | branch worktree-agent-a96914a7f2c210071, bb83b57 |
| E2 | tier0-extensions-agent-resumed | jobs 3945, 3971, 3981, 3986, 3995, 4001, 4003, 4005 |
| F, F2 | tiered-integration-agent(-resumed) | worktree hetero-tiered-integration, d75c0da |

Arms and runs.

| Legacy | Descriptive name | Where / id |
|---|---|---|
| Exp A, B, C, D, E (`--exp`) | scatter-gravity-two-body / -unit-mass-scale-init / -star-plus-light / -orbit / -clustered-baseline | `scripts/train_scatter_field.py` EXPS |
| Exp E / E (as baseline) | scatter-gravity-clustered-baseline | ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt; it 60769 |
| S | tiers-sequential-frozen | upper-tier-module-agent, ckpt ck_*/tiered_S.pt |
| S+R | tiers-sequential-frozen-plus-rollout-in-loop | f0d4f0a |
| S+J | tiers-sequential-then-joint-finetune | job 3976 |
| Exp 1 | tiers-sequential-frozen-retry-large-data | job 3963 |
| Exp 2 | tiers-sequential-then-joint-finetune | job 3976 |
| Exp 3 (Z, L, Er) | tiers-lever-ablations | job 3979 |
| Exp 4 / JE | tiers-joint-finetune-plus-energy-term | job 3985 |
| JT | tiers-joint-finetune-tier0-frozen | |
| JM | tiers-joint-finetune-mean-removal-ablation | |
| JL | tiers-joint-finetune-long-horizon-energy-penalty | job 4010, ckpt ~/augur_tiered_c2/ck_JL/tiered_J.pt |
| ENM, EN-M (Exp 5b) | tiers-energy-functional-moment-inputs | job 4002, ckpts ck_ENM/tiered_S.pt, ck_ENM/j/tiered_J.pt |
| ENMT | tiers-energy-functional-tier0-frozen | ck_ENMT/tiered_J.pt |
| JA | tiers-joint-finetune-l2sp-anchor-plus-energy-term | |
| gv_R | tier0-gravity-velocity-recurrent-inputs | |
| gv_NG | tier0-gravity-no-velocity-no-recurrent-inputs | ~/augur_tiered_c2/gv_NG.pt (from ~/augur_slj_hv/ckpt/gv/E_gv_NG.pt) |
| gv_EN | tier0-gravity-energy-gradient-mesh | |
| Exp 11 NGa | tiers-on-gv-NG-tier0-sequential-then-joint | job 4024 |
| Exp 8/9, 10, 12 | not recoverable from master; see integration worktree experiment-log (job 4019) | |
| P | lj-mesh-energy-gradient-no-residual | 48efe5a, job 3927 |
| PR | lj-mesh-energy-gradient-plus-penalised-residual | 56f5111 |
| V | lj-mesh-non-gradient-inputs-zeroed | 9f08b27 |
| A (LJ) | lj-learned-pair-soft-reciprocity-plus-mesh | 9f08b27 |
| O | lj-exact-pair-plus-learned-mesh | |
| M | lj-mesh-off-pair-only | 291f82f |
| S2 (LJ) | lj-scratch-with-velocity-and-recurrent-inputs | seed 4739, 291f82f |
| PH | lj-potential-mesh-plus-per-particle-head | job 3995 |
| A (hole-9) | dense-unet-larger-grid | hole 9 driver |
| B (hole-9) | tiled-ghosts-far-path | |
| C (hole-9) | tiered-hierarchy | |
| arm A (tier 0) | tier0-ext-field-warm-start-windowed-gravity | job 3945 |
| B1 | tier0-per-particle-head-drag | job 3971 |
| B1g | tier0-per-particle-head-material-gated | jobs 4025, 4029 |
| A0 | tier0-no-ext-control | job 3986 |
| A1 | tier0-ext-gain-100 | job 3986 |
| H1, H2 | tier0-harvested-windows-fine-tune-attempts | |
| H3 | tier0-harvest-plus-baseline-replay-third-order-far-field | job 4005 |
| Fa | tier0-fine-tune-low-lr-short | |
| Fb | tier0-fine-tune-body-frozen-new-input-channels | |
| FbL | tier0-fine-tune-lora | |
| FbH3 | tier0-fine-tune-body-frozen-on-harvest-plus-replay | |
| S1 (tier 0) | tier0-from-scratch-harvest-plus-replay | job 4023 |
| S2 (tier 0) | tier0-from-scratch-density-stratified-harvest | job 4036, ckpt ~/t0n/results/tier0_ext/S2.pt |
| S3 | tier0-from-scratch-density-stratified-harvest-matched-iterations | job t0m (pre-registered) |
| Bh | contact-head-unbounded | job 4001 |
| Bh2 | contact-head-tanh-bounded-plus-penalty | job 4006 |
| Bh4, Bh5, Bh6 | contact-head-power-diagnosis / -longer-horizon / -on-frozen-sdf-bounded | job 4020 |
| B2 (contact) | contact-sdf-gather-bounded-heads-retrain | job 3380 |
| A2 (contact) | contact-kernel-convolution-bounded-heads-retrain | job 3382 |
| D1-D6 (error correction) | softcap-recurrent-field / recurrent-contraction / recurrent-noise / near-field-residual / longer-horizon-lower-lr / shift-jitter | docs/debugging/error-correction-plan.md |
| S1, S2 (substep) | substep-uniform-kdk-unrolled / substep-symmetric-respa-slow-grid-fast-pair | docs/debugging/substep-theory.md |
| S2 (tiling table) | strip-width-2 | experiment-log.md job 2989 entry |
| H1-H7 (adaptive eval) | accuracy / efficiency / robustness / generalization / scaling / tileability / interpretability claims | adaptive-eval-agent-preamble.md |

Rows whose mapping is by name inference from the log text (not from a commit) are: JT, JM, JA, Bh4-Bh6, Fa, FbH3, Exp 1, Exp 11 NGa. Verify before relying on them.

## Research: how agents and harnesses handle identifiers and hand-offs
Method: web search and fetch, text only (see security log). Only the sources marked "fetched" were read; abstract-level sources are marked. Nothing found tests the user's claim directly.

Supported
- A subagent starts with fresh context and sees only the parent's delegation message; rules and names must be restated in that message. Claude Code sub-agents docs (fetched, first 100k chars): https://code.claude.com/docs/en/sub-agents
- Vague or terse hand-offs cause misread tasks and duplicated work; Anthropic's multi-agent research system post recommends explicit objective, output format, boundaries, and passing "lightweight references" to external artifacts rather than relaying content (fetched): https://www.anthropic.com/engineering/multi-agent-research-system
- Inter-agent misalignment is a documented failure cluster: MAST taxonomy, 14 modes, 1,600+ annotated traces, 7 frameworks (abstract only; identifier loss not itemised): https://arxiv.org/abs/2503.13657
- LLM code comprehension depends on identifier names: renaming degrades understanding, and misleading names are worse than uninformative ones (search snippets and third-party summary only, numbers unchecked): https://arxiv.org/pdf/2510.03178.pdf, https://arxiv.org/pdf/2606.31725
- Naming-style rewrites change model behaviour (about 9% on average in TokDrift, attributed to tokenization misalignment; snippet only): https://arxiv.org/pdf/2510.14972

Not supported
- No source shows LLMs deliberately compress names to save tokens. Documented RLHF length bias points toward longer outputs: https://arxiv.org/html/2511.12573v1
- No source shows a tokenization cost that makes "A2" cheaper in a way a model optimises; no measured tokens-per-identifier data was found.
- No study isolating single-letter versus descriptive names in agent hand-offs was found.

Unknown
- Why agents abbreviate labels. Candidate mechanisms, all untested: terse-output style from post-training; in-context anchoring (a label in a table is reused and shortened); compact labels being common in training data for experiment matrices; this repo's own "minimum words" rule in ~/.claude/CLAUDE.md, which plausibly pushes agents to abbreviate; tokenization cost.
- How often hand-offs lose names; no rate published.
- A cheap local test: same hand-off with and without the terseness rule, count name drift.

Verdict: the causal claim ("token optimization clipping") is an unsupported hypothesis; brevity pressure plus anchoring fits better. The consequences the user cites (poor information transfer between layers of command, no semantic cue in opaque labels) are supported by the hand-off and renaming literature. The instruction stands regardless.

Practice recommendations
1. Give the full name in every delegation prompt with "use this exact string".
2. Keep this glossary as the canonical registry; pass its path to subagents.
3. Require returned reports to list exact names and artifact paths.
4. Check mechanically (scripts/check_experiment_names.py), not by model consistency.
5. Avoid near-collision names; make the distinguishing token the first one.

## Security log
- Web research: WebSearch/WebFetch only, performed by a read-only subagent. Pages treated as data; no instructions followed. Nothing downloaded, installed or executed. Fetched: the three URLs marked fetched above; the others were search results only.
- No public code or prompts were run. No scanners were needed or installed (nothing to scan). Verdict for every item: read-only, nothing run.
