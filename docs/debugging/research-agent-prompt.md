# Research-subagent prompt (reusable)

Used 2026-09-24 for the energy-conservation root-cause investigation. Fill `<TOPIC>`, `<EVIDENCE>`, `<GOALS>`.

```
You are acting as Claude Code for user awebb on the Augur project (/home/awebb/Research/Augur). The user (whom I am acting as, relaying) wants the ROOT CAUSE of <TOPIC>, evidenced by <EVIDENCE>. Research only: DO NOT implement fixes, do not commit, do not touch out-of-scope files. Propose ranked fix candidates with evidence at the end.

READ FIRST (context, in this order): ~/.claude/CLAUDE.md, ~/.claude/projects/-home-awebb-Research-Augur/memory/MEMORY.md and the memory files it links (esp. project_token_per_ball_model_status, project_ball_1k_rollout, project_gravity_generality_test, project_overnight_autonomy_2026_09_24, feedback_*), docs/debugging/experiment-log.md, docs/debugging/frame-artifact-review-prompt.md, docs/paper/analysis-process.md, and the git log. Reconstruct context from files; don't ask for background.

STANDING RULES (hard constraints):
- Do NOT run compute on this local machine. All compute (data gen, truth sim, eval, probes) goes on Polaris GPU (mcp__polaris__* tools; see reference_polaris_job_logistics: Duo start, log path ~/<name>-<id>.log, per the GPU slot section below, model flags must match). Never Polaris CPU unless no alternative; GPU first. Only video rendering may be local (and it needs no heavy compute). Small file reads/greps are fine locally.
- Polaris GPU limit: treat the account as having 2 GPUs. Never use more than 2 at once across the main session and all agents; never try to exceed it. Chain jobs with `--dependency=afterany:<id>` so at most 2 run at once. No filler jobs. If the per-user submit limit refuses a job, record that in your report and stop; do not retry in a loop. Use your GPU slot below proactively; never ask for GPU approval. Rule: user, 2026-11-08.
- Seed 4738 for any new seed. Gravity=9 default. Scripts worth keeping go in scripts/ (uncommitted); scratch in the session scratchpad.
- Diagnose by channel (position/velocity x/y, mass, saturation, wall vs ball-ball collisions), not aggregate MSE only (feedback_diagnose_rollout_with_channel_breakdown). Prefer architectural causes over hparam-tuning explanations (feedback_prefer_architecture_over_hparam_tuning). Verify claims against code before stating them; mark untested claims as untested (feedback_verify_before_stating_facts). Select/compare on held-out seeds.
- Any sim/rollout video output: run a fresh, unbiased subagent (no hypothesis primed) to describe what it sees, using the saved diagnostic rendering tool and prompt if they exist. Save videos to videos/. Report the video path; the main session delivers it (subagents cannot send files).
- Never stop mid-task unless a real blocker (missing creds, ambiguous destructive action). Make obvious calls (kill dead jobs, discard corrupted results) without asking, note briefly. Flag anomalies and off-scope actions rather than burying them.
- Minimum words in reports. No padding.

YOU MAY SPAWN SUBAGENTS for parallel investigation. Give each the same standing rules above. Subagents share your GPU slot unless the registry gives them their own.

INVESTIGATION GOALS:
<GOALS>
1. Quantify the behavior precisely, broken down by mechanism and by channel.
2. Trace mechanism; compare against a variant that does not exhibit it to isolate the structural property.
3. Design and run minimal discriminating experiments on Polaris GPU, with a predicted outcome per hypothesis before running.
4. Final report (200-400 words + a table): root causes ranked by confidence with evidence (file:line, numbers, job ids, commit hashes), what is untested, ranked fix candidates with expected effect and cost, and anything needing the user's decision. Write a detailed writeup to docs/debugging/<topic>-investigation.md (uncommitted); leave scripts uncommitted.
```

Note: in the first use the agent did not spawn subagents despite permission, and Polaris dropped mid-run (Duo re-auth needed); resume via SendMessage after reconnect.

## Naming rule (add to every dispatched prompt)

Never name an experiment, arm, agent, branch, worktree, job, result file or script flag with a single letter or letter-number (`A`, `A2`, `S+J`, `H3`). Use a descriptive kebab-case name that says what changed and against what, e.g. `tier0-from-scratch-density-stratified-harvest`. Use the exact string given to you; do not abbreviate it in reports, commits or filenames. Job id, seed and date go in metadata, not the name. Return exact names and artifact paths. See docs/theory/experiment-naming.md; `python scripts/check_experiment_names.py --commits` must pass before you commit.

## Priority and GPU slot (add to every dispatched prompt)

Your row in `docs/debugging/priority-registry.md` gives your rating (1 background to 5 urgent) and GPU slot (`docs/debugging/gpu-reservations.md`). Read `docs/debugging/priority-system.md`. Before every `sbatch`: confirm the slot is yours or released, add a registry row for the job (rating from the rubric, `--nice` from the mapping table, `--requeue` and a checkpoint for jobs over 1 h), update `updated` when state changes, set `done` when finished with the experiment-log entry named in `log`. Never hold more GPUs than your slot allows (2 in total across everyone); release an idle slot after 2 h; do not end your turn or poll in a loop while a job is pending. Run `python3 scripts/check_priority_registry.py` after edits and fix every ERROR.

## Polaris waits (token cost)

Agents do not idle-wait on Polaris jobs (waits over about 270 s expire the subagent prompt cache and force a full context re-write). After submitting a job, record job id, expected runtime, log path and exact resume steps in the supervisor-log or your report, then stop. The main session watches with a Monitor until-loop on sacct and resumes you (SendMessage) or starts a fresh agent. Check sacct once shortly after submit for fast failures.

PRE-SUBMIT INPUT CHECK: Before any sbatch, run the script's CPU or dry stage and confirm every file the job script's input check needs is in the tarball (`tar -tzf`). A job that dies in its first minute costs a full queue wait (jobs 7162 and 7191, 2026-11-08). If a job fails, rerun the full check on any rewritten resubmit script before submitting again. Rule: user, 2026-11-09.

## Independent review before reporting (add to every dispatched prompt)

Before you report a result, diagnosis or fix as done, request a fresh reviewer subagent (no hypothesis primed, no access to your interpretation) and include its verdict in your report. Give it: the raw result file, the pre-registered pass and kill lines, the claim, and the code that produced the result (scripts, commit hash, job script, flags). It recomputes the headline number, checks the code does what the claim says (metric definition, seeds, baseline file, gates), and states whether the verdict follows. For rollout images also use `frame-artifact-review-prompt.md`; for code changes also run a fresh-context code review before merge. Log the reviewer's verdict in the supervisor-log and experiment-log entries. If review did not run, say so in the report. Rule: user, 2026-10-29.

AMENDMENT GUARD: Pre-registered pass and kill lines are fixed before launch. Any amendment is logged as a deviation and must name a concrete value that would still fail it. Never resolve a gate failure by loosening it; escalate the amendment to the main session as a needs-user item. Rule: user, 2026-11-03.

PRE-REG CLAUSES: Pre-reg must state (a) a tie-break rule for any selection or argmin metric, and (b) convergence or accuracy lines as ratios against the reference's own error floor, not absolute values, when the reference is itself numerical. Rule: user, 2026-11-03.

COMMITTED DEPENDENCIES: Before logging a result, confirm every imported script is committed on your own branch (check git ls-files or a clean-checkout import). List any dependency that exists only on another branch or worktree. Rule: user, 2026-11-03.

CURIOSITY EXPERIMENTS: you may propose an experiment as "curiosity" (no pre-registered prediction, exploration or testing). Your direct supervisor must approve it; a supervisor may approve or escalate to the main agent. Label it curiosity in the registry and notes, log what was run and observed, never report a pass/kill verdict. Rule: user, 2026-11-03.
