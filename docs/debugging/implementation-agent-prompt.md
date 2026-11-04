# Implementation/testing-subagent prompt (reusable)

Companion to research-agent-prompt.md; same standing rules. Used 2026-09-24 for the energy-conservation fixes. Fill `<TASK>`, `<INVESTIGATION_DOC>`, `<FIXES>`.

```
You are acting as Claude Code for user awebb on the Augur project (/home/awebb/Research/Augur). Implement and test <TASK>, following the ranked recommendations in <INVESTIGATION_DOC>. Do the fixes in the stated order: <FIXES>.

READ FIRST: ~/.claude/CLAUDE.md, ~/.claude/projects/-home-awebb-Research-Augur/memory/MEMORY.md and linked memory files, docs/debugging/experiment-log.md, <INVESTIGATION_DOC>, docs/debugging/frame-artifact-review-prompt.md, git log. Reconstruct context from files; don't ask for background.

STANDING RULES: same as docs/debugging/research-agent-prompt.md (no local compute; Polaris GPU only, per the GPU slot section; no inline multiline Python; seed 4738; gravity 9; diagnose by channel; verify before stating; held-out seeds; unbiased video-review subagent + save to videos/ and report path; never stop mid-task; minimum words; flag anomalies).

IMPLEMENTATION RULES:
- Match existing code style; new behaviour opt-in behind flags so existing checkpoints/scripts still work; add tests alongside existing tests. Tests run on Polaris, not locally.
- Work on a git branch/worktree, not the current branch. Commit each fix + its results with the key metric in the message after each experiment completes. Do not push, merge, or open PRs.
- Long runs (many hours+) must save periodic full-state checkpoints; confirm the runner does so before launch.
- Before each experiment, state predicted outcome; after, compare to prediction.
- Compare against soup B baseline and CentralForceDynamics on identical held-out seeds; report err@5/10/20, energy drift vs truth, momentum error, wall restitution vs speed, pair head-on KE ratio, and 1000-ball x 500-step behavior (first non-finite step, floor collapse).
- You may spawn subagents; give them the same rules.
- Final report: 200-400 words + table; decisions needed from the user listed separately.
```

## Naming rule (add to every dispatched prompt)

Never name an experiment, arm, agent, branch, worktree, job, result file or script flag with a single letter or letter-number (`A`, `A2`, `S+J`, `H3`). Use a descriptive kebab-case name that says what changed and against what, e.g. `tier0-from-scratch-density-stratified-harvest`. Use the exact string given to you; do not abbreviate it in reports, commits or filenames. Job id, seed and date go in metadata, not the name. New `choices=` values in script flags follow the same rule. See docs/theory/experiment-naming.md; `python scripts/check_experiment_names.py --commits` must pass before you commit.

## Priority and GPU slot (add to every dispatched prompt)

Your row in `docs/debugging/priority-registry.md` gives your rating (1 background to 5 urgent) and GPU slot (`docs/debugging/gpu-reservations.md`). Read `docs/debugging/priority-system.md`. Before every `sbatch`: confirm the slot is yours or released, add a registry row for the job (rating from the rubric, `--nice` from the mapping table, `--requeue` and a checkpoint for jobs over 1 h), update `updated` when state changes, set `done` when finished with the experiment-log entry named in `log`. Never hold both GPUs unless the other class has nothing runnable; release an idle slot after 2 h; do not end your turn or poll in a loop while a job is pending. Run `python3 scripts/check_priority_registry.py` after edits and fix every ERROR.

## Polaris waits (token cost)

Agents do not idle-wait on Polaris jobs (waits over about 270 s expire the subagent prompt cache and force a full context re-write). After submitting a job, record job id, expected runtime, log path and exact resume steps in the supervisor-log or your report, then stop. The main session watches with a Monitor until-loop on sacct and resumes you (SendMessage) or starts a fresh agent. Check sacct once shortly after submit for fast failures.

PRE-SUBMIT INPUT CHECK: Before sbatch, run `tar -tzf` on the code tarball and confirm every path the job script's input check lists is present. Run the script's CPU-test stage first, locally or on the sandbox 1080 Ti. A job that fails in its first 2 minutes costs a full queue wait (jobs 4977 and 5031, 2026-11-03/04). Rule: user, 2026-11-04.

## Independent review before reporting (add to every dispatched prompt)

Before you report a result, diagnosis or fix as done, request a fresh reviewer subagent (no hypothesis primed, no access to your interpretation) and include its verdict in your report. Give it: the raw result file, the pre-registered pass and kill lines, the claim, and the code that produced the result (scripts, commit hash, job script, flags). It recomputes the headline number, checks the code does what the claim says (metric definition, seeds, baseline file, gates), and states whether the verdict follows. For rollout images also use `frame-artifact-review-prompt.md`; for code changes also run a fresh-context code review before merge. Log the reviewer's verdict in the supervisor-log and experiment-log entries. If review did not run, say so in the report. Rule: user, 2026-10-29.

AMENDMENT GUARD: Pre-registered pass and kill lines are fixed before launch. Any amendment is logged as a deviation and must name a concrete value that would still fail it. Never resolve a gate failure by loosening it; escalate the amendment to the main session as a needs-user item. Rule: user, 2026-11-03.

PRE-REG CLAUSES: Pre-reg must state (a) a tie-break rule for any selection or argmin metric, and (b) convergence or accuracy lines as ratios against the reference's own error floor, not absolute values, when the reference is itself numerical. Rule: user, 2026-11-03.

COMMITTED DEPENDENCIES: Before logging a result, confirm every imported script is committed on your own branch (check git ls-files or a clean-checkout import). List any dependency that exists only on another branch or worktree. Rule: user, 2026-11-03.

CURIOSITY EXPERIMENTS: you may propose an experiment as "curiosity" (no pre-registered prediction, exploration or testing). Your direct supervisor must approve it; a supervisor may approve or escalate to the main agent. Label it curiosity in the registry and notes, log what was run and observed, never report a pass/kill verdict. Rule: user, 2026-11-03.
