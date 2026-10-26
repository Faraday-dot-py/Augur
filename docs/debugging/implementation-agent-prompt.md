# Implementation/testing-subagent prompt (reusable)

Companion to research-agent-prompt.md; same standing rules. Used 2026-09-24 for the energy-conservation fixes. Fill `<TASK>`, `<INVESTIGATION_DOC>`, `<FIXES>`.

```
You are acting as Claude Code for user awebb on the Bounce project (/home/awebb/Research/Bounce). Implement and test <TASK>, following the ranked recommendations in <INVESTIGATION_DOC>. Do the fixes in the stated order: <FIXES>.

READ FIRST: ~/.claude/CLAUDE.md, ~/.claude/projects/-home-awebb-Research-Bounce/memory/MEMORY.md and linked memory files, docs/debugging/experiment-log.md, <INVESTIGATION_DOC>, docs/debugging/frame-artifact-review-prompt.md, git log. Reconstruct context from files; don't ask for background.

STANDING RULES: same as docs/debugging/research-agent-prompt.md (no local compute; Polaris GPU only, one job at a time; no inline multiline Python; seed 4738; gravity 9; diagnose by channel; verify before stating; held-out seeds; unbiased video-review subagent + videos/ + SendUserFile; never stop mid-task; minimum words; flag anomalies).

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
