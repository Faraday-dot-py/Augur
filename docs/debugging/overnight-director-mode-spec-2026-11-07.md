# Overnight director mode: spec (DRAFT, awaiting user confirmation)

Expires automatically at 09:00 PST 2026-11-07 (epoch 1794070800). After that, normal behavior resumes and this file becomes the top item for the human to review.

## Priorities (highest first)

1. Keep all 8 Polaris GPUs busy. The 5-minute idle-check cron and filler burns stay in force. Nothing below may displace this.
2. Convert burn GPUs into useful research, by acting as the human research director for the Augur project.

## Hard constraints

- No destructive actions: no deleting, overwriting or force-resetting files, branches, results or jobs (cancelling a filler burn to start a real job in its place is the only job cancel allowed). When unsure whether something is needed, commit it and flag it for cleanup later.
- Every job still self-stops at 09:00 PST 2026-11-07, uses QOS student-pleiades, 1 GPU, no nodelist, 1 CPU, no inter-GPU jobs.
- Standing rules apply: pre-registered pass/kill line before launch (or label "curiosity" with supervisor approval), experiment-log entry when a job finishes, independent review before reporting a result, commit scripts and results with the key metric, descriptive kebab-case names, seed 4738 first.

## Life cycle for each new idea (the human's practice)

1. Broad idea.
2. Validity check: is it already in the "already tried" list in docs/debugging/experiment-log.md? Does it fit the core requirements (generalization, tileable, O(N) with pp=2, interpretability)?
3. Feasibility test: smallest run that can fail fast, with a pass/kill line written first.
4. Experiment: full run only if the feasibility test passes.
5. Results: logged, independently reviewed, committed.
6. Decision: go, kill, or park, following the human's past decisions on similar results.

## Decision source

A fresh subagent reads the human's past prompts and decisions about job results (transcripts and the decisions memories) and writes a short decision-rules file. The head agent follows those rules where they clearly apply. Where they do not clearly apply, the head agent parks the decision in a needs-user list instead of guessing.

## Already-finished results waiting on this process

- Neuron-graph extra training seeds bundle (6163): log downloaded, not yet logged or reviewed.
- Sharpness-aware minimisation with Adam (6167): early kill at iteration 1000, not yet logged or reviewed.
- Replay on second training base, fresh-seed eval (6168): evaluation inconclusive, later block says PASS after two failed steps, needs a close read.
- Solar n/k/m regimes second seed (6160): all four regimes KILL on replication.
- Arm A scatter-core drift, fresh seeds (6169): KILL, logged, no independent review.
- Interp and PM baseline completion bundle (6164): INCONCLUSIVE_PARTIAL.
- Still running: tier0 third and fourth training seeds (6165, 6166).

## Open questions for the human

Answers (02:3x PST 2026-11-07):
- Launch rights: full auto. Feasibility tests, then full experiments when feasibility passes, under pre-registered pass/kill lines.
- Unclear calls: park in the needs-user list, pick another clearly covered idea.
- Idea source: existing queues and fresh brainstorming; every fresh idea passes the validity check against the already-tried list.
- Supervisor role: head agent directs, supervisor agent executes approved jobs via its saved prompt.
Still open: the user has not yet confirmed the written spec as a whole.

## Expiry and handoff

At 09:00 PST: delete the idle-check cron, stop submitting fillers, switch back to normal behavior, and put this file's outputs (needs-user list, decisions taken, jobs launched, GPU-hours used) at the top of the human's morning review.
