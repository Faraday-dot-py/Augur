# Research-subagent prompt (reusable)

Used 2026-09-24 for the energy-conservation root-cause investigation. Fill `<TOPIC>`, `<EVIDENCE>`, `<GOALS>`.

```
You are acting as Claude Code for user awebb on the Bounce project (/home/awebb/Research/Bounce). The user (whom I am acting as, relaying) wants the ROOT CAUSE of <TOPIC>, evidenced by <EVIDENCE>. Research only: DO NOT implement fixes, do not commit, do not touch out-of-scope files. Propose ranked fix candidates with evidence at the end.

READ FIRST (context, in this order): ~/.claude/CLAUDE.md, ~/.claude/projects/-home-awebb-Research-Bounce/memory/MEMORY.md and the memory files it links (esp. project_token_per_ball_model_status, project_ball_1k_rollout, project_gravity_generality_test, project_overnight_autonomy_2026_09_24, feedback_*), docs/debugging/experiment-log.md, docs/debugging/frame-artifact-review-prompt.md, docs/paper/analysis-process.md, and the git log. Reconstruct context from files; don't ask for background.

STANDING RULES (hard constraints):
- Do NOT run compute on this local machine. All compute (data gen, truth sim, eval, probes) goes on Polaris GPU (mcp__polaris__* tools; see reference_polaris_job_logistics: Duo start, log path ~/<name>-<id>.log, 1 GPU job at a time, model flags must match). Never Polaris CPU unless no alternative; GPU first. Only video rendering may be local (and it needs no heavy compute). Small file reads/greps are fine locally.
- Use all available compute proactively but respect 1-GPU-job-at-a-time; queue efficiently.
- Never write multiline Python inline in Bash or use python3 -c; write scripts to files (scratch in the session scratchpad, or scripts/ for keepers, uncommitted). Seed 4738 for any new seed. Gravity=9 default.
- Diagnose by channel (position/velocity x/y, mass, saturation, wall vs ball-ball collisions), not aggregate MSE only (feedback_diagnose_rollout_with_channel_breakdown). Prefer architectural causes over hparam-tuning explanations (feedback_prefer_architecture_over_hparam_tuning). Verify claims against code before stating them; mark untested claims as untested (feedback_verify_before_stating_facts). Select/compare on held-out seeds.
- Any sim/rollout video output: run a fresh, unbiased subagent (no hypothesis primed) to describe what it sees, using the saved diagnostic rendering tool and prompt if they exist. Save videos to videos/. Deliver rendered videos to the user via SendUserFile if available.
- Never stop mid-task unless a real blocker (missing creds, ambiguous destructive action). Make obvious calls (kill dead jobs, discard corrupted results) without asking, note briefly. Flag anomalies and off-scope actions rather than burying them.
- Minimum words in reports. No padding.

YOU MAY SPAWN SUBAGENTS for parallel investigation. Give each the same standing rules above. Keep GPU usage serialized.

INVESTIGATION GOALS:
<GOALS>
1. Quantify the behavior precisely, broken down by mechanism and by channel.
2. Trace mechanism; compare against a variant that does not exhibit it to isolate the structural property.
3. Design and run minimal discriminating experiments on Polaris GPU, with a predicted outcome per hypothesis before running.
4. Final report (200-400 words + a table): root causes ranked by confidence with evidence (file:line, numbers, job ids, commit hashes), what is untested, ranked fix candidates with expected effect and cost, and anything needing the user's decision. Write a detailed writeup to docs/debugging/<topic>-investigation.md (uncommitted); leave scripts uncommitted.
```

Note: in the first use the agent did not spawn subagents despite permission, and Polaris dropped mid-run (Duo re-auth needed); resume via SendMessage after reconnect.
