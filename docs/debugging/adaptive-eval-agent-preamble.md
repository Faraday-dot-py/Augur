# Shared brief for the adaptive-far-field evaluation agents (2026-09-25)

Project: Bounce. Requirements (docs/paper/core-requirements.md): (1) generalization to any particle sim, (2) tileable, (3) O(N) or less, (4) interpretability (untested).
System under test ("ALF" = adaptive learned far field): momentum-symmetric dual tree (scripts/dual_tree.py) + learned node-pair error estimator
(scripts/dual_estimator.py, scripts/est_train.py) + end-to-end audit + geometric fallback (scripts/adaptive_force.py). Read the four newest sections of
docs/debugging/experiment-log.md first (oracle test, dual tree, node-pair estimator, kernel-agnostic labels), then ~/.claude/CLAUDE.md,
~/.claude/projects/-home-awebb-Research-Bounce/memory/MEMORY.md (esp. project_adaptive_far_field, project_core_requirements, feedback_*), and the code.

## Shared tools (already written, smoke-tested on Polaris job 3006)
- scripts/kernels.py: RadialKernel (pair force, Jacobian, exact_accel, phi/pair_sums); analytic (softened gravity eps 0.5), inv_distance, yukawa, learned (checkpoints/gravity_central_v1.pt).
- scripts/nbody_ic.py: IC = uniform, flyby, three, plummer, disk, clumpy (N-scaled, seed 4738, virialised from the kernel).
- scripts/est_train.py: train_estimator / save / load; checkpoints/est_<kernel>.pt (analytic and yukawa exist on Polaris; trained on uniform, flyby t0, flyby t1000 only).
- scripts/adaptive_force.py: AdaptiveForce(kernel, head, mode in geo | geo_audit | est | adaptive, target=0.01 audit rel_l2, audit_k, audit_every, lam_max, ...). `stats` (per-call mode, lam, theta, cost = kernel evals per particle, time, audit error) and `events` (fallback / probe).
- scripts/nbody_rollout.py: KDK leapfrog (dt 0.05, 1 force eval/step), providers exact | geo | geo_audit | est | adaptive | bh (target-based Barnes-Hut, not momentum-conserving) | mesh (uniform mesh + cutoff 4). Diagnostics: energy, |P|, angular momentum, COM, radial quantiles; snapshots + `--ref` comparison (position error, radial quantile ratios, density-map correlation); checkpoints + `--resume`; JSON in results/rollout_<tag>.json, snapshots results/rollout_<tag>_snaps.npz (git-ignored).
- Baselines to compare against, always on identical initial conditions: exact all-pairs (truth), uniform mesh, target-based BH, geometric dual tree with the same audit (geo_audit = the ablation of the learned part), estimator without audit (est).
- Exact reference cost: analytic kernel about 0.5 s/step at N=100k; learned/other kernels are much slower (keep N <= 20k for exact learned-kernel rollouts).

## Rules
- All compute on Polaris GPU (mcp__polaris__* tools; load them with ToolSearch "select:mcp__polaris__polaris_run,mcp__polaris__polaris_upload,mcp__polaris__polaris_download,mcp__polaris__polaris_submit_job,mcp__polaris__polaris_job_status,mcp__polaris__polaris_status"). Nothing heavy locally. Upload files individually to bounce/<path> (remote_path is relative to home), run from ~/bounce with PYTHONPATH=., real job log is ~/<job-name>-<id>.log. ONLY ONE GPU JOB PER USER RUNS AT A TIME and other agents share it: jobs will queue. Keep each job under ~40 min, use unique job names and result-file prefixes (your test id), never scancel jobs you did not submit, poll with short sleeps (<= 100 s per polaris_run call).
- Never write multiline Python inline in Bash / python3 -c; write files. Seed 4738 unless several seeds are the point (then 4738, 9000, 12000).
- Do NOT edit existing files in scripts/ or model/ (other agents use them concurrently). New code goes in new files prefixed with your test id. If you find a bug in a shared module, copy the module to a prefixed file, fix it there, and report the bug precisely.
- Commit only your own new files and results with `git add <explicit paths> && git commit` (retry if .git/index.lock exists; never `git add -A`, never switch branches, never push). Commit message carries the key metric. Add a short results section to a new file docs/debugging/adaptive-eval-<id>.md (do not edit experiment-log.md; the coordinator merges).
- Before each experiment write down the predicted outcome; after, compare. Report negative and ambiguous results as plainly as positive ones; do not tune the system to pass. Mark untested claims as untested. Verify numbers against the JSON before stating them. Chaotic trajectories diverge pointwise regardless of force accuracy, so judge long rollouts by conservation laws and statistics (radial quantiles, density-map correlation), and use pointwise error only for short horizons and against a noise-floor control (exact run with a 1e-6 position perturbation).
- Any rollout that produces visual output (density frames, videos): render diagnostic frames of exact vs candidate side by side (existing render_*.py scripts may be reused; render locally only if it needs no heavy compute, otherwise on Polaris), then dispatch a FRESH subagent with no hypothesis primed to describe what it sees using docs/debugging/frame-artifact-review-prompt.md; save videos under videos/. Subagents cannot send files; list the paths in your report.
- Never stop mid-task unless there is a real blocker (Polaris disconnected and needs the user's Duo approval, etc.). Report a blocker immediately in your final answer with what was completed.
- Final answer: at most 350 words plus one table: what you ran (job ids, commits), the numbers that answer your test's questions (each against its stated pass criterion), what failed or is untested, and anything needing the user's decision. Minimum words otherwise.

## What "better" means (claims being tested)
H1 accuracy: at the same audited force-error target, rollouts conserve energy, momentum, angular momentum at least as well as the baselines and reproduce exact-run statistics within the chaos floor.
H2 efficiency: fewer kernel evaluations per particle and lower wall time than geo_audit at equal audited error on high-contrast states, and not worse on uniform density.
H3 robustness: broken or shifted estimators are detected and handled (no silent failure), and healthy estimators rarely trigger false fallbacks.
H4 generalization: unseen initial conditions, particle counts and force laws still meet the target error via the audit, keeping the cost advantage or degrading gracefully to the geometric rule.
H5 scaling: cost per particle is constant in N and wall time is linear.
H6 tileability and H7 interpretability are exploratory.
