# Scatter-field hardening plan (2026-10-04)

Goal: get the scatter-field model into its best state before resuming interp work (interp is PAUSED by user until this finishes).
Branch base: interp-architecture @ 737aa1a. Each agent works in its own git worktree/branch; no push/merge/PR.
Model code: scripts/scatter_field.py, train_scatter_field.py, scatter_bh.py, profile_scatter_step.py, test_graph_step.py (CUDA-graph recipe, not integrated).
Reference docs: docs/debugging/scatter-field-{architecture,design-decisions,fixed-budget,overnight-notes}.md, scatter-rollout-review-prompt.md.
Headline checkpoint: Polaris ~/bounce/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt (Exp E, 60769 it; +.itNNNNN intermediates). Selection among them must use held-out seeds (9100, 9200, 9300, 4738).
Current numbers: BH 300 bodies c=10, 100 ticks, err@5/10/20/50/100 .008/.023/.067/.90/4.9. Exp B (10-100 bodies) .0006/.0014/.0039, @100 .374 (CentralForce .0006/.0011/.0025, .278).

## User decisions (2026-10-04)
- "100k" = PARTICLE COUNT. Sweep N at quarter-decades 1,1.78,3.16,5.62,10,...,1e5 (rounded). Rollout length held constant, modest (default 100 ticks; fewer for largest N if cost forces it, state it).
- "Grid" = scale of the WORLD (domain extent) being computed, not just resolution of a fixed grid. Same weights, no retraining, when world size / cell size / resolution change. Sweep world size independently of N, and jointly (constant density vs constant box).
- Scenes: zero-shot first, then fine-tune, then full train. Report all three; user is curious about zero-shot.
- Channel: add signed CHARGE (Coulomb-like, attractive/repulsive) alongside mass. Warm-start from Exp E. Predict time and memory complexity BEFORE measuring; report prediction vs measured.
- Performance problems found by any agent go to the optimization agent and get fixed before retrying.

## Standing rules (all agents)
docs/debugging/research-agent-prompt.md + implementation-agent-prompt.md rules apply: Polaris GPU only (1 GPU job at a time -> timing benchmarks need an exclusive GPU, serialize them), no local compute, no inline python, seed 4738, gravity default, checkpoints for long runs, predict-before-measure, unbiased video review subagent (scatter-rollout-review-prompt.md) + videos/ + SendUserFile from main session, commit scripts+results with key metric, never stop mid-task.

## Agents
0. BASELINE/REGRESSION HARNESS (first, blocks 1 and 4). Frozen eval: pinned ckpt, held-out seeds, err@5/10/20/50/100, dE/E, dp, dL, tick time, peak GPU mem, for N in {2, 10-100, 300 BH}. One script + JSON baseline + pass/fail gate (err within seed noise ~10-15%, no speed regression). Output: scripts/scatter_regress.py, results/scatter_baseline.json, docs/debugging/scatter-baseline.md.
1. OPTIMIZATION (after 0). Inference + training speed. Known leads: per-step CUDA graphs 1.45x (integrate), torch.compile unavailable on Polaris (no Python.h), TF32/bf16 no gain on training, batch not the limiter, launch-latency bound at small batch. Also: kernel fusion by hand, fp16 state, cell-list/neighbor build cost, FFT/mesh cost, memory layout, fewer syncs. Gate: harness pass. Report speedups per N.
2. SCALING (after 0; parallel with 1 in accuracy-only mode, exclusive GPU for timing). N sweep above, world-size sweep, joint sweep, no retrain. Metrics: err@k vs exact/reference where feasible, conservation, tick time, memory, first non-finite step. Perf issues -> agent 1, fix, rerun.
3. SCENES: bounce (walls/contact; expect OOD), black hole, orbit, globular cluster. Zero-shot -> fine-tune -> full train; compare. Pass criteria to be defined per scene by the agent before running (state in doc).
4. CHANNEL (last, own branch, after 1-3 stabilize the base): add charge channel; predict then measure time/memory complexity (per-ball ops, mesh/grid ops, memory vs N and channel count C); warm-start from Exp E.
5. CHAOS-FLOOR: perturbed-truth baseline to separate irreducible divergence from model error at err@50/100 (BH and Exp B).
6. CONSERVATION/LONG-HORIZON: energy, momentum, angular momentum drift, |v|<c cap, close-encounter blowups over 1e3-1e4 ticks at moderate N.
7. OOD STRESS: mass ratios, velocity range, density extremes, softening/close passes, dt, boundaries.
8. PARITY+DOCS (last): JS port (web/scatter) parity after optimization; handoff docs.

## Sequencing
0 -> {1, 2 (accuracy), 3, 5, 6, 7 interleaved around GPU lock} -> 4 -> 8.
Launch order now: 0, 5 (no code changes, analysis only). Then 1, 2, 6, 7, 3 as 0 completes. 4 and 8 later.
Timing-sensitive runs (1, 2 perf, 4) must not overlap other GPU jobs.

## Status log (append below; newest last)
- 2026-10-04 13:30: plan written. Nothing launched yet.
- 2026-10-04 13:45: agent 0 DONE (branch scatter-regress-harness 103be23, baseline job 3296). Launched agent 1 optimization (ac96d685efadf8898, branch scatter-opt) and agent 2 scaling (ad7f2dd079149ecc1, branch scatter-scaling). Agent 5 chaos-floor (a3afdebef86797bbc) resumed after Polaris reconnect. Pending launch: 6 conservation, 7 OOD, 3 scenes, then 4 charge, 8 parity.
