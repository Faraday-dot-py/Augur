# learned-per-body-property-channels double-iterations retrain: pre-registration (stage 1)

Written 2026-11-23 before the run. Pilot decision: the project is being archived; this is the one last experiment. It resolves the undertrained flag from job 13166 (registered verdict INCONCLUSIVE-undertrained, raw verdict KILL).

Parent pre-registration: docs/debugging/architecture-push-prereg/learned-per-body-property-channels.md (stage 1). Its plateau-guard paragraph already names this step: "one extension at double length, candidate and control together, only if flagged." The control is not retrained; `init_by_seed` rescoring of single-law-gravity-control on the same scenes is part of the trainer.

## What changes
Only the iteration count: `--stage-iters 1200` (was default 600), 7 curriculum stages, so 8400 iterations (was 4200). Each stage doubles, so the curriculum shape and the learning-rate and rollout-length schedules (functions of the stage length) stretch to the new length. Seed 4738, warm start single-law-gravity-control (`E_ms_kp_pot_v_g128.pt`), same recipe, same eval seeds 9100/9200/9300, same scenes, same tarball (md5 7a0120f3aca6599f3d4eeb4ceaf353be), same code (scripts/train_scatter_mass_curriculum.py md5 7881197abd19461c9b39001a6107442a). Launcher: scripts/polaris_learned-per-body-property-channels-mass-conditioned-stage-1-double-iterations.sh, work dir ~/augur_architecture_push_stage1_mass_double_iterations_2026_11_23.

## Lines (unchanged, quoted from the parent pre-registration)
- Star + one planet, q in {10,100,1000}: PASS if da/a <= 0.02, de <= 0.03, dE <= 0.02, finite. KILL if da/a > 0.15 (planet q=1000: > 0.30) or dE > 0.2 or non-finite.
- Star + disk (30-200 bodies + planet), same q: PASS if da/a <= 0.05, de <= 0.05, dE <= 0.05. KILL if da/a > 0.25 or dE > 0.3.
- In-distribution gate (unit-mass base scenes, err@5/10/20 against single-law-gravity-control): PASS <= 1.15x, KILL > 1.5x.
- Verdict: PASS all lines; KILL any kill line; otherwise INCONCLUSIVE. Worst verdict over the three eval seeds.
- Fails-pass value for the stage-1 line: planet da/a 0.03.
- Plateau guard unchanged: INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window; it overrides KILL and PASS-miss only. If the guard fires again at 8400 iterations, the registered verdict stays INCONCLUSIVE-undertrained and the raw verdict is reported alongside. No further extension.

## Comparison reference
Job 13166 (4200 iterations, 894 s): raw KILL, planet da/a at tick 100 0.9-1.6, gate 1.56-1.62x, 342/4200 skipped steps. Reported as is; no amendment.

## Prediction
Written before the run: the 4200-iteration failure was a large margin (planet da/a about 6x the 0.15 kill line, gate just above 1.5x), so doubling is more likely to leave the raw verdict KILL than to rescue it; probability of PASS under 5%, of a guard-free KILL about 60%, of the guard firing again about 35%. Estimate about 0.5 GPU-h (894 s for 4200 iterations, so about 1800 s plus eval), limit 00:50:00.
