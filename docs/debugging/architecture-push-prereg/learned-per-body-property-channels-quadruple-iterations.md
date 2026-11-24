# learned-per-body-property-channels quadruple-iterations retrain: pre-registration (stage 1)

Written 2026-11-24 before the run. User decision 2026-11-24: train the stage-1 mass-conditioned property channels longer. Prior runs: job 13166 (4200 its, raw KILL, guard fired) and job 13236 (8400 its, raw KILL all 3 seeds, guard fired, planet da/a 2.5-9x over the 0.15 kill line).

Parent pre-registrations: learned-per-body-property-channels.md and learned-per-body-property-channels-double-iterations.md. Lines and plateau guard are identical to job 13236 and are NOT amended; quoted verbatim:

Extension note: learned-per-body-property-channels.md allows one extension at double length (job 13236). This run is a second extension, at 8x the original 4200 iterations; it exceeds that allowance and was approved by the user on 2026-11-24.

## Lines (unchanged)
- Star + one planet, q in {10,100,1000}: PASS if da/a <= 0.02, de <= 0.03, dE <= 0.02, finite. KILL if da/a > 0.15 (planet q=1000: > 0.30) or dE > 0.2 or non-finite.
- Star + disk (30-200 bodies + planet), same q: PASS if da/a <= 0.05, de <= 0.05, dE <= 0.05. KILL if da/a > 0.25 or dE > 0.3.
- In-distribution gate (unit-mass base scenes, err@5/10/20 against single-law-gravity-control): PASS <= 1.15x, KILL > 1.5x.
- Verdict: PASS all lines; KILL any kill line; otherwise INCONCLUSIVE. Worst verdict over the three eval seeds (9100/9200/9300). Tie-break: no selection or argmin metric is used; the final iterate is scored.
- Fails-pass value for the stage-1 line: planet da/a 0.03. If an amendment is ever proposed, a value that would still fail the kill line is planet da/a 0.20; none is proposed.
- Plateau guard: INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window; it overrides KILL and PASS-miss only. The raw verdict is reported alongside. No further extension.

## What changes
Only the iteration count: --stage-iters 4800 x 7 stages = 33,600 iterations (4x job 13236's 8400, 8x job 13166's 4200; the task wording said 4x the original but gave the explicit 4800 x 7 = 33,600, which is followed). Seed 4738, same warm start, recipe, eval seeds, scenes, tarball (md5 7a0120f3aca6599f3d4eeb4ceaf353be), trainer code. Launcher: scripts/polaris_learned-per-body-property-channels-mass-conditioned-stage-1-quadruple-iterations.sh, fresh work dir ~/augur_architecture_push_stage1_mass_quadruple_iterations_2026_11_24; it exits 3 on a result json not written by the same job id. One seed, no control retrain (init_by_seed rescoring of the control is part of the trainer).

## Prediction
The guard fires again and the registered verdict stays INCONCLUSIVE-undertrained with raw verdict KILL (13236 already plateaued; skipped-step rate about 8-9% is a stable feature). Probability: PASS under 3%, raw KILL about 85%, guard fires about 60%. Estimate about 1.8 GPU-h (1552 s per 8400 its, so about 6200 s plus eval), limit 3:00:00.
