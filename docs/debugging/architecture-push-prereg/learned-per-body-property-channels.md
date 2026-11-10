# learned-per-body-property-channels: pre-registration (stage 1)

Written before any run. Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md, experiment 10, with section 10a overriding. Priority 3.

Definitions. err@k: mean per-body position error at tick k. da/a: relative error of the planet's semi-major axis. de: eccentricity error. dE: relative total-energy error against the truth's energy at the same tick (never E(0)). q: mass ratio of the heavy body to a light one. single-law-gravity-control: the single-law gravity checkpoint, the control.

## Stage 1 (what runs)
Arm massfeat_curriculum of scripts/train_scatter_mass_curriculum.py: given per-body mass as extra deposit channels, geometric-mean mass normaliser (`norm="gmean"`), 7-stage ratio curriculum (q 1,3,10,30,100,300,1000 x 600 iterations = 4200), unit-mass replay, warm start from single-law-gravity-control (`load_compat` zero-pads the new input columns, so the warm start reproduces single-law-gravity-control exactly). Seed 4738.
Control and its checkpoint: single-law-gravity-control, Polaris `~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt` (verified present, 3200305 bytes, 2026-11-10). The `init_by_seed` entry in the results json (one per eval seed, so each seed's gate uses its own baseline) is single-law-gravity-control scored by the same code on the same scenes.

## Lines (verbatim from spec section 7, experiment 10; numbers from docs/theory/mass-conditioned-scatter-field.md lines 34-40, implemented in `LINES`)
- Spec: "stage 1 uses the pre-registered mass lines (planet semi-major-axis error at most 0.02, in-distribution gate 1.15x; KILL above 0.15 or 1.5x)."
- Star + one planet, q in {10,100,1000}: PASS if da/a <= 0.02, de <= 0.03, dE <= 0.02, finite. KILL if da/a > 0.15 (planet q=1000: > 0.30) or dE > 0.2 or non-finite.
- Star + disk (30-200 bodies + planet), same q: PASS if da/a <= 0.05, de <= 0.05, dE <= 0.05. KILL if da/a > 0.25 or dE > 0.3.
- In-distribution gate (unit-mass base scenes, err@5/10/20 against single-law-gravity-control): PASS <= 1.15x, KILL > 1.5x.
- Verdict: PASS all lines; KILL any kill line; otherwise INCONCLUSIVE.
- Full run (not stage 1; spec text): "PASS same lines without mass given AND rank correlation of the inferred property with true mass at least 0.9. Fails pass: correlation 0.8." Fails-pass value for the stage-1 line: planet da/a 0.03.
- Default KILL of section 6 (err@20 above 1.15x control) is the in-distribution gate row above.

## Seeds and noise
Eval seeds 9100/9200/9300 (scene seed = eval seed + q, 16 scenes per cell, 100 ticks; the verdict is the worst verdict over the three seeds). Fresh seeds 12000/12100/12200 before any result is called real; the trainer takes `--eval-seeds` only through its constant `EVAL_SEEDS`, so the fresh set is run by editing that tuple in a separate, logged job. Second training seed 4739 only after a pass.
Noise floor: fine-tunes of the same recipe differ about 10-15% between seeds (MEMORY: select on held-out seeds). The in-distribution gate (1.15x) sits at that floor, so an in-distribution ratio between 1.05 and 1.25 is not separable from seed noise at one training seed; no second training seed in stage 1.

## Tie-break and selection
No selection: the final-iteration weights are scored; no best-checkpoint pick, no argmin. A value exactly on a line counts as meeting a PASS line and not triggering a KILL line (the code uses `<=` for pass and `>` for kill).

## Plateau guard (spec 10a item 1)
A fixed validation set (32 scenes per group at q=1000 curriculum composition, seed 4738+50000, k=20 rollout loss) is scored every 20 iterations. INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window (`still_falling`); it overrides KILL and PASS-miss verdicts only (a PASS stays PASS). Iteration count kept at 4200 (the design's own); one extension at double length, candidate and control together, only if flagged.

## Prediction (my estimate, written before the run)
Planet cells q 10 and 100 pass; disk and planet q=1000 are the weak cells; the in-distribution gate holds at about 1.0-1.1x. Probability of PASS on all three seeds: 25%; most likely INCONCLUSIVE. Earlier mass fine-tunes failed in-distribution by 1.2-3.9x, so a KILL on the gate is the main risk (about 25%).

## Constraints check (spec section 5)
- pp=2, knn=16: unchanged (variant `ms_kp_pot_v_g128_massfeat` keeps the pair term).
- No hard-wired conservation; energy compared with the truth's E(t): yes (`dE_model` vs truth energy at the same tick).
- O(N): the new channels are additive cell sums; normaliser is a per-scene constant; no new pair work.
- Matched budget: candidate 4200 curriculum iterations against single-law-gravity-control's 60000; the control is not retrained. Flagged: not budget-matched in the sense of section 5.4; the full run must use the control's count.

## Artifacts
Job script: scripts/polaris_learned-per-body-property-channels-mass-conditioned-stage-1.sh. Tarball: `bash scripts/build_architecture_push_stage1_code_tarball.sh`. Results: `results/scatter_mass_curriculum_stage1/massfeat_curriculum.json` on Polaris (keys verdict, verdict_reasons, final_by_seed, init, plateau_falling). Estimated 1.0 GPU-h (0.315 s/it from the nearest logged run, untimed here). Full-run skeleton: none, because the inferred-property trainer does not exist yet.
