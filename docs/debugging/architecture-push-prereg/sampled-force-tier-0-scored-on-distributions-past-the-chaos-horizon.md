# sampled-force-tier-0-scored-on-distributions-past-the-chaos-horizon: pre-registration (stage 1)

Written before any run. Spec: experiment 14, section 10a overriding. Priority 2.

Definitions. Chaos horizon: the tick after which an exact run's positions diverge from a slightly perturbed exact run. Ensemble: 16 exact float64 runs whose initial positions are the scene plus rel x L x gaussian (L = rms radius of the scene). Bin: one of 13 radial bins about the centre of mass, 12 log-spaced from 0.5 to 64 plus an overflow bin. Density profile: bodies per bin. Velocity spread: rms speed per bin relative to the centre-of-mass velocity. A bin is scored when the ensemble-mean count is at least 2; it is inside when the model value lies within the ensemble mean +- 2 standard deviations (ddof 1).

## Stage 1 (what runs)
Score the UNCHANGED control at tick 300. Control: single-law-gravity-control (single-law gravity, control G), Polaris `~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt` (verified). Scenes: single-law-gravity-control's own distribution (half box, half cluster), 16 scenes per seed. The control is rolled once (deterministic) from the unperturbed state. Primary perturbation rel 1e-4 (builder-fixed; between float32 round-off and the 1e-2 end of chaos_floor.py's range); rel 1e-5 and 1e-3 are sensitivity, never decisive. K = 16 members.

## Lines (verbatim, spec experiment 14)
- "PASS at tick 300, at least 80% of radial bins of the density profile and of the velocity spread fall inside the exact ensemble's two-standard-deviation band AND ensemble-mean err@20 at most 1.10x control. KILL under 50% of bins or err@20 above 1.25x. Fails pass: 65%. Stage 1 scores the unchanged control on the same distribution metric; if the control already passes, the experiment is closed as unnecessary."
- Stage 1 operationalisation (builder-fixed): control counts as passing when BOTH the density and the velocity-spread pooled bin fractions are at least 0.80 on EACH of seeds 9100, 9200 and 9300 (worst of six numbers). The err@20 clause is trivial for the control (ratio 1.0 against itself) and is recorded only for the full run. Outputs: CONTROL-PASSES-CLOSED, CONTROL-BELOW-PASS-OPEN, CONTROL-BELOW-KILL-FRACTION (worst below 0.50, the candidate's KILL fraction, shown for scale only).
- A control pass is confirmed on fresh seeds 12000/12100/12200 before the experiment is closed (run in the same job); a disagreement between the two sets is reported as INCONCLUSIVE.

## Noise floor (ratio lines against the reference's own noise floor)
Leave-one-out coverage of the exact members themselves (each member scored against the band of the other 15) is reported as `self_frac`. An exact member inside its own ensemble's band is the ceiling; a control fraction is read against it. If `self_frac` is below 0.80 the 80% line cannot be met even by exact dynamics and stage 1 is reported as INCONCLUSIVE-metric-ceiling (needs-user: a concrete amended line would be 0.9 x self_frac, which still fails the control's value if the control is below it).

## Seeds and tie-break
Eval seeds 9100/9200/9300, fresh 12000/12100/12200, training seed 4738 (no training in stage 1). Inside-band test is `<=` with tolerance 1e-12, so a value exactly at the band edge is inside. A bin with zero ensemble spread passes only on equality. No selection or argmin anywhere.

## Plateau guard
Stage 1 has no training. For the full run: INCONCLUSIVE-undertrained if mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window; iteration count kept at the control's (60000).

## Prediction (my estimate)
The control fails: it over-contracts clumps at mid range (spec), so inner-bin density falls outside the band; expected pooled fractions 40-70%. Probability the control already passes (experiment closed): 15%. Fresh seeds are expected to agree within 10 points.

## Constraints check
- pp=2, knn=16, O(N): stage 1 does not change the model. The exact reference is O(N^2) float64 on 16 x 17 runs, as an evaluation tool only.
- No hard-wired conservation; energy as a distribution is compared with truth only in the full run (energy is not in the stage-1 metric; flagged: the spec's third quantity, energy, is not scored in stage 1).
- Matched budget: not applicable in stage 1.

## Artifacts
scripts/scatter_distribution_ensemble.py, scripts/score_scatter_distribution_control.py, tests/test_scatter_distribution_ensemble.py. Job script: scripts/polaris_sampled-force-tier-0-scored-on-distributions-past-the-chaos-horizon-stage-1.sh (18 results expected: 6 seeds x 3 rels, checked). Estimated 0.5 GPU-h (untimed; spec stage-1 figure is 1). Full-run skeleton: none (the sampled-output trainer does not exist). Results: `results/scatter_distribution_control/summary.json` on Polaris.
