# Tiered hierarchy: 10 proposed research directions, 5 selected (2026-10-29)

Process: two hypothesis agents (mechanism-first, requirements-first) each proposed 5 directions; a third agent selected the best 5. All 10 were checked against the supervisor-log "already tried" list; none repeats a killed lever. Proposers' files: `mechanism-first`, `requirements-first` (job scratch). Nothing here has been run.

Abbreviations: BL = tiled baseline; dE@100 = relative energy drift at tick 100; err@20 = mean position error per body at tick 20; PE/KE = potential/kinetic energy; e44b796 = commit of the tiled closed-loop tier 0 (job 4060, branch `tier0-tiled-closed-loop`).

Caveat: `train_tiered.py`, `tiered_iface.py` and `tier0_close_pair_localise.py` exist only on that branch, not on master.

## Summary

| # | Name | Source | Verdict |
|---|---|---|---|
| 1 | virial-and-loop-work-decomposition | mechanism-first | SELECTED |
| 2 | restart-from-truth-drift-rate | mechanism-first | SELECTED |
| 3 | visited-state-label-replay-tier-0 | requirements-first | SELECTED |
| 4 | integrator-compensation-test | mechanism-first | SELECTED (diagnosis only unless it fires) |
| 5 | dense-core-refinement-residual-tier | requirements-first | SELECTED (conditional, run last) |
| 6 | coherent-radial-force-bias | mechanism-first | dropped |
| 7 | chaos-floor-noise-control | mechanism-first | dropped |
| 8 | calibrated-force-error-channel-on-visited-states | requirements-first | dropped |
| 9 | two-upper-tier-exact-moment-stack-span-scaling | requirements-first | dropped |
| 10 | mass-conditioned-hierarchy-with-merged-mass-squared-moment | requirements-first | dropped |

## Run order

1. One shared frozen-model rollout dump on e44b796, then diagnoses 1, 2, 4 in parallel (no training, about 1 GPU-h, one Polaris job).
2. Fix arms fire only if their diagnosis passes, each as its own run against frozen e44b796. If 2 says visited-state, run 3 (same fix arm as 2's `visited-state-force-replay`, run once). If 1 passes, run the soft curl/drag-penalty fine-tune. If 4 passes, run the dt 0.05 fine-tune.
3. Run 5 last, only if the dense-core loss is still unexplained.

Overlaps: 2 and 3 share a fix arm. 1 and 4 both explain "PE too negative, KE not up", so each can falsify the other.

## Selected

### 1. virial-and-loop-work-decomposition
- Idea: measure virial ratio, closed-loop work of the learned force, and a drag regression on frozen e44b796. If the force does non-conservative work, fine-tune with a soft curl/drag penalty.
- Pros: diagnosis needs no training (0.3 GPU-h); targets the one unexplained signature (PE over-bound, KE not rising; a coupling error in a conservative force would raise KE); pre-registered lines (pass: virial 10% low and loop work >= 0.03 or drag >= 40% of the effect; kill: loop work < 0.005 and drag < 10%); the KE claim is currently stitched from two runs (jobs 4079, 4064) and this measures it on one rollout.
- Cons: fix arm is a soft conservative prior (allowed, but near the no-hard-coded-forces rule, keep it soft and flag it); curl may be near zero.
- Why selected: highest information per GPU-h on the drift gap and untouched by killed levers.

### 2. restart-from-truth-drift-rate
- Idea: restart the frozen model from truth or its own states at ticks 28 and 60, hidden state warmed up or reset; compare d(dPE)/tick to separate per-tick bias, hidden-state drift and the visited-state effect.
- Pros: inference only (0.5 GPU-h); each cause has a distinct pre-registered signature; tests the open item (free-rollout force error 1.5-1.6x BL vs +6-12% teacher-forced); gates the fix arm of 3.
- Cons: an 8-tick warm-up may not be a clean hidden-state reset; three seeds in the densest third is a small sample.
- Why selected: cheapest way to choose among the remaining hypotheses.

### 3. visited-state-label-replay-tier-0
- Idea: roll out without gradient, label visited states at ticks 20-100 with exact forces (training-time only), fine-tune on one-step force/energy-work error mixed with teacher-forced batches, 2-3 rounds.
- Pros: reaches ticks 28-100 at constant memory (BPTT ran out of memory at k=22); not a longer-horizon penalty or aggregate energy loss, so distinct from killed levers; inference stays O(N), halo exchange kept; pass: force error <= 1.25x BL, dE@100 <= 2.5x BL, err@20 <= 1.0x BL, plain err@20 <= 1.10x e44b796.
- Cons: O(N^2) labels exist on training scenes only (state this); without held-out rounds feedback can drift off-distribution; 3-4 GPU-h; wrong lever if 2 shows a per-tick bias.
- Why selected: the only fix aimed at the named visited-state effect. Both proposers suggested it; run once.

### 4. integrator-compensation-test
- Idea: run the frozen model with two substeps of dt/2; compare oracle force at dt 0.1 vs 0.05. If the model learned to compensate Verlet error, retrain at dt 0.05.
- Pros: 0.2 GPU-h; resolves a log contradiction (hole-1 gravity cut drift 3.5x by halving dt, 0.0149 to 0.0043, but the close-pair localisation ruled dt out from oracle sign alone, ignoring compensation); pass: change >= 30%, kill: < 10%.
- Cons: dt/2 on weights trained at dt 0.1 is not a clean compensation test; fix arm is hyperparameter-flavoured at 1.9x cost, so it stays gated.
- Why selected: very cheap and can falsify 1 or 2.

### 5. dense-core-refinement-residual-tier
- Idea: add a zero-init residual tier below tier 0 at half cell size, active only above a smooth density gate, trained on dense scenes against the frozen tier-0 residual (design hole 8B).
- Pros: architectural, not a loss or hparam change; targets the densest third (all of the PE loss, 76% of energy-work error); resolution is the one untried change among killed levers; touches generalization (dense eval N 727-1098 vs training quantiles up to 273), O(N) via sparse activation, tileability via halo-consistent moments, interpretability via the refinement map; pass lines include a cost line.
- Cons: 5-6 GPU-h, the most of the five; teacher-forced force gain is already 0.98-1.02 at every nearest-neighbour distance, which argues against resolution as the cause; density gate must be differentiable; exact force for the densest 10% closes only 11% of the err@20 gap, so expect a drift gain, not an err@20 gain.
- Why selected: the architectural alternative if 1-4 find a mechanism no loss or data fix reaches.

## Dropped

### 6. coherent-radial-force-bias
- Idea: measure a neighbourhood-averaged inward force bias; add a multi-scale pooled-force loss if present.
- Pros: cheap diagnosis (0.4 GPU-h); analytic Plummer/uniform-sphere probes give effective G and softening (useful for generalization); may explain the 1.55x interior offset at 1e6 bodies.
- Cons: teacher-forced gain already 0.98-1.02 at every distance, so mostly duplicates 2; fix arm is another loss term, same family as killed work-bias, clump-centre and pair-potential terms; overlaps 1 on the same dump.
- Would change decision: 2 shows a per-tick bias at dense states and 1 shows no curl/drag; then run only its diagnosis, including the analytic probes.

### 7. chaos-floor-noise-control
- Idea: inject matched noise into exact force to separate chaos from systematic drift; if chaos dominates, use a tolerance-hinge loss.
- Pros: cheap (0.4 GPU-h); sanity-checks whether the 2.5x line is well specified.
- Cons: drift is already a same-sign loss of about -0.035 on every seed, so systematic, and the control would mostly confirm that; err@20 already passes; fix arm (k up to 40, 3 GPU-h) is another loss-shape lever like killed long-horizon penalties.
- Would change decision: a seed-dependent drift sign, or a result in 2 implicating chaos amplification.

### 8. calibrated-force-error-channel-on-visited-states
- Idea: train a measurement-only head predicting per-cell squared force error on teacher-forced and visited states; judge by Spearman correlation with realised error and drift.
- Pros: 1-1.5 GPU-h; builds the design's unbuilt error-estimate channel and an OOD detector; no force action.
- Cons: does not move the drift gap; its value is as a gate input for 5, which is not built; error is spread across particles, so per-cell predictability is doubtful; depends on the visited-state fix.
- Would change decision: 5 is built and needs a non-density gate (the design says a density gate fails for solids), or interpretability becomes an explicit milestone.

### 9. two-upper-tier-exact-moment-stack-span-scaling
- Idea: build a two-upper-tier stack with exact moment merge; evaluate at spans 256, 512, 1024; re-run RESPA m=4 with a Verlet inner step.
- Pros: only proposal testing the log-depth range and O(N) claim beyond one upper tier; resolves the RESPA confound.
- Cons: 5-7 GPU-h; upper tiers add only 4-10% over exact external field, so it will not move the drift gap (proposer agrees); chaining a second upper tier through `ModelLower` is unverified (classes at `tiered_iface.py` lines 287-318); better after tier 0 is repaired.
- Would change decision: tier 0 meets the drift line, or span scaling is needed for the paper or handoff.

### 10. mass-conditioned-hierarchy-with-merged-mass-squared-moment
- Idea: run the committed mass-conditioning curriculum on the closed-loop tiled recipe, with the mass-squared moment merged exactly through tiers.
- Pros: code and 10 CPU tests exist (`train_scatter_mass_curriculum.py`, commit 0f1c7d8); about 2.5 GPU-h; closes a generalization gap (planetesimal disk is out of range); pass lines pre-registered.
- Cons: unrelated to the drift gap; earlier mass fine-tunes failed the in-distribution gate (1.2-3.9x vs 15% allowed); tier-0 base still fails drift, so it would inherit that; warm start must be the closed-loop checkpoint.
- Would change decision: a free GPU with the drift question parked, or the planetesimal-disk showcase becoming the priority.
