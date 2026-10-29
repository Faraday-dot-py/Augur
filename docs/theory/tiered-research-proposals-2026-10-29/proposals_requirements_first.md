# Tiered hierarchy: 5 requirements-first directions

Lens: judge and improve the hierarchy against generalization (G), tileable (T), O(N), interpretability (I). Standing rules respected: no hard-coded conservative force, no pairwise pp=8, no sweeps as main strategy, energy term = similarity to truth E(t), CFD ignored.

Facts checked: closed-loop tiled tier 0 (job 4060, e44b796): tick-20 err 0.75-0.82x tiled baseline, drift@100 3.9-4.75x (line 2.5x). Teacher-forced force gain 0.98-1.02 at every nn-distance (job 4079), yet PE too negative at r 0.5-4, densest third only, growing after the 28-tick training horizon. Free-rollout force error 1.5-1.6x baseline vs +6-12% teacher-forced (log 10-26 12:30 entry). Upper tiers add 4-10% over exact external field (log 10:25). Only ONE upper tier has ever been built (tiered_iface.py Lower classes: Zero/Range/Model; hole 6 two-tier stack never run: untested). Earlier dense `MultiLevelScatterField` (scatter_field.py:826) was 3-10x worse than dense v1 (scatter-field-architecture.md finding 4); it is a different design (learned NGP levels, no exact moment merge), so not a precedent for exact-merge tiers.

---

## 1. visited-state-label-replay-tier-0 (DAgger-style, no BPTT)

Idea: roll the closed-loop tiled tier 0 forward 100 ticks without gradient, store the states it actually visits (ticks 20-100), label them with exact force (training scenes are small, O(N^2) labels are data-side only), and fine-tune tier 0 on one-step force/energy-work error at those states mixed with ordinary teacher-forced batches. Repeat 2-3 rounds.

Observation: free-rollout force error 1.5-1.6x baseline vs +6-12% teacher-forced; over-contraction of dense clumps grows after tick 20-28 (job 4079); position error already beats baseline, so remaining deficit is off-distribution states beyond the unrolled horizon (max k=28; k<=22 OOM'd for BPTT).

Mechanism: removes the train/visit mismatch directly at ticks 28-100 with a constant memory cost (one-step graphs), which BPTT cannot reach. New vs killed levers: not a longer-k penalty (no unroll), not rollout-in-loop for tier 1 (that was exposure for tier 1, "no gain"), not a loss term on aggregate energy; it changes the data distribution of tier 0. Not a hparam sweep.

Experiment (seed 4738 first, held-out 9100/9200/9300, tiled 4x4 x 64-unit, same scenes as job 4060): start from e44b796 checkpoint; 3 rounds x (100-tick roll of 24 train scenes, label, 1500 iterations). Arm vs frozen e44b796.
- Pass: free-rollout force error <= 1.25x baseline (from 1.5-1.6x) AND drift@100 <= 2.5x baseline AND tiled err@20 <= 1.0x baseline AND plain err@20 <= 1.10x e44b796.
- Kill: drift@100 > 3.5x baseline after round 3 OR tiled err@20 > 1.25x.
Cost: ~3-4 GPU-h (job 4060 was 1.35 h for 28-tick BPTT; rollouts no-grad are cheap).
Requirements: G (visited-state robustness, horizon), T (keeps halo exchange as tier-0 definition). O(N) unchanged at inference. I neutral. No violation.

## 2. dense-core-refinement-residual-tier (hole 8B, untested)

Idea: add a residual tier BELOW tier 0 at half cell size, active only in cells above a smooth density gate g(rho), trained on dense scenes against the frozen tier-0 residual (truth - tier 0), exact-moment-consistent with tier 0 and blended at the activation boundary.

Observation: dense bins hold 76% of the energy-work error (2.1x baseline, 3.3x ticks 10-30, force error only +9%); the whole PE loss sits in the densest third (-0.065 of |E0|, job 4079); dense scenes are OOD-ish (eval N 727-1098 vs training window quantiles up to ~273, log 10-26 23:40). Tier-0 cell h=0.5 (mass-conditioned doc). Resolution as cause is untested, since all loss terms and the corrector failed while no resolution change was tried.

Mechanism: near-field of tight clumps resolved at finer deposit/readout, active cost proportional to dense cells (O(N)); tier structure also supplies a clean "where was fine detail needed" signal. Differs from killed levers (loss terms, corrector head, joint fine-tune): changes resolution/capacity where the error lives, not the objective. Caution: exact force for densest 10% closed only 11% of the tick-20 err gap (corrector oracle), so expect gain on drift, not tick-20 error.

Experiment: gate cheap check first (1 GPU-h): re-score job-4060 tier 0 at 2x grid inference-only is impossible (weights tied to h), so train refinement tier from zero-init output head (identity at start) on dense harvests (mid/dense quantiles), 2 seeds of scenes, 4738 first. Eval: dense-bin energy-work error, drift@100, err@20, mid-density beta, cost vs gate fraction.
- Pass: dense-bin energy-work error <= 1.5x baseline (from 2.0-2.3x) AND drift@100 <= 2.5x baseline AND tiled err@20 <= 1.0x baseline AND active-tile cost <= 1.3x tier 0 on a 12.5%-occupancy scene.
- Kill: dense-bin work error > 1.9x OR err@20 > 1.1x after full training.
Cost: ~5-6 GPU-h. Requirements: G (dense OOD), O(N) (sparse activation), T (halo-consistent), I (explicit "this core needed refinement" map). Risk: tier index no longer scale order; gate must be differentiated (phi*grad g). Violates none.

## 3. calibrated-force-error-channel-on-visited-states

Idea: add a small head (no action on forces) on frozen tier 0 predicting per-cell squared force error, trained on BOTH teacher-forced and own-rollout visited states (exact labels); export it as the "error-estimate channel" from the design notes. Judge it as an interpretability and OOD-detection requirement first; only then (separate later decision) feed it to tier 1 or the refinement gate in direction 2.

Observation: error-estimate channel is in the design (hole 3 amendments) but never built; error is spread (top 10% density closes 11% only), so whether a per-cell estimate has any predictive power is open; drift metric (|E_model(t)-E_truth(0)|/|E0|) is scene-level and currently has no per-scene predictor; OOD density (3x beyond training) fails silently today.

Mechanism: gives the hierarchy a quantitative reliability map; if it correlates with realised drift it becomes the opening-criterion analogue and a per-run monitor for generalization. Distinct from the killed residual corrector (that tried to correct forces and barely learned: val ratio 0.97-0.99); this only measures.

Experiment: freeze job-4060 tier 0, train head on teacher-forced + visited states of ~1.7M particle-ticks (reuse diagnosis harness), seed 4738; evaluate on held-out 9100/9200/9300 and on dense OOD scenes.
- Pass: per-cell Spearman >= 0.5 with realised squared force error on held-out visited states AND scene-level Spearman >= 0.6 between summed predicted error (ticks 0-20) and drift@100 over 72 scenes AND same Spearman >= 0.4 on OOD dense set.
- Kill: per-cell Spearman < 0.3 or scene-level < 0.3.
Cost: ~1-1.5 GPU-h. Requirements: I (primary), G (OOD flag). O(N) (one more head). No violation (no force action; no conservative prior).

## 4. two-upper-tier-exact-moment-stack-span-scaling

Idea: build the first stack with TWO upper tiers (tier 1 = 2x2 merge of tier-0 windows, tier 2 = 2x2 merge of tier 1) with exact moment merge, polynomial-cell local expansions and one-step-lagged top-down hand-off; evaluate on spans 256 (current), 512, 1024, with the multi-rate (RESPA m=4) arm re-run with a Verlet inner step to remove the confound recorded for its earlier MIXED result.

Observation: only a single upper tier and 4x4x64 windows (span 256) tested; upper tiers add 4-10% over exact external field there; tiered-hierarchy arm earlier KILLED for far .575 vs .438 tiled-ghosts-far-path and cost 2.9x (first-generation, tier 0 pre-matched-budget); RESPA m=4 gave 1.04-1.06x err at 0.53-0.54x cost but dE unresolved (non-Verlet inner step). Whether the stack scales (tier count ~log2(span/64), cost ~4/3) is the O(N)/T/G claim of the architecture and is unmeasured beyond one tier.

Mechanism: demonstrates the log-depth range claim; separates upper-tier supply error (small) from tier-0 error (the real gap), so that hierarchy gains can be quoted as scaling results rather than as drift fixes. Honest expectation: will not fix drift (upper tiers are not the bottleneck); it is the requirement-proof experiment.

Experiment: use the matched-60k + closed-loop tier 0; tier 1 and tier 2 sequential-frozen fit against exact-field residual; seed 4738 train, eval scenes at span 256/512/1024 (direct sum / tree truth). Arms: one-tier-with-bigger-top-tile, two-tier stack, two-tier + RESPA m=4 (Verlet inner).
- Pass: far-field force error at span 1024 <= 1.15x the exact-external-field arm AND wall-time per particle at 1024 <= 1.5x that at 256 AND RESPA arm dE@100 <= 1.1x its synchronous twin.
- Kill: far-field error > 1.3x exact-field arm at 1024 OR cost per particle grows > 2x from 256 to 1024.
Cost: ~5-7 GPU-h (tier 2 data gen + fit ~3 h, eval runs ~2-3 h). Requirements: O(N), T, G (span). I (per-tier potentials are separable and inspectable). Violation: none (no pairwise; gradient-form potential kept). Untested claim to verify first: that existing `ModelLower` supports chaining a second upper tier (tiered_iface.py:287-318 read only by name).

## 5. mass-conditioned-hierarchy-with-merged-mass-squared-moment

Idea: run the already-written per-body-mass design (docs/theory/mass-conditioned-scatter-field.md; code and 10 CPU tests committed, never trained) on the closed-loop tiled tier-0 recipe of record, and thread the mass-squared moment through the exact tier-k 2x2 merge so upper tiers see "who is in this cell" (star vs crowd). Arms: features only, curriculum only, both.

Observation: models take unit mass only; planetesimal-disk test is out of range (log 10-26 13:52); earlier mass fine-tunes failed in-distribution gate (1.2-3.9x vs 15% allowed, s46); design pre-registered but "code ready, not run", and its tier-k merge claim is untested through the hierarchy. Heterogeneous mass support is a listed requirement piece.

Mechanism: moments are mass-weighted additive sums, so tiers aggregate heterogeneous mass exactly and tileably (m0 is a scene constant, not a tile statistic); kernel path carries the linear-in-mass part. Zero-init warm start protects the unit-mass regime.

Experiment: existing `train_scatter_mass_curriculum.py` arms, seed 4738, then evaluate through the tiled stack on star+disk scenes spanning two tiles (tier 1 required).
- Pass (existing lines, unchanged): planet scenes q in {10,100,1000}: da/a <= 0.02, de <= 0.03, dE <= 0.02; disk scenes da/a <= 0.05, dE <= 0.05; unit-mass gate <= 1.15x; plus new line: tiled over untiled da/a <= 1.5x for star+disk straddling a tile seam.
- Kill: any existing kill line (da/a > 0.15, planet q=1000 > 0.30, dE > 0.2, gate > 1.5x) or tiled/untiled > 2.5x.
Cost: ~1 GPU-h (curriculum estimate, unmeasured) + ~1.5 GPU-h tiled eval = ~2.5 GPU-h. Requirements: G (mass range, primary), T (m0 constant), O(N) (additive sums), I (mass-weighted moments readable). No violation (kernel linear in mass is existing soft structural prior, flagged; no new hard-coded force). Caveat: tier-0 retrain no-velocity checkpoint was KILLED, so the warm start must be the matched-60k/closed-loop checkpoint, not job 4050.

---

Order by value per GPU-h: 1, 3, 5, 2, 4. Directions 1-2 target the open drift gap; 3-5 close requirement gaps (I, G/mass, O(N)/span) that no current result covers.
