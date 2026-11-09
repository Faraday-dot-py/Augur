# Top-8 first-pass triage (2026-10-29)

Read-only triage of rows 1-8 of docs/debugging/unexplored-directions-audit-2026-10-29.md. Nothing was run. One worker per row; the supervisor re-checked these citations against source: scripts/scatter_field.py:649-669 (truth integrator), scripts/train_tier0_tiled_closed_loop.py:163-164 on branch tier0-tiled-closed-loop (`--seed` default 4738, `--val-seed` 4739), docs/debugging/priority-registry.md:7-8 (jobs 4101/4102 FAILED about 32 s, cause unread), and close-pair-design-proposals.md:41-55 and nbody-comparison-options.md:124-142 via `git show`. Other cites are worker claims, marked "(worker)".

Definitions: BL = tiled baseline model; e44b796 = closed-loop tiled tier 0 of record (job 4060, seed 4738); err@20 = mean position error per body at tick 20; dE@100 = relative energy drift at tick 100; PM = particle-mesh; IAS15 = REBOUND's adaptive 15th-order integrator; CIC = cloud-in-cell; Exp E = frozen learned-kernel scatter model; nn = nearest-neighbour distance; GPU-h = H200 GPU-hours.

## Ranking

| rank | row | direction | verdict | cost |
|---|---|---|---|---|
| 1 | 3 | truth validation vs REBOUND IAS15 | PURSUE | 0 GPU-h, CPU, about half a day |
| 2 | 2 | same-grid plain PM baseline (force-error step only) | PURSUE | about 0.3 GPU-h (worker est.; audit says 3-5 with rollouts) |
| 3 | 1 | interpretability rule recovery | PURSUE | 0.5-1 GPU-h, inference only |
| 4 | 4 | second training seed of closed-loop tier 0 | PURSUE after 4101/4102 are read | about 6.4 GPU-h from-scratch base (audit's 1.4 is fine-tune only) |
| 5 | 8 | exact mesh-pair complement, from scratch | PURSUE low priority, blocked on user constraint call | about 2 GPU-h |
| 6 | 5 | visited-state replay, more rounds | DEFER until 4101/4102 read | about 2.5 GPU-h |
| 7 | 7 | time-reversal consistency loss | DEFER (probe only if wanted) | 0.3 GPU-h probe |
| 8 | 6 | tiered dt 0.05 fine-tune | DEFER | 2.6 GPU-h |

Order logic: rows 3 and 2 are near-free and invalidate or reframe claims everything else rests on; row 1 scores the only unscored core requirement; row 4 measures base-seed variance, which every tiered conclusion lacks.

## Row 3. truth-validation-against-independent-integrator: PURSUE

- Source: nbody-comparison-options.md:124-132 (branch research-nbody-comparison, Option 1); audit row 3.
- Already tried: no. rebound is not installed (pip list, worker); only self-consistency checks of truth (t1/t4/t16, scatter-conservation.md:8,63,67, worker). experiment-log.md:3031,3035 note truth drift with "integrator/softening not investigated" (worker).
- Constraints: none violated; validates the instrument, touches no model.
- Evidence: truth is `rollout_mass_torch`, kick-drift-kick leapfrog, dt 0.1 with 4 substeps (h 0.025), Plummer eps 0.5, fp64, scripts/scatter_field.py:649-669 (verified). t4 vs t16 energy differs 3e-4 on bh300 (scatter-conservation.md:63, worker), so the audit's 1e-6 pass line cannot be met by a 2nd-order scheme on close encounters. experiment-log.md:3044 reports dt 0.1 heated cores about 10x more than dt 0.05 (worker).
- Cost: 0 GPU-h. Smallest experiment: one-pair force check against scatter_field.py:655 to confirm REBOUND softening semantics (Plummer per pair, unverified in its docs), then Kepler e=.5, N=3, Plummer N 300-1000 against IAS15 and against t16.
- Pass/kill (refined): pass = t4 truth agrees with IAS15 to within the t4-vs-t16 floor (about 3e-4 energy) and Plummer r10/r50/r90 inside replica spread; kill = disagreement above that floor or softening mismatch, then fix truth first.
- Needs-user: pip/network access for rebound; accept replacing the 1e-6 line with the t4-vs-t16 floor (supervisor view: yes, the 1e-6 line is unattainable by construction).

## Row 2. same-grid-plain-PM-baseline-learning-value: PURSUE (force-error step only)

- Source: nbody-comparison-options.md:134-142 (Option 2, verified); experiment-log.md:3296 (solar disc kernel-only PM, commit 40ce8f4).
- Already tried: only on the solar disc (3/3 planets bound vs 2/3 for E), no force-error numbers; nothing on expB, bh300, g2 (worker grep).
- Constraints: satisfied as a comparator. A plain PM is a fixed analytic force, so it must not become a model arm ("just use PM" would violate no-hard-coded-conservative-forces and model-learns-dynamics). Tiling untested by this row.
- Evidence: `kernel_acc` at scripts/scatter_field.py:219 (verified); scripts/square_wall_probe2.py:25-45 already computes kernel_only vs net_only vs total against `ops.exact_acc` (worker).
- Cost: about 0.3 GPU-h for static-snapshot force error at N 1e3-1e5, seed 4738 (worker est.); rollout comparison (err@20, dE@100) is a second step and needs a no-net rollout path whose existence is unverified.
- Pass/kill: pass = learned model at least 2x better than same-grid PM at median and 90th percentile on 3 held-out profile families; kill = within 20% at the 90th percentile (claim shifts to speed plus tiling).
- Needs-user: approve the 2x/20% rule (nbody-comparison-options.md:216); whether the pair term counts as part of "Augur" or the comparison is also mesh-only.

## Row 1. interpretability-rule-recovery: PURSUE

- Source: docs/paper/core-requirements.md:5 ("UNTESTED"); architectures.md:79,142,237.
- Already tried: not for the current model. The H7 probe (adaptive-eval-interp.md, c52c074) decoded the far-field error estimator, not the learned kernel; a 2-body force-curve plot for the older LJ model (experiment-log.md:3183-3185) was visual only, no exponent fit (worker).
- Constraints: no violation (inference only). Caveat: the mesh/U-Net path acts through a grid, so "recover the exponent" may be ill-posed there; kmlp and ppmlp are pair-side (scatter_field.py:94-104, worker).
- Evidence: scripts/sf_force_probe.py and scripts/interp_decode.py exist (verified); the 2-body probe covers distances 0.25-14.
- Cost: 0.5-1 GPU-h plus a little new fitting code. Checkpoints for six-law and Yukawa scatter models are not in local checkpoints/ (worker; likely on Polaris, unverified).
- Pass/kill: pass = fit to full-model 2-body force recovers exponent within 10% and softening within 20% for gravity, 1/r and Yukawa; kill = error above 30% or law-dependent; add a "partial" outcome (only the near-field pair term recoverable narrows the paper claim).
- Needs-user: which checkpoints are "the current model" and where; whether full-model 2-body probing counts as recovery or the pair term must be isolated.

## Row 4. tiered-closed-loop-tier-0-second-training-seed: PURSUE after 4101/4102 are read

- Source: supervisor-log.md:123 ("single seed"); architectures.md:43,52.
- Already tried: no. Jobs 4101/4102 FAILED about 32 s, cause unread (priority-registry.md:7-8, verified), and 4101 would re-fine-tune e44b796 itself, so it cannot measure base variance. Earlier seed-4739 arms were killed loss-term fine-tunes on the matched-60k base and are not reusable (worker).
- Constraints: no violation; a seed change only. Tiling/O(N) kept.
- Evidence: script has `--seed`/`--val-seed`, resume every 250 iterations, SIGTERM trap; launcher has requeue (worker; seed flags verified). The launcher itself has no seed flag.
- Cost: audit's 1.4 GPU-h is fine-tune variance only. Base-seed variance needs a from-scratch matched-60k base at seed 4739 (5.06 GPU-h, job 4041, experiment-log.md:3348, worker) plus 1.35 fine-tune, about 6.4 GPU-h, roughly 5.5 h sequential.
- Pass/kill: pass = tiled err@20 <= 1.0x BL and plain err@20 <= 1.10x matched-60k value on seeds 9100/9200/9300, report drift spread; kill = err@20 > 1.25x BL.
- Needs-user: is a fine-tune-only seed (1.35 GPU-h) acceptable, or is the from-scratch base required; fix 4101/4102 first?

## Row 8. exact-mesh-pair-complement-trained-from-scratch: PURSUE at low priority, blocked on a user call

- Source: close-pair-design-proposals.md:41-55 (verified); commit 12a309c.
- Already tried: step 0 PASS (job 3646, nn<1 force error 13.6/13.8/12.0 to 3.7/3.4/4.2%, error-correction-plan.md:70); step 1 KILL (job 3754, swapped into frozen Exp E, 6.5 to 63.2% on expB, error-correction-plan.md:71) (worker). Step 1 tested a swap, not training with the term; from-scratch arm never run. Exp E's near field already relies on large cancellation (worker).
- Constraints: keeps pp=2, O(N) (at most 16 pairs per body, timing UNTESTED), tileable. FLAG: the pair direction is central and reciprocal by construction, inherited from the current pair term. Whether that violates no-hard-coded-conservative-forces is an open user call. Correction to the audit: the proposal's K is learned, so "hard-wired analytic" applies only to the step 0 test.
- Cost: about 2 GPU-h (ctl vs R1, seeds 4738/4739, 4000 iterations); about 60 lines new code. A cheaper pre-check: Exp E kernel with no U-Net term.
- Pass/kill: pass = err@20 <= 0.85x control on both seeds and nn<1 force error <= 0.85x control (plus phase share down 40%); kill = err@20 ratio > 0.95 on either seed. Expect dE/dL gate may fail regardless (control already fails bh300 dE@100, worker).
- Needs-user: central/reciprocal pair form acceptable (else row 9 neighbour-context branch instead); is a dE/dL gate fail acceptable if err@20 passes.

## Row 5. visited-state-replay-more-rounds-longer-ticks: DEFER

- Source: supervisor-log.md:133; experiment-log.md:3374 is only the seam-freeroll "Next (not run)" line, not a replay result; replay results sit on branch tiered-drift-mechanism-diagnoses (worker).
- Already tried: job 4089 (seed 4738, 3 rounds, ticks 20-100): dE@100 1.78-2.29x BL vs 3.89-4.75x for e44b796, tiled err@20 0.70-0.77x BL (worker). More rounds/longer ticks/seeds not run; 4101/4102 failed unread.
- Constraints: no violation (training-data lever; exact labels train-time only). Round count is a sweep axis, so add one arm only.
- Cost: about 1.55 GPU-h for 3 rounds; 5 rounds about 2.5 (worker est.).
- Pass/kill: pass = dE@100 <= 1.5x BL with err@20 <= 1.0x BL on two seeds; kill = 5 rounds within 10% of 4089.
- Reason: the truth-state control (4102) decides whether the gain is visited-state specific; read and rerun 4101/4102 first. Master experiment-log has no replay entry (logging gap, worker).

## Row 7. time-reversal-consistency-loss: DEFER (probe only if wanted)

- Source: error-correction-codebase-proposals.md:52-58 (time-reversal proposal, labelled P5 there); error-correction-plan.md:31.
- Already tried: no code or result in scripts/, model/, docs/ (worker grep); 127 branches not individually inspected.
- Constraints: FLAG: assumes time-reversible dynamics, invalid for dissipative sims, so opt-in only (generalization / energy-similarity rule). Soft symmetry prior. Confound: velocity inputs and recurrent field must be carried through the flip.
- Cost: 0.3 GPU-h probe: return error vs k in {5,10,20,40,100} on e44b796, with an oracle Verlet return-error control as chaos floor.
- Pass/kill: pass = model return error at k=20 at least 2x oracle and growing like dE drift, then one fine-tune with dE@100 down 30% and err@20 <= 1.1x control; kill = at or below oracle.
- Reason: similar loss-term fine-tunes hit the accuracy/dE trade-off. Needs-user: is a per-sim opt-in prior acceptable.

## Row 6. tiered-dt-0.05-fine-tune: DEFER

- Source: supervisor-log.md:104,128-133 (HELD); experiment-log.md:3387.
- Already tried: tiled tier 0 at dt 0.05 no. Inference-only dt/2 (job 4086) uninformative; retraining at half dt helped on older models only (gravity 3.5x drift, LJ 5.9x) (worker).
- Constraints: a hyperparameter change, strains no-param-sweeps and prefer-architecture; evidence points to a training-dt artifact. Treat as a diagnostic only.
- Cost: 2.6 GPU-h at 1.9x inference cost; truth may already be substepped so no new data (unverified).
- Pass/kill: pass = dE@100 <= 2.5x BL, err@20 <= 1.0x BL on three seeds at <= 2x cost; kill = dE@100 within 10% of e44b796.
- Reason: replay already reaches about 2x BL at 1x cost; run dt only if replay plateaus above 2x BL.

## Needs-user (consolidated)

1. Pip/network access for rebound; accept t4-vs-t16 floor replacing the 1e-6 line (row 3).
2. Approve 2x/20% pass-kill rule and what "Augur" means in the PM comparison (row 2).
3. Which checkpoints are "the current model", where they live; full-model probing acceptable as recovery (row 1).
4. Fine-tune-only vs from-scratch seed (row 4); read, then resubmit 4101/4102 (rows 4, 5).
5. Is the central/reciprocal pair form hard-coded-conservative (row 8, also rows 9, 38); is a dE/dL gate fail acceptable.
6. Per-sim opt-in time-reversal prior acceptable (row 7); does a dt change count as a diagnostic (row 6).
7. Log replay results to master experiment-log (row 5 logging gap).
