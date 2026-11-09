# Director feasibility probes 2026-11-07

Written by the head agent under docs/debugging/overnight-director-mode-spec-2026-11-07.md at about 02:40 PST. Three inference-only feasibility probes, each with a pre-registered pass/kill line fixed before any run. None submitted by the writer; the main session submits. Hard stop 09:00 PST 2026-11-07 (epoch 1794070800) in every script. Seed 4738, 1 GPU, QOS student-pleiades, no nodelist, 1 CPU, no inter-GPU. All three are eval-only, so there is no training state to checkpoint; instead each saves results per cell or per family, skips finished work on rerun, forwards SIGTERM, and exits 4 at the deadline.

Amendment guard applies: any change to a line below is a logged deviation naming a concrete value that would still fail it.

## Validity check (spec step 2)

Already-tried list read: docs/debugging/experiment-log.md through line 3635 (last entry: 2026-11-07 overnight results). Audit rows 1 (law recovery), 2 (plain PM baseline), 3 (IAS15 truth check) and 4 (tier-0 second seed) are already logged there and were skipped even though the 2026-10-29 audit still lists them as never run. Rows 11, 10 and 32 have no entry. Picked rows are numbered as in docs/debugging/unexplored-directions-audit-2026-10-29.md.

Constraints checked for all three: pp=2 O(N) kept (probes measure it, none changes it); no hard-coded conservative force added (probe 2 only measures what a union list would change in a frozen model); no parameter sweep used as a strategy (the N and mass ladders in probes 1 and 3 are attribution and scaling measurements, not tuning); architecture first (all three test an architectural cause: density range, list symmetry, cell-list scaling). Past decisions followed: rule 15 (diagnostic follow-ups approved quickly), rule 6 and 10 (noise floors stated).

## Probe 1: interior-force-offset-probe (audit row 10)

Question: the model/exact radial force ratio is about 1.55 everywhere in the interior at 1e6 bodies (experiment-log.md line 3356, "not investigated"). Is it mass density out of the training range, the pair term, or the grid kernel (G, softening, tail)?

Setup: uniform disk, radius 24, eps 0.5, G=1, exact fp64 point sum as truth (targets subsampled to 8192 above N=1e5). Learned model = pinned kern/E_frozen.pt, seven force calls (one plus six warm). Cells: per-body mass 1 at N = 1e3, 1e4, 3e4, 1e5, 1e6; total mass held at 1000 at N = 1e4 (body mass 0.1) and 1e5 (body mass 0.01). Variants per cell: learned, learned with the pair term zeroed, plain PM with the analytic kernel. R_int = sum(a . rhat) / sum(a_exact . rhat) over bodies at r < 0.75 R, mean over scenes.

- PASS: offset reproduced (R_int of the learned model >= 1.30 at N=1e5 or at N=1e6, per-body mass 1) AND attributed: one of three explanations accounts for at least 90% of the excess (R_int - 1). Density: at the same N with total mass held fixed, excess is at most 10% of the unit-mass excess. Pair term: with the pair term zeroed, excess is at most 10% of the full excess. Grid kernel: plain PM excess is at least 90% of the learned excess.
- KILL: offset absent: R_int within 0.90-1.10 at every completed N >= 1e5.
- INCONCLUSIVE: anything else, for example R_int 1.20 at both N (neither offset clause nor KILL band met), or 1e6 out of memory with N=1e5 at 1.11 or above but below 1.30.
- Noise floor: 1 to 4 scenes per cell (1e6 has one, 1e5 two); per-scene ratios are stored. Clause definitions (review clarification, before any run): excess = R_int - 1; "at most 10%" means absolute value of the comparison excess <= 0.1 x the learned unit-mass excess; density clause uses N=1e5 (fixed-total-mass cell vs unit-mass cell); if the offset is only at 1e6 the density clause cannot be computed and the verdict is INCONCLUSIVE unless the pair-term or kernel clause holds; pair-term and kernel clauses use the N with the largest learned excess among N >= 1e5. Prediction PASS 45%, INCONCLUSIVE 35%, KILL 20%.
- Cost: under 1 GPU-h (not timed). Script scripts/interior_force_offset_probe.py, job scripts/polaris_interior-force-offset-probe.sh (time limit 3 h).
- CPU smoke (random weights, N=1000): runs, plain-PM ratio 1.002 (the reference code path is right); the learned ratio is 0 with random weights, so the learned path has not been checked on real weights.

## Probe 2: reciprocal-neighbour-list-step0 (audit row 32, close-pair proposal R4 step 0)

Question: the pair term keeps each body's 16 nearest neighbours, so a pair can be held by one body only. In dense cores, does a union list (pair kept if either body selects it) change force error for close bodies enough to justify a training arm?

Setup: frozen kern/E_frozen.pt, dense pair path, N=1000, mass 1, v 0, 16 scenes per family: compact_cluster (Plummer core 1), plummer (core 4), two_clump. Replace the top-16 pair term by the union-list pair term (same per-source weights, same frozen weights, same momfix, no 2k cap) and compare squared force error against fp64 exact for bodies whose nearest neighbour is closer than 1. Also reported: fraction of in-window pairs held one-sided, fraction dropped by both lists, momfix mean-acceleration magnitude over rms |a|. In-script check: the single-sided rebuild must match model.pp_acc to 1e-4 or the script aborts.

- PASS (the step-0 gate opens; says nothing about err@20): compact_cluster pooled SSE ratio (union / knn) <= 0.98 AND at least 12 of 16 compact_cluster scenes with ratio <= 1.0.
- KILL: compact_cluster pooled ratio > 0.99 (the proposal's own "proceed only if above 2%" with a 1% margin; 0.995 would still fail).
- INCONCLUSIVE: ratio in (0.98, 0.99], or pooled <= 0.98 with fewer than 12 consistent scenes.
- Flag, not a gate: all-body SSE ratio > 1.05 in any family means the union list hurts the model that was trained with the top-16 list.
- Prediction KILL 55%, PASS 35%, INCONCLUSIVE 10%. Source of the prediction: close-pair doc says expB has no truncation and only cluster scenes could gain.
- Cost: under 0.2 GPU-h (not timed). Script scripts/reciprocal_neighbour_list_step0.py, job scripts/polaris_reciprocal-neighbour-list-step0.sh (time limit 1 h).
- CPU smoke (random weights, 2 scenes): runs and passes the rebuild check; ratios are 1.0 because random pair weights are zero, so the effect size has not been checked on real weights.
- If PASS, the training arm is NOT launched; it is on the needs-user list (hard-wired reciprocity versus the no-hard-coded-conservative-forces rule, rule 16).

## Probe 3: collapsed-pair-term-scaling-probe (audit row 11)

Question: O(N) with pp=2 is a core requirement. The adaptive multilevel pair term was timed only at N=1e5 (193 ms). In a collapsed state does it stay linear to 1e6 and 3e6?

Setup: Plummer cloud with core scale 1 (collapsed) and core scale 4 (reference), N = 1e5, 3e5, 1e6, 3e6, adaptive path forced, top-16. Time model.pp_acc alone and one full model.force (cuda-synchronized, 3 calls after one warm-up, 1 call at 3e6). Output: seconds per call, ns per body, adaptive-path statistics, log-log slope of pair-term time against N.

- PASS: collapsed log-log slope (completed cells with N <= 1e6; the 3e6 cell is reported only, slopes are written after every cell) <= 1.15 AND collapsed pair-term time at N=1e6 <= 10 s per call AND collapsed ns per body at 1e6 <= 2.0x the reference cloud's.
- KILL: collapsed slope > 1.40, or out of memory at N <= 1e6.
- INCONCLUSIVE: anything else (for example slope 1.15-1.40, or only N <= 3e5 completed).
- Noise floor: single scene per cell; timing repeats stored implicitly in the mean of 3. Prediction PASS 50%, KILL 20%.
- Cost: under 0.5 GPU-h up to 1e6. Reviewer estimate (numpy replica of the adaptive criterion, not timed): about 40% of bodies at 3e6 fall back to chunked all-pairs, so the 3e6 cell may not finish inside the 3 h limit; a missing 3e6 cell does not change the verdict. Script scripts/collapsed_pair_term_scaling_probe.py, job scripts/polaris_collapsed-pair-term-scaling-probe.sh (time limit 3 h).
- CPU smoke (random weights, N = 2e4 and 6e4): runs; slope 1.12 (collapsed), so the CPU number is not evidence for the GPU line.

## What was not picked and why

- Held items (dt 0.05 fine-tune, curl-and-drag penalty, time-reversal loss, more replay rounds): release conditions unmet or ambiguous, listed in director-needs-user-2026-11-07.md.
- Audit row 17 shift-consistency probe: the probe is already measured (error-correction-plan D6, 6-10% at half a cell) and the near neighbour (grid-origin jitter) raised dE 5.8x; the fine-tune would need new training code that cannot be tested before the deadline.
- Neuron-track rows 35 and 41: track paused by the user (rule 19).
- Brainstormed new idea: none passed. The candidates (a learned error-estimate channel, row 21) need new model code and 1-1.5 GPU-h of training, which cannot be smoke-tested without a GPU before the window.

## Utilization note

These probes total under 2 GPU-h against about 8 x 6 h of idle GPUs. Underutilization is still an anomaly and is flagged; no further non-repeat, non-sweep single-GPU idea with a testable script was found in the time available.

## Results

None yet. When a probe finishes: log in experiment-log.md, independent reviewer on raw JSON plus script plus this line, commit with the headline number.

## Review (fresh reviewer, 2026-11-07, scripts read plus CPU smoke with random weights)

No crash-level bugs. Fixes applied before commit: probe 3 slopes written after every cell and gated on N <= 1e6, time limit 3 h, missing-input check for scatter_scaling_ops.py; probe 1 stores per-scene values and seeds the 1e6 target subsample, clause definitions clarified above. Noted, not changed: probe 2 "nn < 1" selects 92-99% of bodies (so the gate is close to an all-body SSE ratio and KILL is likelier than the stated prediction), frac_pairs_one_sided uses held pairs as denominator, no per-family resume in probe 2 (cheap), momfix() helper unused. Not verified: any real-weights or GPU behaviour, the probe 2 rebuild assert with nonzero weights (it aborts the job if it trips), SIGTERM behaviour under SLURM.
