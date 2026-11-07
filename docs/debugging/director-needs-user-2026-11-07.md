# Director needs-user list 2026-11-07

Items the head agent did not launch or decide because they rest on user intent, an unmet release condition, or an ambiguous rule (spec: docs/debugging/overnight-director-mode-spec-2026-11-07.md; rules: docs/debugging/user-decision-rules-for-director-mode.md). Nothing here is submitted.

1. dt 0.05 tiered fine-tune (audit row 6). Held; release condition "replay plateaus above 2x BL" (memory project_decisions_2026_11_03; rule 18). Latest evidence: replay dE@100/BL 2.41/2.01/1.21 (overnight results, experiment-log.md line 3625 onward) and fresh-seed eval INCONCLUSIVE (4818, err@20/BL 1.012). Whether that counts as a plateau is not stated. Needs dt 0.05 truth data (about 2.6 GPU-h). Decision wanted: release or keep held.
2. Curl-and-drag penalty fine-tune (row 22). Held by the coordinator; diagnosis INCONCLUSIVE; a soft prior near the no-hard-coded-conservative-forces rule (rule 16). Decision wanted: release, and whether a soft penalty is acceptable.
3. Time-reversal consistency loss (row 7). Memory says "per-sim opt-in OK, stays deferred". Even the 0.3 GPU-h eval-only return-error probe is under that deferral. Decision wanted: may the probe run.
4. More visited-state replay rounds (row 5). User held further replay rounds after the seed-4739 EARLY_KILL (rule 4), and the second-base replay result is mixed (item 1 above). Decision wanted: more rounds or not.
5. Noisy-observation state assimilation (row 19). Error-correction plan: "only if a deployment with observations is in scope". Scope is the user's call.
6. If the reciprocal-neighbour-list probe (docs/debugging/director-feasibility-2026-11-07.md, probe 2) passes: the training arm makes pair reciprocity exact, which may count as hard-wiring conservative structure (rules 16 and 17; audit constraint question rows 8, 9, 38, never answered). Decision wanted before any training arm.
7. Shift-consistency loss (row 17). Not run. Unclear whether it counts as already tried: the probe exists (D6, 6-10% at half a cell) but grid-origin jitter raised dE 5.8x. Decision wanted: treat as tried or allow a probe-then-loss arm.
8. Neuron track (rows 35, 41). Paused by the user 2026-10-29 (rule 19); not reopened.
9. Spec inconsistency: the spec says "no nodelist", but the eight scripts in docs/debugging/overnight-8gpu-plan-2026-11-06.md and the tiered scripts pin `--nodelist=gpu01`. The three new scripts omit the nodelist per the spec. Decision wanted: which is binding.
10. Standing-rules note dated 2026-11-06 says Polaris GPU nodes are down. Memory says a burn test later passed. The note is still in standing-rules.md. The main session must check node status before submitting; the writer did not.
11. The 2026-10-29 audit table is stale for rows 1-4 (all logged since 2026-11-03). Marking them in the audit file would be an edit to a prior record, so it was not done. Decision wanted: annotate or leave.
12. Spec still marked "awaiting user confirmation" as a whole (Open questions section). Launch rights were answered, the written spec as a whole was not.
13. Utilization: only three short inference probes (under 2 GPU-h) found for roughly 48 idle GPU-hours. Whether to spend more on repeats (for example more seeds of the still-INCONCLUSIVE items) rests on intent; filler burns continue meanwhile.

- Interior-force-offset probe: literal KILL, but reviewer says it measured a different quantity than the 1.55x (see experiment-log.md director RESULTS section). Decide: leave row 10 open and queue a bh_edge_force_check variant rerun, or accept KILL.
