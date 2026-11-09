# Neuron-graph seed-robust verdicts (slow-patch rate token, plastic-synapse STDP probe)

Status 2026-11-08: diagnosis done from existing result files (no GPU); seed-robust run submitted as Polaris job 7193 (slot 2, afterany:7192), not yet finished. Pre-registration: experiment-log.md, entry "2026-11-08 neuron-graph-seed-robust-verdicts: PRE-REGISTRATION". Diagnosis script: scripts/neuron_seed_robust_diagnosis.py. Windowed attention (KILL x3) and bit-output (PASS x3) were not touched.

## Diagnosis

Sources: seed 4738 slow-patch from .claude/worktrees/timesfm-neuron-ideas/results/neuron_graph_slow_patch/seed4738/results.json; seed 4738 plasticity from ~/polaris-mcp-files/neuronplas/seed4738/results.json; seeds 4739/4740 from job 6163 (Polaris results.json, downloaded to ~/polaris-mcp-files/seedrobust/).

### Slow-patch rate token (err@20 = position error at tick 20, f1@20 = spike F-score at tick 20, slope300 = per-cell predicted-vs-true rate slope at 300 ticks)

| seed | N=10 slope300 variant / baseline / control | N=10 f1@20 variant / baseline / control | N=10 err@20 ratio variant/baseline (line 1.03) | N=100 err@20 ratio | verdict |
| --- | --- | --- | --- | --- | --- |
| 4738 | 1.29 / 0.15 / 0.07 | 0.735 / 0.814 / 0.703 | 0.943 | 0.957 | PARTIAL |
| 4739 | 1.33 / 0.17 / 0.16 | 0.557 / 0.818 / 0.676 | 0.958 | 0.945 | PARTIAL |
| 4740 | -0.01 / 0.04 / 0.01 | 0.498 / 0.854 / 0.228 | 1.021 | 1.053 | KILL |

Findings:
1. Distance from the line. The slope line (0.3) is far from baseline noise (baseline slope 0.04-0.17, sd 0.07 across seeds) but the variant's own seed spread is 0.76: two seeds sit at 1.3 (four times the line), one at 0. The KILL at seed 4740 is a collapsed rate head (predicted/true rate ratio 0.13 and 0.06), not a near-miss. Variation is far larger than baseline noise; it is training-run bimodality.
2. f1@20 line is confounded; the variant missed it in all 6 cells. The 48-tick-segment control (baseline architecture, same training segments) misses the f1 line at all three seeds (control minus baseline at N=10: -0.11, -0.14, -0.63). The variant also misses it everywhere, so PASS looks out of reach (empirically at 3 seeds, not proved impossible), and the rate ratio overshoots at two of three seeds (N=10 ratio 1.48, 2.08). The swing between the two non-KILL verdicts and KILL is the rate head, not the confound; the confound makes the f1 check hard to meet at every seed.
3. Frozen baseline. Seeds 4739/4740 compared against a baseline re-trained at the same seed (paired); seed 4738 used the shared frozen checkpoint. Baseline err@20 varies 8.60-9.44 at N=10 (sd 0.42, 4.6%), so the +3% err line is about 0.65 baseline sd wide; within-seed pairing removes most of it (ratios 0.94-1.05 across 6 cells; only seed 4740 N=100, 1.053, exceeds the 1.03 line).
4. Control instability: seed 4740's control had a loss spike to 23.9 at the start of rollout-in-loop training (seed 4739: 0.46), and a control f1@20 of 0.23, so the control is itself seed-sensitive.

### Plastic-synapse STDP probe (dR = probe response change in mV; contrast = forward minus reversed)

| seed | plastic contrast N=10 p1 (paired t) | N=10 p2 | N=100 p1 | N=100 p2 | fixed-arm contrast (range) | final g forward / reversed at N=10 (teacher 2.35 / 0.20) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4738 | -0.12 (-2.0) | -0.42 | 0.00 | -0.12 | -0.27 to +0.09 | 1.82 / 0.51 | KILL |
| 4739 | +0.11 (+1.6) | +0.48 | +0.18 | +0.04 | 0.01 to 0.12 | 2.52 / 0.05 | PARTIAL |
| 4740 | +1.25 (+7.4) | +0.81 | +1.73 | +0.40 | -0.00 to +0.01 | 3.39 / -0.35 | PARTIAL |

Findings:
1. The first KILL clause fires when the plastic contrast is <= 0 in all four cells (the second clause, fixed contrast >= 0.75x plastic in all four cells, did not fire at any seed); at 4738 it is -0.12/-0.42/0.00/-0.12 (paired SE 0.06-0.11, wrong-signed at t -2.0 and -3.9 in two cells; N=100 probe 1 is -0.005, t -0.06, so the KILL is knife-edge on that cell). Seeds 4739 and 4740 are positive in all four. Seed-to-seed sd of the contrast is 0.7 at N=10 probe 1, against per-seed paired SE 0.06-0.17: variation comes from training runs (how much of the learned synapse state m the cells read out, tracking final g separation 1.3 / 2.5 / 3.7), not from the 12-circuit eval sample.
2. The fixed arm (trained per seed, so already paired) has contrast -0.27 to +0.12 (near zero except 4738 N=100 probe 2, -0.27, larger than the plastic -0.12); the second KILL clause did not decide at any seed.
3. The PASS line cannot be met as written: "forward >= 0.5 x teacher" fails in every cell of every seed (teacher forward dR is itself heavy-tailed, 1.2 to 25.8 mV, driven by post-cell spikes counted at 30 mV and by circuit draw, since eval circuits change with the seed), and "no-induction small (<= 0.25 x forward)" divides by a forward mean that is near zero in weak seeds (4739 N=10 probe 1: forward +0.09, none +0.67). A 24-check conjunction of that kind cannot separate signal from noise.
4. Seed 4740's student overshoots (reversed g -0.35, below the teacher's 0.2 floor), another sign this is a training-run property.

Conclusion: for both ideas the original lines are valid as lines but the verdict class is decided by whether a given training run happens to learn the mechanism, and the PASS line looks unreachable at 3 seeds (slow-patch: f1 vs a baseline that the control itself fails; plasticity: teacher-relative and ratio checks). A seed-robust protocol therefore resolves "does it work reproducibly" but will most likely return FAIL for both unless the lines are amended.

## Protocol (pre-registered in experiment-log.md, lines unchanged)

Seeds 4738-4747. 4738-4740 re-run so all ten use identical code and a same-seed re-trained baseline; 4741-4747 were never run for these ideas. Per-seed verdict from the unchanged scorers. Aggregate: PASS if PASS seeds >= ceil(0.6 n); otherwise FAIL (FAIL-KILL if KILL seeds >= ceil(0.5 n), a 5-5 split goes to KILL; else FAIL-PARTIAL); n >= 8 required. Seeds 4741-4747-only check as a robustness cross-check; disagreement is a needs-user item. Seed power: with 10 seeds the rule passes 97% of the time at a true per-seed pass rate of 0.8, 38% at 0.5, 4.7% at 0.3. Effect size needed relative to seed noise: slow-patch slope300 about 0.5 above baseline (baseline seed sd 0.07); plasticity contrast about 1.0 mV (seed sd 0.7).

Job 7193: script scripts/polaris_neuron-graph-seed-robust-verdicts.sh, slot 2 after 7192, est 3.0-3.4 h (per seed about 17 min; unverified eval time), log ~/augur-neuron-graph-seed-robust-verdicts-7193.log (download with polaris_download), results ~/augur-neuron-seed-robust/results/neuron_seed_robust/. Resume (if requeued the job resumes itself by skipping finished steps); after COMPLETED: download the log and results.json files, run scripts/score_neuron_seed_robust_verdicts.py --root <dir>, spawn a fresh reviewer, log the RESULT entry in experiment-log.md.

## Proposed amendments (NOT applied; needs-user list)

1. Slow-patch: replace "f1@20 >= frozen baseline - 0.03" with "f1@20 >= same-seed 48-tick control - 0.03", keep every other check. A value that still fails: seed 4739 N=10 (variant 0.557 vs control 0.676, gap -0.119) and any seed whose slope300 is below 0.3 (seed 4740, -0.013). Under it, seed 4738 N=10 (0.735 vs 0.703) would pass that check; the rate ratio (1.48 and 2.08 at N=10) would still fail the full PASS.
2. Plasticity: replace "forward >= 0.5 x teacher" and "no-induction <= 0.25 x forward" with "plastic forward-minus-reversed contrast paired t >= 2 at probe 1 at both N and the fixed-arm contrast <= 0.25 x plastic". A value that still fails: seed 4738 (t = -2.0 and -0.06) and seed 4739 N=10 probe 1 (t = +1.63); seed 4740 would pass those two replaced checks (7.4 and 8.0) but still fail others (N=10 probe 2 forward t, N=100 probe 1 reversed t, N=100 probe 2), so no seed reaches a full PASS.
Both proposals were written after seeing the data, so they would be post hoc; the as-registered verdict classes stay on record.

## Independent review
Diagnosis reviewed by a fresh agent 2026-11-08 (recomputed all table numbers from raw files, all match). Corrections applied: err ratio sentence, "unreachable" softened to "missed in all cells at 3 seeds", plasticity KILL has two clauses (neither second clause fired), fixed-arm range -0.27 to +0.12, seed 4740 would not fully pass the amended plasticity line, scorer fixed (crashed seeds count toward n; unseen-seed check now compares pass/fail class, min 6 of 7). Untested: the claim that teacher dR is driven by 30 mV spike counting. The result run (job 7193) still needs its own review.
