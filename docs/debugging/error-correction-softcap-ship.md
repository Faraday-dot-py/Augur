# Ship the D1 softcap as inference default for contact-C ckpts, combined with the D5-control recipe (2026-10-13)

Follow-up option #3 of error-correction-plan.md (Batch 1). Branch `worktree-agent-a51e0b89567ace982` (master f0e6238 + C line 2146183 + D1/D5/error-growth scripts copied from the sibling worktrees, which were uncommitted). Seed 4738, gravity 9, eval seeds fresh: 9400, 9500 (selection) and 9600 (report); D1 used 9100-9300/9000.

## What changed (one flag)
- `ScatterField.softcap` (default 0 = off): in `_force`, the fed-back field is mapped `s*tanh(field/s)` before it is concatenated to the net input. Net output, integrator, scatter inputs unchanged. This equals D1's pre-hook (D1 squashed the same input channels).
- `scatter_contact_scenes.build(name, dev, ckpt, softcap=0.0)`; `softcap="auto"` (the `eval` CLI default, `--softcap`) = ckpt metadata key `softcap` if present, else 100 for option C, else off. Training and every existing caller stay off. E, A, B: off. No ckpt was rewritten (C.pt has no metadata; the C default is resolved by name).
- `scatter_regress.py run --softcap s` (default 0) for the harness.
- Tests: `scripts/scatter_softcap_test.py`; E with the flag off is bit-identical to pre-change commit f0e6238 (positions and velocities, 50 ticks, 64 bodies). Job 3558 log. Anomaly: this needs `torch.use_deterministic_algorithms` (the scatter's `index_add_` atomics make plain E rollouts differ run to run, max 0.19 in position at 50 ticks, job 3549); the harness numbers are therefore not bit-reproducible by default.

## Predictions (written before the sweep)
P1 C s=0: contact runaway 40-70%, err@100 ~1e3. P2 s=70/100: contact runaway 0, err@20 +10-25%; s=140: partial runaway (10-40%). P3 gravity |a|>200 at s=100 persists in a large fraction of C scenes; I expected part of it to be close-encounter legitimate or edge related. P4 D5ctrl is already 0% on contact, so s=100 changes it <3%; D5ctrl+s100 lowers gravity runaway to <=20%. P5 no dE row recovers.

## Results (jobs 3558 sweep, 3565 harness, 3575 contact eval/render)
Contact c_id, 24 scenes per seed, 100 ticks. Runaway = max|a| > 200 at ticks >= 21. err in sim units, off100 = fraction of bodies with |x|>=32 at tick 100.
| ckpt | s | runaway sel(9400+9500) / 9600 | err@20 sel / 9600 | err@100 median sel / 9600 | off100 sel / 9600 |
|---|---|---|---|---|---|
| C | 0 | .50 / .67 | 6.94 / 5.46 | 384 / 1017 | .87 / .94 |
| C | 70 | 0 / 0 | 8.55 / 6.44 | 138 / 5.5 | .13 / .08 |
| C | 100 | 0 / 0 | 7.77 / 5.71 | 120 / 5.4 | .27 / .14 |
| C | 140 | .15 / .04 | 7.11 / 5.60 | 166 / 47.9 | .44 / .41 |
| D5ctrl | 0 | 0 / 0 | 10.75 / 9.58 | 113 / 18.3 | .50 / .35 |
| D5ctrl | 100 | 0 / 0 | 10.63 / 9.29 | 136 / 22.6 | .46 / .35 |
s=100 chosen on 9400+9500 (s=70 costs +23% err@20, s=140 re-admits runaway); 9600 confirms. err@20 cost of s=100 on C: +12% (sel), +5% (9600). Medians over 48 scenes are dominated by seed 9500 (ejected balls), so the err@100 medians differ strongly by seed.

Gravity (geometry-free grid net, as D1/diagnostic; expB 12 + cluster 6 scenes per seed, 150 ticks; truth max|a| = 6.1 median / 8.5 max on expB and 18.9 / 60.5 on cluster, never > 200):
| ckpt | s | expB runaway sel / 9600 | cluster runaway sel / 9600 | expB err@100 mean | cluster err@100 mean |
|---|---|---|---|---|---|
| C | 0 | .79 / .92 | 1.0 / 1.0 | 755 | 1311 |
| C | 100 | .79 / .67 | 1.0 / 1.0 | 40 | 97 |
| C | 140 | .58 / .50 | 1.0 / 1.0 | 39 | 94 |
| D5ctrl | 0 | .04 / 0 | .50 / .50 | 14.0 | 5194 (one scene, field 8e18) |
| D5ctrl | 100 | 0 / 0 | .25 / .17 | 14.4 | 31 |
| D5ctrl | 140 | .04 / 0 | .50 / .17 | 14.4 | 31 |

### Residual gravity runaway by channel
- Truth never exceeds |a| 60, so every |a|>200 is model error (not legitimate close encounters).
- C s=0: ~50-90 bodies per scene exceed 200, 36-64 of them already off-grid (r>=32): ejection cascade, raw field ~2e6-6e6.
- C s=100 (and 140): exactly one body per scene exceeds 200 (median 1), none off-grid, worst body has nn>=1 in 93-100% of scenes: an isolated in-grid force spike of |a| ~250 (expB) / ~600 (cluster), i.e. 40x truth for expB and 30x for cluster. Raw net output stays 380-960 (the output is unbounded, only the input is capped). Onset: tick 67 median on expB, 24 on cluster. So the bound removes the cascade (err@100 755 -> 40) but a single-cell gain error remains; this is a property of the C weights (C grid net on gravity err@20 3.5-3.9 vs E 0.024), not of recurrence.
- D5ctrl+s100: expB clean; cluster 17-25% of scenes with one body at |a| ~220 (truth ~40), the worst body has nn<1 in 75-100%: a near-pair overshoot of ~5x, not a runaway. s=100 removes the catastrophic scene (cluster err@100 5194 -> 31).

### Harness rows (grid net of the ckpt, plain ScatterField; baseline scatter_baseline_v2.json; timing/memory rows of this run share the node and are not meaningful: peak_mem FAIL on bh300/expB_batch1 also fails for E s=0)
| run | err@5/10/20 rows | bound/virial/vmax rows | dE@100 rows | nr100 dE@1000 | bh300 dE@1000 | overall |
|---|---|---|---|---|---|---|
| E s=0 | PASS (all 9) | PASS | PASS (bh300 .0408) | .0144 | .451 | FAIL (memory rows only) |
| E s=100 | PASS (all 9) | PASS | bh300 dE@100 .0604 FAIL (+48%) | .0282 | .419 | FAIL |
| C s=0 | FAIL x9 | FAIL x7 | FAIL | 8271 | 2.3e4 | FAIL |
| C s=100 | FAIL x9 (err@20 two_body .78 vs .0091) | FAIL (bh300 vmax/c passes .996) | FAIL | 18.6 | 11.3 | FAIL |
| D5ctrl s=0 | FAIL x9 | FAIL | FAIL | 5.04 | .895 | FAIL |
| D5ctrl s=100 | FAIL x9 | FAIL | FAIL | 4.33 | 3.84 | FAIL |
Failing rows `nr100 dE@1000` and `bh300 dE`: still failing for every C/D5ctrl variant (softcap moves them from 1e4 to 10-20 for C, not to E levels). E is moved by the flag on bh300 dE@100 (+48%), nr100 dE@1000 x2: reason the flag is not turned on for E.

### Contact function (eval seeds 9100-9300, c_id)
| ckpt | err@20 / @100 | tunnel@20 / @100 | KE ratio@100 | wall e_eff (truth 1) |
|---|---|---|---|---|
| C | 5.1 / 1770 | .003 / .76 | 80 | 1.13@8, .91@15, none elsewhere |
| C + s100 | 5.61 / 89.5 | .013 / .003 | .45 | erratic: 2.68@2, 1.27@4, .94@6, 1.03@8, .62@10, 1.03@12, 1.38@15 |
| D5ctrl | 8.13 / 81.3 | .286 / .295 | .18 | 3.16@2, 1.53@6, 1.18@8, .94@10, .74@12, .78@15 |
| D5ctrl + s100 | 7.86 / 86.2 | .295 / .300 | .16 | 3.2@2, 1.09@8, .95@10, .86@12, .72@15 |
Energy rows (dE@20 model -29 to -76 vs truth -2.5) unchanged by any of it. D5ctrl loses the wall (29% tunnelled at tick 20, 70% on box15).

### Unbiased image review (fresh subagent, grids only, c_id scenes 0-2, ticks 0-100)
C+s100: model balls stay inside the container, separate non-overlapping rings, no wall/obstacle crossing, positions diverge from truth within ~10 ticks and error plateaus (~3-6), no late jump. D5ctrl+s100: same early divergence, but model balls pile into tight overlapping clumps, one clump leaves the container in scene 0, error keeps climbing in 2 of 3 scenes (to ~10-13 by tick 100). Consistent with the table (D5ctrl tunnelling/lost wall, no runaway).

## Verdict
- Contact C: s=100 default is confirmed on fresh seeds (runaway 0 on 120/120 scene-runs, tunnel@100 .76 -> .003, off-grid .9 -> .14-.27) at +5-12% err@20. Shipped as `auto` default for option-C ckpts; off for E.
- Combination with the D5-control recipe: contact adds nothing (D5ctrl has no runaway to remove); gravity: removes the expB runaway and the cluster catastrophe, leaves a mild near-pair overshoot in 17-25% of cluster scenes. But D5ctrl costs +50% err@20 and a 29% wall tunnel rate vs C+s100, so C+s100 is the better contact model and D5ctrl+s100 the more robust gravity one. Neither passes any harness row beyond what C did; dE rows still fail (owned elsewhere).
- Remaining defects are a gain/overshoot problem of the fine-tuned weights, not recurrence growth; the follow-up is gain control (D2b), not a different bound.

## Files
`scripts/scatter_field.py` (flag), `scripts/scatter_contact_scenes.py`, `scripts/scatter_regress.py`, `scripts/scatter_softcap_{test,eval,summarize,table}.py`, raw `results/softcap/{C,D5ctrl}.json`, logs `results/softcap/*.log`, `results/contact/eval_{C_sc100,D5ctrl_sc0,D5ctrl_sc100}.json`, videos `videos/contact_c_id_{C,D5ctrl}_sc100{,_grid}.{mp4,png}`. Polaris: ~/augur-softcap.
