# Protected first phase (lr warmup + lower peak lr) for Exp E's warm-start recipe (2026-10-15, branch lr-warmup)

Question: long-rollout-root-cause.md found the first Adam step of Exp E (init = Exp B checkpoint, fresh optimizer, lr 5e-4) took expB force error 0.94% -> 5.7% and 60k iterations never recovered (6.1% final). Does protecting the first phase keep the force accurate and improve the rollout?
Definitions: B = batch, N = bodies, nn = distance to a body's nearest neighbour, err@k = pooled mean per-body position error at tick k (sim units), dE/dL = relative energy / angular-momentum drift, nr100 = 100-body no-relativity conservation regime, bh300 = 300-body relativistic black-hole regime, Adam eps = the 1e-8 term in the Adam denominator, "force error" = rms|a_model - a_exact| / rms|a_exact| of the learned acceleration at truth states.

## Recipe (all arms identical except lr schedule)
Exp E recipe from `scripts/polaris_scatter_bh_train.sh`: variant ms_kp_pot_v_g128 (pp=2, O(N), architecture untouched), init = `init_B_ms_kp_pot_v_g128.pt`, 1500 train scenes x 50 ticks, batch 16, k 4 -> 30, squared loss, grad clip 1.0, cosine to 5% of peak, 60769 iterations (the iteration count E reached in its 9000 s budget), seed 4738, fresh Adam. Runner `scripts/scatter_lrwarm.py` (trainer change: `--lr-warmup`, `step_hook`; lr at iteration i = peak * min(1, (i+1)/warmup) * cosine).
- ctl: peak 5e-4, no warmup (E's original schedule; same-schedule control, retrained here).
- w1: linear warmup from 0 over 2000 it, peak 2.5e-4 (half of E's).
- w2: linear warmup from 0 over 2000 it, peak 1e-4.
Schedule design (not a sweep; two arms): B's own cosine ended at 1e-3 x 0.05 = 5e-5, i.e. B's weights are a minimum under steps of ~5e-5; E's first step at 5e-4 is 10x that. w2 = 2x B's final lr (the smallest peak that still leaves room to learn the cluster/10-200-body mix), w1 = 5x B's final lr, half E's. Warmup 2000 it = 3.3% of the run.

## Pre-registered criteria (written before launch)
Metric: teacher-forced force error on bodies with nn < 1 (all |x| < 26), expB scenes seed 9300 (12 scenes, ticks 2-49, recurrent field carried along truth), logged by the training hook at iterations 0, 1, 2, 5, 10, 50, 200, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000 and at the end. Reference value at iteration 0 (= B) expected ~0.9-1.0%.
- PASS (per arm): force error nn<1 <= ~1.5% (up to 1.65%) at every logged iteration after warmup; AND expB err@20 and err@100 each improve >= 20% vs ctl on fresh held-out seeds (expB 9500, cluster 9400; docs-comparable seeds 9300/9200 also reported); AND nr100 dE@1000 <= .05; AND bh300 dE/dL rows (harness dE@100, dL@100 and conservation dE@1000) no worse than ctl; still pp=2, O(N), architecture unchanged. Headline arm chosen on seeds 9500/9400 only. Replicate seed 4739 if PASS.
- KILL and report: a warm arm's nn<1 force error grows > 3x within the first 10 steps (checked automatically after a 30-iteration smoke run of each arm; exit code 2), or expB err@20 gain < 10% for both warm arms.

## Predictions (before launch)
1. ctl reproduces the jump: force error nn<1 >= 4x step-0 by step 10 (5-6% at step 1-10), stays 6-7% to the end; expB err@20 ~.025, @100 ~1.4-1.5 (Exp E's numbers).
2. w1 and w2: no jump in the first 10 steps (ratio <= 1.5x). The force error is then expected to rise as lr reaches its peak, because finding #3 of the root-cause doc (squared rollout loss trades bulk accuracy for tail hedging) is lr-independent in direction: w1 ends 2-4%, w2 1.2-2.5%. P(PASS) ~ 30% for w2, ~10% for w1; most likely outcome is a partial win (err@20 -10% to -40% vs ctl) that fails the 1.5% force criterion.
3. Mechanism: Adam's first update is a sign update (|dtheta| = lr for every parameter whose |grad| >> 1e-8), so a fresh optimizer moves every weight by the same amount regardless of gradient size; I expect > 90% of parameters to have |grad| > 1e-8 after clipping (eps not the cause; counterfactual first steps with eps up to 1e-4 still jump, eps 1e-2 does not), and attribution to be spread over the U-Net (largest parameter share) rather than concentrated in kmlp/ppmlp. Zero-init of the last layers is not a mechanism here (the init is the trained B checkpoint; out-layer weights non-zero), logged via init parameter abs-means.

## Results (job 3664 after 3647 failed on a script bug (unbound $4) following a passed smoke test; seed 4738; raw `results/lrwarm/`, summarizers `scripts/scatter_lrwarm_summarize.py`, `scatter_lrwarm_cons.py`)
Force error on nn<1 bodies (%, 12 expB scenes seed 9300, ticks 2-49) by training step:
| arm | 0 | 1 | 2 | 5 | 10 | 50 | 200 | 1000 | 2000 | 10000 | 30000 | 60000 | final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctl (5e-4, no warmup) | .71 | 4.86 | 9.38 | 15.4 | 14.1 | 6.8 | 7.5 | 6.8 | 7.7 | 8.7 | 9.0 | 7.3 | 7.1 |
| w1 (warmup, 2.5e-4) | .71 | .71 | .71 | .71 | .73 | 1.52 | 9.05 | 9.13 | 7.5 | 8.1 | 9.0 | 7.4 | 7.3 |
| w2 (warmup, 1e-4) | .71 | .71 | .71 | .71 | .71 | .97 | 5.45 | 9.69 | 11.9 | 9.0 | 9.4 | 7.8 | 7.7 |
Final checkpoints, fresh seeds (expB 9500, cluster 9400), force all-body rel_rms % / nn<0.25 %, err@20, err@100:
| model | expB force / nn<.25 | expB @20 | expB @100 | cluster @20 | cluster @100 |
|---|---|---|---|---|---|
| frozen E | 6.24 / 10.05 | .0249 | 1.348 | .0681 | 4.337 |
| ctl | 6.06 / 9.79 | .0246 | 1.305 | .0705 | 4.287 |
| w1 | 6.19 / 9.98 | .0246 | 1.360 | .0701 | 4.402 |
| w2 | 6.91 / 10.55 | .0249 | 1.334 | .0694 | 4.376 |
(seeds 9300/9200: ctl .0245 / 1.473, w1 .0258 / 1.531, w2 .0261 / 1.532.) Control reproduces frozen E to within noise, so the recipe is reproduced.
Conservation (1000 ticks, median of 3 evals): nr100 dE@1000 ctl .143, w1 .068, w2 .199; bh300 dE@1000 ctl 1.35, w1 .72, w2 .31 (dL ctl 8.8, w1 5.8, w2 6.9). Evals of one checkpoint vary up to 1.5x on bh300; no arm meets nr100 dE <= .05. Regression harness: all arms PASS vs frozen E on err rows (expB err@20 ctl -4.2%).
Mechanism (control step-1 probes, 30-iteration smoke): force nn<1 .71% -> 4.9% after one Adam step. Updating any single module group alone raises it only to .78-1.4%, so the jump is the sum of small moves across all modules (net.dec/enc/inp/out, kmlp, ppmlp), not one layer. Counterfactual first step at lr 5e-4 with Adam eps 1e-8/1e-6/1e-4/1e-3/1e-2: 4.9/4.9/4.3/3.1/1.7%: eps matters only at 1e-3 and above (>= 1e5x the default), so default eps is not the cause; the step is sign-like (every weight moves ~lr). Zero-init last layers not applicable (init is trained B).

## Verdict: NEGATIVE (KILL criterion met, err@20 gain 0% vs required >= 10%)
Warmup does exactly what it was designed to do for the first ~50 iterations (ratio 1.00 to step 10 vs ctl 6.9-21x; the 3x-in-10-steps kill did not trigger), but the force error then rises to 5-9% by iteration 200-2000 in both warm arms as the lr reaches its peak, and ends at the same 7.3-7.7% as ctl. err@20 and err@100 are indistinguishable from ctl and frozen E. So the first Adam step is not the cause of the plateau: it is the squared-rollout-loss optimum reached under any lr tried (1e-4 to 5e-4) from B. Prediction 2 (partial win) missed on the high side: no err gain at all; prediction 1 and 3 held (eps not the cause; spread across modules). Not replicated at 4739 (no arm passed). Conservation differences between arms are within the 3-eval noise range except nr100/bh300 w1/w2 looking lower, not a pass.
Follow-up: the degradation is in the objective, not the optimizer; combine with the Euclid-loss run (job 3642, other agent) only as a separate test, and a lower lr floor than 1e-4 was not tried (a still-lower peak is a sweep, not run).
