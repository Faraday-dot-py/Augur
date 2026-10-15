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

## Results
(pending)
