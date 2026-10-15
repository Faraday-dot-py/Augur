# Energy-similarity loss term on the Euclid rollout loss (2026-10-14, uncommitted)

Branch/worktree `loss-energy` (`.claude/worktrees/loss-energy`; base commit 6d9a530 = the uncommitted Euclid loss + losstest runner copied from `loss-tests`).

## Design
- Term (`train_scatter_field.energy_sim`): at each supervised rollout tick s = 1..k (k = unroll steps), per scene, (E(model state at s) - E(truth state at s)) / (K0 + |U0|), E = `sf.energy` (exact softened pairwise, eps 0.5) applied identically to model and truth states, K0/U0 = truth kinetic/potential at the window start (scale only). Loss adds lam * mean_scenes,s |.|. Model masses detached (gradient only via pos/vel). The target is the truth's energy at the same tick, never E(0) and never dE/dt = 0: a model that follows a dissipative/driven truth scores 0.
- Weight principle (no sweep): lam = 0.1 * lp0 / le0, lp0/le0 = mean Euclid position term / energy term of the INITIAL model (before any update) over 16 training batches, k = 4. Computed separately per setup.
- Arms (one change each vs the same-schedule Euclid control; control logs the term without gradient): control = Euclid; A = Euclid + term; B (only if A shows bias) = A + lam * |batch-mean signed error|.
- Schedule (= loss-tests): ms_kp_pot_v_g128 (pp=2, knn=16), Exp E data 2000 scenes x 30 ticks seed 4738, B (batch) = 16, k 4->20, 4000 it, cosine lr; scratch lr 1e-3, warm from Exp B ckpt lr 5e-4. Replicate seed 4739 if a clear effect.
- Eval: held-out 9100/9200/9300 (24 ab + 24 cl scenes x 100 ticks each); err@5/10/20/50/100; step-20 Euclid median/mean/p99; energy-similarity error vs truth at ticks 20/100; `scatter_regress.py` compare; `scatter_conservation.py run --steps 1000 --regimes nr100,bh300` per scene (4 scenes, seeds 9100/9200/9300/4738): conservation dE@100/1000 and truth-similarity |E_m - E_t4| at 100/1000.
- Property test (`scripts/scatter_esim_test.py`): truth with per-tick velocity factor f (0.98, 0.995 drag; 1.01 injection; 1 conserving), 8 scenes x 20 bodies; tracker = truth + 1e-3 noise, conserver = f=1 integration.

## Predictions (written before any run)
- Property test PASS: tracker esim ~1e-4, conserver esim = |truth drift| (~1e-2-1e-1) on drag/injection; on f=1 both ~0. Conservation penalty shows the reverse (tracker large).
- On conserving scenes the term equals |E_model(t) - E0| up to the truth's 4e-5 drift, i.e. it is a conservation-shaped penalty there, but only inside the 4-20-tick supervised window starting from truth states.
- A vs control: err@20 within +-10% (noise) on ab and cl; esim@20 on held-out down 20-60%; nr100 dE@1000 down 0-50% from the Euclid control's ~2 but NOT to the squared control's .125-.39; bh300 dE@1000 unchanged within noise. P(kill criterion passed) ~20%. Reason: the 1000-tick drift is a slow per-tick bias of the free rollout; a 10%-weighted term on <=20-tick truth-started windows constrains it weakly (E1's soft conservation dE loss at w1 reached nr100 .045 but cost +33% err@20).
- Warm: smaller effect than scratch (Euclid warm control already near B's energy).

## Arm B trigger and prediction (written after A, before B)
A shows bias: ab signed in-window error at tick 20 = +3.0e-4 of 4.7e-4 abs (scratch, both seeds; control -1e-4/-3e-4), and at ticks 100/1000 nearly all of |E_m - E_t| is positive (heating) in every arm incl. control. B run (scratch + warm, seed 4738). Prediction: in-window signed error -> ~0 but nr100/bh300 dE@1000 within noise of A (still 0.7-2), err@20 within +-10% of A; kill criterion still fails (85%). Reason: the long-horizon heating is not driven by the in-window bias (sign of in-window bias differs between setups while the 1000-tick sign is always +).

## Results
Arm A (term on Euclid), seeds 4738 + 4739, 4000 it, held-out 9100/9200/9300. Job 3627 (all arms + replicates + summary); fixed summarizer crash (it globbed regress_*.json).

Property test (`scatter_esim_test.py`) PASS: tracker esim ~1.2e-4 on drag 0.98/0.995, injection 1.01, conserving; conserver esim = truth drift (0.077 / 0.023 / 0.115) on non-conserving, 0 on conserving; conservation penalty is the reverse (tracker 0.077 on drag 0.98). The term rewards following truth E(t), not conserving.

Held-out (ab = 24 scenes, cl = 24 scenes; seed 4738 / 4739), Euclid control -> +esim:
- scratch err@20 ab 0.0097/0.0095 -> 0.0116/0.0120 (+20%, worse); cl 0.0424/0.0437 -> 0.0454/0.0544 (worse). esim@20 ab 3.9e-4/3.8e-4 -> 4.7e-4/4.7e-4 (worse), esim@100 ab 0.0089/0.0197 -> 0.013/0.016 (noise).
- warm err@20 ab 0.0066/0.0064 -> 0.0069/0.0068 (+5%); cl 0.0364/0.0356 -> 0.0357/0.0357 (same). esim@20 ab 4.5e-4/6.0e-4 -> 3.0e-4/3.4e-4 (better, 30-45%); cl 4.5e-3/3.4e-3 -> 4.1e-3/3.2e-3 (~5-10%). esim@100 ab 0.024/0.0137 -> 0.019/0.0147 (noise).
- Regression harness: all 8 runs pass accuracy/timing rows (-50..-74% err vs baseline) and FAIL bh300 dE_rel_model@100 / dL_rel_model@100 (same as control, 3-6x baseline).
- Conservation, median of 4 scenes, dE@1000 (nr100 / bh300): scratch ctl 0.84,1.20 / 2.30,2.21; scratch esim 0.87,1.12 / 1.96,2.04; warm ctl 1.24,0.84 / 1.72,1.37; warm esim 1.30,1.04 / 1.59,1.81. Truth-similarity |E_m-E_t| equals dE (truth drifts 4.5e-5 / 3.6e-4). No separation from control beyond seed spread (0.4-1.8 within one arm).

Verdict: A does NOT pass the kill criterion (nr100 dE@1000 stays 0.8-1.3 vs squared control .125-.39). Prediction check: err@20 within +-10% held for warm, missed for scratch ab (+20%); esim@20 down 20-60% held for warm only (scratch got worse); dE@1000 unchanged held. The in-window (<=20 tick, truth-started) term does not constrain the free-rollout 1000-tick heating, as predicted. No headline change.

Arm B (+ signed-bias term): job 3631 submitted (scratch + warm, seed 4738) but pending behind QOSMaxGRESPerUser at time of writing; results to be appended. Prediction above stands.
