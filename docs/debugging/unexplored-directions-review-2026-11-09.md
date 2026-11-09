Architecture reviewed: MultiScatterField (scripts/scatter_multi.py `MultiScatterField(sf.ScatterField)`): CIC-deposit grid, width-32 5-level U-Net potential head, FFT learned kernel, pp=2 knn-16 pair term, Verlet, 20-float law descriptor theta (4 ordered-pair slots x 5), two species, no momentum fix; mixed six-law training (pow1, pow2, pow3, yuk8, yuk128, lj; pow1.5 and yuk32 held out). This is what main_s4738 / main_s4739 are (docs/theory/supervisor-log.md:154). The task premise "token-per-ball" is wrong: token-per-ball is the legacy model (docs/architectures.md sec 3).

# Unexplored-directions review, 2026-11-09 (review only; nothing launched, no scripts written)

Sources mined: experiment-log.md, unexplored-directions-audit-2026-10-29.md, rule-free-force-recovery.md, sae-random-pool-dose-response.md, error-correction-plan.md, priority-registry.md, gpu-slot-chains.md, .remember/today-2026-11-08.md, memory index, and `git log --all` (near-field-retrain branch 173ed13/f4822a4/780c8f1; interp commits f426fb6, 50d522d, 09bf374).

Common facts used below
- Training recipe and runner: scripts/train_scatter_multi.py exists ONLY on branch near-field-retrain (173ed13); `git ls-files` on this branch shows 0 matches. Full-state ckpt every 500 it, SIGTERM exit 143, --resume, --init-ckpt, --only, --per-shape (that branch, lines 64-84, 160-200). Needs porting and committing before any retrain (committed-dependency rule).
- Cost anchor: 0.315 s/it, 80k it = about 7 h (experiment-log.md:3755). Fine-tunes of 3000 it = about 0.26 h.
- Baselines: main_s4738 / main_s4739 on Polaris ~/augur_nfr/ckpt (experiment-log.md:3476). Gravity err@20: 4738 7.89 (one-scene blow-up), 4739 .106, Exp E .0407 (experiment-log.md:3435).
- Standing facts: 4738 dec2 is exactly zero (32/32 channels dead), dec3 mean_abs 3741 vs 0.078 in 4739 (experiment-log.md:3517). Unbiased image review: 4738 had 44 BAD scenes, 41 of 48 in power-1 (experiment-log.md:3435). Rule-free recovery KILL (2 PASS of 16): model curves deviate from truth for pow3, lj (e_curve 1.3 vs floors .27/.17) and pow1 (.34 vs .017) (docs/debugging/rule-free-force-recovery.md "Why it fails").

## Ranked candidates (1 GPU each; checkpoints noted)

### 1. mixed-law-retrain-without-amplitude-10-power-1-scenes (night 1, slot 1)
- Tests: removing the high-amplitude power-1 scenes stops the dec3 blow-up and the 4738 failure.
- Hypothesis: the BAD scenes and dead dec2 in 4738 come from amplitude near 10 power-1 scenes (amp is 10^U(-1,1), train_scatter_multi.py:4), not from mixing laws.
- Evidence: experiment-log.md:3435 (41/48 BAD in power-1), :3436 ("Not retried: ... dropping the amplitude-10 power-1 scenes"), :3517 (dec3/dec2); audit row 16.
- Not tried: the 3436 line says so; no entry after it.
- One change vs frozen baseline (same recipe, seed 4738, 80k it, init E_frozen.pt): pow1 scenes with amplitude above 10^0.5 dropped (cut value in needs-user).
- Pass: seed 4738 gravity err@20 <= 0.0509 (1.25x Exp E, the near-field pre-reg regression limit), dec2 live (mean_abs > 0.1), image-review BAD <= 5 of 48, every trained non-pow1 law tick-1 force within 1.25x of main_s4739. Kill: gravity err@20 > 0.106 (the 4739 value) or dec2 all dead or BAD >= 20. Between = INCONCLUSIVE. A value that fails pass: gravity 0.07.
- Tie-break: n/a (no argmin).
- Script: port train_scatter_multi.py plus an amp-cap flag; eval scatter_multi_eval.py / score_multi_nearfield.py (same branch).
- Cost: about 7 h + 0.5 h eval = 7.5 GPU-h. Checkpoints: yes, runner has them. Priority 3.
- Caveat: 80k it of 72k scenes is about 17.8 epochs, not 50 (needs-user).

### 2. mixed-law-dec3-activation-blowup-by-law-and-amplitude (night 1, slot 1, before candidate 1; CURIOSITY, no verdict)
- Tests: where dec3 activation magnitude comes from in main_s4738 (law, amplitude, tick, species).
- Hypothesis: dec3 mean_abs 3741 is driven by large-amplitude power-1 scenes; if it is spread across laws, candidate 1 is the wrong fix.
- Evidence: experiment-log.md:3517 (layer stats not stratified by law or amplitude).
- Not tried: layer stats exist but never stratified (job 4833 entry); nothing else.
- Reuse: scripts/network_utilization.py layer-stats mode with a per-law/amplitude loop (small new script or flag). Pre-reg: observations only, no pass/kill. Needs supervisor curiosity approval.
- Cost: 0.3 GPU-h inference. No checkpoint needed.
- Chain: runs first in slot 1 so candidate 1's pre-reg can name the right fix.

### 3. two-body-curve-scenes-finetune-vs-matched-control (night 1, slot 2 after 7269)
- Tests: adding exact-force two-body scenes to training fixes the model's two-body force curve.
- Hypothesis: curve errors for pow1, pow3, lj are a data-coverage gap (isolated pairs at r 0.25-14 are rare in N=10-300 scenes), not capacity.
- Evidence: rule-free-force-recovery.md "Why it fails" (fitter not the cause; model curves deviate); experiment-log.md:3475-3477 (pow1 exponent 0.108, 4739 worse).
- Not tried: 2-body appears only as evaluation (log:3475, :3685); no entry trains on two-body scenes (grep 2026-11-09).
- Design: fine-tune main_s4739 3000 it, seed 4738, two arms: control (original data) vs treatment (10% of batches two-body scenes, separation log-uniform 0.25-14, trained laws and amplitudes as before). Metric: rule_free_force_curves.py e_curve on the held-out interleaved grid, pow1, pow3, lj; plus gravity err@20.
- Pass: e_curve for all three laws <= 0.5x control AND gravity err@20 <= 1.15x control. Kill: any of the three > 0.8x control OR gravity > 1.30x control. A value that fails pass: pow3 at 0.6x control. Tie-break n/a.
- Script: scripts/rule_free_force_curves.py (exists, committed); new two-body generator flag in the ported trainer.
- Cost: 2 x 0.26 h train + about 0.6 h per curve extraction (job 7190: 37 min for 16 cells) = about 1.7 GPU-h. Checkpoint: arms under 1 h each; runner has them anyway.

### 4. mixed-law-shift-equivariance-probe (night 1, slot 2)
- Tests: force change under a half-cell global shift for the mixed checkpoints (audit row 17, stage 1).
- Hypothesis: sub-cell shift sensitivity (6-10% on Exp E, error-correction-plan.md:53) is law-independent and present in main_s4738/9.
- Evidence: error-correction-plan.md:53, :64; audit row 17.
- Not tried: only on Exp E (D6, job 3526); not on the mixed model.
- Pass (proceed to candidate 5): median relative force change > 2% on >= 3 of 6 trained laws. Kill: > 2% on <= 2 of 6. Value that fails pass: 1.5% on all six.
- Script: new small probe (scripts/probe_shift.py is an unrelated file; do not reuse).
- Cost: 0.3 GPU-h inference. No checkpoint.

### 5. shift-consistency-loss-finetune-vs-matched-control (slot 2, afterok candidate 4 pass)
- Tests: soft penalty on force difference between a scene and its sub-cell-shifted copy (audit row 17 stage 2).
- Hypothesis: it halves the shift sensitivity without worsening energy drift (tileability).
- Evidence: audit row 17; error-correction-codebase-proposals.md:31-37; plan :64 ("needs same-schedule control, count dE rows as failing"). Risk: jitter augmentation made nr100 dE@1000 x5.8 (plan :53), so dE rows are in the pass line.
- Not tried: row 17 NEVER-RUN.
- Pass: probe value <= 0.5x control AND err@20 <= 1.05x control AND dE@100 <= 1.10x control. Kill: err@20 > 1.15x control OR dE@100 > 1.5x control. Value that fails pass: probe 0.6x control.
- Cost: about 1.5 GPU-h (audit estimate; 2 x 3000 it fine-tune plus eval). Checkpoint: not needed (under 1 h per arm).

### 6. galilean-boost-augmentation-finetune-vs-matched-control (night 2)
- Tests: random constant-velocity boosts of whole scenes as an opt-in soft prior reduce spurious velocity dependence (audit row 18).
- Hypothesis: boost invariance cuts angular-momentum drift; valid here because all six laws are central and velocity-independent (scatter_multi.py:accel).
- Evidence: audit row 18; error-correction-codebase-proposals.md P4 (velocity-dependence of near-body error).
- Not tried: NEVER-RUN (nearest: solar-scale-translation-augmentation-retrain, a different model, INCONCLUSIVE). Opt-in prior only (not applied if a future law is velocity-dependent).
- Pass: dL@100 <= 0.7x control AND err@20 <= 1.10x control. Kill: err@20 > 1.15x control. Fails pass: dL 0.8x control.
- Cost: about 1.5 GPU-h; new --boost flag. Boosted truth is P+ut, V+u, so the 32-extent acceptance rule applies (needs-user: boost range).

### 7. mixed-law-gravity-replay-curriculum-retrain (night 2, slot 1)
- Tests: extra pow2-single scenes interleaved through training keep the gravity path from regressing.
- Hypothesis: the regression (.0407 to .106 / 7.89) is forgetting of gravity, not law-conditioning limits.
- Evidence: experiment-log.md:3435-3436 ("Not retried: ... gravity-replay curriculum"); audit row 16.
- Not tried: stated in log.
- Pass: gravity err@20 <= 0.0509 AND every trained law tick-1 force <= 3x its single-law comparator (the original near-field line). Kill: either fails. Fails pass: gravity 0.06.
- Cost: 7.5 GPU-h; ckpt in runner. Should run after candidate 1 (so effect of the amp cut is known; tests individually).

### 8. single-law-smallest-width-sae-dose-response (afterok job 7256 verdict; conditional)
- Tests: interpretability protocol (TopK SAE + dose-response ablation) on the smallest sufficient single-law width from job 7256.
- Hypothesis: with one law and a minimal net, top-k SAE features are causally specific (ablation error >> matched-random), which the mixed model only showed from k=5 (sae-random-pool-dose-response.md, commit f426fb6).
- Evidence: memory project_interp_single_law_plan_2026_11_08; experiment-log.md:3495-3511.
- Not tried: single-law interp is the sanctioned next step, nothing run; 7256 verdict pending.
- Pass: top-5 ablation raises median force error >= 5x the matched-random mean AND above the random max, on >= 2 of 3 held-out scene sets. Kill: < 2x on >= 2 of 3 sets. Otherwise INCONCLUSIVE. Fails pass: 3x. Tie-break n/a.
- Script: sae_interp.py, sae_ablation_random_pool_dose_response.py (exist). Width list from 7256.
- Cost: 1-2 GPU-h. No checkpoint needed.

### 9. mixed-law-training-with-more-law-diversity (night 2, slot 1)
- Tests: adding four more law shapes to training improves held-out pow1.5 / yuk32.
- Hypothesis: held-out error (1.6-2.5x single-law comparator) is a diversity limit.
- Evidence: experiment-log.md:3430-3435 ("needs more law diversity or a structural prior"; held-out 1.6-2.5x). Data variant only, not a sweep.
- Not tried: stated as untested.
- Pass: held-out pow1.5 and yuk32 tick-1 force <= 1.3x cmp_pow1.5_single / cmp_yuk32_single. Kill: either > 1.6x. Fails pass: 1.45x.
- Cost: 7.5 GPU-h; new LAWS entries; after candidates 1 and 7.

### 10. adaptive-depth-3e6-argmax-body-brute-force (any gap)
- Tests: why default and adaptdeep differ by 1.37e-2 at collapsed 3e6.
- Evidence: experiment-log.md:3765-3768 (reviewer: cause untested).
- Not tried: listed as open.
- Pass: the argmax body's adaptdeep top-16 set equals brute-force up to distance ties. Kill: adaptdeep set differs from brute-force beyond ties (neighbour at gap > 1e-6). Fails pass: a different 16th neighbour at distance gap 1e-5.
- Cost: 0.2 GPU-h. No checkpoint.

## Chains (two GPUs at once at most)
- Slot 1 (about 8 h, free): 2 (0.3 h) -> 1 (7.5 h). Night 2: 7 (7.5 h), then 9.
- Slot 2 (7256 then 7269, then about 4 h): 4 (0.3 h) -> 3 (1.7 h) -> 5 (1.5 h, only if 4 passes) -> 10 (0.2 h). Candidate 8 afterok the 7256 verdict. Candidate 6 night 2 after 5.
- Dependencies by afterok; 7269 is not recorded in gpu-slot-chains.md (only in .remember/today-2026-11-08.md).

## Needs-user
1. Premise correction (token-per-ball is not the main_s4738 model).
2. Epoch rule: 80k it is about 17.8 epochs of 72k scenes; 50 epochs would be about 3.7x (about 26 h). Keep the baseline recipe?
3. Amplitude cut for candidate 1 (proposed: drop pow1 amp above 10^0.5).
4. Boost range for candidate 6.
5. Approve curiosity label for candidate 2.
6. Port train_scatter_multi.py from near-field-retrain onto this branch (committed dependency).
7. Confirm ckpt/main_s4738.pt and main_s4739.pt are still on Polaris ~/augur_nfr/ckpt.
8. Yukawa lambda selection (log:3507): reviewer noted no tie rule; whether to re-score with one (an amendment, not done).

## Rejected (one line each)
- Neighbour-context pair branch (audit row 9): architecture change, not training/data/loss; out of scope without a go.
- Exact mesh-pair complement (row 8): user rejected 2026-11-03.
- Mass-conditioned curriculum (row 12): tiered/hierarchy code, not this recipe.
- Softening-conditioned input from scratch (row 31): eps already in theta (scatter_multi.py params_of / LAWS).
- Interior-offset rerun on main checkpoint: KILL on Exp E at 1e6, ratio 1.0104 (log:3742-3746); no new information.
- Plain-PM comparison for mixed model: Exp E comparison done (log:3471-3472); no mixed-law prediction.
- Third mixed seed 4740: pure replication; candidates 1 and 2 test the cause.
- Re-running rule-free recovery or Yukawa selection unchanged: KILL, lines not loosened (log:3507-3511, rule-free doc).
- Pairwise pp=8, token-per-ball, tiered/replay/Muon/SAM/solar, neuron-graph, LJ, CFD: other architectures or lines, or banned.

## Review
Independent reviewer ran (fresh agent, 2026-11-09). It found wrong line numbers (3433/3434/3438/3511-3513, supervisor-log 49-62, network_utilization:17), non-partitioning lines in candidates 4, 8, 10, and the nearest-neighbour note for candidate 6; all fixed above. No candidate was found to be already tried. Numbers (.0407, 7.89, .106, 44 BAD, 41/48, dec2, dec3) confirmed. The 17.8 epochs figure is arithmetic from 72k scenes (log 3435).
