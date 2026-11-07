# Optimizer sweep proposal for the tiered model (research only, 2026-11-03)

Status: proposal. Nothing run, submitted or committed. Branch unexplored-directions-audit. Written under the research-agent prompt (docs/debugging/research-agent-prompt.md): research and writing only, no Polaris waits.

## Summary (for listening)

This is a sweep: ten different ways of training the same model, each compared against the same frozen baseline. It is a sweep, and the standing rule is that sweeps are not the main strategy. So I frame it narrowly. It asks one question: is the energy drift in the tiered model an artifact of how the weights are updated, or is it built into the architecture and the states the model visits? We expect the answer to be "built in", and we pre-register that prediction. A pile of null results is the expected outcome and is useful, because it closes the optimizer lever for good.

Two findings change how to read the running Muon job. First, the model is mostly convolution layers, and the Muon hybrid only touches plain two-dimensional weight matrices. If most parameters are convolution kernels, the running job barely changes the training, and a null result from it would say little about optimizers in general. I could not verify the parameter split locally. Second, the strongest outside evidence favors an optimizer that adapts to curvature over Muon for fitting forces: one 2026 preprint on interatomic force fields found Muon worse than Adam and SOAP best.

If we only run three: SOAP, Muon applied to flattened convolution kernels, and sharpness-aware training. The last one targets the visited-state mechanism directly instead of tuning anything. Drop first: the second-order Sophia, then Adan, Lion, and the Adam beta and schedule variants. Total cost if all ten run is roughly 14 to 17 GPU-hours, packed into two or three jobs.

## 1. Definitions

- BL: frozen tiled baseline, the model whose numbers every ratio divides by. Job 4060 (commit e44b796) is the tier-0 model of record; its optimizer is plain torch Adam, learning rate 2e-4, cosine decay to 5% of peak, gradient-norm clip 1.0, 5000 iterations, unroll horizon k ramping linearly 8 to 28 ticks, initialised from the matched-60k checkpoint (scripts/train_tier0_tiled_closed_loop.py lines 117, 224, 273, 387).
- err@20: mean position error per body at tick 20. dE@100: relative energy drift at tick 100 against the truth simulator's own E(t), as in the 4975 pre-registration.
- Held-out evaluation seeds 9100/9200/9300: the three scene sets all tiered results are scored on. Training seed 4738: the seed of job 4060. Seed 4739: second training seed (job 4818).
- Adam: adaptive moment estimation; AdamW: Adam with decoupled weight decay. Muon: updates each 2-D weight matrix by an orthogonalised momentum step. SOAP: Shampoo preconditioning run through Adam in the preconditioner's eigenbasis. Shampoo: Kronecker-factored second-moment preconditioner. Lion: sign-of-momentum update. Sophia: diagonal-Hessian-clipped update. Adan: Adam with Nesterov-style extra gradient-difference term. Prodigy: Adam variant that estimates its own learning rate. SAM: sharpness-aware minimisation. GPU-h: H200 GPU-hours. s/it: seconds per training iteration.

## 2. The open problem and what an optimizer could plausibly touch

Open problem (docs/architectures.md section 1, experiment-log): tiled tier-0 energy drift dE@100 is 3.9-4.75x BL (second training seed 4.04-5.77x), pass line 2.5x, while force error is at or below BL and err@20 is 0.75-1.0x BL. Free-rollout force error is 1.5-1.6x BL against only +6 to +12% teacher-forced: a visited-state effect, cause not located. Loss-term levers, joint fine-tune, rollout-in-loop, frozen-base corrector, checkpoint soups and noise injection all failed.

Why an optimizer could still matter, in two hypotheses (neither tested):
- H-conditioning: gradients through 28-tick unrolls are noisy and badly scaled (documented gradient-quality problems; v26 long-unroll hurt, 40-step fine-tune helped). Adam's diagonal scaling may settle in a sharp, drift-prone basin that a curvature-aware or orthogonalising update would avoid.
- H-flatness: the model is fitted on truth-state starts plus 28-tick closed-loop states, but evaluated at 100 ticks in states it never trained on. A flatter minimum is more tolerant of the shift in input states.

H-architecture (visited-state and no conservation structure, the project's prior) predicts all ten arms fail. Pre-registered prediction for the whole sweep: at most one arm passes; 80% confidence that none does.

## 3. Finding on the running reference (job 4975)

Job 4975 builds the hybrid in scripts/train_tier0_tiled_closed_loop.py class HybridMuon: Muon on parameters with ndim == 2, Adam on everything else. The scatter-field model is built from nn.Conv2d layers (scripts/scatter_field.py lines 16-38, 807-809: weights are 4-D) plus small nn.Linear MLPs for the kernel and pair terms (lines 85, 93: 2-D). So Muon acts only on the Linear MLPs; every convolution stays on Adam. Untested: the actual count of 2-D versus 4-D parameters in the tier-0 config (the job log or one parameter-count print will settle it). If conv weights dominate, 4975 is close to a no-op test and a KILL from it must not be read as "Muon does not help"; it would only mean "Muon on the small MLPs does not help". Muon for conv layers is normally done by flattening each kernel to a 2-D matrix of shape (out channels, in channels x 3 x 3). This is the reason option 2 below is not a duplicate of 4975. Flagged, nothing changed in 4975.

## 4. The ten options

Common to all arms unless stated: job 4060 recipe, one lever changed, same data, init, 5000 iterations, k 8-28, same probes. Cost basis: job 4060 fine-tune 1.35 GPU-h for 5000 iterations (about 0.97 s/it, from the audit table, unexplored-directions-audit-2026-10-29.md row 4); job 4818 total 6.39 GPU-h included the 60k base. Optimizer overhead multipliers from the 2025 diffusion-on-dynamical-systems benchmark (arXiv 2510.19376: Muon 1.45x, SOAP 1.72x per step versus AdamW). Our step is dominated by 28-tick unrolled rollouts through a U-Net, so the optimizer share is smaller and real multipliers should be lower; untested, so I budget the benchmark figures as upper bounds. Common pass/kill line (verbatim from 4975, amendment guard applies): PASS if on seeds 9100/9200/9300 dE@100 <= 2.5x BL and err@20 <= 0.82x BL on all three; KILL otherwise. Caveat inherited from 4975: frozen BL err@20/BL is already 0.8215 at seed 9100, 0.0015 over the err line, so even a copy of 4060 fails by that margin on that seed. Every arm shares that caveat; do not amend.

Availability: torch 2.11.0 is present in the user environment (torch.optim has Adam, AdamW, NAdam, RAdam, Muon and Adafactor to my knowledge; the local interpreter in isolated mode could not import it, so class list not re-verified here). The pytorch-optimizer package (pip name pytorch-optimizer, github kozistr/pytorch_optimizer) lists Lion, Sophia, Adan, SOAP, Shampoo, Prodigy, Muon variants (AdaMuon, NorMuon), SAM and Schedule-Free, per its README table, fetched 2026-11-03; class names and version not checked. Schedule-Free AdamW also exists in the facebookresearch schedule_free package (pip schedulefree; from memory, not re-fetched). Everything below needs one smoke instantiation before launch.

### 4.1 SOAP
- Mechanism: keeps Shampoo's Kronecker-factored gradient covariances per weight, then runs Adam inside their eigenbasis, refreshing the basis every few steps. Conv kernels are reshaped to matrices.
- Why it might help: addresses H-conditioning directly: it whitens the correlated, anisotropic gradients that long unrolls produce, which diagonal Adam cannot. Best external evidence of fit: a 2026 preprint on interatomic force-field training (arXiv 2607.02499, from the search snippet; not opened) reports SOAP most robust and Muon worse than AdamW; the 2025 dynamical-systems benchmark (2510.19376) reports SOAP and Muon about 18% lower final loss than AdamW.
- Implementation cost: low-medium. Swap in the optimizer class, add one flag, preconditioner frequency 10, check the checkpoint and resume path stores its state (the recipe requires resume and SIGTERM save).
- Compute: 1.35 x up to 1.7 = 1.4-2.3 GPU-h.
- Pass/kill: common line.
- Risk: moderate. Eigenbasis refresh can spike at the start of a short 5000-iteration run; extra memory is small for this model size.

### 4.2 Muon on flattened convolution kernels (all-conv Muon)
- Mechanism: orthogonalise (Newton-Schulz iteration) the momentum of each conv kernel viewed as a 2-D matrix; Adam for biases, output layer and any 1-D parameters.
- Why it might help: same as SOAP on conditioning, and it is the version of Muon that actually covers the model (section 3). Evidence in force fitting is mixed (see SOAP).
- Implementation cost: low. One-line change to the parameter split in HybridMuon plus reshape wrapper; torch Muon only accepts 2-D tensors, so it needs a view and care that the hidden recurrent-field output layer keeps Adam.
- Compute: 1.35 x up to 1.45 = 1.4-2.0 GPU-h.
- Pass/kill: common line, plus read together with 4975 (if 4975 already passes, drop this arm).
- Risk: moderate. Closest to duplicating 4975 in spirit; justified only by section 3. Run only after 4975's parameter-count check; if conv parameters are under 10% of the total, 4975 already covers the model and this arm is a duplicate, so skip it.

### 4.3 Sharpness-aware minimisation (SAM) around Adam
- Mechanism: each step first ascends to the locally worst point within a small ball of radius rho in weight space, then applies Adam with the gradient taken there. Seeks flat minima.
- Why it might help: targets H-flatness, the one hypothesis aimed at the visited-state shift instead of fitting speed. A flatter solution is more tolerant of small input-distribution changes, which is what free rollout produces.
- Implementation cost: medium. Needs two forward-backward passes per iteration through the unroll and a closure; the recipe's gradient clip and skip-step logic have to wrap the second pass.
- Compute: 2x = 2.7 GPU-h.
- Pass/kill: common line.
- Risk: moderate to high. Doubled cost; rho (0.02 to 0.05 is typical in the literature) is itself a knob, so use a single fixed value and do not sweep it. Weight-space flatness of a chaotic rollout may not map to energy drift.

### 4.4 Schedule-Free AdamW
- Mechanism: replaces the learning-rate schedule with an iterate average plus interpolation, so the final iterate behaves like a schedule-annealed one without knowing the horizon.
- Why it might help: our fine-tune has a hand-set cosine schedule and a moving horizon k; removing the schedule tests whether the end-of-run decay lands in a bad spot. Weak link to drift; mostly a convergence-robustness test.
- Implementation cost: low (small dependency, but the optimizer needs train and eval mode switches before evaluation and checkpointing, an easy place to get wrong).
- Compute: about 1.35 GPU-h.
- Pass/kill: common line.
- Risk: moderate. Averaging overlaps the already-tried checkpoint soups (partial result); a null is likely.

### 4.5 Lion
- Mechanism: update is the sign of an interpolation of momentum and current gradient, scaled by a learning rate; stores one moment instead of two.
- Why it might help: sign updates equalise per-parameter step size and ignore gradient-magnitude spikes, which could help the spiky unrolled gradients. No evidence in physics fitting.
- Implementation cost: low (pytorch-optimizer). Needs lr about 3-10x lower than Adam and larger weight decay; one fixed choice (lr 5e-5, weight decay 0.01), no tuning.
- Compute: about 1.3 GPU-h.
- Pass/kill: common line.
- Risk: high that the fixed lr is wrong; a single un-tuned setting makes a null uninformative.

### 4.6 Adan
- Mechanism: Adam with an extra momentum term on gradient differences (Nesterov-like lookahead).
- Why it might help: could damp oscillation in the noisy unroll gradients. Weakest rationale of the ten.
- Implementation cost: low (pytorch-optimizer).
- Compute: about 1.4 GPU-h.
- Pass/kill: common line.
- Risk: high of a null with no mechanism learnt. First to drop with Sophia.

### 4.7 Sophia (Gauss-Newton-Bartlett or Hutchinson variant)
- Mechanism: divides momentum by an estimate of the diagonal Hessian and clips the result; Hessian refreshed every ~10 steps.
- Why it might help: curvature-aware like SOAP, but diagonal.
- Implementation cost: high. The Hutchinson estimator needs Hessian-vector products through a 28-tick unroll (double backward, large memory and an unrolled-graph double-backward path that has not been tried here); the Gauss-Newton-Bartlett variant assumes a classification-style sampled label and does not fit a squared-error rollout loss.
- Compute: 1.35 x (2 to 3) = 2.7-4 GPU-h, uncertain.
- Pass/kill: common line.
- Risk: high. Drop first.

### 4.8 Adam with lower second-moment decay and tighter clipping
- Mechanism: betas (0.9, 0.95) instead of the default (0.9, 0.999), and gradient clip 0.3 instead of 1.0, so the second-moment estimate follows recent (more recent k) gradient scale and spikes are cut.
- Why it might help: horizon k changes every iteration, so gradient scale drifts; a 0.999 second-moment window is about 1000 iterations long, 20% of the whole run, and lags the changing scale. Cheap, mechanistic, and a standard remedy in large-model training.
- Implementation cost: very low (two argument changes). Bundles two changes; label it as bundled (feedback_test_each_fix_individually permits it with a reason: both act on the same stale-scale mechanism); ablate only if it passes.
- Compute: 1.35 GPU-h.
- Pass/kill: common line.
- Risk: low cost, low expected gain; classic hyperparameter tuning, which is why it is dropped early.

### 4.9 Prodigy (parameter-free Adam)
- Mechanism: estimates the distance from init to solution online and sets the learning rate from it; no peak learning rate to tune.
- Why it might help: removes the one hand-set number (lr 2e-4, never tuned for this recipe per the log) from the comparison; the 2510.19376 benchmark describes it as a robust parameter-free alternative. Tests whether the 2e-4 choice is the problem.
- Implementation cost: low (pytorch-optimizer or the prodigyopt package); pairs with the cosine schedule already present.
- Compute: about 1.4 GPU-h.
- Pass/kill: common line.
- Risk: moderate. It tends to pick a large lr that, with a fine-tune from a good init, can wreck the init; kill probe will show this early. It is an lr-tuning proxy, so it is a pure hyperparameter arm.

### 4.10 Adam with warmup and longer decay to zero (schedule arm)
- Mechanism: peak lr 1e-4 (half), 250-iteration linear warmup, cosine to zero rather than to 5% of peak.
- Why it might help: the k=28 end of the ramp is trained at the smallest lr, but the very start jumps to peak lr at k=8 with no warmup; a warmup would protect the matched-60k init. Evidence: v29/v30 short fine-tunes at lr 3e-4 helped and longer ramps did not, so schedule sensitivity is recorded.
- Implementation cost: very low (schedule function, scripts/train_tier0_tiled_closed_loop.py line 117).
- Compute: 1.35 GPU-h.
- Pass/kill: common line.
- Risk: low cost, low gain, and a bundled change of three schedule numbers; hyperparameter tuning in form.

Total if all ten run: roughly 13.5-23 GPU-h by the upper-bound multipliers (about 14-17 using likely overheads). Training seed 4738 only in stage 1.

## 5. Recommended ordering

1. Prerequisite (free): read the 4975 log or a parameter-count print for the 2-D versus 4-D split. Decides whether option 2 stays.
2. Tier A, architecture of the update or mechanism-targeted: SOAP, all-conv Muon, SAM. Run these first.
3. Tier B, stale-scale and schedule fixes with a stated mechanism: Adam low-beta tight-clip, Adam warmup schedule, Schedule-Free AdamW.
4. Tier C, only if A and B show any movement: Prodigy, Lion.
5. Never run unless something else moves: Adan, Sophia.

Early stopping rule: run the 4060-default early probe at the same iteration in every arm; an arm that is worse than the init by the same margin the trainer already uses is cut. If tier A all KILL with drift within seed noise of BL (dE@100/BL 3.9-5.8x, the observed spread between training seeds 4738 and 4739), skip tiers B and C: the optimizer lever is closed and the savings are about 7-9 GPU-h.

## 6. Batching on Polaris

Rule: batch arms and seeds (feedback_package_tests_together); GPU audit says one training process uses about 25 of 141 GB and is launch-bound, two concurrent processes were approved on 2026-10-14 but check the audit doc for the concurrency result before relying on it.
- Job 1 (tier A): SOAP, all-conv Muon, SAM as three concurrent processes on one GPU, seed 4738, then scoring of all three on seeds 9100/9200/9300 in the same job. About 6.4 GPU-h serial, about 2.5-3.5 h wall if concurrency scales.
- Job 2 (tier B): the three Tier B arms, about 4 GPU-h serial.
- Job 3 (tier C and survivors): Prodigy and Lion, plus the second-training-seed re-run (base 4739 from job 4818) of any arm that passed or came within 20% of the line. Adan and Sophia not scheduled.
- All jobs: requeue, resume, SIGTERM trap, checkpoint every 250 iterations (recipe default), registry row with rating 2 per 4975, nice from the mapping table, GPU slot confirmed, sacct check shortly after submit, no agent idle-waits; the main session watches.
- A single-process stacked batch of arms is not available in this trainer; separate processes are the practical packaging.

## 7. Held-out-seed selection and tie-break

- Eval seeds 9100/9200/9300 are used for pass/kill only, exactly as in 4975. With ten arms scored on the same three seeds, the best arm will look good partly by chance (feedback_select_on_heldout_seeds: single fine-tunes differ 10-15% by noise).
- Selection among survivors uses a fresh seed set not used in any prior tiered scoring: 12000/12100/12200 (the 12000+ range was used as held-out for token-model checkpoints; verify that no tiered result used it before launch). The winner is the arm with the lowest median dE@100/BL on the fresh set; ties within 5% go to the cheaper, simpler arm. Then confirm that one arm once on 9100/9200/9300 and on the second training seed 4739.
- Never argmin across arms on 9100/9200/9300, never pick among checkpoints of one run (final checkpoint only).
- Pre-registered noise floor: a difference under 25% in dE@100/BL between two arms is noise (4060 versus 4818 spread). Also report each ratio against truth's own drift (0.016-0.019 absolute) as a floor-referenced ratio, per the pre-reg clause.
- Denominator caveat (audit row 8): use the frozen baseline file results/loss_term_arms/baseline_hier_eval_matched60k.json, as 4975 does; the alternative re-rolled baseline gives 3.3-3.7x instead of 3.9-4.8x.
- Independent reviewer on every result before reporting; log each arm, including KILLs, in experiment-log.md (feedback_log_every_experiment_to_notes).

## 8. This is a sweep: constraint check

Constraint check against recorded constraints:
- Generalization, tileable, O(N), interpretability (project_core_requirements): satisfied; a training-time optimizer leaves the deployed model unchanged, and tiling behaviour is what is being scored. It adds no interpretability.
- pp=2 O(N), no pairwise, no hard-coded conservative forces: satisfied, nothing added to the model or loss.
- Test each fix individually against a frozen baseline: satisfied for eight arms; the Adam low-beta tight-clip arm and the warmup schedule arm bundle changes, with a stated mechanism, ablate on pass.
- Select on held-out seeds: satisfied by section 7.
- Package tests together: satisfied by section 6.
- Prefer architecture over hyperparameter tuning: strained. Lion, Adan, Adam betas/clip, Prodigy, the schedule arm and Schedule-Free are hyperparameter-flavoured, and Sophia is too. SOAP, all-conv Muon and SAM change the update geometry, not a number, but still do not change the architecture. The architectural explanation is voiced plainly: the evidence (drift unchanged by every training-side lever; visited-state effect; teacher-forced error near baseline) points to architecture and visited states, not to optimization.
- No parameter sweeps as the main strategy: violated if this is run as a hunt for a passing optimizer. The only exception clause in the rule is "OOD probing or robustness testing". Compliant framing: a pre-registered robustness test of one conclusion, "the tiered drift does not depend on the optimizer", with the expected outcome stated as all KILL. The sweep is a bounded negative-control set closing an untried lever (the log shows no optimizer swap tried), not the main strategy; the main strategy remains the architectural and visited-state work in the tiered research directions note. To keep that framing honest: no per-arm learning-rate tuning (one fixed setting per arm), no second round of arms chasing a near miss, and a stop after tier A if it is flat. If the user prefers the rule to be taken literally, run only the three-arm shortlist.

## 9. If only three

1. SOAP: best outside evidence for force fitting, addresses conditioning, about 1.4-2.3 GPU-h.
2. All-conv Muon (after the parameter-count check): the only version of Muon that covers a convolutional model; makes 4975 interpretable either way, about 1.4-2.0 GPU-h.
3. SAM: the only arm aimed at the visited-state shift rather than at fit speed, about 2.7 GPU-h.

About 5.5-7 GPU-h total, one Polaris job of three concurrent processes. Drop order, first to last: Sophia, Adan, Lion, Adam low-beta tight-clip, warmup schedule, Prodigy, Schedule-Free AdamW.

## 10. Not verified

Parameter counts by tensor rank; class names and versions in pytorch-optimizer and schedulefree; whether 12000-series seeds are unused for tiered scoring; the concurrency result in the GPU audit; the optimizer overhead share in our unroll-dominated step; the content of arXiv 2607.02499 (search snippet only). The 2025 benchmark is on diffusion models and the force-field preprint is on interatomic potentials; neither is a deterministic rollout surrogate, so transfer is an assumption.

Sources: arXiv 2510.19376 (https://arxiv.org/abs/2510.19376), arXiv 2607.02499 (https://arxiv.org/abs/2607.02499), https://github.com/kozistr/pytorch_optimizer.
