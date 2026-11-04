# Reactive optimizers and the tiered model (research note, 2026-11-03)

Status: research only. Nothing run, nothing committed. Source read: the project page https://xingjianbai.com/reactive-optimizers/ (via fetch summary). The arXiv paper (listed on the page as 2610.33826) and the code repo (https://github.com/xingjian-bai/opal) were NOT opened; all claims about them below are from the page and are unverified against the paper.

## Summary (for listening)

A reactive optimizer is a training optimizer whose update rule is rewritten by a language-model agent while a single training run is in progress. The agent watches probes it wrote itself, then changes learning rates, momentum, clipping, or which optimizer a layer uses. The authors report gains over tuned fixed recipes on language-model, image-generation and toy tasks, at roughly 1.01 to 1.14 times the wall-clock if the agent works in the background.

For the tiered model, this is a training-time tool only. It cannot correct a rollout, and putting an agent in the rollout loop would break the generalization and interpretability requirements. The tiered model's open problem is energy drift, about two to five times the baseline, with force error already below baseline. Earlier levers (extra loss terms, joint fine-tuning, rollout-in-loop training, weight averaging, noise injection) did not fix it, which points at the visited-state effect and architecture, not at the optimizer. So the expected benefit is small.

Recommendation: do not pursue it as a main line. If we want to try it once, run a single closed-loop fine-tune where an agent may only adjust learning rate, gradient clipping and the unroll-length ramp, judged against the existing drift and error lines. Label it as low priority, and treat a null result as expected. Swapping to a plain static optimizer such as Muon is the cheaper control and should come first if anyone wants to test the optimizer lever at all.

## 1. What reactive optimizers are

Definitions: AdamW = Adam with decoupled weight decay. Muon = an optimizer that orthogonalizes the update for matrix weights. FID-50K = Frechet Inception Distance on 50,000 generated images (lower is better). OPAL = "OPtimizer with an Agent in the Loop", the authors' implementation.

Mechanism (from the page):
- An LLM agent, zero-shot (no memory from earlier runs), edits two programs during one training run. `analysis.py` computes statistics it chooses: per-group gradient norms, update-to-weight ratio, cosine between consecutive gradients, loss spikes. `updater.py` is the code run at every step that sets the update for each parameter group.
- The agent sees the full history of the current run at each decision (about 100 decisions in the largest run).
- Observed behaviors: multi-stage per-group learning-rate schedules including mid-run increases; moving attention and MLP matrices from AdamW to Muon; keeping some optimizer moments in float32 where bfloat16 rounding was dropping updates; per-tensor clipping and a gate that holds back unusually high-loss batches; on toy landscapes, momentum raised on slopes, lowered near the minimum, reset after overshoot; on noisy-label CIFAR-10, a validation-selected weight average.
- Usage is a wrapper around an existing optimizer: pass model, scheduler, total steps, a goal string and a dollar cap; call step with the loss.
- No equations or pseudocode are given for the update rule; the rule is whatever code the agent writes.

Claimed results (page; mostly single runs, a few repeated three times):
- 162M-parameter GPT speedrun track: matches or beats the best of five leaderboard recipes (bar chart only).
- Goldfish Amharic language model: official recipe 83.48 bits per sentence; OPAL from it 81.11; best swept static recipe 80.89; OPAL from the sweep point 80.73.
- LightningDiT-XL/1 on ImageNet: FID-50K 1.83 vs 2.18 (with guidance), 4.33 vs 5.15 (without). RAEv2-B: 3.19 vs 3.39.
- 34 toy landscapes: mean log10 distance to optimum -11.6 vs -2.2 (best of 12 common recipes) vs -7.4 (best of 382,080 swept recipes).
- Noisy-label CIFAR-10: test loss 0.685 and 78.6% accuracy vs 2.493 and 73.5% for AdamW trained straight through.

Costs and limits (page):
- Wall-clock: 1.30x to 4.39x if training pauses for each agent decision; 1.01x to 1.14x if the agent runs asynchronously (edits land a median 65 updates late).
- Money: about $6.42 of agent calls for about 100 decisions, on top of about 74 H100-hours of training.
- Gains shrink when runs are short or the recipe is already well tuned. Reproducibility is reduced. Some agent changes were reverted later. Sharded and compiled training at scale is unfinished.

## 2. The tiered model's relevant facts (checked against docs/architectures.md section 1)

- Tiers are identical-architecture grid models; tier 0 is the scatter-field model on particles, upper tiers see exact merged moments. Training is sequential and frozen: tier 0 first, then residual training for each higher tier. Best tier-0 recipe: matched 60,000 iterations then closed-loop tiled fine-tune with 28-tick unrolls (job 4060, commit e44b796).
- Open gap: energy drift at tick 100 (dE@100) is 3.9-4.75x the baseline (BL), about 2x the exact reference's own drift, while force error is below BL. Free-rollout force error is 1.5-1.6x BL against only +6 to +12% teacher-forced: a visited-state effect, cause not located.
- Unroll-length history: stepwise ramp no gain, from-scratch long-unroll hurt (v26), short fine-tunes at 40-step unrolls helped (v29/v30). Training on long unrolls has known gradient-quality problems.
- Already tried and failed (do not repeat): loss-term levers (energy-similarity, work-bias, clump-centre, pair-potential, tail/CVaR/Huber, more noise injection), joint fine-tune of tiers (drift 6x), rollout-in-loop training, frozen-base residual corrector, removing recurrent inputs, checkpoint soups (partial), kernel fine-tune through unrolls. Grep of experiment-log.md found no run that tested an optimizer swap (AdamW to Muon), per-group learning rates, or an adaptive optimizer; those are untried. Honest caveat: the log is 3,586 lines and I searched by keyword, not by reading all of it.
- Recorded constraints: generalization, tileable, O(N), interpretability; pp=2 (no pairwise, no pp=8); no hard-coded conservative forces; do not use parameter sweeps as the main strategy; prefer architectural causes over hyperparameter tuning; test each fix individually against a frozen baseline; select on held-out seeds.

## 3. Candidate uses

Constraint check applies to all: a reactive optimizer changes only how weights are trained, so the deployed model keeps the same architecture and cost. It does not add pairwise terms or hard-coded forces. It does not help interpretability, and it does not add generalization by itself. The agent's validation-based choices are a risk for the held-out-seed rule.

### Use A: agent-controlled closed-loop fine-tune (training wrap)
- What: wrap the optimizer in the job 4060 recipe (matched-60k start, tiled closed-loop, 28-tick unrolls). Agent may change learning rate, clipping, and loss-spike gating only.
- Constraints: satisfies all four (training only) and the O(N) deployed cost. Violates or strains: "no param sweeps as main strategy" (it is automated tuning) and "prefer architecture over hparam tuning". Reproducibility loss conflicts with exact-metrics tracking, mitigated by logging every agent edit.
- Expected benefit: low. The gap is not a convergence failure (validation loss and force error are already good); the page itself says gains vanish on tuned recipes and short runs. Possible small help with the gradient spikes of unrolled training.
- Cheapest test: one arm, one seed (4738), same budget as job 4060, against the frozen job 4060 numbers; agent sees training-set probes and a held-out-scene validation split that is disjoint from eval seeds 9100/9200/9300.
- Pass/kill (one line): PASS if on seeds 9100/9200/9300 dE@100 is at most 2.5x BL with err@20 at most 0.82x BL on all three; KILL otherwise. Tie-break: if the agent finishes with several checkpoints, take the final one, no argmin on eval seeds. Do not log the agent's edits as an amendment path.

### Use B: agent-controlled unroll-length ramp (training wrap, curriculum)
- What: the agent decides when to lengthen the unroll (28 to 40 and beyond) from its own gradient-norm and gradient-cosine probes, instead of the fixed ramp that gave mixed results (stepwise no gain, from-scratch hurt, fine-tune helped).
- Constraints: same as A. Closest to the reactive-optimizer strengths (it reacts to instability).
- Expected benefit: low to moderate; the one place the project's own history shows schedule sensitivity.
- Cheapest test: same as A but the agent controls only the ramp; fixed-ramp control is the existing v30-style 40-step fine-tune.
- Pass/kill: PASS if err@20 and dE@100 both improve over the fixed-ramp control by more than the 10-15% seed noise recorded in the fine-tune notes on all three held-out seeds; KILL otherwise.

### Use C: per-tier optimizers
- What: separate agent per tier during sequential training (tier 1 residual training).
- Constraints: same as A. Tiers share architecture, so the optimizer need not differ.
- Expected benefit: negligible. Teacher forcing suffices for tier 1 (exposure-bias gap 1.056) and upper tiers add only +4 to +10% error. The drift is in tier 0.
- Verdict: reject, not worth a test.

### Use D: rollout-time correction
- Not applicable: an optimizer has no role at inference. An agent that watches a rollout and edits it would hard-code corrections (violates generalization, interpretability, and the no-hard-coded-forces rule) and costs LLM calls per run. The page's analysis-probe idea is already how our agent workflow operates (diagnostics by channel). Reject.

### Use E: static optimizer swap as the control
- What: Adam to Muon (or per-group rates; the job 4060 baseline is plain Adam, not AdamW) for matrix weights in the same closed-loop fine-tune, no agent. This is the page's most reproducible finding and tests whether the optimizer lever matters at all, for a fraction of the cost.
- Constraints: same as A, minus reproducibility loss.
- Cheapest test: one arm, seed 4738, job 4060 budget.
- Pass/kill: same line as Use A. If this control fails, skip A and B.

## 4. Recommendation

1. Do not make reactive optimizers a research line for the tiered model. The evidence says the remaining drift is a visited-state and architecture problem, and every non-architectural lever tried so far failed.
2. If one optimizer experiment is wanted, run Use E first (static, reproducible, cheap). Only if it moves dE@100 meaningfully, run Use B (the ramp-controlling agent), because that is where the training history shows schedule sensitivity. Use A is dominated by E and B.
3. Reject C and D.
4. Before any run: read the paper and repo (not done here); confirm the agent can be run with held-out seeds hidden; check the priority registry and GPU slot; pre-register the lines above verbatim. Per the standing curiosity rule, propose it as "curiosity" if the supervisor prefers no pass/kill commitment; the lines above are the non-curiosity version.

Open questions not answered here: whether OPAL can control a custom closed-loop training loop with tiled rollouts; its agent cost for a 4-hour job; whether LLM-agent calls are permitted inside Polaris jobs (network access not checked).
