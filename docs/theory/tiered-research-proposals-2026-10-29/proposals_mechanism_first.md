# Mechanism-first proposals for the tiered architecture

Reference model: tiled closed-loop tier 0, e44b796 (branch tier0-tiled-closed-loop, job 4060). Eval scenes: seeds 9100/9200/9300, 24 scenes x 100 ticks, Verlet dt 0.1. Training seed 4738 first, then 4739.
BL = tiled baseline (err@20 0.067, dE@100 0.010; its 0.010 is flattered, exact force through the same dt-0.1 Verlet drifts 0.016-0.019).
dE@100 = mean over scenes of |E_model(100) - E_truth(0)| / |E_truth(0)|. PE = potential energy. KE = kinetic energy. err@k = mean position error per body at tick k.
Constraints re-read and satisfied by all five: no hard-coded conservative forces (soft penalties only), O(N) model (exact-force labels are training-time only), no pp=8, one lever per run, energy term is similarity to truth E(t) never E(0), CFD untouched, no param sweeps.

Facts that frame the set (from the log, verified against supervisor-log.md and experiment-log.md; not re-run):
- Free-rollout force error of e44b796 is 0.039-0.041, BELOW the baseline's 0.052, yet dE@100 is 3.3-4.75x baseline. So the drift is not random force error; it is a small systematic component.
- Drift is a roughly linear loss of about -0.035 |E0| per seed, growing after tick 20; located in PE at separations 0.5-4, densest third by local density (dPE -0.038 at tick 100); pair counts 5-7% too many there; teacher-forced force gain 0.98-1.02 at every nearest-neighbour distance.
- PE is too negative by 0.038 but total E is low by about 0.035, so KE did not rise to pay for the tighter binding (derived from two different runs: localisation job 4079 and audit job 4064; not measured on one rollout). Over-binding without matching KE is not what a conservative force with a coupling error would do.
- Exact force through dt 0.1 Verlet has the opposite sign (+0.0285). The model's -0.038 is therefore not integrator error alone.
- Untested in the log: whether tier 0's learned force has curl; whether the loss at ticks 28+ is chaos-dominated (stated only as "likely cause, untested" in the residual-corrector entry).

---

## 1. virial-and-loop-work-decomposition (dissipative or curl component in the learned force)

Idea: test whether the learned tier-0 force contains a non-conservative part (curl and/or velocity-correlated drag) that does net negative work on bound clumps each orbit. If yes, add a soft penalty on that part to the existing head.

Targets: drift with force error below baseline; PE too negative with KE not rising; linear growth after tick 20 (work per orbit accumulates).

Mechanism: a conservative force with a coupling error would tighten clumps AND raise KE (virial). Tighter PE with unchanged KE means energy is removed by the force law itself. A curl part gives net work around closed orbits; a drag-like part (a_err correlated with -v; the tier 0 sees velocity and recurrent channels in closed loop) removes KE. Both are invisible to a per-particle force-error metric.

Experiment (frozen e44b796, no training; reuse tier0_close_pair_localise.py rollout dump, 7d57992):
1. Per-clump virial ratio 2KE/|W| vs tick, model vs truth, densest third.
2. Loop work: integrate the model force on circles of radius 1, 2, 4 about clump centres at visited states (ticks 30, 60, 90), divided by the closed-loop integral of |F||dl|. Truth is zero by construction.
3. Regress the force error a_model - a_true at visited states on -v and on position-only terms; report drag coefficient and fraction of work error explained.
Pass line (mechanism confirmed): virial ratio at tick 100 at least 10% below truth, AND (mean |loop work|/closed-loop |F||dl| >= 0.03 OR drag regression explains >= 40% of the signed work error). Kill line (mechanism rejected, go to proposal 2): loop work < 0.005 and drag share < 10%.
If confirmed, fix arm: fine-tune e44b796 with soft penalty on grid-force curl (finite-difference on the acceleration grid) plus a one-sided penalty on v . (a_model - a_true) < 0 at visited states, closed loop k 8-28, seed 4738. Pass: dE@100 <= 2.5x BL, err@20 <= 1.25x BL, plain err@20 <= 1.10x e44b796. Kill: dE@100 > 3.5x BL or err@20 > 1.25x BL at the iteration-750 probe.

Difference from killed levers: removing velocity/recurrent inputs (killed) changed inputs and was trained teacher-forced; the hole-1 gradient-mesh gravity arm (dE .029 vs .015, 5.6x slower) replaced the head. This diagnoses the current head first and penalises softly; soft curl penalty is permitted by the standing rule.

Cost: diagnosis 0.3 GPU-h; fix arm about 2.5 GPU-h (job 4060 was 1.35 h plus penalty).
Requirements: interpretability (gains a named measurable defect), generalization neutral, tileable preserved (penalty is local), O(N) preserved. No violation.

## 2. coherent-radial-force-bias (small inward bias hidden under random error)

Idea: split the free-rollout force error into a coherent part (neighbourhood-averaged, radial about the local clump centre) and a random part, find the bias that over-contracts clumps, and train with a loss on the coarse-grained force rather than only per-particle.

Targets: force error 0.039-0.041 below baseline but over-contraction at r 0.5-4 in the densest third (pair counts 1.05-1.07x truth); tiered error is mostly coherent (share 0.36-0.54, loss analysis 75d8622). Also the unexplained constant 1.55x model/exact interior force offset at 1e6 bodies (experiment-log, kpad entry) suggests a scale/softening mismatch of the grid force; connection untested.

Mechanism: per-particle squared error is dominated by random scatter, so a 1-2% inward bias (gain 1.00-1.02) costs almost nothing in the loss but integrates into extra binding. Deposit plus bilinear gather gives a smoothed effective kernel, so effective softening or coupling can differ slightly from the truth eps 0.5.

Experiment: (a) frozen model, visited states ticks 30-100: coherent radial bias b(s) = <(a_model - a_true) . r_hat> averaged over neighbourhoods of 2, 4, 8 units, versus the random rms; (b) isolated Plummer and uniform-sphere probes: model radial force vs analytic over r 0.25-8 to read off effective G and effective softening.
Pass line (confirmed): b/rms >= 0.2 at scale 4 in the densest third, with the same sign as the PE excess (inward). Kill line: b/rms < 0.05 (then the bias is not coherent radial; proposal 1 or 3).
Fix arm if confirmed: add a multi-scale pooled-force loss (force error averaged over 2/4/8-unit neighbourhoods, radial component) to the closed-loop fine-tune, weight set so the term is at least 20% of the loss at start (the killed clump-centre position term was 0-6% of loss and barely moved the model). Pass: dE@100 <= 2.5x BL, err@20 <= 1.25x BL, mid-density bias within +/-0.15. Kill: dE@100 > 3.5x BL.

Difference from killed levers: clump-centre term (killed, 6336ef3) penalised position error of 16-NN means with a tiny share; this targets force bias directly, after measuring that it exists.

Cost: diagnosis 0.4 GPU-h; fix arm about 1.5 GPU-h (one fine-tune of 3000 iterations).
Requirements: generalization (effective-G check on analytic probes), interpretability. Tileable, O(N) unchanged. No violation.

## 3. integrator-compensation-test (learned force absorbing the dt-0.1 Verlet error)

Idea: check whether tier 0 learned a force that compensates the Verlet step error at the short training horizon, a compensation that overshoots later. Resolve the apparent contradiction between two logged results.

Targets: oracle exact force through dt 0.1 Verlet gives dPE +0.0285 (pairs fly apart) while the model gives -0.038, i.e. 0.066 away from the integrator-only value; yet hole-1 gravity (non-tiled) halving dt cut drift 3.5x (0.0149 to 0.0043) and err@20 about 40%. The localisation entry concluded dt is not the cause from the oracle sign alone; that inference ignores compensation.

Mechanism: truth is generated finer than dt 0.1, so the net must output F plus a shadow correction that depends on (v . grad)F and F . grad F. Fitted over 28 ticks this correction is fine; in the densest clumps (short orbital period relative to dt) it is wrong in sign or size and accumulates.

Experiment: (a) inference only, frozen e44b796, integrate with two substeps of dt/2 (same weights): does dPE move toward 0 or flip? (b) oracle: exact force at dt 0.1 vs dt 0.05 vs truth, restricted to the densest third, to size the compensation budget. Pass line (confirmed): inference-only dt/2 changes dE@100 by >= 30% (either direction) with the same weights. Kill line: change < 10%.
Fix arm if confirmed: closed-loop tiled fine-tune at dt 0.05 (truth subsampled at dt 0.05), seed 4738. Pass: dE@100 <= 2.5x BL, err@20 <= 1.25x BL, cost <= 2x. Kill: dE@100 > 3.5x BL or cost > 2.5x.
Not a param sweep: one pre-registered change motivated by a measured mechanism; also tests whether the 2x cost can be traded by the cached-pair substep route used in LJ (lj-retrain-dt-0125-cached-pair-substeps, 1.24x).

Cost: diagnosis 0.2 GPU-h; fix arm about 2.6 GPU-h (1.9x job 4060).
Requirements: generalization (dt-robustness), tileable preserved, O(N) preserved (cost x2 constant). Possible violation: none of the four; mild tension with prefer-architecture-over-hparam, hence gated on diagnostic (a).

## 4. restart-from-truth-drift-rate (is post-tick-28 growth accumulation, hidden state, or visited-state?)

Idea: decompose the growth of dPE after the 28-tick training horizon into three causes by restarting the model from truth states, and, if visited-state is the cause, train with force labels at visited states instead of longer BPTT.

Targets: drift grows after tick 28; free-rollout vs teacher-forced force error gap (1.5-1.6x vs +6-12%, before closed-loop training); open item "a loss term that reaches beyond tick 28" with memory limits (OOM at k=22 for joint fine-tune).

Mechanism: three candidates give different signatures. Per-tick bias at dense states: a fresh restart from truth at tick 60 shows the same -dPE/tick as continuing. Hidden-state drift: a restart with a warm-started hidden field (8 teacher-forced ticks) shows a smaller rate than continuing. Visited-state effect: restart from the model's own tick-60 state with fresh hidden shows the same rate as continuing, and the rate from truth states is smaller.

Experiment (frozen e44b796, no training): for start ticks 28, 60: run 20 ticks from (i) continuing model state, (ii) truth state with 8-tick hidden warm-up, (iii) model state with hidden reset-and-warm-up; report d(dPE)/dtick in the 20-tick window, densest third, 3 seeds. Pass line (visited-state is the cause): rate(i) ~ rate(iii) >= 2x rate(ii). Other outcomes: rate(ii) ~ rate(i) means per-tick bias (go to proposals 1-3); rate(iii) < 0.5 rate(i) means hidden state.
Fix arm if visited-state: visited-state-force-replay. Roll the model without grad to tick 100, store states at ticks 28-100, train with one-step force regression against exact direct-sum force at those states (labels are training-time only, N <= 1000), mixed 30% into the existing closed-loop batches. No trajectory loss beyond 28, so no chaos target and no BPTT cost. Pass: dE@100 <= 2.5x BL, err@20 <= 1.25x BL, free-rollout force error <= 0.052. Kill: dE@100 > 3.5x BL.

Difference from killed levers: rollout-in-loop and long-horizon penalties used trajectory or energy losses with BPTT; this supervises forces at far visited states.

Cost: diagnosis 0.5 GPU-h; fix arm about 2 GPU-h.
Requirements: generalization (training on own distribution), O(N) model preserved (direct-sum labels only at training). No violation.

## 5. chaos-floor-noise-control (separate chaos from systematic drift, then use a chaos-tolerant loss)

Idea: measure how much of the late-tick position error and of dE is explained by random force error of the observed size acting on a chaotic system, using exact force plus injected noise as a control; if position error is chaos floor, change the loss so late ticks stop contributing noise gradient.

Targets: residual-corrector KILL (a57452c) with "likely cause, untested: the 40-tick squared-position loss is dominated by chaotic late ticks"; energy-similarity term halved drift at tick 40 on seed 4738 but cost 1.29x err@20 (7c6d709); err@20 passes while dE fails, so the two may be different problems.

Mechanism: beyond the Lyapunov time a squared position loss pulls toward one unpredictable trajectory, giving noisy gradients that mask the systematic signal. If random 4% force noise on exact force reproduces err@20 and err@50 of the model but yields drift near the exact-force Verlet floor, then position error is irreducible chaos and the model's extra dE is systematic bias (proposals 1-3). If noise alone yields drift of about 0.03, the metric itself is noise-sensitive at this horizon and the target line (2.5x BL) is mis-specified.

Experiment: (a) exact force through dt 0.1 Verlet plus Gaussian relative force noise of 3%, 4%, 5% (matched to the measured 0.039-0.052 rel error, spatially uncorrelated and also correlated over 4 units), 3 seeds x 24 scenes: report err@20/50/100 and dE@100. (b) truth perturbed by 1e-4 initial noise to get the Lyapunov time t_L and the chaos-floor err@k. Pass line (chaos explains positions, drift is systematic): noise control err@50 within 20% of the model's, AND noise control dE@100 <= 0.5x the model's. Kill line: noise control dE@100 >= 0.8x the model's.
Fix arm if pass: tolerance-hinge loss, per-tick penalty max(0, |x_model - x_truth| - c * floor_k) with floor_k from (b), k up to 40 with gradient checkpointing (as job 4021), fine-tune e44b796, seed 4738. Pass: dE@100 <= 2.5x BL with err@20 <= 1.25x BL. Kill: dE@100 > 3.5x BL (no better than the 4x control).

Difference from killed levers: long-horizon k penalty variants used energy and position penalties without a chaos floor; this first measures whether a floor applies.

Cost: diagnosis 0.4 GPU-h (CPU-feasible at N <= 300 but run on GPU per project rule); fix arm about 3 GPU-h (k up to 40 with checkpointing).
Requirements: generalization, interpretability (gives a principled horizon). Tileable and O(N) unchanged. No violation.

---

Ordering: run the shared rollout dump once for 1, 2, 4 (about 1 GPU-h total diagnostics); 3 and 5 need their own cheap runs. Only the diagnostic whose pass line fires gets its fix arm. Log each outcome, failures included, in experiment-log.md before reporting.
