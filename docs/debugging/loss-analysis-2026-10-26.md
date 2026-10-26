# Loss-function analysis for the scatter-field / tiered gravity model (2026-10-26)

Status: Part 1 pre-registration written BEFORE any measurement. Results, literature and recommendation are appended below as they land.

## Definitions
- e_i(k): Euclidean position error of particle i at rollout tick k, |x_model - x_truth|. err@k = mean_i e_i(k).
- worst-1% share: sum of e_i^2 over the 1% of particles with the largest e_i, divided by sum of all e_i^2 (pooled over scenes; also reported per scene).
- excess kurtosis of e_i (0 = Gaussian; large = heavy tail).
- training-loss scalar for a k-tick window from t0 = 0: lp = sum_steps sum_i,dims (x_model - x_truth)^2 / (N * 2k), lv likewise for velocity, loss = lp + 0.1 lv (+ mass term, not measured here), as in scripts/train_scatter_field.py.
- local density n_i: number of truth particles within radius 3 (simulation units) of particle i at tick k. Bins: lowest 50% of particles, middle 40%, top 10% ("dense").
- coherent error c_i: mean error vector over the 16 nearest truth neighbours of i (including i); incoherent residual r_i = e_vec_i - c_i.
- direction: radial = along the vector from the centroid of the 16 nearest truth neighbours to i (positive = outward), tangential = perpendicular to it, along-velocity = along the truth velocity.

## Models and scenes (existing checkpoints, CPU rollouts)
scatter-gravity-clustered-baseline (E_ms_kp_pot_v_g128.pt) and tier0-matched-60k-iterations; also tier0 tiled (tile 32). Multiclump scenes, span 64, N = 300 and N = 1000, 100 ticks, held-out seeds 9100.., truth = tiered_truth rollout.

## Pre-registered rule (Part 1)
The tail-term idea (any loss term that up-weights the worst particles: top-k, CVaR, L-p with p > 2, focal) is DROPPED if the pooled worst-1% share of squared error is under 20% at BOTH ticks 20 and 50, for both models, at both N. If only some cells exceed 20%, the idea is kept only for those cells and flagged as scene-dependent. Predictions (prior, before measuring): worst-1% share 8-18% at tick 20 (50-70% of the squared error in the top-10% density bin), incoherent residual larger than coherent part for the baseline, tier0 dense bin with a positive signed kinetic-energy-error bias.

## Part 1 results (CPU rollouts, no GPU; scripts/loss_error_roll.py, scripts/loss_error_analyze.py; raw results/loss_analysis/stats_n300.json, stats_n1000.json)
Setup as pre-registered, except: 8 scenes at N = 300 and 4 scenes at N = 1000 (seeds 9100+), 100 ticks, multiclump, span 64, dt 0.1. Arms: baseline = scatter-gravity-clustered-baseline; tier0 = tier0-matched-60k-iterations (single window); tier0-tiled = same tier 0, tile 32. Density bins use truth density at tick k. N = 1000 in span 64 is far denser than training (baseline err@20 0.47 vs 0.10 at N = 300), so N = 1000 is an out-of-distribution stress case.

Worst-1% share of squared position error (pooled), tick 20 / tick 50:
| arm | N=300 | N=1000 |
|---|---|---|
| baseline | 0.25 / 0.17 | 0.39 / 0.15 |
| tier0 | 0.21 / 0.19 | 0.12 / 0.10 |
| tier0-tiled | 0.11 / 0.12 | 0.16 / 0.12 |
Worst-5% share (tick 20): baseline .54/.65, tier0 .48/.29, tiled .36/.38. Excess kurtosis at tick 20: baseline 16 / 45, tier0 12 / 7, tiled 3.4 / 9.3. Baseline N = 1000 tick 5: one blown-up particle holds 86% of squared error (kurtosis 463), a single-event outlier, not a distribution property.

Coherent share of squared error (neighbourhood-mean part), tick 20: baseline 0.08 / 0.11, tier0 0.19 / 0.43, tiled 0.54 / 0.36 (N = 300 / 1000). The tiled arm's error is mostly coherent (whole neighbourhoods displaced together), the baseline's is incoherent scatter.
Direction (rms at tick 20, N = 300): radial / tangential / along-velocity = baseline .117/.095/.122, tiled .186/.172/.189: no preferred direction, error is isotropic in position.
Density: dense top-10% bin holds only 12-26% of squared position error (tiled 26% at N = 300 tick 20) yet its mean signed kinetic-energy error (model minus truth, per particle) is the largest and negative in every arm and both N at ticks 20 and 50 (e.g. N = 1000 tier0 -19.9 at tick 20, -46.9 at tick 50; tiled -14.8 / -37.5; baseline -5.2 / -8.9; units mass x speed^2/2). Low-density half is slightly positive. This is a one-signed bias concentrated where density is highest, invisible in position-error shares.
Training-loss scalar: position term lp_proxy = mean e^2 / 2 is 0.011 (baseline) / 0.015 (tier0) / 0.032 (tiled) at tick 20, N = 300; it tracks err@20 and says nothing about sign, coherence or density.

### Verdict against the pre-registered rule
Rule: drop the tail-term idea if worst-1% share < 20% at ticks 20 and 50. Result: tick 50 is under 20% in all 6 cells (0.10-0.19). Tick 20 is under 20% in 4 of 6 cells; baseline fails it (0.25, 0.39) and tier0 at N = 300 is borderline (0.21). The rule is therefore NOT cleanly met for the baseline. Applied where it matters (the tiled hierarchy that fails): 0.11 / 0.16 at tick 20 and 0.12 / 0.12 at tick 50, so tail terms are DROPPED for the tiered failure. For the baseline at tick 20 the tail share is real but is driven by few close-encounter particles / a single blow-up scene, which Euclid and Huber already addressed (Euclid: p99 -52%, loss-function-tests.md). Caveats: top-1% of 300 particles is 3 particles per scene; ticks >= 50 are chaos-dominated (floor), so the shares there do not measure learnable bias; truth-density binning moves particles between bins as clumps disperse.

## Part 2
See docs/debugging/loss-literature-review-2026-10-26.md (25 arXiv abstract pages opened, abstract-level only; gaps listed there: pushforward specifics, CVaR/top-k in simulators, curriculum, FMM-inspired learned gravity not covered).

## Part 3: ranking against the measured failure (systematic, signed, dense-region, coherent in the tiled arm; not outlier)
Prior negatives checked: Euclid (position win, dE worse), global energy-similarity (weak: a 20-tick global term does not constrain slow bias; kill), force-anchor (helps warm only, fails dE), relabel curriculum, Huber/focal (inert), long-horizon penalty (dE .0212 vs needed .0125, com_err +18-42%).
| rank | candidate | targets the measured failure? | prior overlap | verdict |
|---|---|---|---|---|
| 1 | local signed kinetic-energy-work term: per density bin / k-neighbourhood, squared mean (not per-particle) signed error of dKE and of path work over the window | yes: signed, local, density-weighted | global energy-similarity was scene-total and 10%-weighted; local-neighbourhood form untried | run |
| 2 | coherent-displacement term: squared k-nearest-neighbour-mean position and velocity error (clump-centre-of-mass error), weight upweighting the coherent part | yes for the tiled arm (coherent share .36-.54) | com_err used only as eval metric | run |
| 3 | soft curl penalty on the learned acceleration field (soft prior) | plausible root cause of work bias | lj mesh energy-gradient and potential-head work exist; gradient mesh gave no benefit for gravity (dE .029 vs .015) | low priority |
| 4 | Sobolev term on field gradient | speculative | none | low |
| 5 | sliced-Wasserstein on per-particle dE or pair distances | matches distribution not bias; literature: invariant-measure paper | none | low |
| - | tail/CVaR/top-k, heteroscedastic, more noise, Huber, hard HNN | no (Part 1: tail share < 20% in tiered arms; systematic) | Huber/focal inert | drop |

### Design 1 (rank 1): local-work-bias term
Loss = existing (lp + 0.1 lv + lm) + w_work * mean over bins b of (mean_{i in b} [dKE_model_i - dKE_truth_i])^2, where dKE_i = 0.5 m_i (|v_i(t_k)|^2 - |v_i(t_0)|^2) accumulated at every unrolled step, bins = quantiles of truth local density (k-nearest-neighbour count already computed by the model) per scene, w_work set so the term starts at about 10% of lp (value recorded, not swept). Signed mean per bin so one-sided bias cannot cancel into noise. O(N) (reuses knn 16).
Test plan: separate run vs frozen baseline = tier0-matched-60k-iterations recipe with identical seed 4738, data, schedule; control = same fine-tune with term off. Stage A: 4000-iteration warm fine-tune from the tier-0 checkpoint, both arms, about 0.3 GPU-h per arm (earlier 4000-iteration arms took 540-1200 s). Stage B only if A passes: full 60,000-iteration retrain, 5.1 GPU-h, plus second seed 4739 on the other GPU. Total plan 0.6 GPU-h for stage A, +10.2 GPU-h for stage B.
Pre-registered PASS (held-out seeds 9100/9200/9300, teacher-forced diagnosis of the 1.66M particle-tick set plus tiled rollouts): dense-bin energy-work error <= 1.5x baseline (currently 2.1x), tiled energy drift at tick 100 <= 0.012 (baseline-level; currently 0.019), plain err@20 <= 1.10x control, force error no worse than +10%, mid-density beta within +-0.15. KILL otherwise; stop at stage A if dense-bin work error is not <= 1.7x.

### Design 2 (rank 2): coherent neighbourhood term
Add w_coh * mean_i |mean_{j in knn16(i)} (x_model_j - x_truth_j)|^2 (plus velocity at 0.1) per unrolled step. Same stage plan, cost, and control. PASS: tiled clump-mean error (com_err) <= 0.7x current tiled value and plain err@20 <= 1.10x control and energy drift not worse than control; KILL otherwise. Run separately from Design 1 (not bundled), after Design 1 stage A, since both fit one GPU-batch together (batch per feedback: package tests together, one job, separate arms).

No training was launched (CPU analysis only; sandbox and Polaris GPUs not used).
