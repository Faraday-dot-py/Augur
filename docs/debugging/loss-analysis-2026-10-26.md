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
