# Chaos-floor investigation (plan agent 5), 2026-10-04

Question: how much of the scatter-field model's late-horizon error (BH 300 bodies c=10: err@50/100 .90/4.9; Exp B 10-100 bodies: .025/.374) is irreducible chaotic divergence of the exact sim versus model error.

Method: `scripts/chaos_floor.py` (Polaris GPU). Exact reference = gravity_sim.rollout_torch math (float64, kick-drift-kick, 4 substeps/tick, eps .5, dt .1; BH relativistic c=10), re-implemented with selectable dtype/substeps (asserted equal to gs.rollout_torch in-script). Per scene: initial positions perturbed by rel * L * N(0,1) (L = rms radius of the scene, same direction for all magnitudes) with rel in 1e-6..1e-2; float32 truth (centered; Exp B also absolute coords with centre 500); substeps 8 and 2 truth. err@k = mean over bodies of |dx|, averaged over scenes, vs unperturbed float64 ref. Same for model. Splits: bodies with a close encounter (min pair distance < eps=.5 over substeps, cumulative up to tick k) vs not; Exp B by N.

## Prediction (written before any run)

- BH (cold collapse, free-fall ~ 35 ticks, collapse near tick 30-50, then violent relaxation): strongly chaotic. Truth-noise floor from a 1e-6 relative perturbation: err@20 ~1e-5, err@50 ~1e-3..1e-2, err@100 ~0.3-3. f32 truth floor ~ a 1e-7 perturbation: err@100 ~0.05-1. Model err@20 (.067) is >>100x above any floor => model error. err@50 (.90): model error still dominant (equivalent perturbation ~1e-4..1e-3 relative, vs 1e-7 f32 noise). err@100 (4.9): near saturation of the system scale (~10-20 units); perturbations >=1e-4 reach it, so ratio -> ~1 there, but that is saturation not irreducibility at the f32 level.
- Exp B (10-100 bodies, mostly regular with occasional close encounters, Lyapunov rate ~.3-1 per time unit over 10 time units => e^3..e^10 growth): err@5/10/20 are model-dominated (ratio >50x vs f32 floor). err@100 .374: f32 floor ~1e-3..1e-2 (ratio ~30+), perturbation 1e-4 gives ~.01-.1. Model error is equivalent to a ~1e-3 relative perturbation at k=100 and a much larger one (>1e-2) at k=5-20, i.e. equivalent-perturbation is NOT constant in k (model error is not mostly chaotic amplification of a fixed initial error).
- Close-encounter bodies carry most of the late-horizon floor in both.

## Results (job 3297, Polaris H200; scripts/chaos_floor.py, scripts/chaos_floor_table.py; results/chaos_{bh,expb}.json)

Reimplemented sim == gs.rollout_torch (BH max diff 0.0; Exp B 2e-10). BH: 4 scenes (seeds 9100/9200/9300/4738, 300 bodies, c=10). Exp B: 240 scenes (seeds 9000/12000/9100/9200/9300 x 48; 4738 excluded, it is the training seed). Model checkpoints: BH `checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt`; Exp B `checkpoints/budgetB/sfv/B_ms_kp_pot_v_g128.pt` (reproduces the reported .0006/.0014/.0040/.0257/.415). Perturbation = rel x scene rms radius L (L ~ 11 BH). Floors are pure position err vs the unperturbed float64/4-substep truth.

### BH 300 bodies, err@k (mean per-body position error, sim units)

| k | model | f32 truth | sub8 truth | pert 1e-6 | pert 1e-5 | pert 1e-4 | pert 1e-3 | pert 1e-2 | model/f32 | model/sub8 | model/pert1e-5 | equiv rel pert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | .00795 | 1.0e-6 | 9.2e-5 | 1.6e-5 | 1.6e-4 | .0016 | .016 | .16 | 8000 | 87 | 50 | 5e-4 |
| 10 | .0209 | 1.5e-6 | 1.8e-4 | 2.1e-5 | 2.1e-4 | .0021 | .021 | .20 | 14000 | 118 | 101 | 1e-3 |
| 20 | .0622 | 3.2e-6 | 5.4e-4 | 3.9e-5 | 3.9e-4 | .0039 | .039 | .36 | 20000 | 114 | 160 | 1.6e-3 |
| 50 | .912 | 4.3e-5 | .015 | 5.9e-4 | .0059 | .057 | .46 | 1.9 | 21000 | 60 | 155 | 1.5e-3 |
| 100 | 4.73 | .0056 | 1.1 | .076 | .54 | 1.8 | 3.4 | 6.1 | 850 | 4.4 | 8.8 | 9e-5 (saturated) |

### Exp B, 10-100 bodies, 240 scenes

| k | model | f32 truth | sub8 truth | pert 1e-6 | pert 1e-5 | pert 1e-4 | pert 1e-3 | pert 1e-2 | model/f32 | model/sub8 | model/pert1e-5 | equiv rel pert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | .0006 | 9.6e-7 | 3.0e-5 | 1.6e-5 | 1.6e-4 | .0016 | .016 | .16 | 626 | 20 | 3.8 | 3.8e-5 |
| 10 | .0014 | 1.4e-6 | 5.3e-5 | 1.7e-5 | 1.7e-4 | .0017 | .017 | .17 | 1020 | 27 | 8.2 | 8.2e-5 |
| 20 | .0040 | 2.3e-6 | 1.1e-4 | 2.3e-5 | 2.3e-4 | .0023 | .023 | .23 | 1730 | 37 | 17 | 1.7e-4 |
| 50 | .0257 | 1.0e-5 | 6.0e-4 | 8.8e-5 | 8.8e-4 | .0088 | .087 | .68 | 2540 | 43 | 29 | 2.9e-4 |
| 100 | .415 | 1.8e-4 | .013 | .0019 | .018 | .17 | .96 | 2.6 | 2240 | 31 | 23 | 2.3e-4 |

(Exp B float32 truth in absolute coords, centre 500: .000053/.000076/.000125/.00053/.0096, 50x the centered float32 floor, still 40x below the model at every k.) Equiv rel pert = 1e-5 x model/pert1e-5, valid while the floor is linear in perturbation (checked: pert1e-4/(100 pert1e-6) = 1.0 for k<=50 in both, .23 at BH k=100 = saturated).

### Findings

1. Model error is not irreducible chaos at any k<=50 in either setup. At the practical numerical-noise level (f32 truth, ~1e-7 rel) the model is 600-21000x above the floor everywhere; BH even at k=100 (850x, but see 3).
2. Equivalent initial perturbation is not constant in k: it grows 5e-4 -> 1.6e-3 (BH, k 5-20) and 3.8e-5 -> 2.9e-4 (Exp B, k 5-50). If the model error were just a fixed initial-condition-like error amplified by the dynamics it would be flat. It is instead error injected every step (model force error) and then amplified. Early on the model error (e.g. BH err@5 .008) is 50x the 1e-5 floor and its growth from k=5 to 50 (115x) is similar to the truth's amplification of a perturbation (BH pert 1e-5: 37x from k=5 to 50; model 115x), so late error is "step error x chaotic amplification", not separable post hoc.
3. BH err@100 = 4.73 is at the chaos ceiling: a 1e-3 relative perturbation already gives 3.4 and 1e-2 gives 6.1 (saturation ~6, positions decorrelated). The truth sim itself is not converged at k=100: halving the substep size (4 -> 8) changes the truth by 1.1 (sub2 vs 4: 2.2), and a 1e-5 perturbation by .54. So BH err@100 cannot be used to rank models/checkpoints below ~1; err@50 (.91 vs truth-discretization floor .015, 1e-5 floor .006) is still model-dominated and is the latest informative BH horizon. Model's own sensitivity (1e-5 perturbation of the model rollout): err@50 .04, @100 1.69 (similar to truth's: .0059/.54 scaled by ~3-7x, the model is more chaotic than the truth).
4. Exp B err@100 = .415 vs floors .018 (1e-5), .013 (sub8), 1.8e-4 (f32): model-dominated, 23-30x above the truth's own discretization floor. The CentralForce gap (.278) is real (both are far above floor). Equivalent perturbation ~2-3e-4 relative at k=50-100.
5. Truth substep sensitivity is the largest noise term: sub8-vs-sub4 floors (BH .015@50, 1.1@100; Exp B 6e-4@50, .013@100) exceed f32 noise by 100-1000x. The model is trained to the 4-substep truth, so this is a reference-definition issue, not model error; any Exp-B-style number below ~.01 at k=100 or BH below ~.015 at k=50 is within reference ambiguity.
6. Regimes. BH: bodies with a close encounter (min pair distance < eps=.5 before tick k) are 45% at k=5 and 82% at k=20, so the split is weakly discriminating there; model error is 3-6x higher for close-encounter bodies at k=50 (.95 vs .16) and the truth floor is 8-100x higher (sub8 .016 vs .0002 @50): the no-encounter bodies are model-dominated by 800x vs sub8 at k=50, close bodies by 60x. Exp B: 14% of bodies close by k=5, 74% by 50, 93% by 100; at k=100 close-body floor (pert1e-5 .0196, sub8 .0144) is 11x the no-close floor (.0018/.0012), and model error is 5x (.438 vs .087): close encounters produce both. No dependence on N (10-30/31-60/61-100: model .33/.44/.43 @100, floors similar).
7. Lyapunov: pert 1e-5 curve growth from tick 1: BH x1.04/1.4/2.6/39/3550 at k=5/10/20/50/100; Exp B x1.0/1.1/1.5/5.6/118. Local exponent d ln(err)/dt: BH .54 (t<=2), .94 (t 2-5), .91 (t 5-10) per time unit (unit time = 10 ticks); Exp B .23, .44, .62. The built-in per-rel fit (30x window) returned None for Exp B/1e-2 because growth never reached 30x before plateau; use these windowed slopes. Effective BH Lyapunov time ~1.1 time units (11 ticks); Exp B 1.6-4.4 time units. Note: truth in Exp B is much less chaotic over 10 time units (e-folds ~4.8) than the BH collapse (~9.1).

### Caveats
- Only 4 BH scenes (all 300-body c=10 collapse); no variance estimate beyond per-scene logs.
- Perturbation is on positions only, isotropic gaussian, scaled by rms radius; velocity perturbation untested. Equivalent-perturbation numbers depend on that definition.
- Close-encounter threshold .5 (=eps) is loose in dense BH cores; stricter thresholds untested.
- Reference ambiguity (substeps) means "model error" at late k is measured against one particular discretization.
