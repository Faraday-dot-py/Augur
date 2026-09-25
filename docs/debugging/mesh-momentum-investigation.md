# Particle-mesh far field: momentum non-conservation (2026-09-25, job 2975)

Probe: `scripts/mesh_momentum_probe.py` (uncommitted), log `~/mesh-momentum-2975.log` on Polaris. Analytic force, seed 4738, cutoff 4, uniform scaled init.

## Root cause
`scripts/gravity_1b.py` `far_accel` (original): deposit was nearest-cell (`floor` + bincount), gather was bilinear (cell-centre offset -0.5). Not an adjoint pair. The FFT kernel is antisymmetric, so sum_c n_c F_c = 0 for the NGP counts n_c, but the particles feel sum_i sum_c W(x_i,c) F_c = sum_c m_c F_c with m_c the bilinear-weighted mass, m_c != n_c. Net force = sum_c (m_c - n_c) F_c != 0. This is a systematic, position-dependent mismatch (not random), so the momentum error is a net force on the cloud and integrates.

## Static probe (step 0, no dynamics; sum_i a_i of mesh acceleration, fp64 sum)
| N, grid | ngp-bilinear (old) | CIC-CIC | NGP-NGP |
|---|---|---|---|
| 20k, 64 | 4.73 | 1.3e-4 | 1.9e-4 |
| 20k, 128 | 1.04 | 3.1e-4 | 2.9e-4 |
| 20k, 256 | 2.26 | 1.4e-4 | 4.8e-5 |
| 20k, 512 | 0.676 | 3.6e-4 | 3.3e-4 |
| 1M, 1024 | 57.6 | 0.043 | - |
Adjoint pairs are at fp32 roundoff. Old mode is 1e4-1e5x larger, ~1e-4..4e-4 of N*mean|a| (non-monotonic in grid). Exact sum of true force is ~8e-6, near field ~4e-6.
Force accuracy vs exact all-pairs (mean rel err, 20k): grid 64/128/256/512 old 0.125/0.092/0.068/0.049, CIC 0.124/0.088/0.066/0.047, NGP 0.132/0.097/0.073/0.053. CIC is not worse.

## Rollout (10k, grid 256, 1000 steps dt 0.1, analytic near+mesh vs exact truth)
| variant | P @20 | P @200 | P @1000 | dE/E0 @1000 | gap @1000 |
|---|---|---|---|---|---|
| truth | 4e-4 | 6e-4 | 1.3e-3 | - | - |
| ngp-bilinear (current) | 1.12 | 1.75 | 9.73 | -0.002 | 136 |
| CIC | 2.3e-4 | 1.7e-3 | 0.041 | -0.001 | 133 |
| NGP | 2.4e-4 | 1.7e-3 | 0.039 | +0.008 | 134 |
| ngp-bilinear + subtract mean mesh accel | 3.9e-4 | 4.5e-4 | 9.6e-4 | +0.003 | 135 |
| CIC + subtract mean | 2.7e-4 | 3.1e-4 | 5.6e-3 | +0.006 | 135 |
(step = 0.1 t; the earlier "~100 by t=200" in the log means step 2000; stored npz shows model+mesh 161, analytic+mesh 62 at step 2000.)
Gap is chaos-saturated for all variants; momentum fix does not change it. Energy differences are small at this horizon.

Note: |P| growth for CIC (0.041 by step 1000) is residual fp32 accumulation of mesh pair antisymmetry (kernel aliasing at offset +-grid, clamped edge cells, kernel/near boundary), ~30x smaller than old, ~100x smaller than a cell-scale effect. Cause not isolated further (untested).

## Cost
1M, grid 1024, per mesh call: 0.013 s (old) vs 0.014 s (CIC). Two calls per step: ~+0.002 s/step on 0.10 s/step (job 2970).

## Ruled out / not the cause (by evidence)
- Learned force: analytic+mesh shows same jump.
- Kernel near/far mask, model clamp at d>150, FFT padding (2*grid zero-padded already, gather clamps only at bbox edge): none can break momentum given an adjoint deposit/gather, since NGP-NGP and CIC-CIC give ~0 with the same kernel. They affect accuracy only (untested individually).
- fp16/fp32: fp32 floor ~1e-4 observed, not 1e0.

## Fix (implemented, opt-in)
`scripts/gravity_1b.py --far-mode {ngp-bilinear (default), cic, ngp}`; `far_accel(..., mode=)`; shared `cic_corners` used for both deposit and gather. Recommend `cic` (same cost, slightly better force accuracy, P at roundoff, no post-hoc hack). Mean-subtraction also works (P ~1e-3) and is cheap but treats the symptom (removes the mean of the mesh accel, which is not the true zero-net-force distribution) and is exact only for equal masses; useful as an extra guard. Not applied to a 1M run yet (would need re-run of job 2970 config with `--far-mode cic` to confirm |P| 27 stays ~27, and to see if the abrupt collapse at step 1000 changes).

## Decision for user
Make `cic` the default (changes results of earlier mesh videos/jobs 2970, 2972, 2973 collision/flyby runs, which used ngp-bilinear) and re-run those? Nothing committed.

## Follow-up (2026-09-25): creep isolated, defaults changed, reruns

Job 2976 (probe, 2000 steps, 10k, grid 256, analytic near+mesh vs exact truth; |P| at step 200/1000/2000; truth 5.9e-4/1.3e-3/8.7e-4):
| variant | 200 | 1000 | 2000 |
|---|---|---|---|
| ngp-bilinear (old) | 1.53 | 46.6 | 241 |
| CIC fp32 FFT | 1.7e-3 | 0.040 | 0.077 |
| CIC fp64 FFT | 1.9e-4 | 4.8e-4 | 4.3e-4 |
| CIC fp32 + meansub | 2.6e-4 | 7.2e-3 | 5.3e-3 |
| CIC fp64 + meansub | 1.8e-4 | 8.1e-4 | 8.8e-4 |
Static: 20k bodies grid 256 mesh sum 1.4e-4 (fp32) -> 2.5e-5 (fp64); 1M grid 1024 0.043 -> 0.0019; cost 0.014 -> 0.015 s/mesh call.
The residual creep of CIC was fp32 roundoff in the mesh (counts, kernel FFT, convolution), not boundary/zero-pad or the near/far handoff (those cannot break momentum with an adjoint deposit/gather; fp64 removes the creep with them unchanged). With fp64 the mesh |P| is at the level of the truth's own fp32 near-field summation. Position gap and energy are unchanged by any variant.

Defaults now: `far_accel(mode="cic", fft64=True)`; CLI `--far-mode {cic,ngp,ngp-bilinear}`, `--far-fft32` to opt out. `far_field_accuracy.py`, `gravity_collision.py` pick the new default via `make_accel`.

Reruns (job 2977, one script `scripts/polaris_gravity_cic_reruns.sh`, cic + fp64, same seed/args as 2970/2972/2973; npz gitignored):
| run | old (ngp-bilinear) | new (cic fp64) |
|---|---|---|
| 1M uniform 10k steps, |P| @100/1000/2000/5000/10000 | 27 / 110 / 132 / 162 / 796 | 0.061 / 0.050 / 0.048 / 0.171 / 0.149 |
| 1M KE start / peak / end | 8.8e7 / 1.14e9 (step 1000) / 2.80e8 | 8.8e7 / 8.05e8 (step 1000) / 2.78e8 |
| 1M wall | 1030 s | 1042 s (peak 25 GB) |
| collision E0 -4.197e5, truth end -2.717e5 | model end -2.809e5 | model end -2.830e5; |P| start 6.1e-4, truth end 2.7e-4, model end 1.0e-3 |
| flyby E0 -1.365e5, truth end -5.53e4 | model end -5.81e4 | model end -5.917e4; |P| 622 start = truth end = model end (622) |
(Old collision/flyby |P| was not recorded.) Videos: videos/gravity_1m_far_cic_10k.mp4, collision_cic_10k.mp4, flyby_cic_10k.mp4.
