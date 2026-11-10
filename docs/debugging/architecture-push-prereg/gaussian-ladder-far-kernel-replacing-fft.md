# gaussian-ladder-far-kernel-replacing-fft: stage 1 pre-registration

Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md (sections 5, 6, 7; 10a overrides). Registered before launch; do not edit lines after launch.

## Lines (copied from the spec)
- PASS: err@20 (mean position error after 20 ticks) at most 1.15x the control AND tiled-versus-single-grid force difference at most 4% with no global term AND exponent read from the amplitudes within 10% on pow2 (potential of 1/r^2 force law).
- KILL: err@20 above 1.30x the control.
- Fails the pass line (INCONCLUSIVE): exponent error 15% (any value above 10% without a KILL is INCONCLUSIVE; 15% would fail).
- Control: Exp E (ScatterField with the FFT far kernel), `augur_nfr/kern/E_frozen.pt` (also `augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt`; both verified present on Polaris with polaris_ls). The multi-law model `augur_nfr/ckpt/main_s4739.pt` is fitted per law for kernel fit error and exponent only; it is not rolled out.

## What is run
- Ladder: K(r) = sum over l of a_l exp(-r^2 / 2 s_l^2), 7 levels, s_l = 0.5 x 2^l (units), amplitudes free in sign, fitted once by weighted least squares to the frozen learned kernel on the 2G x 2G lattice (weights 1/(|K| + 0.01 max|K|), r up to the lattice half-diagonal). Each level is a separable Gaussian blur, stencil plus or minus 4 coarse cells, on a grid coarsened by 2^max(l-1, 0), then bilinearly upsampled. No FFT. Fixed numbers: 7 levels, s0 0.5, 4 taps. None is tuned.
- Exponent: slope of log|a_l| against log s_l over levels 2 to 6; force exponent = potential exponent + 1.
- Tiled difference: kernel acceleration on 8 pow2 scenes (seed 9100) computed tile by tile (tile 64 cells, halo 32 cells, density outside tile+halo dropped, no global term) against the single grid; relative L1.
- Rollout: Exp E against Exp E with the ladder replacing its kernel and the U-Net frozen at E's weights, 48 pow2 scenes per seed, 20 ticks, seeds 9100/9200/9300. No training.
- Per-law kernel fit error and exponent for main_s4739 (pow1, pow2, pow3, yuk8, yuk128, lj): expected force exponents 1, 2, 3; yuk and lj report fit error only.

## Seeds and reality rule
Eval seeds 9100/9200/9300. Before a result is called real: fresh seeds 12000/12100/12200. A second training seed (4739) only after a pass (there is no training in stage 1).

## Tie-break rule
Ladder amplitudes are one closed-form fit (no argmin over candidates). If a selection among equal-valued options is ever needed, the lower index wins (lowest level, lowest seed).

## Ratio lines against the reference noise floor
Every ratio is the mean over seeds of ladder / E per seed, printed beside E's own seed spread (max - min) / mean. A ratio within that spread is reported as "inside the noise floor", not as a difference. The 1.15 and 1.30 lines are applied to the mean ratio as written.

## Prediction (stated before launch)
- Rollout ratio: likely near 1 to 1.2. Stated: PASS on err@20 is plausible but not certain.
- Exponent on pow2: likely FAILS. A local probe on a 1/r potential read 1.60 against 2.0 (20% off) because the fit gives oscillating amplitudes with sign changes. Expected outcome is therefore INCONCLUSIVE, not PASS.
- Tiled difference: at risk. The widest Gaussians (s up to 32 units) exceed the 16-unit halo, so the 4% line may fail.

## Plateau guard (section 10a)
No training in this stage, so not applicable. Stated for completeness: INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window, keeping the control's iteration count.

## Constraints check (spec section 5)
- pp=2, knn=16 pair term: unchanged (same model, only the far kernel path is replaced). Satisfied.
- No hard-wired conservation: none added. Satisfied.
- One change against the frozen control: the far kernel only; the U-Net is frozen at E. Satisfied.
- Matched training budget: no training. Satisfied.
- No sweeps: levels, widths, taps, tile, halo fixed above. Satisfied.
- Seed 4738 first; eval 9100/9200/9300 then 12000/12100/12200: stated above.
- At most 2 GPUs: one GPU, about 0.5 GPU-h; supervisor chains it.
- Requirements Til (tileable), Int (interpretable), O(N): ladder cost is linear in grid cells; amplitudes are readable. The tileability is what the 4% line tests.
- Checkpoints: not needed (under 1 h, per-cell resume in the result json).
- Rendered diagnostic and fresh reviewer: required when rollout output is produced; the supervisor runs them after the job.
- Names: descriptive kebab-case.

## Job
scripts/polaris_gaussian-ladder-far-kernel-replacing-fft.sh (about 0.5 GPU-h). Full run: none authorised; no trainer for a ladder-kernel retrain is written.
