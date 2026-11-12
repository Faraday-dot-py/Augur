# neighbour-context-pair-branch-replacing-central-pair-term: pre-registration (stage 1, written 2026-11-10, before any run)

Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md section 7 item 9; section 10a overrides (keep control iteration count; plateau guard). Branch single-law-readability-and-neighbour-pair-branch. Not yet launched.

Definitions. Exp E: pinned single-law gravity checkpoint ms_kp_pot_v_g128 (grid 128, extent 64, unit mass). Pair term: small learned network over the 16 nearest neighbours inside radius 2 (pp=2, knn=16). err@k: mean per-body position error after k self-fed ticks (tick = one step, dt 0.1). Momentum fix (momfix): Exp E subtracts the mass-weighted mean acceleration after the force is formed. Pair frame: unit vector d-hat from body i to neighbour j and its 90-degree rotation d-hat-perp.

## Change
Stock pair net: input [r, log(r+0.05)], output one radial magnitude. Candidate (flag `pairctx`, opt-in, off by default; old checkpoints load unchanged): input adds the neighbour's relative velocity along d-hat and d-hat-perp, and local density (log(1 + number of the selected neighbours with r < 2) of the query body, from the same top-16 list); output is a free 2-vector (radial, tangential) in the pair frame, times window (1-(r/2)^2)^2 and neighbour mass. Rotation-equivariant by construction; direction not forced central, no velocity independence.
Initialisation (interpretation of "zero-initialised"): the radial path copies Exp E's pair net; new input columns and the tangential output row are zero. The candidate therefore starts exactly equal to Exp E (zero-initialised residual, allowed); the tangential output starts at 0. A literal zero-everything start would delete Exp E's pair term and confound the comparison.
Dense pair path only (N <= 512, as in Exp E training and eval scenes); other pair modes raise NotImplementedError.

## Design
Both arms warm-start from Exp E and fine-tune with the same recipe (`scripts/train_scatter_ft.py`: `--fix ctrl` stock pair net, `--fix pairctx` candidate), same data and batch stream per training seed. Control (matched): ctrl fine-tune. Reference only: frozen Exp E (not the pass/kill denominator, so "more training" is separated from the change). 1500 iterations, 1000 scenes x 50 ticks, batch 16, k 4->20, lr 2e-4 cosine, loss cap 3 (existing ft defaults). Iteration count identical for both arms.
Training seeds 4738 and 4739 both run in stage 1: the spec line is "on both seeds". (This departs from the spec rule "second seed only after a pass"; flagged to the supervisor.)
Eval: Exp E-distribution scenes (`scatter_ft_data.base`), 32 scenes x 30 ticks per seed, eval seeds 9100/9200/9300; fresh 12000/12100/12200 (`--eval-seeds`) before any result is called real. err@20 per arm = mean over the 3 eval seeds; ratio = candidate / control, per training seed.
Controls and paths (Polaris): Exp E `$HOME/E_ms_kp_pot_v_g128.pt` (3,200,305 B, verified with polaris_ls 2026-11-10; same size as `$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt`). Code dir `$HOME/augur_pair_context` (not yet on Polaris; code tarball uploaded by the supervisor). Job script scripts/polaris_neighbour-context-pair-branch-replacing-central-pair-term.sh.

## Lines (copied from spec section 7 item 9, unchanged)
- PASS: err@20 at most 0.90x control on both seeds AND tangential output below 5% of radial on gravity data.
- KILL: err@20 at or above 0.95x or momentum error above control.
- Fails pass: 0.93x. Between: INCONCLUSIVE. No loosening after launch.

Mechanical reading (fixed here, not a change to the lines):
- err@20 ratio per training seed = candidate / control. KILL clause applies to the mean of the two per-seed ratios (>= 0.95); PASS needs both per-seed ratios <= 0.90.
- Tangential-to-radial = rms of the tangential pair-force component over rms of the radial component, pooled over pair entries on truth states at ticks 0, 10, 20, held-out seeds; both training seeds must be < 0.05.
- Momentum error = per scene |sum_i m_i a_i| / sum_i m_i |a_i| of the full acceleration with the momentum fix switched off, mean over scenes and ticks 0/10/20 (momfix would zero it in both arms). "Above control" = mean over the two seeds of candidate/control ratio strictly greater than 1.0. Tie-break: equality counts as not above.
- Tie-break for err@20: exactly 0.90 passes, exactly 0.95 kills (lines read literally).
- A value that would still fail: err@20 ratio 0.93 on one seed.
- Noise floor: the control's err@20 spread (coefficient of variation) across the 3 eval seeds and across the 2 training seeds is reported next to the ratios; a ratio inside that spread is described as within noise in the write-up, the line is still applied literally.
- Plateau guard: if the candidate's mean validation loss over the last 10% of iterations (iterations 1350-1499) is more than 3% below its mean over the preceding 10% window (1200-1349), a non-PASS result is labelled INCONCLUSIVE-undertrained, not KILL; one extension at double length for candidate and control together is then allowed. Validation loss: 32 held-out scenes (seed 9000), 20-tick rollout loss, logged every 37 iterations (4 points per window); control's guard is reported too.

## Prediction (stated before running)
err@20 ratio 0.97-1.03 (no meaningful gain; the earlier frozen-base residual with the same inputs cut one-step error 44% with no rollout gain), tangential/radial well under 5% (it starts at 0 and the loss has little incentive to grow it on central gravity), candidate momentum error slightly above control. Expected label KILL (momentum clause) or INCONCLUSIVE; PASS probability about 10%.

## Constraints check
- pp=2, knn=16 kept (same radius, same top-16 list). Satisfied.
- No hard-wired conservation added; momfix is Exp E's existing step, kept identical in both arms. The candidate removes a hard-wired central, velocity-independent pair form. Satisfied.
- O(N): extra inputs come from the same top-16 list (relative velocity, count of neighbours inside r<2): cost per body O(knn), independent of local density beyond the existing cap of 16. Dense path is O(N^2) at N <= 512, as in Exp E; the cell/adaptive-depth versions are not implemented (stated limit for the full run to scale).
- Seeds: 4738 first (plus 4739, see above); eval 9100/9200/9300 then 12000/12100/12200. One change against a matched control; no sweep.

## Already tried (checked 2026-11-10)
Frozen-base residual with the same inputs: one-step error -44%, no rollout gain; never trained into the model (spec section 7 item 9). No pair-branch fine-tune in docs/debugging/experiment-log.md as of this write (not re-grepped line by line; log not edited by this worker).

## Cost
Stage 1: 4 trainings x 1500 its. Logged rate: 0.2-0.3 s/it at batch 16 (docs/debugging/scatter-scenes.md:34; eager 5.1 it/s at B16, scatter-optimization.md:40) -> about 375 s each, plus data generation and validation about 150 s each, plus eval 5 min: about 0.65 GPU-h (spec 0.5; slightly above, below the 0.7 flag). Full run (6000 its, 1500 scenes) about 2 GPU-h; draft script marked NOT YET AUTHORISED.
Checkpoints: full state (model, optimiser, iteration, RNG, validation log) every 50 iterations and on SIGTERM; `--resume`; per-cell resume skips finished cells. Job time limit 1.5 h for a ~0.65 h estimate; each cell under 15 min, so the 1 h checkpoint rule is covered by the above anyway.
