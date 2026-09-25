# Energy conservation investigation (soup B, free-rollout token model), 2026-09-24

Research only. Nothing here changes the model. Scripts (uncommitted): `scripts/energy_probe.py`, `energy_probe2.py`,
`energy_probe3.py`, `energy_finetune.py`, `polaris_energy_*.sh`, `summarize_energy_log.py`. Logs: `results/energy_probe_2943.log`,
`energy_probe2_2944.log`, `energy_ft_2946.log`. All compute on Polaris GPU (jobs 2943-2947). Truth = torch fp64 re-implementation of
`bounce.py` (`scripts/ball_1k_rollout.py`), gravity 9 along +x, energy E = 0.5|v|^2 - 9x (contact spring energy excluded unless stated).

## Findings

1. Truth is fine. Total energy incl. spring PE drifts only 1401.5 -> 1394.9 (step 30, |v|max 54) and -1137 at step 100 (|v|max 126) for
   1000 balls: penalty contact at |v| up to ~100 is close to conserving in truth, so the failure is the model.
2. The model has gravity baked in (no g input). Running with g=0/1 truth gives identical model trajectories (KE 24.66 at step 5 for both);
   g=0 energy "gain" numbers in `energy_probe_2943.log` are a comparison artifact, not a finding.
3. Free flight is accurate to ~30 speed. Teacher-forced free-flight dE error per token-step: ~0 up to speed 20, -5 at 30-50, -22 at >=50
   (4 balls, n=317). Single-ball free flight dvx 1.37 (truth 1.35) for speed 1-10, 1.10 at 20, 0.77 at 80 (speed-dependent drag).
4. Energy tracks truth to step ~20 in every config (n=317, 1000 balls: E -1423 vs -1405 at step 20) and departs when balls reach the
   floor (first wall hits at speed 10-30) and start colliding at speed >30.
5. WALL response is the dominant energy error and degrades with incoming speed, inside the training range. Single ball, hidden zero,
   8 cells from the wall, n=800 (`energy_probe2`, truth restitution 1.000 at every speed):

   | incoming speed | 3 | 6 | 10 | 15 | 20 | 30 | 40 | 60 | 80 |
   |---|---|---|---|---|---|---|---|---|---|
   | y-wall restitution (model) | 1.65 | 0.95 | 0.81 | 0.64 | 0.43 | 0.41 | 0.54 | 0.60 | 0.59 |
   | x-floor dE (model; truth 0-3) | 0.0 | 0.6 | -4.7 | -14 | -67 | -97 | -472 | +1296 | -1524 |

   Training speed distribution (h44 dataset, 1.09M token-frames): mean 8.0, p90 14.1, p99 19.7, max 35.7; 32% > 10, 7% > 15, 0.9% > 20.
   Loss of energy starts at 10, well inside the data. Beyond ~35 the response is unstructured (sign flips: +1296 at 60, -1524 at 80).
   Removing `wall_head` at inference gives restitution ~ -1.0 (ball passes through), so the head is what produces the bounce, and
   it under-delivers 2*v_in.
6. Teacher-forced 1000-ball wall class (truth states, all speed bins): dE error -88 (speed<5) ... -260 (20-30) ... -750 (>=50) per
   token-step; hidden zero vs carried identical, so it is not hidden-state exposure.
7. PAIR response: momentum is NOT conserved. Isolated head-on pair in an empty box: model total-momentum error 4-10 velocity units
   (truth 0.0; `pair_curve`), although `pair_impulse` is antisymmetric. The softmax-attention message, GRU and `delta_head`
   act per-token and are not antisymmetric (`model/token_free.py:_core`, only the `pair_head` term is antisymmetric). Head-on relative KE
   ratio (truth 1.00): 0.43 (s=3), 0.10 (10), 0.01 (20). Offset-0.6 collisions at low speed GAIN energy (ratio 10.0 at s=3, 3.0 at s=6).
8. Pair tunnelling above ~13 speed: neighbour graph is built once per step on pre-step positions with radius 4.0
   (`neighbor_radius`); a pair closing at 2*s*dt > 4 (s > 13) can cross contact inside one step with no edge, so no pair force is applied
   (`energy_probe3` shows s=40 identical for offset 0 and 0.6). In the 1000-ball run speeds are 20-100 after step 20.
   Teacher-forced pair dE error grows with speed: +18 (5-10), +26 (20-30), +45 (30-50), +213 (>=50) per token-step (gain).
9. Result in the ball_1k run: dissipative wall bounces + pair damping remove KE (model KE/ball 30-100 vs truth 2344 late, floor pile,
   as seen in the video), while at speeds >~40 wall/pair extrapolation produces spurious gains (max speed 100+ at step 50, non-finite at
   step 69; pair gain +213/token-step; positive feedback: gain -> higher speed -> larger gain).
10. In-distribution too: 20x20, 4 balls, from truth start, hidden carried: model E -95.2 -> -119.3 at step 100 (truth -96.3). Zeroing hidden
    each step is worse (-154), so hidden state helps (not the culprit).
11. Fine-tunes (state-space, GPU-generated truth, soup B init), `energy_ft_*` (job 2946):
    - wide (box 20-200, init speed up to 40), 1-step loss: wall/pair curves NOT fixed (y-wall restitution 0.15/0.07/0.83/0.39, pair momentum
      err 1.5-3.4); training loss plateaued ~0.9-1.5 (fit poor).
    - wide, 8-step loss: y-wall restitution 1.07-1.20 up to speed 15 then 0.78 (20), 0.75 (30); x-floor still -19..-107 (speed 3-20);
      pair momentum error 8.6-9.6; n=20 energy still leaks (-132 at step 100); 1000-ball E -1819 vs -1275 at step 50.
    - control (same 8-step loss, original 20x20 data only): energy GAIN and blow-up (4 balls n=317: E +14095 at step 100; 100 balls
      n=100: +1487 at step 50; 1000 balls speed max 453 at step 30). A small weight change flips the extrapolation from loss to gain.
    So more/wider data through the same architecture does not restore elasticity, and out-of-range behaviour is set by noise in the
    fit, not by structure.
12. CentralForceDynamics (`model/central_force.py`, energy -3402 vs +2.6e6 at 10k) conserves because (a) force is a function of
    pair distance only, hence a potential exists and momentum is exact by antisymmetry; (b) velocity Verlet is symplectic
    (dv = 0.5 dt (a0 + a1)); (c) no velocity input, no hidden state, no unbounded/saturating features, no free per-token delta_head.
    The token model has none of these: dp/dv are unconstrained network outputs fed by |v|-dependent inputs.

## Root causes (ranked)

1. (high) Unconstrained per-token velocity/position deltas: wall and pair impulses are learned regression outputs, not a conservative
   force. Evidence: wall restitution falls to 0.4-0.8 and x-floor dE to -5..-100 within the training range, pair momentum error 4-10,
   fine-tuning on wider data did not restore it, and the control fine-tune diverged to energy gain (findings 5, 7, 11).
2. (high) Speed extrapolation beyond the training distribution (max 35.7, p99 19.7): 1000-ball gravity fall reaches 30-100 in 20-40 steps,
   past where wall/pair/free-flight responses were ever supervised (findings 3, 5, 9).
3. (medium-high) Radius-graph cutoff 4.0 vs per-step displacement 6-15 cells: pair contacts missed entirely at speed >13 (finding 8).
4. (medium) L2 loss with chaotic targets rewards velocity shrinkage; existing `--speed-weight` partly counters (documented in the
   experiment log). Consistent with dissipation but not isolated here (untested).
5. (low) Hidden-state growth, y-velocity bias, fp16, mass channel: not supported (finding 10, no fp16 used, no mass channel in this path).

## Job 2947 results (single-ball `wall` and two-ball `pair` fine-tunes, 3000 iters, lr 5e-4, from soup B; log `results/energy_ft2_2947.log`)

| run | final loss (ema) | y-wall restitution s=3/10/20/40 (truth 1.0) | x-floor dE s=10/20/40 (truth ~1/3/0) | pair relKE s=3/10/20 (truth 1.0) | n20 4-ball E@100 (truth -96) |
|---|---|---|---|---|---|
| soup B | - | 1.65/0.81/0.43/0.54 | -4.7/-67/-472 | 0.43/0.10/0.01 | -119 |
| wall K=1 | 0.87 | 1.08/1.12/0.61/0.77 | -106/-160/-346 | 0.56/0.09/0.03 | -161 |
| wall K=8 | 2.60 | 1.85/1.22/1.10/1.23 | +5/+18/-186 | 0.60/0.11/0.03 | -106 |
| pair K=1 | 0.24 (last iter 0.003) | -0.27/0.52/0.48/0.54 | +99/-86/-134 | 0.16/0.42/0.02 | +30 |
| pair K=8 | 0.17 (last iter 0.01) | 0.18/0.69/0.55/0.68 | +25/+236/-42 | 1.62/0.83/0.46 | +117 |

Reading:
- Pair terms are fittable in-range (pair-only loss reaches 0.003-0.01, vs wall-only 0.08 at K=1 and 2.6 at K=8, and wide-mix 0.9-9),
  but the fit is not the elastic response the probe wants: head-on relKE stays 0.02-0.4 at s=10-20 (s>=13 also has no graph edge),
  and momentum error stays 1.3-6. Caveat: the probe runs 30 free steps and relKE includes x-velocity divergence from model gravity error, and
  training-time pair scenes are s<=12, so this is not a clean isolation.
- Wall: even with wall-only data the elastic bounce is not reached: K=1 restitution 0.6-0.8 above s=15 and x-floor dE -60..-350 already at s=3-10;
  K=8 reaches restitution ~1.0-1.2 up to s=30 but x-floor dE becomes positive/noisy (+5..+70) and is -186 at 40 (OOD to the trained 40 max).
  Wall loss does not go near zero, so the (wall_head + GRU) path is not fitting the stiff impulse as one shot, even in range.
- Specialising one contact type breaks the others (catastrophic forgetting): pair-only runs wreck walls and n20 energy goes positive
  (+30, +117 at step 100; x-floor dE +20156 at s=60). No run improves both walls and pairs. The shared unconstrained delta head has no
  invariant that ties the terms together.
- Speed-shrinkage hypothesis (root cause 4): NOT supported. Longer-horizon L2 (K=8) raised restitution rather than lowering it
  (wide K=1 0.39-0.5 at s=15-30 vs K=8 0.78-1.03; wall K=1 -> K=8 0.6 -> 1.1 at s=20). Shrinkage is not the mechanism; the sign of the bias
  moves with training horizon and noise, consistent with an unconstrained head.

## Final ranking

| # | Cause | Confidence | Key evidence |
|---|---|---|---|
| 1 | Wall/pair impulses are unconstrained regression outputs (no potential, no antisymmetry outside pair_head, softmax/GRU/delta_head unconstrained) | high | restitution 0.4-0.8 in range; pair momentum error 4-10; wide/wall/pair fine-tunes trade errors between contact types or flip to energy gain; control blow-up |
| 2 | Speed range: fall in a 317 box gives 30-100 vs train p99 19.7 / max 35.7 | high | free flight ok <30; floor dE +/-1000 beyond 40; blow-up at 1000 balls step ~69 |
| 3 | Neighbour radius 4.0 < per-step displacement (fast pairs tunnel) | medium-high (mechanism verified in graph construction and probe3 identical s=40 rows; not ablated with a larger radius) | |
| 4 | L2 chaos-shrinkage | not supported | K=8 raised restitution |
| 5 | Hidden growth, fp16, y-bias | ruled out / not applicable | zeroed-hidden worse; same behaviour teacher-forced |

## Fix candidates

1. (cheap, ~1 GPU-hour) neighbor_radius >= max displacement + 2r, or swept-pair edges (use pos and pos+v*dt); test: head-on s=20/40 relKE. Expected: removes tunnelling, not the energy law.
2. (medium) Conservative contact: pair force f(d) only (CentralForce-style, exact momentum, symplectic) plus a wall potential term (function of wall distance only) added to, and eventually replacing, pair_head/wall_head; keep the network for residual (non-contact) dynamics. Expected: bounded energy at all speeds; matches the one property (potential + Verlet) that separated CentralForce. Cost ~1 day, needs training from scratch or head replacement. Not tested here.
3. (medium) Include gravity/speed-normalised inputs and train on speed range >= 100 with an energy-drift loss. Expected: weak on its own (finding 11).
4. (cheap) Guard clamp (speed cap, containment) as used in realtime_sim: hides the blow-up, not the collapse.
