# scatter-field-three-part-exact-swap-drift-attribution: pre-registration

Registered before launch; do not edit lines after launch (amendments are logged deviations naming a value that would still fail). Rating 2, GPU slot: queued behind the slot tails, one GPU job at a time.

## Question
Which of the three scatter-field parts (the FFT kernel, the U-Net potential residual, the pair term) contributes the most closed-loop drift error, and does the dominant part change over rollout time? Checked first: docs/debugging/experiment-log.md has no closed-loop exact-formula swap of each part (the nearest are the Oct 16-19 exact mesh-pair complement, teacher-forced only, and the Oct 21 core-drift note that put the core bias in the U-Net term, measured on forces not by swap; scatter-field-vs-cf-gap-investigation.md has sf_floor, sf_nonet, sf_prior_pp8 which remove parts by retraining or zeroing, not by exact swap).

## Models (frozen, no training)
- exp-e-frozen: `~/augur_nfr/kern/E_frozen.pt` (260,419 parameters; ScatterField grid 128, extent 64, pp 2.0, knn 16, split kernel, potential head, Verlet, momentum fix on). Loaded strict as `scatter_field.ScatterField`; learned kernel (no analytic prior), learned pair weight.
- width8-single-law-control: `~/augur-single-law-network-sizing/ckpt/width8.pt` (57,986 parameters; `scatter_multi.MultiScatterField` with a U-Net of width 8, zero law descriptor, no momentum fix; built by `single_law_network_sizing.build_model(8, ...)`, loaded strict). Both files verified present on Polaris with polaris_ls; both checkpoints were also loaded locally on CPU.

## Definition of exact, per part (per body, direct sums in fp64, unit masses, eps 0.5, G 1; the scatter-field convention of sf_floor and sf_prior_pp8, not the token-ball gravity=9)
- F: exact softened force, sum over j of (x_j - x_i)/(r^2 + eps^2)^1.5.
- exact kernel K_exact: the model's own mesh pipeline (CIC scatter, FFT convolution, 2h central difference, CIC gather) with the analytic split kernel -1/sqrt(r^2+eps^2) x (1 - (1 - min(r/pp,1)^2)^2).
- exact pair term P_exact: all pairs with r < pp (not only the 16 nearest), weight r/q^3 minus the analytic continuous mesh share a_cont (the `prior` pair weight of ScatterField and the base variant of scripts/scatter_r1_step0.py on branch r1-step0, results/r1_step0). The learned pair arms keep the model's own 16-neighbour path, so knn truncation counts as pair-term error.
- exact U-Net replacement U_exact = F - K_exact - P_exact: whatever closes the gap between the exact force and the analytic mesh plus analytic pair complement (the mesh discretisation correction). Approximation: it is a per-body vector, not a grid potential, and it depends on the exact force at the rollout's own current state each tick (an oracle).
- Arm = which of the three are exact. Total acceleration per body = U + K + P (+ momentum fix for exp-e-frozen, as the model does). Consequence: total force error = (K - K_exact) + (U - U_exact) + (P - P_exact), additive in force, so the force error of each arm is exactly the sum of the parts left learned.
- The U-Net always receives the learned kernel acceleration as input and its recurrent field is always advanced, so a swap changes only the force that part adds. Untested alternative: feed the U-Net the exact kernel (R1 step 1 showed the U-Net compensates its own mesh response, 6.5% to 63%, so this variant would add a distribution-shift effect).
- Caveat that follows from the definition: U_exact is defined against the exact kernel, so the learned U-Net partly cancels the learned kernel's own deviation. Swapping only one of the two can make the error larger (negative gain); this is a result, not a bug, and is what the interaction numbers report.

## Arms (8 per model)
all learned; exact kernel; exact U-Net; exact pair term; exact kernel + U-Net; exact kernel + pair term; exact U-Net + pair term; all exact. A ninth, non-attributed reference run, exact-force velocity-Verlet at dt 0.1 in fp64, is run alongside.

## Scenes and rollout
Held-out seeds 9100/9200/9300 and fresh seeds 12000/12200/12100. Per seed 8 scenes of each of two families from `scatter_error_growth.make_scenes`: expB (10-100 unit-mass bodies, scale init, gravity_sim generator) and cluster (80-300 bodies, `scatter_field.make_cluster_dataset`). 48 scenes per seed set per family pair: 3 seeds x 2 families x 8 = 48 scenes per set, 96 per model. Truth = exact N-body from the same generators (fp64 kick-drift-kick, 4 substeps per tick, dt 0.1, eps 0.5, 150 ticks, `scatter_error_growth.exact_traj`); same generators, eps, dt and tick horizon as the sf_floor and error-growth diagnostics so numbers compare. Model arms are rolled in Verlet exactly as `ScatterField._step` with positions and velocities carried in fp64.

## Metrics
- Rollout position error vs truth at every tick 0-150 per arm: mean over bodies of the displacement norm, mean over scenes (the err@k of the earlier tables).
- Relative force error by part: rms over bodies inside |x| < 32 of (learned part - exact part), divided by the rms exact force, for kernel / U-Net / pair term / total; teacher-forced along the truth trajectory (recurrent field carried, every tick, tabulated at ticks 1,5,10,20,30,50,75,100,125,150) and along the all-learned arm's own trajectory.
- Time-resolved attribution per tick: gain_i = err(all learned) - err(part i exact); Shapley average over the 8 arms of the error reduction with the parts in a coalition exact (average over the six swap orders); share_i = Shapley_i / (err(all learned) - err(all exact)), so the three shares sum to 1; dominant part per tick = largest share; crossover = tick at which the dominant part changes (reported with how many ticks the new leader holds it); interaction = sum of the three single-swap gains over the all-exact gain, at ticks 20, 50, 100. 95% bootstrap intervals over scenes for the single-swap gain fractions at ticks 20, 50, 100.
- Figures: err per arm, part shares with dominance strip and crossover annotations (both models, held-out and fresh), and force error by part.

## Unit check (fail loudly; stop and report on failure)
Per group, all must hold or the script exits 3: (1) the three learned parts reproduce the model's own `force()` output at truth states ticks 0, 50, 100 to relative max 1e-5; (2) the all-exact arm reproduces F teacher-forced to relative rms 1e-5; (3) the all-exact arm's closed-loop rollout matches the independent fp64 exact-force Verlet reference to 1e-4 position error at tick 20; (4) my exact kernel path equals the model's own prior-kernel mesh path to relative max 1e-5. Note (2) and (3) hold by construction in fp64 and mainly test the plumbing (flag selection, padding, momentum fix); (1) and (4) test the decomposition against independent model code.
Soft cross-check: the U-Net-exact term over the exact force should be near the 11.5% mesh floor of sf_floor (job 3261).

## Floor and ratio clause
The truth is itself numerical (4-substep integrator, dt 0.1 single-step arms). The floor is the all-exact arm's error against truth (the exact-force integrator floor, about .002 at err@20 on expB in the earlier notes). Every err is also printed as a ratio to that floor. The 50% and 25% lines below are applied to err(all learned) as written, except at a tick where the floor exceeds 25% of err(all learned): there the gains are computed as (err(all learned) - err(swap)) / (err(all learned) - floor) and the row is flagged.

## Pre-registered lines (applied separately at err@20, err@50, err@100, per model and per scene set)
Gain fraction g_i = (err(all learned) - err(part i exact)) / err(all learned).
- DOMINANT drift source = the part with the largest g_i, if g_i >= 0.50.
- FAILS (no dominant source) = no single swap has g_i > 0.25.
- Between: reported as intermediate (no dominance, not a fail).
- Tie-break: largest gain wins; if the top two gains differ by under 5 percentage points and both reach 0.50, report a tie and no single winner.
The verdict per model is read on held-out seeds pooled over both families; fresh seeds are the reality check (a verdict is called real only if fresh gives the same outcome at that k). Per-family rows are reported as additional detail.

## Prediction (stated before running)
- exp-e-frozen: no single swap reaches the 0.50 line at err@20 (intermediate or fails), because the learned U-Net partly cancels the learned kernel's deviation, so single swaps of either can lose (negative or small gain), and the largest single gain is the pair term or the kernel+U-Net pair swap. By Shapley share the U-Net term is the largest at err@100 (above 40%) and the pair term is relatively larger at err@5-20; expected dominant-part change from pair term to U-Net between ticks 5 and 40. Consistent with the Oct 21 core-drift note (U-Net drives core drift) but the earlier evidence was a force measurement, not a swap, so this is a hypothesis.
- width8-single-law-control: same ordering, with a larger kernel share than exp-e-frozen (the narrow U-Net has less to correct, the kernel's error is exposed); err(all learned) at tick 20 within 1.5x of exp-e-frozen.
- Unit check passes.

## Compute and seeds
Evaluation only. Seed for any random draw 4738 (bootstrap, torch seed). Estimate 0.2-0.5 GPU-h on one GPU (untimed: CPU smoke of 2 scenes per group, 25 ticks measured 10-20 s per group for exp-e-frozen, 4-5 s for the control; GPU rate not measured). Per-group files; a rerun skips finished groups, SIGTERM exits 143 after the current group. No training state, so no model checkpoint is needed.
