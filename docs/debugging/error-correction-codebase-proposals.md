# Long-rollout error correction for the scatter field (Exp E): codebase-grounded proposals

Research only, 2026-10-12, uncommitted. Nothing run. Every claim marked UNTESTED is an inference from reading code or prior docs. Model = Exp E `E_ms_kp_pot_v_g128` (60769 it). Line refs are `scripts/scatter_field.py` (sf) and `scripts/train_scatter_field.py` (tsf) on master d1607f9 unless stated. Contact code is on branch `worktree-agent-a98bdf87385c8eb37` (`.claude/worktrees/agent-a98bdf87385c8eb37/scripts/scatter_contact.py`).

Built on (not repeated): `scatter-correction-research.md` (mechanism table 1-13, D1-D3 oracle diagnostics, guard band, L2 frozen corrector, interlaced deposit, far-quad/thermostat/parareal/Kalman rejected), `scatter-closed-loop-relabel.md` (negative: relabel err@20 .0461 vs control .0298, base .0383; any fine-tune breaks nr100 dE@1000 .017 -> .18), `scatter-contact-c.md` (a0 runaway 17 -> 1150 over ticks 21-24, C only). Rules applied: no hard-coded conservation, one fix per run, same-schedule control for any shared-weight fine-tune, O(N) and tileable, select on fresh seeds.

## 0. Where error enters (code map)

| # | entry point | file:line | note |
|---|---|---|---|
| a | Recurrent state is unbounded and always cold at train window start | sf:453-454 (`x = cat([x, field])`), sf:470 (raw net output becomes next `field`), tsf:166 (`field = model.init_field`), tsf:161 (window start t0 in [0,10] of 30-tick data) | The net never sees a warm field older than k_end=20 ticks in the main train loop. Exp E zero-shot is finite to 1e4 ticks (research mech 8), so this bites fine-tunes (C, A2, B first run), not E itself. Carried-vs-reset field in relabel ablation was within noise on accuracy (.0461 vs .0440), so the state is not the accuracy lever. |
| b | Verlet carries a second state, `a_prev` | sf:501-507, tsf:171 (a_prev None at window start) | Same cold-start issue; in C the geometry pass writes the field and the a0 pass reads it (`scatter_contact.py:351-352` on branch), a state mismatch that could amplify (UNTESTED). |
| c | Grid edge | sf:138/152 (corner weights zeroed outside, not renormalised), sf:405-409 (`neg_grad` zero-pads phi), sf:378-382 + 477-479 (cosine taper to far monopole, opt-in `far`, off by default sf:69), tsf scenes centred (`to_tensors` sf:609) | Spike +4960 at x 30.5 vs exact ~120, exists in kernel gradient alone (2074) and U-Net alone (2460), moves with the edge. Zero-pad in `neg_grad` only touches the last cell (x > 31.5), so it cannot explain 29-31 (derived from code, UNTESTED). Hypothesis H-edge: edge cells are almost never populated in training (scenes centred, clusters sigma 5-10, `scatter_bh.py` clips to +-29), so the learned edge response is untrained, not mis-padded. UNTESTED. |
| d | Mesh phase error | sf:124-127 `_corners`, sf:157-170 scatter, sf:172-181 gather, sf:204-206 FFT convolve | Static function of sub-cell position (research mech 7); constant for slowly moving bodies, so it acts as a coherent force bias and accumulates linearly in orbit energy rather than averaging out (UNTESTED). |
| e | Velocity-dependent force | sf:161 (scatter channels `m*vx`, `m*vy`), sf:90 (`cin`) | Truth gravity is v-independent; measured v-dependence 3% (r=1) to 100% (r=8) of true force (research mech 3). Source of torque (dL/L@100 .30-.67) and plausible heating. |
| f | Mean-force removal | sf:482-484 | Fine, but redistributes any beyond-grid / edge imbalance uniformly (research mech 2). |
| g | Energy heating nr100 dE@1000 .017 (truth 4e-3), bh300 .48, nr100 1e4 .39 | measured by `scatter_regress.py:156-193` `conservation_rows` | Systematic, mostly bias, not chaos (research mech 1). |
| h | Contact: restitution 1.13@8, .91@15, post 2.3-2.7 | `scatter_contact.py:342-358` (branch): `base + FMAX*tanh((a_geo - a0)/FMAX)`, FMAX=40 (`:26`) | Only the geometry difference is bounded; a0 (E field net + prior) is not. |

## 1. Proposals (new or sharpened)

Each: mechanism, hook, per-tick cost, O(N)/tileable, test, risk. "Control" = same-schedule truth-window fine-tune (relabel doc: improves accuracy 15-35% but breaks dE rows), so any shared-weight training proposal is compared to the control and to the base, and dE rows are reported as failing for any fine-tune unless they pass.

### P1. Grid-origin jitter (inference only, frozen E)
- Mechanism: shift the mesh origin by a per-tick pseudo-random sub-cell offset delta in [0,h)^2 (seed 4738). Scatter and gather use the same shifted coordinates, so the physics is unchanged; the phase-dependent mesh error (d) becomes a per-tick random draw instead of a coherent bias for slow bodies. Pair term (sf:241-376) uses real positions and is untouched.
- Hook: add `+ self._jit` (buffer, shape (1,1,2) in cell units) inside `_corners`/`_corners4` (sf:124-127, 142-155) before `floor`; `kernel_acc` unchanged (translation-invariant convolution). Update `_jit` in place each step so CUDA-graph replay (sf:421-443) stays valid.
- Cost: ~0 (one add). O(N). Tileable (each tile may jitter independently if tiles share the global offset; use one global offset per tick).
- Test: eval only, opt flag `jit` through `apply_opt` (sf:1076). `scatter_regress.py run` + `compare` vs `results/baseline_v2_regress.json`, rows: err@5/10/20, nr100/bh300 dE@1000, dL, bound/virial. Fresh seeds not used in selection (check `results/` for unused seeds; do not reuse 9000/12000/4738 for the headline). Prediction (UNTESTED): err@20 within +-5% (noise), dE@1000 changes by more than 20% in one direction; sign is genuinely unknown because random phase also adds diffusion (random-walk heating ~ sqrt(t) vs coherent ~ t).
- Risk: could raise short-horizon noise; Verlet `a_prev` was computed on a different phase, giving a small non-symplectic mismatch. Cheapest experiment on the list, so run first.

### P2. Shift-equivariance probe, then shift-consistency loss
- Mechanism: true force is translation-invariant (any free-space sim; not gravity-specific). The model is exactly equivariant only to cell-multiples times the U-Net stride (2^levels = 32 cells for g128), so tile seams and sub-cell phase both appear as `a(x + t) != a(x)`. Loss: `|a(x + t) - a(x)|` over tokens, t = random sub-cell plus random integer cell shift (including non-multiples of 32), scene kept >=8 cells inside the grid. Truth-free, works at any horizon, a soft penalty not a hard constraint (non-translation-invariant sims, e.g. walls, are excluded by scene selection or flag).
- Hook: probe: new function beside `self_force_probe` (sf:708-721), force-only on cold field. Loss: tsf after line 165, one extra `model.force` call (no unroll) on shifted tokens, added to `loss` at tsf:175 with weight lam_s; mind `graphtrain` (tsf:139): use eager `model.force` for the extra call.
- Cost: inference 0. Training +1 force fwd/bwd per iteration (~+30-50%, UNTESTED). Directly targets core requirement 2 (tileable).
- Test: (1) probe on frozen E first: report median `|a(x+t)-a(x)|/|a|` for t in {0.25h, 0.5h, 1, 5, 17.3, 32 cells}; if < 2% everywhere the proposal is dead and costs nothing. (2) Warm-start fine-tune at lr 5e-5 (relabel doc's open item), 2000 it, seed 4738, vs same-schedule control; gate: `scatter_regress.py compare`, plus new rows: probe above and a tiling row (larger-than-train domain from `scatter_scaling.py`).
- Risk: lowers equivariance error without lowering rollout error (precedent: pp8/prior halved force error and was 73% worse in rollout); fine-tune drift on dE rows.

### P3. Edge-exposure augmentation (tests H-edge)
- Mechanism: random global translation of training scenes so edge cells carry mass. Truth unaffected (exact sim is translation-invariant). If H-edge holds the spike should shrink without any padding change.
- Hook: tsf:162-165 after gathering `Pw, Vw`: `Pw += t`, `pos` accordingly, t uniform so max |x| of the scene stays < 30 (bodies outside the grid get zero model force by sf:138, so keep scenes inside, otherwise the loss punishes the unfixable far field). Same offset for the whole window.
- Cost: inference 0; training free. O(N), tileable-positive.
- Test: first an eval-only discriminator, no training: run the square-wall probe (jobs 3444/3445) on E with `neg_grad` replicate-padded (sf:405-409) and with a 4-cell zero guard (pad the scattered input, run net, crop; needs padding to a multiple of 32, so G'=160, +56% net cost, UNTESTED). If the spike at x 29-31 survives both, padding is exonerated and the augmentation run is justified. Then one warm-start run vs control. Metric: `a(x)` at 28-34 vs exact, reflection rate (3-4.5% vs truth .15%), tracers at r>70 (32/400 -> 0).
- Risk: net edge handling is also partly shared with interior features; fine-tune drift; edge response may need `far` handoff beyond x=32 anyway (existing opt-in).

### P4. Galilean-boost augmentation (kills spurious velocity dependence)
- Mechanism: for gravity, adding a constant velocity u to every body (positions advance by u t) leaves truth forces unchanged, so training on boosted windows pushes the learned force to ignore the scene's bulk drift, and penalises the v-input channels (sf:161) wherever they carry no signal. Targets mech 3 (dL) and plausibly heating (e). Soft: data augmentation, not an architecture constraint; sims that are not Galilean invariant (drag against a medium, walls, relativistic regime beyond |u| << c) must use a per-sim flag or a small |u| range.
- Hook: tsf:163-165, `Vw += u`, `Pw += u * dt * (tt - t0)`. Positions must stay inside the grid, so |u| t_max < slack; clip u accordingly (clusters here have speed ~1-3).
- Cost: 0 inference and 0 training. O(N), tileable.
- Test: new probe, force vs boost on frozen E (the 3-100% v-dependence figures exist in the conservation docs; the probe script is not located by me, so verify). One warm-start run vs control. Rows: dL/L@100, dE@1000, err@20. Prediction (UNTESTED): dL down >30%, accuracy within +-10%.
- Risk: only shifts the *bulk*; local dispersion (the m|v|^2 blind spot in research L1) is not addressed. Conflicts with generalisation claim (requirement 1) if the sim is not Galilean; state as an opt-in training prior.

### P5. Time-reversal consistency loss (truth-free long-horizon dissipation/heating term)
- Mechanism: roll k ticks, flip velocities, roll k ticks, penalise return error `|x_back - x_0|`. A reversible symplectic integrator on a conservative force returns exactly (up to chaos amplification, controlled by small k); systematic heating, field memory and velocity-dependent torque show up as irreversibility. Needs no exact sim, so can run at k = 40-100 where truth targets are meaningless (chaos floor), unlike truth-window fine-tunes. Reversal with a cold field at the flip confounds; carry the field through the flip (field is not time-reversal-aware, so this itself measures the recurrent-state asymmetry).
- Hook: new function modelled on `long_penalty` (tsf:108-124), called at tsf:178-179.
- Cost: training only, +2k forward ticks without truth. O(N). Not valid for dissipative sims (contact); flag.
- Test: diagnostic first, eval only: measure return error vs k on frozen E, nr100/bh300/expB, both field-carried and cold. If return error at k=20 is below the err@20 floor the loss has no gradient (kill). Then one warm-start run at lam chosen once (not swept) vs control; dE rows reported as in the relabel doc.
- Risk: weaker signal than `long_penalty`'s direct dE (which already cost +29-76% accuracy at lam 3); can conflict with legitimate irreversibility in close encounters.

### P6. Recurrent-state contraction: stability loss and a zero-init forget gate
- Mechanism: the tick map `field_{t+1} = net([scatter, kernel acc, field_t])` has no bound on gain from `field_t`. Runaway in C (x2-6 per tick, ticks 21-24) is positive feedback. Two soft remedies: (i) stability loss: with `delta` = small random perturbation of the field input (norm 1e-2 x field rms), penalise `relu(|net(x, f + delta) - net(x, f)| / |delta| - 1)^2`, a one-sample Lyapunov-gain estimate per unroll tick; (ii) a learned gate `field_in = field * (1 - relu(tanh(conv1x1(x))))`, conv zero-init, so it is exactly identity at init (bit-equal to E) and can only attenuate. Neither hard-codes physics; the gate is architecture, the loss is soft.
- Hook: gate at sf:453-454 (new 1x1 conv, `cin` params only, a few hundred weights); stability loss at tsf:170-174, one extra net pass per unroll step with perturbed `field` (eager; the graphed callables `fns[s]` hold static buffers so reuse needs care, UNTESTED).
- Cost: gate ~0 inference (1x1 conv on G^2). Loss +1 net pass per unroll tick in training (~+50%). O(N) (grid-sized), tileable.
- Test: needs the reproducer: C checkpoint on the branch (blow-up tick 21-24). Run 1: freeze everything except the gate, train with the stability loss only (no truth change), evaluate C past tick 25 (first non-finite tick, max |a0| per tick, err@100, `eval_scatter_cl.py` on the relabel branch already logs field rms and |a| max). Run 2 on E: gate-only vs identity, exact equality expected at init, then rows. Because base weights are frozen in run 1, the fine-tune-drift confound (mech 13) is avoided.
- Risk: the relabel ablation says state is not the accuracy lever, so benefit may be confined to fine-tuned/contact models; a gate that attenuates always would lose the memory the recurrent net uses (E without recurrence was worse in Exp A/B, check `experiment-log.md`).

### P7. Soft saturation of the grid-net acceleration (data-scaled, not a hand FMAX)
- Mechanism: C bounds only the geometry difference with FMAX=40; the grid net a0 is unbounded. Replace `acc_net` (sf:471) by `s * tanh(acc_net / s)` with `s = kappa * max|ak| + floor`, where `ak` is the learned-kernel acceleration already computed at sf:451 (a per-scene scale from the model's own mesh solution, not a gravity formula). Near identity when |acc_net| << s.
- Hook: sf:471-473. Apply to the net's potential-gradient term only, not the pair term (sf:480-481) and not the kernel term.
- Cost: one amax + tanh on a G^2 map, ~0. O(N), tileable (per-tile s differs, so use a global max over tiles, small reduction).
- Test (no retraining): zero-shot on the C and A2 checkpoints (branch): first non-finite tick, max |a| growth/tick past 20, err@20 unchanged (identity region), then on E: fraction of cells where |acc_net| > 0.5 s must be ~0 on harness rows so E is bit-near-equal (UNTESTED, measure kappa from that fraction, not from sweeps). If zero-shot cures C's blow-up this is a diagnostic win; if not, runaway sits in the state path (go to P6).
- Risk: "a hand clamp masks, does not fix" (research); bounded a with unbounded field feedback can still blow up through the state. Close encounters inside `pp` radius are not affected.

### P8. Restitution / wall-energy soft loss for contact (branch)
- Mechanism: C gains energy on posts (2.3-2.7) and loses it at v=15 (0.91). Add a loss on per-bounce kinetic-energy ratio from truth windows that contain a wall hit: `(KE_after/KE_before - e_truth)^2` per ball, plus the existing trajectory loss. Soft, uses truth from `contact_truth` (`scatter_contact.py:156`), no reflection formula in the model.
- Hook: contact training loop in the branch (loss assembly near the unroll; trainer file not read by me, locate before use).
- Cost: training only. Gate: restitution table and tunnel rates in `scatter-contact-c-table.md`, in-horizon err@20 not worse than C. UNTESTED.
- Risk: only addresses in-horizon bias; the beyond-horizon failure is P6/P7. Do after them.

### P9. State-noise / pushforward (kept low: partly covered)
- Mechanism: Gaussian noise on `pos, vel` at the window start (tsf:165), sigma ~ one-step error (3e-3 per relabel doc) so training sees its own drift. The relabel run already used sigma 1e-3 together with other changes and lost, so a clean single-change run is only informative if P1-P6 don't land. Cost 0. Test: same as P2. Risk: reduces accuracy (MP-PDE finding in research doc).

## 2. Ranking

| rank | proposal | hook | cost / tick | test plan (short) | risk |
|---|---|---|---|---|---|
| 1 | P1 grid-origin jitter | sf:124-127, 142-155 (+ sf:1076 flag) | ~0 | frozen E, harness + cons rows, fresh seeds, one run | sign of dE effect unknown; may add noise |
| 2 | P2 shift-equivariance probe -> loss | new probe near sf:708; loss at tsf:175 | 0 inference; +30-50% train | probe first (kill if <2%); then fine-tune vs control | force-vs-rollout mismatch; dE drift |
| 3 | P3 edge-exposure augmentation (after padding discriminators) | tsf:162-165; sf:405-409 | 0 | eval-only padding/guard probes, then one fine-tune | H-edge wrong; drift |
| 4 | P6 recurrent contraction (gate + stability loss) | sf:453-454; tsf:170-174 | ~0 gate; +50% train | frozen-base gate-only run on C ckpt, then E | benefit only fine-tuned/contact models |
| 5 | P7 data-scaled acc saturation | sf:471-473 | ~0 | zero-shot on C/A2 ckpts, E near-identity check | masks instead of fixes |
| 6 | P4 Galilean boost augmentation | tsf:163-165 | 0 | v-dependence probe, one fine-tune vs control | requirement 1 (non-Galilean sims); only bulk |
| 7 | P5 time-reversal loss | new fn like tsf:108-124 | train only | return-error diagnostic first | weak signal; contact incompatible |
| 8 | P8 restitution soft loss | branch scatter_contact.py:342-358 + trainer | train only | restitution/tunnel table | in-horizon only |
| 9 | P9 state noise | tsf:165 | 0 | single-change fine-tune | accuracy loss |

Ordering logic: ranks 1-2 are free to falsify (no training), rank 3 has a free discriminator, rank 4-5 have a known reproducer (C ckpt), 6-9 need fine-tunes that the control shows damage dE.

## 3. Common test protocol

- Baseline frozen: E ckpt and `results/baseline_v2_regress.json`; train seed 4738 (tsf:291); headline on fresh held-out seeds, not 9000/12000 used in tsf `EVAL_SEEDS` (tsf:68) and previous selections (feedback_select_on_heldout_seeds).
- One change per run; control for every shared-weight change; fine-tune lr 5e-5 (the relabel doc's open item, not swept).
- Gate: `scatter_regress.py compare`; count `nr100@1000 dE` and `bh300 dE` rows as failing for any fine-tune (relabel doc's open rule).
- Horizon rows: run every candidate to 5x its training horizon; log max |a| per tick and first non-finite tick.
- Judge by channel (feedback_diagnose_rollout_with_channel_breakdown) and against the chaos floors (research table 1a); bh300/cluster at tick 100+ and contact beyond tick ~20 are at the floor and not correctable.
- After any rollout/video output: fresh hypothesis-free subagent review (standing rule).
- Checkpoints: any run > a few hours must save full state (`tsf:192-194` saves model+opt every `log_every`; the C branch trainer must be checked before launch).

## 4. Open items (not decided by me)

- Whether P4/P5 augmentations (assume Galilean/time-reversal) count as acceptable soft priors under requirement 1; recommend per-sim opt-in flags.
- Whether to run P1 before D1-D3 oracle diagnostics (it is cheaper; independent).
- I did not locate the velocity-dependence probe script or the contact trainer; verify before P4/P8.
