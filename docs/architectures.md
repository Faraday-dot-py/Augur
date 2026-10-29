# Augur architectures reference

Audience: another LLM or engineer with no prior context. Every architecture below is a learned simulator for a system of interacting particles ("bodies"): given positions and velocities at tick t, predict them at tick t+1, then feed predictions back in (autoregressive rollout). Each section has the same fields. Every number is from a logged experiment; the source file is given in brackets. Where a claim was not re-checked against code or logs it is marked UNVERIFIED. Single-seed results are marked as such.

## Shared definitions

- Tick: one simulation time step (dt 0.1 unless stated).
- Body / token / ball: one particle. Mass is 1 unless stated.
- err@k: mean position error per body after k autoregressive ticks, in simulation length units, against an exact reference simulation. Lower is better.
- Baseline (BL): the reference model a result is compared against (named per section). "1.2x BL" means 20% worse.
- dE@k: relative energy drift at tick k, |E(k) - E(0)| / |E(0)| (or the comparison to truth's own drift when stated). Learned forces are not conservative by construction, so drift is a core failure axis.
- Teacher-forced: evaluation where each step starts from the true state. Free-rollout: each step starts from the model's own previous output.
- CIC: cloud-in-cell, bilinear spreading of a body's quantity onto the 4 nearest grid cells; its transpose (bilinear gather) reads a grid field back at a body.
- Halo: overlap region a tile reads from its neighbours but does not own.
- Chaos floor: the err@k obtained by re-running the exact simulator from a slightly perturbed start; no model can beat it at that horizon.
- Four project requirements, used as the scorecard in each section: (1) Generalization: one approach works across particle systems with different force laws. (2) Tileable: a model trained on a small region simulates a large one by tiling. (3) O(N): cost linear in body count. (4) Interpretability: learned rules are recoverable from the model.

---

## 1. Tiered hierarchy (tier 0 scatter-field tile + upper tiers)

### Summary
A stack of identical-architecture grid models at geometrically increasing spatial scale. Tier 0 is the scatter-field model (section 2) operating on particles inside a tile. Tier k+1 operates on a tile that spans a 2x2 block (2D; 2x2x2 in 3D) of tier-k tiles and sees only aggregated moments, not particles. Forces from all tiers are summed. Analogy: Barnes-Hut or FMM (fast multipole method) with learned kernels and a learned near field.

### State and I/O
- Tier-0 input: particles (position, velocity, mass) inside a tile plus halo; per-cell recurrent hidden field from the previous tick.
- Upper-tier input: a grid of exact merged moments built from child cells (mass, momentum, second-order position moments, velocity dispersion), same G x G grid size at every tier; a parent cell is the 2x2 merge of child cells; a tile spans 2x the linear extent of its child tile.
- Output per tier: a scalar potential phi_k (as a local-expansion polynomial per coarse cell for upper tiers). Force = -grad(phi_k) via matched scatter/gather finite differences, which makes the summed force curl-free by construction when the potential head is used. Mass-weighted mean force is removed per tier over the closed domain (exact total momentum conservation).
- Rollout state: particles + every tier's hidden field (+ optional per-particle hidden vector at tier 0 only).

### Forward pass (one tick)
1. Build moment grids bottom-up: tier 0 scatters particles; each higher tier merges 2x2 child cells exactly.
2. All tiers read the same state at tick t and evaluate independently (order irrelevant; outputs summed). Each tier covers a distance band beyond a multiple of its child tile size (opening-criterion analogue); gravity needs about log2(domain / tile) tiers, top tier a single tile. Short-range laws (LJ) need tier 0 only.
3. Tier 0: CIC deposit -> U-Net -> field -> gather -> integrate (see section 2).
4. Upper tier: grid net on moment channels -> local-expansion coefficients (value, gradient, Hessian) per coarse cell; each child evaluates the Taylor expansion at its position.
5. Sum forces from all tiers, apply momentum fix, integrate (Verlet).
6. Single ownership: a particle takes output only from the tile whose interior contains it; halo outputs are discarded. Input-side halo cells are read-only copies.
7. Tiled tier-0 recipe in use: 32-unit cores + 16-unit halo, tiles exchange the recurrent field in the halo each tick.

### Training
- Sequential and frozen: train tier 0 first on scenes no larger than its reach (tile + halo) with a random smooth external field imposed so it sees bulk acceleration; freeze; train tier 1 against the residual (truth minus frozen lower-tier output); repeat upward.
- Weights shared across tiles within a tier, different between tiers. Same architecture at every tier.
- Best tier-0 recipe so far: matched 60,000-iteration budget, then fine-tuned inside the 4x4 tiling with closed-loop rollouts to 28 ticks and recurrent-field halo exchange (job 4060, commit e44b796, branch tier0-tiled-closed-loop).

### Code
- Tier 0: `scripts/scatter_field.py` (`ScatterField`). Multi-level prototype: `MultiLevelScatterField` in the same file. Tiered training/interfaces (`train_tiered.py`, `tiered_iface.py` with `ZeroLower`, `RangeLower`, `ModelLower`) exist only on branches (tier0-tiled-closed-loop and an integration branch), not on master.
- Design: `docs/theory/heterogeneous-tiered.md` (ten identified weak points, each walked through).

### Verified pros (experiments)
- Matched-budget tier 0 closes the mid-density bias: plain err@20 1.13x/1.12x BL (two seeds), mid-density force-bias slope beta -0.118/-0.052 versus -1.46/-1.78 at 10k iterations, so the earlier bias was undertraining. [supervisor-log 10-26 10:27]
- Upper tiers add little error: only +4 to +10% over an exact external field. [supervisor-log 10-26 10:25]
- Closed-loop tiled tier 0 with halo exchange: tiled err@20 0.82x/0.82x/0.75x BL on held-out seeds 9100/9200/9300 (matched-60k alone was 1.66-1.82x); own-state force error 0.039-0.041 versus BL 0.052. Single training seed. [supervisor-log 10-27 00:50]
- Seams are not the failure: particles changing owner window are 1.2-2.2% of particle-steps and carry 0.6-3% of the energy-work error; relative force error is flat across seam-distance bins (0.079-0.090 vs BL 0.052-0.057). [supervisor-log 10-26 18:25]
- Teacher forcing suffices for tier-1 training: exposure-bias gap 1.056 +- 0.051. [supervisor-log 10-26 19:10]
- RESPA multi-rate (inner substeps for the near term): m=4 gives 1.06x err at 0.54x cost. Sparse tile route: 4.7x faster at a 1024-unit span with 12.5% occupancy. Forces agree to 2.9e-4. [supervisor-log]
- Potential-gradient mesh removes drift in the liquid-argon-style (LJ) setting: |dE/N| <= 0.0084 versus -0.26/-0.29 for the unconstrained mesh. [heterogeneous-tiered.md]
- Gravity half-timestep retrain cut err@20 about 40% and drift 3.5x (0.0149 to 0.0043) at 1.9x training cost. [supervisor-log 10-26 02:30]

### Verified cons (experiments)
- Energy drift stays high. Closed-loop tiled tier 0: dE@100 3.9-4.75x BL (pass line 2.5x); absolute 0.033-0.038. The exact pairwise reference itself drifts 0.016-0.019, so the real gap is about 2x truth, not 4x. [supervisor-log 10-27 00:50, energy-drift audit]
- End-to-end hierarchy with the matched-60k tier 0: 1.5-1.7x BL err@20 even with the exact external field, drift 4-5x. [supervisor-log 10-26 11:30]
- Upper tiers on a frozen tier 0: force error +28-29%, err@50 +4/+3/-1%. Joint fine-tune: energy drift 0.06-0.07 versus 0.01 (6x), nothing gained over sequential. Rollout-in-loop training: no gain, bias check |z| about 6. [supervisor-log 10-25 20:15, 10-26]
- Removing velocity and recurrent inputs from tier 0 (two seeds, 6 h each): err@20 1.56-1.85x, drift 6.0-7.3x, even though teacher-forced force error was within 6-12% of BL. [supervisor-log 10-26 18:10]
- Frozen-base residual corrector (job 4059): err@20 1.60-1.92x, drift 4.7-5.3x; validation-loss ratio 0.97-0.99 (the head barely learned). [supervisor-log 10-26 20:25]
- Loss-term levers all failed: energy-similarity (one seed dE@40 halved but err@20 1.29x; other seed dE@40 1.69x worse), local work-bias, clump-centre, pair-potential-by-separation (dE@40 1.51x, err@20 1.13x), energy-functional tiers (best far-field relative error .66 to .55). [supervisor-log 10-27 01:15, 10-28]
- Free-rollout force error is 1.5-1.6x BL while teacher-forced error is only +6 to +12%: a visited-state effect, cause not located. Located error: mid-range clump over-contraction (r 0.5-4) in the densest third; close-pair softness ruled out (force gain 0.98-1.02). [supervisor-log 10-26 11:40, 10-28 19:49]
- Cost: one arm of the hierarchy ran 2.9x the baseline cost; hierarchy arm lost to the existing far path on far-field error (0.575 vs 0.438/0.443). [supervisor-log 10-25 19:10]
- Tier-0 variants with an extra external-field channel produced velocity blow-up from about tick 35 via that channel. [supervisor-log 10-25 23:30]

### Structural limits (inferred from the evidence above; not independently proven)
- Upper tiers see moments only, so sub-cell / dense-core error cannot be repaired by them; no refinement tier has been built.
- Frozen sequential training prevents upper tiers from correcting tier-0 rollout error; joint training lets tier 0 co-adapt.
- Learned forces carry no conservation guarantee for gravity; the potential-gradient head fixes LJ drift but the gravity free-rollout drift persisted after input removal.

### Requirement scorecard
- Generalization: partial. Gravity only; a mixed six-law retrain failed; per-body mass untrained.
- Tileable: partial. Tiled err@20 beats BL; drift fails.
- O(N): partial. Designed linear (about 4/3 of tier-0 cost in 2D); sparse route measured 4.7x faster at one span; no full scaling curve for the hierarchy.
- Interpretability: untested.

---

## 2. Scatter field (single-tier grid model)

### Summary
Deposit bodies onto a grid, apply a convolutional network to the grid, read per-body accelerations back, integrate. Cost O(N + G^2) per tick (N bodies, G x G grid). No per-body network pass and no all-pairs interaction.

### State and I/O
- Per-body: position (relative to the grid centre), velocity, mass. Mass is not predicted.
- Grid: G x G, extent E, cell size h = E/G. Bodies outside the grid get zero force.
- Grid input channels: scattered [mass, mass*vx, mass*vy] density (scaled by 1/h^2), optional 2 kernel-acceleration channels, optional previous output field (recurrent), optional epsilon channel.
- Grid output: either 2 acceleration channels, or (potential mode) 1 scalar potential whose finite-difference gradient gives acceleration, plus optional hidden channels fed back next tick.

### Forward pass (one tick), as in `ScatterField._force` and `._step`
1. CIC scatter of [m, m*vx, m*vy] onto the grid.
2. Optional kernel term: FFT convolution of the density with a learned radial kernel (small MLP of distance), producing a far-field acceleration; the kernel potential is taken one extra cell beyond each grid side (`kpad="ext"`) to remove an edge artifact.
3. Concatenate density, kernel acceleration, recurrent field.
4. Grid net: `LocalNet` (stack of convolutions) or `UNet` (levels = min(levels, log2(G)-2)); output layer zero-initialised so training starts from zero learned force.
5. Acceleration = field channels, or -grad(phi) by central differences in potential mode; add the kernel acceleration.
6. Bilinear gather at each body.
7. Optional: far multipole term blended in near the grid edge; optional learned near-field pair term over k nearest neighbours (cell-list or adaptive).
8. Momentum fix: subtract the mass-weighted mean acceleration (exact total-momentum conservation).
9. Integrate: v += dt*a; x += dt*(v_old + v_new)/2, or velocity-Verlet when enabled; optional substeps.

### Training
- Pinned gravity checkpoint: 2D gravity, extent 64, unit mass (file name `ms_kp_pot_v_g128`, G = 128, kernel + potential mode). Gate used for new variants: err@5/10/20 <= 1.15x this baseline.
- Tiled inference: 32-unit cores + 16 halo with a coarse global far term.
- Reference simulations: exact direct N-body for small N; no reference exists at 1e9+.

### Code
- `scripts/scatter_field.py` (`ScatterField`, `LocalNet`, `UNet`, `MultiLevelScatterField`). Design notes: `docs/debugging/scatter-field-architecture.md`, `scatter-field-design-decisions.md`, `scatter-baseline.md`.

### Verified pros (experiments)
- Linear scale on one H200 GPU: 1e9 bodies 1.33 s/tick (100 ticks) to 1.39 s/tick (300 ticks), 14-16 GB, 16-bit body state; 1e10 bodies 13.4 s/tick, 87.3 GB peak, 1.34 ns/body/tick. Speed and memory only; no accuracy claim exists at that size. [README, experiment-log job 4068]
- Small-N gravity accuracy at 100,000 bodies versus exact: err@5/10/20/50/100 = 0.073 / 0.266 / 1.327 / 8.31 / 18.94. [experiment-log job 4081]
- Edge artifact fixed with no retrain via `kpad="ext"`: edge-band max|a|/exact 113.5x -> 1.0x; err@5/10/20 unchanged (.0030/.0082/.0242). [experiment-log 2026-10-27]
- Tiling with ghost cells (halo 16) plus a coarse far term passes against one G=256 grid; seam free-rollout showed no seam artifact. [docs/debugging/adaptive-eval-tiling.md, experiment-log seam entry]
- Potential-gradient variant with a 4-fold-symmetrised U-Net: core drift .087 -> .0015/.0022/.0067 per tick (target .007) on three seeds; err@20 0.9x; but dE@1000 1.10-1.16x (gate 1.15x), 3-4x U-Net cost, opt-in only. [push-state notes]
- Solar system (star, 8 planets, moon, belt) with the edge fix: 0/8 planets lost by tick 1000 (median period error 17%); 5/8 lost by tick 2000. [experiment-log job 4068 follow-up]
- Reproducible acceptance gate exists (err@5/10/20 <= 1.15x baseline). [scatter-baseline.md]

### Verified cons (experiments)
- LJ molecular dynamics: both arms failed (jobs 3916/3917). The mesh branch injects 0.2-1% of true force that depends on velocity/history and does net work: NVE E/N drift -0.26/-0.29 versus -0.009 with the mesh switched off. The pair term alone is 1.5-4.2x worse than the dedicated LJ model. [scatter-lj.md on branch lj-scatter-field, not on master]
- Mixed six-law retrain failed on both seeds: gravity err@20 7.89 / 0.106 versus 0.0407 for the pinned checkpoint; predator-prey tick-1 force 8.8-9.4x worse. [experiment-log 2026-10-23]
- Seven scaling/out-of-distribution fixes failed the 15% gate; mass fine-tunes failed in-distribution by 1.2-3.9x. [scatter-review-4-6.md]
- Ghost-cell closure failed energy: dE@1000 1.246x and 1.660x versus the 1.15x gate (kept opt-in). Near-field retrain: both seeds failed (gravity regressed; predator-prey 9x). [scatter notes]
- Contact: one arm over-damped (restitution .49-.57), another exploded after tick 25. Euclid-distance loss worsened energy drift; learned integrator gave at most 2-11% of err@20. [scatter-contact notes, experiment-log]
- Core drift at 1e9: window centre reaches y = 13.7 by tick 300; fp16 lattice stripes after tick about 190. A roughly 1.55x interior force offset against exact is unexplained. [experiment-log job 4081 follow-up]
- Heavy central star: unit-mass assumption; mass ratio 1000:1 gave binary err@2 of 216 (2.3 when time-step-matched). [scatter-ood.md]
- Chaos: err@100 is unusable in black-hole scenes (reference noise floor 1.07 versus model 4.8). [scatter-chaos-floors.md]

### Structural limits (inferred)
- No per-body mass channel: deposit and kernel assume unit mass; fixing it needs new inputs (`docs/theory/mass-conditioned-scatter-field.md`, code and 10 CPU tests written, untrained).
- Grid is a smoothed mesh: sub-cell separations rely on the nearest-neighbour pair term; the densest 10% of bodies hold 12-26% of squared error.
- Bodies beyond the grid feel zero force (tracers at r > 70: 24 -> 149 lost per 400 with the edge fix).
- Mesh branch inputs (velocity, history) with no conservation structure -> net work, momentum not conserved by the mesh alone (1M-body run |P| 27 -> 796).

### Requirement scorecard
- Generalization: fails beyond unit-mass gravity.
- Tileable: passes for gravity (ghost halo + far term); untested for LJ.
- O(N): mesh part linear to 1e10 bodies. The exact kNN pair term was O(N^2); a cell-list fix exists, its scaling result not re-verified here (UNVERIFIED).
- Interpretability: untested (a 3D walkthrough illustrates the forward pass only).

---

## 3. Token-per-ball model

### Summary
Each ball is a persistent token with position, velocity and a learned hidden vector. Neighbours within a fixed radius interact through attention and a summed pair head; the update is Verlet-style. Built for bouncing balls with walls and contact. The final variant (`conservative_contact`) replaces the learned interaction with distance-only conservative pair and wall forces.

### State and I/O
- Token: (x, y, vx, vy, 32-d hidden state). Mass, radius and friction are not inputs (identical balls).
- Neighbour graph: cell-list radius graph (fixed cutoff).
- Output: per-token position and velocity update.

### Forward pass (one tick)
1. Build the radius graph with a cell list.
2. Attention over neighbours plus a summed pair head produce a per-token interaction update (design doc: softmax attention cannot express forces that scale with neighbour count; the summed pair head partly compensates).
3. `conservative_contact` variant: distance-only learned pair force and wall force, learned uniform gravity, 8 velocity-Verlet substeps. Pair momentum is exact by construction.
4. Update position and velocity.

### Training
- Lineage: an observation-loop family (tokens read from a rendered grid each step) was abandoned for free-rollout training. The weight-average of the free-rollout fine-tune lineage ("soup B", checkpoints v30-v37) was the headline model; then `conservative_contact` (`cons_pure_100.pt`).
- Evaluated on held-out seeds; steps 3-20 are the usable benchmark (chaos floor beyond).

### Code
- `model/token_free.py`, `model/token_graph.py`, `model/token_model.py`, `scripts/train_conservative.py`, `scripts/tiled_sim.py`. Specs: `docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md` and the 2026-09-23 free-rollout, track-query and territory-masking specs. Live web demo serves the conservative variant.

### Verified pros (experiments)
- Short-horizon accuracy (4 balls): conservative variant err@5/10/20 = 0.12 / 0.31 / 2.39 versus 0.28 / 0.67 / 3.19 for the weight-average checkpoint. [energy-fix-results.md]
- Energy fixed: wall restitution 1.00-1.04 for speeds 3-80 (previous 0.41-1.65); head-on pair kinetic-energy ratio 0.99-1.00 (previous 0.01-0.43). [energy-fix-results.md]
- Size generalization: trained at <= 200 balls; err@20 1.44 / 1.09 / 1.06 at 1x / 4x / 8x ball count. 50,000 balls (500x training count): err@20/50/100 = 0.94 / 9.3 / 71.5, kinetic energy per ball within 2% of truth, finite. [experiment-log tile_box_test, job 3116]
- Tiling: strip-tiled versus global rollout differs only at fp32 noise (halo >= 12: max 0 to 6e-5); exact translation equivariance in the interior. [experiment-log jobs 2988-2989]
- Scale: 1e9 balls, 300 ticks, 12.75 s/tick (3,825 s), 121 GB peak on one GPU; fitted cost exponent 0.96-0.99 from 1M to 100M balls. [README, bench_scaling_big.json]

### Verified cons (experiments)
- Observation-loop family lost tokens ("dropout") on 3-66 of 48 test seeds depending on loss/architecture variant; the best (track-query attention) still lost 4/48. [experiment-log 2026-09-22/23]
- Gravity at 100-1000 bodies (unmodified model, small training budget so not proof of impossibility): err@5/10/20 0.50 / 0.52 / 2.14 versus constant-velocity 0.073 / 0.27 / 0.95; energy at tick 20 -4882 versus truth -3867. A 10,000-step 1000-body run reached energy +2.59e6 while truth stayed at -10437. 3-8 bodies worked (0.0016 / 0.0039 / 0.0103). [experiment-log jobs 2928, 2932]
- Floor pile collapse: 1000 balls, err@50 about 57, err@100 167, non-finite at step 69; 50,000 balls develop a mottled floor pile after about step 50. [project_ball_1k_rollout]
- Chaos floor: err@20 is about 2.1-2.3 even for the true simulator restarted from the model's initial state; step 20+ is not a usable benchmark. Pair contacts hold about 64% of the error at 11% of token-steps.
- A separately learned central-force model did not transfer to balls (no walls or contact).

### Structural limits (from `docs/paper/sim-specific-assumptions.md`; not independently re-tested)
- Fixed-radius graph gives no long-range force; error grows 22x once a cloud exceeds a cutoff of 100. Unbounded radius is O(N^2).
- Walls, contact radius, disc rasterisation and identical balls are baked into the inputs; the conservative variant also assumes distance-only central forces, so friction, tangential and velocity-dependent forces are unrepresentable.
- Halo must grow with substeps x contact reach + speed x dt, so stiffer simulations need wider halos.

### Requirement scorecard
- Generalization: partial. Holds across ball counts and speeds; fails on gravity at 100-1000 bodies.
- Tileable: passes for contact; fails for long range.
- O(N): passes for contact (12.7 ns/ball/tick); not for unbounded-range forces.
- Interpretability: untested as recovery of rules; the conservative variant exposes a 1D pair-force curve, a wall curve and a learned gravity (8.9-9.1) that can be inspected.

---

## 4. Grid next-frame models (windowed attention, then dilated-conv flow-warp)

### Summary
Dense predictors on a rendered image of the system: the state is a 50 x 50 pixel grid, and the model predicts the next frame. Two generations: a Swin-style windowed-attention network, then a dilated residual convolutional network that predicts a per-pixel flow field and warps the current frame.

### State and I/O
- Input/output: 3 channels per pixel: PROB (mass/occupancy splat), VX, VY. Balls are 1-2 pixels wide.
- History: optional stack of previous frames (tested up to 3).

### Forward pass (flow-warp generation, class `AugurNextFrameModel` in `model/net.py`)
1. 7 dilated residual convolution blocks with dilations 1-2-4-8-4-2-1 (about 33-pixel receptive field), no downsampling.
2. Heads output a per-cell flow field and a bounded correction.
3. `grid_sample` warps the current frame by the flow (advection), then the correction is added.
4. Both heads are zero-initialised so the model starts as the identity (copy last frame).
5. Windowed-attention generation: 2x2 patchify, Swin-style windows with relative position bias (code removed from master at commit f0e709b; recoverable from `f0e709b^`).

### Training
- Loss: mean squared error on next-frame channels with per-region normalisation; scheduled sampling per step; gradient clipping; horizons 3-12.
- Iterations of the flow-warp model were fixes to successive artifacts, each documented in `docs/debugging/findings-*.md`.

### Verified pros (experiments)
- Step-1 mean squared error versus copy-last-frame baseline: flow-warp 0.80x, 0.68x (best), 0.71x; windowed attention never beat 0.90x (typically 1.04-1.35x). [experiment-log 2026-09-21/22]
- Translation-equivariant with no downsampling; the windowed-attention checkerboard disappeared. [experiment-log 2026-09-21]
- Mass was exactly constant in the best flow-warp run; the VX/VY means stayed pinned at +0.0162 / -0.0098 for steps 0-30. [findings docs]

### Verified cons (experiments)
- Windowed attention: checkerboard visible by rollout step 1-2, dominant by 16-20; longer horizon only delayed it; 200 epochs was worse (1.18x vs 1.04x); richer history left onset unchanged (about step 8) and regressed step-1 to 1.29x.
- Flow-warp early version: period-2 oscillation (Jacobian eigenvalue about -1.15 to -1.2); border flow 2.20x interior (fixed to 0.72x); later versions showed velocity drift (0.016 -> 1.43 by step 12), tile-like quilting from mass renormalisation, and a flat two-tone band by step 7-13.
- Peak decay, never solved: a single ball under bilinear warp keeps 13% of its peak by step 10 and under 1% by step 16; bicubic keeps 0.24 at step 10. The MSE loss prefers a blurred blob to a sharp ball shifted by 2 pixels.
- A global peak-preserving loss held peak height (0.54-0.85) but missed most balls (recall < 0.5 at step 1). A tile-level mass loss passed a synthetic probe (win rate 0.753) but failed real retraining (diagonal fan by step 8-16). A 3-frame history model showed moire from step 8 and 1.5-3x ball overcount. Focal reweighting was inert.
- Honest horizon (recall >= 0.5): 6 steps at 20-50 balls, 4 at 125 balls, 1-2 at 250 balls. [findings-honest-horizon.md]

### Structural limits (inferred from the failures above)
- Pointwise regression on a dense field of a chaotic multi-ball system rewards blur; fixing blur produced "give up" behaviour and vice versa.
- Resampling every autoregressive step diffuses mass; each stopgap (renormalisation, thresholds, bicubic) created the next artifact.
- No per-ball identity or state; velocity is only a field value. Cost scales with grid area rather than body count; ball size is tied to pixel size. Receptive field about 33 pixels gives no long-range term.

### Requirement scorecard
- Generalization: partial. Grid-size agnostic by design; density generalization failed (horizon falls from 6 to 1-2 steps at 250 balls).
- Tileable: partial. Local convolutions, but global mass renormalisation and VX/VY recentering are frame-wide, not tile-local; tiling untested.
- O(N): fails or untested (cost is O(grid area)).
- Interpretability: untested.
