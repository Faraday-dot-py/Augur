# Feasibility: z channel (3D) and mass channel for the conservative-contact model (2026-09-25)

Research only; nothing implemented, no compute run. Static read of master (a867b60 lineage, HEAD e3d0a4d), tags `archive/gravity-direction`, `archive/tall-box` (commit 0f73d65), and the `pages-publish` worktree `.claude/worktrees/pages/web/`. Claims marked (untested) are reasoning, not measurement.

## 0. Ground facts (verified in code)

- No physical mass channel exists anywhere. "mass" in older grid code (model/net.py, token_detect.py, losses) means rendered PROB density. Truth is unit mass: bounce.py:130 docstring "force (== acceleration, unit mass)". Grid is 3 channels PROB/VX/VY (bounce.py:30-31). Gravity-test bodies are all mass 1 (project_gravity_generality_test).
- Conservative model = `TokenFreeDynamics(conservative_contact=True)`: only `pair_force` MLP (token_free.py:105), `wall_force` MLP (:106, one MLP shared by all four walls), `gravity` param (:107, shape 2, x10 scale), 8 Verlet substeps (`_conservative` :149-162). 8836 params. Nodes have no state features: no hidden, no velocity input.
- Pair force is `pen * MLP(pen) * 100`, pen = (2r-d)/(2r), applied `+f*rel/dist` to dst per directed edge (:141-146). Both directions are in the edge list so it is antisymmetric = Newton 3rd exactly (tests/test_token_free.py:168 `test_conservative_pair_momentum_exact`).
- cons_pure gravity is a learned free parameter (8.90/~0.07); direction is a parameter, not an input. Gravity-direction branch (`archive/gravity-direction`) replaces it with `gravity_map = Linear(2,2)` on a gdir input; generalizes to held-out angles (err@5/10 same, err@20 2.6-4.1 vs 0.7-2.1, noise-confounded, per its own doc).
- Tall-box branch (0f73d65) already parameterizes wall extents (`height`/`width`, `hx,hy` in contact_accel) - template for a third extent.

## 1. Touchpoints assuming 2D

### Truth sim / GPU generators
- bounce.py: state dict keys x,y,vx,vy (:204-213 `init_balls`, :161-165 `integrate`); `wall_force` 4 walls (:94-108); `ball_pair_forces` dx,dy (:111-126); `compute_forces` (:129-154, gravity on x only :138); `splat_ball/splat_all` 2D disc grid (:45-83). Python-list, CPU, O(m^2). Grid is 2D by construction: a 3D grid (n^3) is not viable for the sizes used, so 3D means dropping the grid, not extending it.
- GPU truth: `scripts/ball_1k_rollout.py:24-46` `forces` (dense N x N; `p[:,0]`,`p[:,1]` indexing, 4 wall branches, gravity on axis 0) and `truth` :49-60; `scripts/energy_probe2.py:26-38` `truth_states`; `energy_probe.py:28` `energy` (KE unit mass, PE = -g*x); `energy_probe2.py:105-122` `spring_pe`/`truth_conservation`; tall-box `scripts/ball_rect.py` `forces/contact_potential/energy_per_ball`. `forces` is nearly dimension-agnostic (`d.norm(-1)`, broadcasting) except wall/gravity lines.
- Init: `model/dataset.py:10,24-35` scenario makers (2 coordinates, `bounce.init_balls`).
- Gravity axis convention: axis 0 ("x is the gravity axis") in truth, energy, eval, render (y-inverted plots `render_ball_1k_compare.py:31-40`).

### Dataset / representation
- `model/token_dataset.py:93-94` state dict x,y,vx,vy; `token_losses.token_state_loss` (:59-92) stacks `["x","y"]`/`["vx","vy"]` and divides by 2 (:78-81 area). `scripts/train_conservative.py:28-51` `sample_scene` (2D coordinates, 4 walls' worth of geometry, angle math), `scene_loss` :54-70 builds the target dict with x,y,vx,vy; `dyn.gravity` fast-lr group :100-102.
- Rasterize/detect pipeline (`token_rasterize.py:4-16`, `token_detect.py`, `token_model.init_tokens`) is 2D grid-bound. The free-rollout path does not use it (`step_free(render=False)`, token_model.py:312-323), so it can stay 2D-only and be asserted off for 3D.

### Model
- `wall_features` / `wall_contact_features` (token_free.py:7-28): 4 walls, x,y. Legacy path only.
- `_radial_mlp` reuse: wall MLP is one net for all walls, so z-walls need no new weights (token_free.py:134-136).
- `contact_accel` (:131-147): `d = stack([x, n-1-x, y, n-1-y])` (4 walls), `torch.stack([f0-f1, f2-f3])` (2 comps), gravity (2,). Needs 6 walls, 3 components, gravity (3,).
- `build_radius_graph` (token_graph.py:4-21): dimension-agnostic (dense N x N, `diff.sum(-1)`). `build_radius_graph_cells` (:24-52): 2D-only: 3x3 offsets (:39), key `cell[:,0]*stride+cell[:,1]` (:35-37). 3D needs 27 offsets and a two-stride key. Neighbour count per probe x3; per-step cost: graph is rebuilt 9 times per step (initial + 8 substeps, :152-156).
- Legacy `_core` attention path: `node_dim = 2 + wall_dim + core_dim`, `edge_dim = 2`, `delta_head Linear(.,4)` (:89-96), `pair_head Linear(5,..,4)` (:126), `pair_invariants` tangent construction (:31-47), `mirror_sym` (:169-175). All hard 2D. Only used by soup B / contact_residual; not needed for the conservative model. Recommend guard `if dims==3 and not conservative_contact: raise`.
- `TokenModel` ctor (token_model.py:162-166) passes n; needs `dims`, box extents; `step_free` :312-323 is generic except the rasterize call.
- Constants: `TokenFreeDynamics.n` is a scalar box side (square); tall-box shows the (height, width) pattern; z needs a depth extent.
- `contain_state` (scripts/realtime_sim.py:21-34): generic on last axis (it works on the columns of `positions`), so it works in 3D as is, with `n` a scalar box. `Sim.spawn`/`clear` hard-code (0,2) shapes and `[x, y]` (:51-67). Speed cap `velocities.norm(dim=1)` generic.

### Losses / training
- `token_state_loss`: `/2` is a hardcoded "mean over 2 coords" normalization (:78-79); with 3D it needs `/dims` (or the loss scale changes by 1.5x, mixing 2D and 3D scenes). Rewrite to take pos/vel tensors instead of the 4-key dict (`train_conservative.py:65-67` builds the dict from columns).
- Energy/momentum metrics: `energy_probe.energy` (unit mass), `eval_energy_fix.py:56-58` (momentum `vel.sum(0)`, energy mean), `energy_probe2.wall_curve/pair_curve` (2D placement, `[[400,400]]`, gravity mismatch `[2*G*DT*steps, 0.0]` :95) all need axis/dims/mass generalization.

### Rendering / video
- `scripts/render_*` scatter `p[:,1], p[:,0]` with inverted y (render_ball_1k_compare.py:31,40-42), `render_model_hires.py` density imshow, `render_gravity_*`, `render_tiled_video.py`, `render_collision.py`: 2D projections. 3D would need a projection choice (orthographic x-y with z as colour/size, or 3D matplotlib; the hi-res streaming renderer `render_model_hires.py` is a 2D density image, so a 3D version would be new). Memory rule ([[feedback_local_render_memory_bounds]]) applies: stream frames.
- `scripts/gravity_1b.py` strip-tiler is 2D x-strips; a 3D 1B run would need slabs (not required for feasibility).

### Web port (read-only look; another agent is editing)
- `web/js/model.js`: pos/vel stride 2 everywhere (`px, vx, a, aWall, aPair` :94-98, step loops :173-198, `containState` :265-291 with `c < 2`); `forPairs` 2D cell hash 3x3 (:105-134) -> 27 cells; `accel` 4 walls/`gx,gy` (:137-163); `startTrace` (:213-250) records 2-vectors, 4 wall entries, `unit`; gravity getters `gx/gy` (:66-67) and `set/get "gravity"` x10. Hidden `H=32` passthrough is unused (README) - can be dropped.
- `web/js/sim.js`: `pos = Float32Array(2*maxBalls)`, spawn (x,y) only (:27-62).
- `web/js/physics.js`: ghost truth 2D: `xs,ys,vxs,vys`, O(m^2) pair loop, gravity +x only, energy (:1-104). Needs zs/vzs/fz, z walls, and mass in `_forces`/`energy`.
- `web/js/scene.js`: already Three.js 3D, but the arena is a flat plane (`PlaneGeometry(100,100)` :77) and Three.js up (y) is used for the channel cube stacks (`yi=0.5`, links :213-218); sim (x, y) maps to Three (z, x) (:213, :267, :318). Real z would collide with "up = activation height". Picking (:307-326) projects onto the plane and picks by (x,y) distance: needs true 3D ray-nearest.
- `web/js/arch.js`: diagram assumes 2-vectors (input token pos/vel 2 cells, offset x/y per edge :96-97, wall columns x4, `plate(-1,3.5,80,34)` layout, acceleration sum). z adds a 3rd cell per vector and 2 more wall columns (layout width and labels change).
- `web/tests/verify.mjs`, `physics.test.mjs`, `scripts/export_web_weights.py`, `dump_web_testvectors.py`, `dump_web_physics_ref.py`: shapes (gravity 2 -> 3), 2D states.

## 2. Architecture changes and conservation

### z channel (3D)
- Change: last-dim generalization (`dims`), 6 walls via `d = cat([p, L - p])` with per-axis extents `L`, gravity `Parameter(zeros(3))` or the gravity-direction `gravity_map = Linear(3,3)`, 27-cell neighbour search. Force law, wall MLP, Verlet unchanged.
- Conservation: preserved. Pair force is central, distance-only (potential), antisymmetric: exact total momentum and a potential exist in any dimension; walls are per-axis one-body potentials (energy bounded by the same argument as tests/test_token_free.py:186). Nothing in the structure depends on dims.
- Zero-shot (untested, high prior): truth contact is isotropic penalty + axis-aligned walls, so a 3D truth is the same functional form as the 2D one the MLPs were fitted to. cons_pure's `pair_force` and `wall_force` weights should apply unchanged with gravity vector set by hand (e.g. (g,0,0) or (0,0,g)). What could break: more simultaneous contacts per ball in 3D at equal number density (3D packing), and pair force at pen > 1 (heavier penetration in dense 3D); MLP was trained on 2D scenes up to ~120 balls at density 0.005-0.02 (train_conservative.py:42-44), not 3D packing.
- Gravity along any axis: two routes. (a) treat as a parameter and set it (cheap, exact for uniform fields). (b) port the gravity-direction `gdir` input to 3D (Linear(3,3)); already shown to generalize 2D angles from 4 cardinal directions; 3D version untested and needs training scenes with gravity along all 3 axes (or just verify with hand-set g).
- 3D neighbour search cost: cell-list is O(N*27*k3) vs O(N*9*k2) per probe; with the same contact radius 1.5 and cell side 1.5. Per-step 9 graph builds. Expect ~2-3x slower per step at equal N (untested); the O(N) claim is unaffected. Dense N x N graph fine to ~few thousand.
- Box: the truth gravity axis is axis 0; z-walls are just axis 2.

### mass channel
Design decisions first (physics of truth), because they set the architecture:
1. Contact force independent of mass (same stiffness, a_i = F_ij / m_i) - the natural minimal extension of bounce.py where "force == acceleration".
2. Wall force same (a = F_wall / m).
3. Gravity: uniform acceleration g (equivalence-principle; mass cancels) vs force m*g (identical dynamics, but energy/momentum bookkeeping differ). Pick the former unless the user wants weight-like behaviour.
4. Optionally: mass-scaled radius (visual + hitbox) or mass as gravitating source strength (a_i = sum_j G m_j f(d), i.e. CentralForceDynamics with a source term; the gravity-test bodies were all m=1).

Model options:
- A. Structural (zero new params): `acc = acc_force / m[:,None]` for wall+pair, gravity untouched; MLPs unchanged. Newton 3rd preserved (F_ij = -F_ji on force, mass only in the update): total momentum sum m*v conserved exactly; the pair potential is unchanged, so energy sum(0.5 m v^2 + U) is the conserved quantity; Verlet stays symplectic. cons_pure weights could be reused as-is (untested; likely, if truth is design 1-2).
- B. Learned: pair MLP input (pen, m_i, m_j) symmetrized (e.g. via m_i+m_j and m_i*m_j, or averaged over the swap), F = pen*MLP*100, a_i = F/m_i. Momentum exact; conservative iff F is a gradient of a potential of d at fixed masses, which any symmetric f(d; m_i,m_j) is. Needed only if truth stiffness depends on mass (e.g. Hertz with radius/mass). More capacity, more risk (OOD mass ratios).
- Mass-dependent gravity learned per-mass: not needed under design 3; if mass-scaled (F=mg) the model must not divide gravity by m (order of ops in `contact_accel`).
- Momentum bookkeeping and eval need mass-weighted quantities: `eval_energy_fix.py` momentum `vel.sum(0)` -> `(m*vel).sum(0)`; `energy()` 0.5 m v^2 - m g x.
- Stability (untested, from Verlet/penalty reasoning): effective contact frequency ~ sqrt(k_eff/m); at m=1 the 8-substep truth is stable at k=400. A ball with m<~0.1 (10x lighter) has 3x higher frequency and will need more substeps in both truth and model (`contact_substeps` is a constructor arg, cheap). Heavy balls penetrate deeper (pen > 1 more often): MLP OOD there. Mass ratio should be bounded (suggest 0.25-4 log-uniform for the first round).
- Legacy attention/GRU path would need mass as a node feature; irrelevant to the conservative model.

### combined
Independent and composable: mass only changes the accel scaling, z only the dimension. Only real coupling is the eval/energy bookkeeping and the truth scene generator.

## 3. Data / training / tests

- Truth generators: (a) torch `forces`/`truth_states` get a 3rd column, 6 walls, per-ball `m` (a = F/m), mass-aware energy. ~2-3 h. bounce.py itself (CPU python) is only needed for grid data; do not extend it, or extend it as a reference implementation with a unit test against the torch version.
- Training scenes (train_conservative.py:28-51): 3D versions of single-ball wall (6 walls), head-on/offset pair (random 3D axis), wide multi-ball boxes with a depth extent; mass draws for pairs and multi-ball (include equal-mass to keep old distribution, plus 1:4 and 4:1 pairs, since pair exchange depends on mass ratio).
- Retrain: cons_pure took ~39 min / 50 epochs on one Polaris GPU (energy-fix-results.md; SBATCH time 04:00:00 in polaris_train_conservative.sh). Estimated (untested): 3D-only ~1-1.5 GPU-h (more pairs per scene, 3D truth generation); mass-only ~0.7-1 GPU-h; combined ~1.5-2 GPU-h. One GPU job at a time on Polaris ([[reference_polaris_job_logistics]]); checkpoint every epoch already exists in this runner (train_conservative.py saves each epoch), so the checkpoint rule is satisfied.
- Cheap zero-shot tests first (all on Polaris GPU, minutes, no training):
  1. z: 3D truth vs cons_pure with gravity set by hand along x / y / z; head-on 3D pair (random axis), 6-wall restitution speeds 3-80, 100-1000 balls x 100-500 steps; compare energy drift vs truth, err@5/10/20, momentum. If MLPs transfer, z is done without retraining.
  2. Mass A: truth with m in {0.25,1,4}, model with `acc/m`, same metrics plus mass-weighted momentum; pair head-on with mass ratio sweep 1:1..1:16 checking post-collision velocities.
- Held-out generalization tests to design: unseen mass ratios (train 0.25-4, test 0.1 and 10); unseen gravity axis (train x,y; test z, oblique); box shapes (aspect ratios via the tall-box pattern); density (3D packing 2x training); ball count 10x; combined heavy-light in dense 3D; seeds 9000/12000 as elsewhere ([[feedback_select_on_heldout_seeds]]); real-retrain validation, not only synthetic probes ([[feedback_validate_synthetic_loss_probes_with_real_retrain]]).
- Risks: dense 3D contact (many simultaneous contacts, pen>1), OOD mass ratios and Verlet stability, energy drift at high speed x heavy mass, gravity mismatch offsets (0.88 momentum offset seen in cons_pure from g 8.90 vs 9; a short low-lr fine-tune was already listed as open), chaos limits err@20 as before, test suite updates (tests/test_token_free.py, test_token_graph.py, test_central_force.py assume 2D; new 3D cell-list equivalence test vs dense graph needed).
- Post-run protocol: fresh unbiased subagent video review needs a 3D render tool + hypothesis-free prompt (docs/debugging/frame-artifact-review-prompt.md pattern); none exists for 3D yet.

## 4. Browser impact
- Physics engine is small, mechanical change (loops to 3 comps, 27 cells, mass division): model.js ~1 day incl. verify vectors regenerated from PyTorch (the existing verify.mjs float64 1-step gate makes this safe).
- Ghost truth (`physics.js`) needs the same truth changes and must stay bit-comparable with the Python truth (`dump_web_physics_ref.py`).
- Rendering is the real work: Three.js is already in, but z collides with the current "y = channel height" encoding (scene.js :213,:267); the token-cube-stack view must be re-laid out (e.g. channel stacks anchored beside the ball, or channel view separate from the arena). Picking must be 3D. Arch diagram grows (3-vectors, 6 wall columns, mass input cell).
- Mass UI: per-ball mass in `sim.js` arrays, spawn slider, ball size/colour by mass, mass-weighted energy plot; weights editor unchanged (gravity shape 3).
- Weights: unchanged file if zero-shot works (gravity 3rd entry 0); export script needs shape change only.
- The other agent is editing web/; expect merge conflicts on model.js/scene.js if 3D work starts before that lands.

## 5. Ratings

| item | engineering (h) | Polaris GPU-h | difficulty | notes |
|---|---|---|---|---|
| z, model+truth+evals (zero-shot path) | 10-14 | 0.5-1 | M | 6 walls, 27-cell list, gravity 3-vec, torch truth 3D, metrics; conservation preserved by construction |
| z, retrain if zero-shot fails | +4-6 | 1-1.5 | M | new 3D scene sampler |
| z, 3D render/video tools | 4-8 | 0 local render | M | projection choice needed |
| z, web port | 25-40 | 0 | L | scene layout collision on the Y axis, picking, arch |
| mass, structural (A) | 5-8 | 0.5 zero-shot (+0.7-1 retrain) | S-M | a=F/m; needs truth mass + bookkeeping |
| mass, learned (B) | +6-10 | 1-1.5 | M | only if truth stiffness depends on mass |
| mass, web port | 8-14 | 0 | M | arrays, slider, physics ghost, energy |
| both, model/eval/training | 25-40 | 2-3 | L | dominated by truth, tests, evals, renderer |
| both incl. web | 60-90 | 2-3 | L | |

## 6. Recommended order
1. z zero-shot probe on Polaris (3D torch truth + a ~40-line `dims` generalization of `contact_accel` and a 3D cell list): decisive, cheap; predicts whether any retrain is needed.
2. Mass structural (A) zero-shot probe with equal-and-unequal masses.
3. Only then combined retrain (single run, both channels), then 3D renderer + unbiased video review, then web port last.

## 7. Decisions needed from the user
- Mass semantics in truth: contact stiffness independent of mass (a=F/m)? gravity as acceleration (mass cancels) or as force m*g? does mass change radius? mass as gravitating source in the N-body case?
- Mass ratio range for training and the held-out test (suggest 0.25-4 train, 0.1/10 test); is a substep increase acceptable?
- 3D box: cube of side n, or per-axis extents (use tall-box `height/width` pattern extended)?
- Which gravity axis convention in 3D (keep axis 0 as "down"), and whether to use the gdir input branch (unmerged) or a hand-set vector.
- Merge the unmerged branches (gravity-direction, tall-box) first? Both touch `contact_accel`; doing z on top of them avoids re-doing the same edit.
- Web: proceed in the pages-publish worktree only after the other agent's edits land; how much of the arch view to keep in 3D.
