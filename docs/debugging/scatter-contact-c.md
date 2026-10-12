# Scatter-field obstacle option C: geometry as extra grid-net input channels (2026-10-12)

Branch `worktree-agent-a98bdf87385c8eb37` (on master d1607f9). Option C of obstacle-contact-force-options.md; B (SDF gather) is dropped by the user, B2/A2 are reference rows only. Code: `scripts/scatter_contact.py` (`obstacle="chan"`), `scripts/scatter_contact_rand.py` (randomised scenes), hooks in `scatter_contact_scenes.py`, tests in `scatter_contact_test.py`. Protocol, ball counts/radii, seeds, metrics as scatter-contact.md (eval seeds 9100/9200/9300 x 24 scenes, 100 ticks, train seed 4738, gravity 9).

## What C is
- Per-scene raster on the 128^2 grid (h=.5): 4 input channels `[occupancy, clip(sdf/8,-1,1), normal x, normal y]` (from the existing `rasterize`). They enter the U-Net's first conv through extra weights `net.gin` (zero-init, weights and bias), i.e. conv([x, geo]) = conv_x(x) + gin(geo). Plain E checkpoints load unchanged.
- No per-ball gather, no wall MLP, no reflection/projection. The wall force is whatever the existing potential head (a = -grad phi) outputs because of the geometry channels. Ball-ball: the existing pair head `cmlp` and `gvec/gscale`, unchanged.
- momfix (mean-force removal) would cancel any external wall impulse (a single ball would feel zero). So momfix is applied to the geometry-free force only: a = a0 - mean(a0) + FMAX*tanh((a_geo - a0)/FMAX), a0 = same net with no geometry channels (second pass), a_geo = with channels. FMAX=40 bound (as B2/A2). At init a_geo = a0, so untrained C == Exp E bitwise (unit-tested). Cost: two net passes per force eval.
- Not available to C by construction: ball radius (radius-blind like A; all new families use R=.75), sub-cell geometry (h=.5 vs R=.75), a dissipative force (potential head is conservative; velocity only enters through the scattered momentum channels).
- Training: 4000 `c_train` scenes (80 ticks truth), 8000 it, batch 16, lr 1e-4, loss cap 100, k 4->20, init from E (`checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt`). Training distribution is the wide randomised one (user requirement), so C differs from B2/A2 in architecture AND training scenes; B2/A2 were trained on 4 narrow families.

## Scene generator (`scatter_contact_rand.py`)
Container: box (half-sizes 4.5-8, offset +-4), disc (r 5-8), or open (tilted floor slab, no walls). 0-4 obstacles: circle, rotated rectangle, regular polygon (3-8), irregular convex polygon (hull), circumradius s in [.8,3.2], centre anywhere in the container with y <= 3.5 (may touch walls/each other). Free space reachability: largest connected component of ball-admissible cells must be >= 35% of the empty container (so no sealed-off scenes); balls are placed only in it. 2-15 balls.

| family | what is held out |
|---|---|
| c_train / c_id | in-distribution (c_id = fresh seeds) |
| c_big | s in [3.8,5.2] (> train max 3.2), larger containers |
| c_small | s in [.3,.6] (< train min .8; smaller than the ball) |
| c_pos | obstacle centres y in [4.5,10] (train: y <= 3.5) |
| c_count | 6-8 obstacles (train max 4) |
| c_nonconvex | L, 4-6-point star, crescent (never convex-only) |
| c_cont | triangle / hexagon / rotated-rectangle container (train: box, disc, open) |

Also evaluated: the six scatter-contact.md families (box15, box_var, post_circle, post_rect, disc, post_poly). For C these are mostly inside the training law (disc containers, polygons occur in c_train), so they are comparison rows, not held-out for C. Extra metrics added: err_x/err_y (gravity is along +x), energy change per ball (0.5|v|^2 - 9x) at ticks 20/100 model vs truth, momentum error |sum(Vm-Vt)|/n.

## Predictions (written before any C result)
1. Untrained C == E bitwise: unit test.
2. c_id: err@20 4.5-6.5 (E zero-shot ~7, B2 ~4.5-5), tunnel@100 < 5% (E ~95%). The conservative potential head cannot dissipate, so KE ratio@100 will be far above B2's .003-.009 (0.2-1.0), wall restitution 0.7-1.0 where it holds the ball at all.
3. Held-out: c_big and c_pos within 1.5x c_id err@20 (the net is convolutional and translation-equivariant up to 2^4 cells); c_small worst on tunnelling (>10%, obstacle < ball and ~1 cell); c_count and c_nonconvex err@20 1.3-2x c_id, tunnel 5-15%; c_cont between.
4. Risk: heads blew up at k~20-21 earlier (A2 after tick 25 on A); the FMAX bound should prevent divergence, but late-tick drift of balls out of the grid or into solids is possible.

## Results
(see below once run)
