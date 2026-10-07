# Obstacle / contact force options for the scatter-field model (design notes, untested)

Context: scatter-scenes agent (#7) found bounce needs wall + contact terms; `BounceScatterField` (scripts/scatter_bounce.py) computes per-ball distance to each box wall, which does not generalise to arbitrary shapes. Goal: arbitrary solid shapes at arbitrary positions, ball-ball contact through the same system. Nothing here is built or measured. Date: 2026-10-06.

## Common setup
- Obstacles are SOLID: rasterize filled occupancy and a signed distance field (SDF, negative inside) on the existing 128^2 grid (h=.5), computed once per static scene. Moving obstacles out of scope for now. Restitution channel for obstacles: later.
- Wall force is an external force: add it AFTER `momfix` (scatter_field.py:375) or momfix cancels it.
- Do not hard-code conservation/centrality (standing principle); the contact force is learned, geometry is an input.
- Tunnelling risk: ball displacement per tick (dt .15, v=15 -> 2.25 units) can exceed thin-solid thickness.
- Ball-ball contact is separate: radius-aware pair term, feed overlap r-(Ri+Rj) to `ppmlp`-style MLP via the cell-list kNN; needs per-body radius state.

## Options for turning the obstacle field into a force
A. Learned kernel convolution of occupancy: phi_wall = K_wall * occupancy (FFT path), force = -grad phi_wall.
   + reuses kernel code, O(grid), smooth, any shape, no distance transform.
   - conservative potential: elastic only, no damping/friction; force is a sum over blurred solid not a true normal; h=.5 resolution limit; thickness dependence near kernel range.
B. SDF gather + small MLP (like ppmlp): sample d and grad d (normal n) at each ball; MLP(d, v.n, v.t, ...) -> normal + tangential force.
   + exact sub-cell depth and normal; sees velocity so restitution/friction learned; cheap per ball; restitution channel later = extra gathered material channel input.
   - sees only nearest surface (concave corners, thin features ambiguous; n bad on medial axis; mitigate with 3-5 SDF stencil samples); needs distance transform per static scene.
C. Occupancy/SDF as extra U-Net input channels; force via existing potential/acc head.
   + most general, no new machinery, sees wider context.
   - mesh-resolution (h=.5) vs sub-cell contact; velocity only as blurred scattered momentum; potential head cannot dissipate; needs retrain with zero-init new channels; hardest to learn precisely.
D. B + C together. Most capable; two mechanisms to validate (test separately first).
E. Post-step projection along n + learned restitution reflection.
   + exact anti-clipping.
   - hard-coded geometric rule, not a learned force; keep only as diagnostic/backstop.

## Decision (2026-10-06)
Build B as main path (with stencil input, wall force after momfix, material channel slot reserved), with A as the ablation. Hold C and D until B/A show a gap. E only as backstop. Test individually vs Exp E zero-shot and BounceScatterField on box scenes + at least one non-box shape.
