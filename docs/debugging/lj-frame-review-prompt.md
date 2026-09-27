# LJ frame review prompt (reusable)

Regenerate the sheet first:

```
PYTHONPATH=. python3 scripts/render_lj_grid.py /tmp/lj_grid.png 0,10,50,150,400,800 gas=results/lj_gas.npz liquid=results/lj_liquid.npz ...
```

Then dispatch a fresh agent (no context about the model, physics or known issues) with:

---

Read the image at `<path>`. It is a grid of frames from a 2D particle simulation on a periodic square box. Each row is a
different run, each column a different frame index (left to right = increasing time). Every dot is one particle drawn
at its true size, coloured by its speed (blue slow, red fast; one fixed colour scale per row).

Describe, in plain terms, what you see:
- What does each row look like at the first frame vs the last? How does the arrangement of particles change over time?
- Is there any ordered/regular structure (lattice, rows, grains), clustering, empty regions, or uniform gas-like spread?
  When does it appear or disappear?
- Do speeds (colours) look uniform, or are some regions/particles faster or slower than others?
- Do any rows look physically odd (particles overlapping, streaks, tearing, sudden changes between adjacent frames, particles
  piled on the box boundary or at a single spot)?
- Do rows differ from each other in ways beyond the obvious?

Do not assume a cause; just describe what is visible, in a paragraph or two per row.

---
