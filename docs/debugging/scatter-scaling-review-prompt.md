# Scatter-field large-N rollout review prompt (hypothesis-free)

Render: `python scripts/render_scaling_grid.py --npz results/scatter_scaling_dump_<N>.npz --out videos/scatter_scaling_n<N>.png`, then dispatch a fresh agent with:

---
Read the image at `<path>` (and, if given, the second image of the same run). It shows one simulated scene of many bodies attracting each other. Top row: a density map of the reference simulation at six time steps (left to right, increasing time; brighter = more bodies per pixel on a log scale). Middle row: the same for a learned model's rollout from the same initial state. Bottom row: a histogram of per-body position error (log10, simulation length units) at each step.

Describe plainly what you see: how the reference evolves, how the model evolves, at roughly which step they start to look different and in what way (shape, size, extent, brightness, symmetry, bodies leaving the frame), whether the error distribution is broad or concentrated and how it shifts, and anything else notable. Do not assume a cause. Report in a short paragraph.
---
