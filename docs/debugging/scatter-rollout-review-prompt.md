# Scatter-field rollout review prompt (hypothesis-free)

Render: `python scripts/render_scatter_rollout_grid.py --npz <A_variant_traj.npz> --out videos/<name>.png`, then dispatch a fresh agent with:

---
Read the image at `<path>`. Each row is a different simulated scene of two bodies attracting each other. Column 1 shows x-y paths: thick faint lines are the reference trajectories, dashed lines are a learned model's rollout of the same scene (dot = start). Column 2 shows the distance between the two bodies vs time step for reference (solid) and model (dashed). Column 3 shows the model's mean position error vs step on a log scale.

Describe plainly what you see: how closely the dashed curves follow the reference, at roughly which step they start to differ, whether differences grow gradually or suddenly, whether any scene behaves differently from the others, and anything else notable. Do not assume a cause. Report in a short paragraph.
---
