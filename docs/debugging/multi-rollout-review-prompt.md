# Multi-law scatter-field rollout review prompt (hypothesis-free)

Render: `python scripts/render_multi_rollout_grid.py --npz <multi_traj.npz> --out videos/multi_rollout_grid_s<seed>.png`, then dispatch a fresh agent with:

---
Read the image at `<path>`. Each column is a different simulated setting. The top four rows each show one scene: x-y paths of six bodies, thick faint lines are the reference trajectories, dashed lines are a learned model's rollout of the same scene. The bottom row shows the learned model's mean position error versus time step on a log scale, one line per scene; scenes flagged BAD have a red border and red error lines.

Describe plainly what you see: how closely the dashed paths follow the reference in each column, at roughly which step they start to differ, whether differences grow gradually or suddenly, which columns behave differently from the others, how many scenes are flagged, and anything else notable. Do not assume a cause. Report in a short paragraph per column.
---
