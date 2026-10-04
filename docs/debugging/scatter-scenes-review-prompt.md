# Scatter-field scenes review prompt (hypothesis-free)

Render: `python scripts/render_scatter_scenes.py --scene <bounce|bh|orbit|globular> --tag <tier> [--video]`, then dispatch a fresh agent (no other context) with the text below and the PNG paths for ONE scene+tier.

---
Read the image(s) at `<paths>`. They compare a simulated reference ("truth", blue / thick lines) with a learned model's rollout of the same scene ("model", red outlines / dashed lines). Rows are different scenes; columns are snapshots at increasing time, the last column (or a side panel) shows the model's mean position error against time on a log scale.

Describe plainly what you see: how closely the model follows the reference, at roughly which time they begin to differ, whether differences grow gradually or suddenly, whether anything is outside the area where the reference stays, whether bodies behave (move, bounce, orbit, cluster) the way the reference does, whether any scene differs from the others, and anything else notable. Do not assume a cause. Report in a short paragraph.
---
