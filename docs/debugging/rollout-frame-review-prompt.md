# Rollout frame review prompt (hypothesis-free)

Render grids with `scripts/rollout_render.py --out videos/<x>.png --steps ... --rows name=results/<tag>_snaps.npz ...`, then dispatch a fresh agent with:

---

Read the image(s) at `<paths>`. Each is a grid of density frames (2D histograms of particle positions, log color scale) from particle simulations: each row is a different simulation variant (row label on the left), each column is a different time step (labeled at the top). Within a column all rows share the same field of view and color scale.

Describe, in plain terms, what you observe:
- What does the content of each row look like at the first column vs later columns?
- Does structure change gradually or suddenly, and at about which columns?
- Do the rows differ from each other, and if so in which columns and how (e.g. shape, extent, density of clumps, smoothness, blockiness, grid or stripe patterns, missing or extra structure)?
- Any regular or repeating spatial pattern in any row, where it first appears and how it evolves?
- Anything else visually notable.

Do not assume a cause. Just describe what is visible, for each image separately, in a few sentences to a paragraph each. Read only these images; do not read any other files.
