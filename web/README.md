# Augur — browser demos

Static site, published via GitHub Pages. `index.html` is a landing page linking
to the current models, each a self-contained static page:

- `token/` — the conservative-contact token model (`results/cons_pure.pt`). See `token/README.md`.
- `gravity/` — the central-force N-body model (`checkpoints/gravity_central_v1.pt`). See `gravity/README.md`.
- `scatter/` — the scatter-field N-body model (`B_ms_kp_pot_v_g128.pt`). See `scatter/README.md`.
- `scatter-tiled/` — the tiled scatter-field model (Exp E `E_ms_kp_pot_v_g128.pt` as 2x2 / 4x4 tiles with ghost bodies, halo exchange and the R1b far kernel), next to the exact sim. Reuses `scatter/js`. See `scatter-tiled/README.md`.

Each subdirectory owns its own `js/`, `weights.bin`/`weights.json`, and `tests/`;
nothing is shared between them beyond the general dark/glass visual style (each
page's CSS is inlined, matching the other's variables and layout conventions
rather than importing a shared stylesheet, since neither page's markup overlaps
enough to be worth factoring out).

Serve locally with `python3 -m http.server` from this directory, then visit
`/`, `/token/`, `/gravity/`, `/scatter/`, or `/scatter-tiled/`.
