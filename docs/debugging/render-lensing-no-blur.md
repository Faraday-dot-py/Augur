# Lensing renders: no blur by default (user request, 2026-09-28)

The user has explicitly asked that future black-hole lensing renders (`scripts/render_nbody_lens.py`)
NOT use blur by default — neither the fixed `--blur`/`--mass-blur` knobs nor the adaptive
`--mass-blur-n` (default 20). Render with `--blur 0 --mass-blur-n 0 --mass-blur 0` unless the
user asks for blur back.

## Known tradeoff — read before assuming no-blur is free

`--mass-blur-n` was added in 8e74abf specifically because at 100k bodies over a 1080^2 grid
(~0.09 particles/pixel), the unblurred density has strong Poisson shot noise. That noise
propagates through the FFT deflection convolution and shows up as a structured-looking but
spurious polygonal facet / windowpane-lattice pattern in the dense core — easy to mistake for a
real lensing caustic.

Verified 2026-09-28 on `results/orbit_bh_100k_first20_8substeps_snaps.npz`
(re-render of `videos/bh_100k_first20_8substeps_lensed.mp4`, `--blur 0 --mass-blur-n 0
--mass-blur 0`, fps 12): the facet/lattice artifact is clearly visible in the collapsed core in
the no-blur render, matching the pre-8e74abf artifact exactly. It was delivered anyway per the
explicit no-blur instruction, but any reviewer reading this render should know the ring/spoke
pattern in the core is a shot-noise artifact, not real structure — and should not silently
re-add blur to "fix" it without checking with the user first.
