#!/bin/bash
# Local re-renders for the README showcase (no GPU). Streams frames to ffmpeg; peak RAM estimates in comments.
# Needs, from Polaris (polaris_download is capped at 1 GiB per file; lens files come in 900 MB parts):
#   results/showcase_bh_1e9_300ticks_lens.npy.part* (+ _windows.npy)  and  results/showcase_galaxy_collapse_1e9_clumpy_lens.npy.part* (+ _windows.npy)
#   results/solar/showcase_solar_orbit_ft2_traj.npz
# No-blur lensing is the project default (docs/debugging/render-lensing-no-blur.md); dynamic lensing recomputes the deflection every frame from the actual density.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p videos
for t in showcase_bh_1e9_300ticks showcase_galaxy_collapse_1e9_clumpy; do
  [ -e results/${t}_lens.npy ] || cat results/${t}_lens.npy.part* > results/${t}_lens.npy
done
# lensed 1e9 renders: memmap of the 1080x1080 float32 histograms (1.4 GB file, a few hundred MB resident), 1080x1080 at 24 fps
python3 scripts/render_nbody_lens_density.py results/showcase_bh_1e9_300ticks_lens.npy results/showcase_bh_1e9_300ticks_lens_windows.npy videos/showcase_black_hole_1e9_lensed_300ticks.mp4 \
  --n 1000000000 --fps 24 --title "1 billion bodies collapsing into a black-hole-like core (lensed view)"
python3 scripts/render_nbody_lens_density.py results/showcase_galaxy_collapse_1e9_clumpy_lens.npy results/showcase_galaxy_collapse_1e9_clumpy_lens_windows.npy videos/showcase_galaxy_collapse_1e9_clumpy_lensed.mp4 \
  --n 1000000000 --fps 24 --title "1 billion bodies: 30 clumps falling together (lensed view)"
# density videos from the stored per-frame histograms of all bodies (about 1.5 GB resident)
python3 scripts/render_bh_density_labeled.py results/scatter_bh_10b_hist.npz videos/showcase_black_hole_1e10_density.mp4 --title "10 billion bodies, learned gravity field" --fps 12
python3 scripts/render_bh_density_labeled.py results/scatter_bh_1b_hist.npz videos/showcase_black_hole_1e9_density.mp4 --title "1 billion bodies, learned gravity field" --fps 12
# planetesimal disk (cached simulation arrays in results/protodisk_video_{real,proxy}.npz), 1920x1080
python3 scripts/render_protodisk_video.py render --dpi 120 --video videos/showcase_solar_system_formation_disk_reference_and_toy_proxy.mp4
# solar system, exact reference vs learned model
python3 scripts/render_scatter_solar.py --traj results/solar/showcase_solar_orbit_ft2_traj.npz --model orbit_ft2 --out videos/showcase_solar_system_reference_vs_learned.mp4 --every 4 --grid videos/showcase_solar_system_grid.png
# next: contact sheet per video (ffmpeg select+tile) and the fresh-reviewer step in docs/debugging/frame-artifact-review-prompt.md; verify with ffmpeg -v error -f null -
