#!/bin/bash
# Local renders of the --kpad ext (edge-artifact-free) showcase runs of Polaris job 4068 (scripts/polaris_showcase_kpadext_rerenders.sh).
# Needs results/showcase_*_kpadext* downloaded and the lens part files concatenated. Streams frames to ffmpeg, no GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p videos
python3 scripts/render_nbody_lens_density.py results/showcase_bh_1e9_300ticks_kpadext_lens.npy results/showcase_bh_1e9_300ticks_kpadext_lens_windows.npy videos/showcase_black_hole_1e9_lensed_300ticks_kpadext.mp4 \
  --n 1000000000 --fps 24 --title "1 billion bodies collapsing into a black-hole-like core (lensed view)"
python3 scripts/render_nbody_lens_density.py results/showcase_galaxy_collapse_1e9_clumpy_kpadext_lens.npy results/showcase_galaxy_collapse_1e9_clumpy_kpadext_lens_windows.npy videos/showcase_galaxy_collapse_1e9_clumpy_lensed_kpadext.mp4 \
  --n 1000000000 --fps 24 --title "1 billion bodies: 30 clumps falling together (lensed view)"
python3 scripts/render_bh_density_labeled.py results/showcase_bh_1e10_kpadext_hist.npz videos/showcase_black_hole_1e10_density_kpadext.mp4 --title "10 billion bodies, learned gravity field" --fps 12
python3 scripts/render_scatter_solar.py --traj results/solar/showcase_solar_orbit_ft2_kpadext_traj.npz --model orbit_ft2 --out videos/showcase_solar_system_reference_vs_learned_kpadext.mp4 --every 4
python3 scripts/render_scatter_solar.py --traj results/solar/showcase_solar_orbit_ft2_kpadext_traj.npz --model orbit_ft2 --out x --grid videos/showcase_solar_system_grid_kpadext.png
python3 scripts/render_nbody_lens_density.py results/showcase_bh_1e9_300ticks_kpadext_lens.npy results/showcase_bh_1e9_300ticks_kpadext_lens_windows.npy videos/showcase_black_hole_1e9_lensed_300ticks_kpadext_lattice_smoothed.mp4 \
  --n 1000000000 --fps 24 --dt 0.1 --lattice-blur --fontsize 13 --title "1 billion bodies collapsing into a black-hole-like core (lensed view)"
python3 scripts/render_nbody_lens_density.py results/showcase_galaxy_collapse_1e9_clumpy_kpadext_lens.npy results/showcase_galaxy_collapse_1e9_clumpy_kpadext_lens_windows.npy videos/showcase_galaxy_collapse_1e9_clumpy_lensed_kpadext_lattice_smoothed.mp4 \
  --n 1000000000 --fps 24 --dt 0.1 --lattice-blur --fontsize 13 --title "1 billion bodies: 30 clumps falling together (lensed view)"
python3 scripts/split_snaps_model_truth.py results/showcase_bh_100k_kpadext_snaps.npz /tmp/bh100k_model.npz /tmp/bh100k_truth.npz
python3 scripts/render_nbody_lens.py /tmp/bh100k_model.npz videos/showcase_black_hole_1e5_lensed_learned_model_kpadext.mp4 --fps 24 --dt 0.1 --blur 0 --mass-blur-n 0 --mass-blur 0 --label "learned force (scatter field), c=10"
python3 scripts/render_nbody_lens.py /tmp/bh100k_truth.npz videos/showcase_black_hole_1e5_lensed_exact_nbody_kpadext.mp4 --fps 24 --dt 0.1 --blur 0 --mass-blur-n 0 --mass-blur 0 --label "exact direct N-body, c=10"
