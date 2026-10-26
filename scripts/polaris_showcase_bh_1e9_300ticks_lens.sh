#!/bin/bash
#SBATCH --job-name=augur-showcase-bh_1e9_300ticks_lens
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@600
#SBATCH --output=augur-showcase-bh_1e9_300ticks_lens-%j.log

# 1e9-body scatter-field black-hole scene, 300 ticks (previous runs: 100), all bodies histogrammed per frame at 1080x1080 for the lensed render; two passes (windows, then histograms); expected ~25 min on one H200
# submit: sbatch --begin=2026-10-26T22:00:00 --nice=10000 scripts/polaris_showcase_bh_1e9_300ticks_lens.sh   (needs ~/showcase_code.tgz = tar of scripts/ model/)
if [ ! -d "$HOME/augur-showcase/scripts" ]; then mkdir -p "$HOME/augur-showcase" && tar -xzf "$HOME/showcase_code.tgz" -C "$HOME/augur-showcase"; fi
source "$HOME/augur-showcase/scripts/showcase_job_lib.sh"
showcase_setup
showcase_run results/showcase_bh_1e9_300ticks.done python scripts/scatter_bh_1b.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --n 1000000000 --steps 300 --mom-scale 16384 --lens-out results/showcase_bh_1e9_300ticks_lens.npy --lens-res 1080 --tag showcase_bh_1e9_300ticks --resume-dir ckpt/bh1e9 --ckpt-every 100 --done-file results/showcase_bh_1e9_300ticks.done || exit 1
# polaris_download is capped at 1 GiB per file: split the lens histogram file into 900 MB parts (cat them back together locally)
if [ -e results/showcase_bh_1e9_300ticks.done ] && [ ! -e results/showcase_bh_1e9_300ticks_lens.npy.part00 ]; then split -b 900M -d results/showcase_bh_1e9_300ticks_lens.npy results/showcase_bh_1e9_300ticks_lens.npy.part; fi
