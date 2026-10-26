#!/bin/bash
#SBATCH --job-name=augur-showcase-galaxy_collapse_1e9_clumpy_lens
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@600
#SBATCH --output=augur-showcase-galaxy_collapse_1e9_clumpy_lens-%j.log

# 1e9-body clumpy (30 clumps plus field) gravitational collapse, scatter field, 200 ticks, lens histograms; expected ~15 min on one H200
# submit: sbatch --begin=2026-10-26T22:00:00 --nice=10000 scripts/polaris_showcase_galaxy_collapse_1e9_clumpy_lens.sh   (needs ~/showcase_code.tgz = tar of scripts/ model/)
if [ ! -d "$HOME/augur-showcase/scripts" ]; then mkdir -p "$HOME/augur-showcase" && tar -xzf "$HOME/showcase_code.tgz" -C "$HOME/augur-showcase"; fi
source "$HOME/augur-showcase/scripts/showcase_job_lib.sh"
showcase_setup
showcase_run results/showcase_galaxy_collapse_1e9_clumpy.done python scripts/scatter_bh_1b.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --n 1000000000 --steps 200 --mom-scale 16384 --ic clumpy --lens-out results/showcase_galaxy_collapse_1e9_clumpy_lens.npy --lens-res 1080 --tag showcase_galaxy_collapse_1e9_clumpy --resume-dir ckpt/clumpy1e9 --ckpt-every 100 --done-file results/showcase_galaxy_collapse_1e9_clumpy.done || exit 1
# polaris_download is capped at 1 GiB per file: split the lens histogram file into 900 MB parts (cat them back together locally)
if [ -e results/showcase_galaxy_collapse_1e9_clumpy.done ] && [ ! -e results/showcase_galaxy_collapse_1e9_clumpy_lens.npy.part00 ]; then split -b 900M -d results/showcase_galaxy_collapse_1e9_clumpy_lens.npy results/showcase_galaxy_collapse_1e9_clumpy_lens.npy.part; fi
