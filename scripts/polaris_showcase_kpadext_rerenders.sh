#!/bin/bash
#SBATCH --job-name=augur-showcase-kpadext
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@600
#SBATCH --output=augur-showcase-kpadext-%j.log

# Edge-artifact-free (--kpad ext) re-runs of the showcase scenes, sequential on one GPU: solar (~5 min), clumpy 1e9 200 ticks (~15 min), black hole 1e9 300 ticks (~25 min), black hole 1e10 100 ticks with histograms (~45 min, 88 GB peak).
# Each step has a done file and is skipped on rerun; 1e9 steps resume from ckpt/, the 1e10 run checkpoints every 50 ticks (about 80 GB, failure to write is non-fatal).
# submit: sbatch scripts/polaris_showcase_kpadext_rerenders.sh   (needs ~/showcase_kpadext_code.tgz = tar of scripts/ model/)
export PYTHONUNBUFFERED=1
D="$HOME/augur-showcase-kpadext"
if [ ! -d "$D/scripts" ]; then mkdir -p "$D" && tar -xzf "$HOME/showcase_kpadext_code.tgz" -C "$D"; fi
cd "$D" && export PYTHONPATH=. && mkdir -p results/solar ckpt checkpoints/scatter_bh
[ -e checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt ] || ln -s "$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt" checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
source scripts/showcase_job_lib.sh
CK=checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
split_lens() { [ -e results/$1.done ] && [ ! -e results/$1_lens.npy.part00 ] && split -b 900M -d results/$1_lens.npy results/$1_lens.npy.part; return 0; }
showcase_run results/solar/showcase_solar_orbit_ft2_kpadext_traj.npz python scripts/scatter_solar.py run --kpad ext --cands "orbit_ft2=E:ms_kp_pot_v_g128:$HOME/bounce-scenes/checkpoints_scenes/orbit_ft2.pt" --steps 2000 --hill-k 4.0 --tag showcase_solar_orbit_ft2_kpadext || exit 1
T=showcase_galaxy_collapse_1e9_clumpy_kpadext
showcase_run results/$T.done python scripts/scatter_bh_1b.py --ckpt $CK --n 1000000000 --steps 200 --mom-scale 16384 --ic clumpy --kpad ext --lens-out results/${T}_lens.npy --lens-res 1080 --tag $T --resume-dir ckpt/$T --ckpt-every 100 --done-file results/$T.done || exit 1
split_lens $T
T=showcase_bh_1e9_300ticks_kpadext
showcase_run results/$T.done python scripts/scatter_bh_1b.py --ckpt $CK --n 1000000000 --steps 300 --mom-scale 16384 --kpad ext --lens-out results/${T}_lens.npy --lens-res 1080 --tag $T --resume-dir ckpt/$T --ckpt-every 100 --done-file results/$T.done || exit 1
split_lens $T
T=showcase_bh_1e10_kpadext
run10() { python scripts/scatter_bh_1b.py --ckpt $CK --n 10000000000 --steps 100 --mom-scale 16384 --kpad ext --hist --tag $T --resume-dir ckpt/$T --ckpt-every 50 && touch results/$T.done; }
showcase_run results/$T.done run10 || exit 1
