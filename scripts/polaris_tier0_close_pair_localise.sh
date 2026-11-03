#!/bin/bash
#SBATCH --job-name=augur-tier0-close-pair-localise
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/adamwebb/augur-tier0-close-pair-localise-%j.log
# Localise the e44b796 tiled closed-loop tier 0's potential-energy loss by pair separation, local density, bound-pair survival and teacher-forced force gain.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_close_pair_localise
mkdir -p $W && cd $W && tar xzf $HOME/closepair_code.tgz
export PYTHONPATH=$W
mkdir -p results/close_pair
python scripts/tier0_close_pair_localise.py --tier0 $HOME/E_ms_kp_pot_v_g128.pt --m60k $HOME/augur_tiered_c2_tier0_matched_60k.pt \
  --new $HOME/augur_tiled_closed_loop/results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo.pt --out results/close_pair/localise.json
echo ALLDONE
