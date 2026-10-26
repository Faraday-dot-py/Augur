#!/bin/bash
#SBATCH --job-name=augur-bh_1e9_edge_free_panels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --begin=2026-10-26T23:30:00
#SBATCH --nice=10000
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-bh_1e9_edge_free_panels-%j.log

# 1e9-body scatter-field black-hole scene, 100 ticks, Exp E weights with kpad=ext (grid-edge force artifact removed), all bodies histogrammed + 128x128 force/potential maps per tick, for the four-panel README figure.
# Expected ~5 min on one H200 (original zero-padded run: 263 s loop). --hist/--grid-dump runs cannot resume mid-rollout (no checkpoint path in scatter_bh_1b.py), so the wrapper requeues a preempted job from tick 0.
# submit: sbatch --begin=2026-10-26T23:30:00 --nice=10000 scripts/polaris_bh_1e9_edge_free_panels.sh   (needs ~/edge_free_code.tgz = tar of scripts/ model/, ~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt)
export PYTHONUNBUFFERED=1
D="$HOME/augur-edgefree"
if [ ! -d "$D/scripts" ]; then mkdir -p "$D" && tar -xzf "$HOME/edge_free_code.tgz" -C "$D"; fi
cd "$D" && export PYTHONPATH=. && mkdir -p results checkpoints/scatter_bh
[ -e checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt ] || ln -s "$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt" checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
source scripts/showcase_job_lib.sh
run_all() {
  python scripts/scatter_bh_1b.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --n 1000000000 --steps 100 --mom-scale 16384 --kpad ext --hist --grid-dump --tag bh_1e9_edge_free \
    && python scripts/bh_edge_panels_extract.py results/bh_1e9_edge_free results/bh_1e9_edge_free_ticks.npz \
    && touch results/bh_1e9_edge_free.done
}
showcase_run results/bh_1e9_edge_free.done run_all || exit 1
