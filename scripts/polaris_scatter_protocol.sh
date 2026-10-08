#!/bin/bash
#SBATCH --job-name=augur-pp8-protocol
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=augur-pp8-protocol-%j.log
set -eo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
cd $HOME/augur_pp8
cp $HOME/augur_pp8_up/scatter_protocol_rescore.py scripts/
python scripts/scatter_protocol_rescore.py --out results/scatter_pp8/protocol_rescore.json
