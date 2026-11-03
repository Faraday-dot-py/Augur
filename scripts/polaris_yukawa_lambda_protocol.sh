#!/bin/bash
#SBATCH --job-name=augur-yukawa-lambda-protocol
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --nice=100
#SBATCH --output=augur-yukawa-lambda-protocol-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/yukawa_lambda_protocol.py scripts/law_recovery_followup.py; do test -f $f || { echo MISSING $f; exit 2; }; done
python scripts/yukawa_lambda_protocol.py --n 1024 --out results/yukawa_lambda_protocol.json
rc=$?
test -f results/yukawa_lambda_protocol.json || { echo MISSING OUTPUT; exit 3; }
echo EXIT $rc
exit $rc
