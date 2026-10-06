#!/bin/bash
#SBATCH --job-name=rollout-q
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=rollout-q-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
START=$(date +%s)
BUDGET=${BUDGET:-2100}
while read -r tag args; do
  [ -z "$tag" ] && continue
  [ -f results/$tag.json ] && continue
  LEFT=$((BUDGET - ($(date +%s) - START)))
  [ $LEFT -lt 120 ] && break
  echo "=== $tag ($LEFT s left)"
  python scripts/rollout_nbody.py --tag $tag --max-wall $((LEFT - 90)) --resume --ckpt-every 250 $args
done < ${TASKS:-scripts/rollout_tasks.txt}
