#!/bin/bash
cd "$HOME/bounce"
while IFS='|' read -r name cmd; do
  [ -z "$name" ] && continue
  while squeue -u adamwebb -h -o %j | grep -q '^scaling-'; do sleep 30; done
  until sbatch --job-name="$name" --output="$name-%j.log" scripts/scaling_job.sh $cmd; do sleep 45; done
  echo "submitted $name $(date)"
done < "$1"
