#!/bin/bash
cd ~
waitgone() { while squeue -u $USER -h -j $1 | grep -q .; do sleep 60; done; }
for s in a b1 b2 c1 c2 d; do
  if [ $s = a ]; then id=3020; else
    until out=$(sbatch augur/scripts/polaris_fallback_$s.sh 2>&1); do sleep 30; done
    id=$(echo $out | awk '{print $4}')
  fi
  echo "$s $id $(date)" >> ~/fallback_chain.out
  waitgone $id
done
echo done >> ~/fallback_chain.out
