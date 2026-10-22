#!/bin/bash
# SIGTERM + --resume check for train_scatter_lj.py (runs on CPU or GPU, ~2 min with small data)
export PYTHONUNBUFFERED=1 PYTHONPATH=.
D=${1:-/tmp/scatter_lj_rt}
rm -rf $D
A="--tag rt --ckpt-dir $D --device ${2:-cpu} --scenes 8 --val-scenes 2 --frames 40 --warmup 50 --iters 40 --batch 2 --k-start 3 --k-end 4 --ckpt-every 10 --val-every 20 --log-every 2"
${PY:-python} -m scripts.train_scatter_lj $A > $D.log1 2>&1 &
PID=$!
while ! grep -q "^it 1[0-9] " $D.log1 2>/dev/null; do sleep 1; kill -0 $PID 2>/dev/null || break; done
kill -TERM $PID
wait $PID
echo "EXIT after SIGTERM: $?"
grep -E "SIGTERM" $D.log1
${PY:-python} -m scripts.train_scatter_lj $A --resume > $D.log2 2>&1
echo "EXIT resume: $?"
grep -E "resumed|^done" $D.log2
ls $D
