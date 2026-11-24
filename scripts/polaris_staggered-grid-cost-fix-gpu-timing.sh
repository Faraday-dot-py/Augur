#!/bin/bash
#SBATCH --job-name=augur-staggered-grid-cost-fix-gpu-timing
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-staggered-grid-cost-fix-gpu-timing-%j.log
# staggered-double-grid-with-quadratic-deposit, GPU timing of branch staggered-grid-cost-fix-same-outputs (8aeb20a): the shared kernel FFT across the two grids. Same eval script, scenes (seeds 9100/9200/9300 and 12000/12100/12200) and timing method as job 9248; trained candidate checkpoint unchanged.
# Inference only, estimate under 0.2 GPU-h (guess). No pass/kill line amended: reports the measured cost ratio against the existing line cost <= 2.2x. Code tarball: staggered_grid_cost_fix_code.tgz built from the branch, uploaded to $HOME.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_staggered_grid_cost_fix_gpu_timing
mkdir -p $W && cd $W && tar xzf $HOME/staggered_grid_cost_fix_code.tgz
export PYTHONPATH=$W
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
CK=$HOME/augur_staggered_double_grid/ckpt/single-law-gravity-checkpoint-g128_ms_kp_pot_v_g128_quad_dual.pt
OLD=$HOME/augur_staggered_double_grid/results
V=ms_kp_pot_v_g128_quad_dual
O=$W/results
mkdir -p $O
for f in $E $CK $OLD/eval_primary.json $OLD/eval_fresh.json scripts/staggered_double_grid_eval.py scripts/shift_average_stage1.py scripts/scatter_field.py model/__init__.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q '_kernel_fft_cached' scripts/scatter_field.py || { echo "MISSING cost fix in scatter_field.py"; exit 2; }
nvidia-smi -L || { echo "NO GPU"; exit 2; }
PID=
trap 'echo SIGTERM; [ -n "$PID" ] && kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
FAIL=0
for S in 9100,9200,9300:primary 12000,12100,12200:fresh; do
  SEEDS=${S%%:*}; TAG=${S##*:}
  if [ ! -s $O/eval_$TAG.json ] || ! grep -q 'seeds' $O/eval_$TAG.json || [ ! -e $O/DONE_$TAG ]; then
    python scripts/staggered_double_grid_eval.py --control $E --cand $CK --variant $V --seeds $SEEDS --out $O/eval_$TAG.json &
    PID=$!
    wait $PID
    rc=$?
    PID=
    echo STEP eval_$TAG $rc
    [ $rc -eq 0 ] && touch $O/DONE_$TAG || FAIL=1
  fi
  [ -e $O/DONE_$TAG ] && python - $OLD/eval_$TAG.json $O/eval_$TAG.json <<'PY' || FAIL=1
import json, sys
import numpy as np
old, new = (json.load(open(p)) for p in sys.argv[1:3])
keys = ("err@20", "dE@100", "probe_half")
for name, d in (("9248", old), ("cost-fix", new)):
    c = np.mean([s["control"]["sec_per_tick"] for s in d["seeds"].values()])
    k = np.mean([s["cand"]["sec_per_tick"] for s in d["seeds"].values()])
    print(f"{name}: control {c * 1e3:.2f} ms/tick cand {k * 1e3:.2f} ms/tick cost ratio {k / c:.3f}", {m: round(float(np.mean([s["cand"][m] for s in d["seeds"].values()])), 4) for m in keys})
for seed in old["seeds"]:
    print(seed, {m: (old["seeds"][seed]["cand"][m], new["seeds"][seed]["cand"][m]) for m in keys})
PY
done
for f in $O/eval_primary.json $O/eval_fresh.json; do [ -s $f ] || { echo OUTPUT MISSING $f; FAIL=1; }; done
echo "FAIL=$FAIL"
exit $FAIL
