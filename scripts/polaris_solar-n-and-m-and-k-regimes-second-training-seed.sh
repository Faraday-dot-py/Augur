#!/bin/bash
#SBATCH --job-name=augur-solar-nkm-second-training-seed
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --nice=100
#SBATCH --output=augur-solar-nkm-second-training-seed-%j.log

# job 4918 recipe at training seeds 4739 then 4740 (env SEEDS), 4 arms per seed concurrently on one GPU. Resumable: trainer full-state checkpoint every 200 its, SIGTERM saves, finished stages skipped, requeue (max 3).
# needs ~/solar_nkm_second_seed_code.tgz (scripts/ model/ tests/ conftest.py results/solar_moon_diagnosis/moon_arm_enlarged_eval_repeat3.json results/solar_nkm_regimes/verdict.json), ~/augur-solar-moon-arms (scene_kinds_256.pt), ~/augur-solar-more-moons (scene cache, moon arm checkpoint), ~/bounce-scenes/checkpoints_scenes/orbit_ft2.pt.
export PYTHONUNBUFFERED=1
DEADLINE=1794070800
REMAIN=$((DEADLINE-$(date +%s)))
[ $REMAIN -le 300 ] && { echo "DEADLINE 2026-11-07 09:00 PST: $REMAIN s left, not starting"; exit 3; }
past_deadline() { [ $((DEADLINE-$(date +%s))) -le 300 ]; }
dlrun() {
  local left=$((DEADLINE-$(date +%s)))
  [ $left -le 300 ] && { echo "DEADLINE: skipping $*"; return 124; }
  timeout -k 120 -s TERM $left "$@"
}
export DEADLINE
export -f past_deadline dlrun
D="$HOME/augur-solar-nkm-second-seed"
OLD="$HOME/augur-solar-more-moons"
PRE="$HOME/augur-solar-moon-arms"
INIT="$HOME/bounce-scenes/checkpoints_scenes/orbit_ft2.pt"
SCENES="$OLD/ckpt/solar_more_moons/scenes_256_0_train-more-moon-situations.pt"
KINDS="$PRE/ckpt/solar_moon_arms/scene_kinds_256.pt"
MOONARM="$OLD/ckpt/solar_more_moons/solar-scale-translation-augmentation-more-moon-situations.pt"
for f in $HOME/solar_nkm_second_seed_code.tgz $INIT $SCENES $KINDS $MOONARM; do test -f $f || { echo MISSING $f; exit 2; }; done
mkdir -p "$D" && tar -xzf "$HOME/solar_nkm_second_seed_code.tgz" -C "$D"
cd "$D" || exit 2
export PYTHONPATH=.
for f in scripts/train_scatter_solar_moon_arms.py scripts/train_scatter_solar_stability.py scripts/eval_moon_survival_enlarged.py scripts/eval_scatter_solar_translation.py scripts/eval_scatter_solar_generalized.py scripts/eval_scatter_solar_stability.py scripts/scatter_solar_generalized.py scripts/verdict_solar_nkm_second_training_seed.py tests/test_scatter_solar_stability.py tests/test_scatter_solar_moon_arms.py tests/test_scatter_solar_regimes.py results/solar_moon_diagnosis/moon_arm_enlarged_eval_repeat3.json results/solar_nkm_regimes/verdict.json; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results/solar_nkm_second_seed
mkdir -p $R ckpt/solar_more_moons ckpt/solar_moon_arms logs
ln -sf "$SCENES" ckpt/solar_more_moons/
ln -sf "$KINDS" ckpt/solar_moon_arms/
export INIT MOONARM R
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo "restart count ${SLURM_RESTART_COUNT:-0}"
python -m pytest tests/test_scatter_solar_stability.py tests/test_scatter_solar_moon_arms.py tests/test_scatter_solar_regimes.py -x -q || exit 1

arm_pipeline() {
  SEED=$1; ARM=$2
  case "$ARM" in
    k-alone) RAMP=k ;; n-and-k) RAMP=k,n ;; n-and-m-and-k) RAMP=k,n,m ;; control-all-full) RAMP=none ;;
  esac
  O=$R/s$SEED
  CK=ckpt/solar_moon_arms_s$SEED/$ARM.pt
  L=logs/${ARM}_s$SEED.log
  mkdir -p $O
  [ -e $O/${ARM}_train.json ] || dlrun python scripts/train_scatter_solar_moon_arms.py --arm $ARM --init-ckpt $INIT --ramp $RAMP --seed $SEED --out-dir $O --ckpt-dir ckpt/solar_moon_arms_s$SEED >> $L 2>&1 || return 1
  [ -e $O/${ARM}_train.json ] || return 1
  [ -e $O/${ARM}_enlarged_eval.json ] || dlrun python scripts/eval_moon_survival_enlarged.py --ckpt $ARM=$CK --seeds 50 --out $O/${ARM}_enlarged_eval.json >> $L 2>&1 || return 1
  [ -e $O/${ARM}_guard_eval/eval_summary.json ] || dlrun python scripts/eval_scatter_solar_translation.py --baseline $MOONARM --baseline-name moon-arm --arm $ARM=$CK --pick $ARM --control moon-arm --offset 0,0 --out-dir $O/${ARM}_guard_eval --out-name eval_summary >> $L 2>&1 || return 1
}
export -f arm_pipeline
ARMS="n-and-m-and-k n-and-k k-alone control-all-full"
SEEDS=${SEEDS:-"4739 4740"}

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader -l 30 >> $R/gpu_util_samples_${SLURM_JOB_ID:-local}.csv &
SMI=$!
trap 'echo "[$(date -Iseconds)] TERM: stopping trainers and evals, requeue"; TERMED=1; pkill -TERM -f "scripts/(train_scatter_solar_moon_arms|eval_moon_survival_enlarged|eval_scatter_solar_translation).py .*$R/s"' TERM
for S in $SEEDS; do
  [ -n "${TERMED:-}" ] && break
  past_deadline && break
  ( for A in $ARMS; do echo "$S $A"; done | xargs -P 4 -L 1 bash -c 'arm_pipeline $0 $1' ) &
  PIDX=$!
  while kill -0 $PIDX 2>/dev/null; do wait $PIDX; done
  if [ -z "${TERMED:-}" ] && ! past_deadline; then
    MISS=""; for a in $ARMS; do [ -e $R/s$S/${a}_guard_eval/eval_summary.json ] || MISS="$MISS $a"; done
    if [ -n "$MISS" ]; then echo "seed $S retry (2 at a time) for:$MISS"; for a in $MISS; do echo "$S $a"; done | xargs -P 2 -L 1 bash -c 'arm_pipeline $0 $1'; fi
  fi
done
kill $SMI 2>/dev/null
awk -F', ' '{u=$2+0; m=$4+0; su+=u; n++; if(u>mu)mu=u; if(m>mm)mm=m} END{printf "GPU samples %d mean_util %.1f max_util %d max_mem_MiB %d\n", n, su/n, mu, mm}' $R/gpu_util_samples_${SLURM_JOB_ID:-local}.csv
DONE=1; for S in $SEEDS; do for a in $ARMS; do [ -e $R/s$S/${a}_guard_eval/eval_summary.json ] || DONE=0; done; done
if [ $DONE = 1 ]; then
  python scripts/verdict_solar_nkm_second_training_seed.py $R results/solar_moon_diagnosis/moon_arm_enlarged_eval_repeat3.json results/solar_nkm_regimes/verdict.json $R/verdict.json
elif past_deadline; then echo "DEADLINE_STOP: unfinished arms, no verdict"; ls $R/s*/; exit 4
elif [ -z "${TERMED:-}" ] && [ "${SLURM_RESTART_COUNT:-0}" -ge 3 ]; then echo "arms unfinished after 3 restarts"; exit 1
else echo "unfinished, requeue"; scontrol requeue $SLURM_JOB_ID; fi
