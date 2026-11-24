#!/bin/bash
#SBATCH --job-name=augur-learned-per-body-property-channels-mass-conditioned-stage-1-double-iterations
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=00:50:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-learned-per-body-property-channels-mass-conditioned-stage-1-double-iterations-%j.log
##SBATCH --dependency=afterany:SUPERVISOR_GPU_SLOT_PREDECESSOR_JOBID
# Stage 1 rerun of learned-per-body-property-channels at double iterations: --stage-iters 1200 x 7 stages = 8400 (job 13166 used 600 x 7 = 4200), all else identical, seed 4738.
# Pre-registration: docs/debugging/architecture-push-prereg/learned-per-body-property-channels-double-iterations.md.
# Estimate about 0.5 GPU-h (job 13166 took 894 s for 4200 iterations). Requeue-safe: full state every 50 iterations and on SIGTERM, resumed automatically.
# A result json in the work dir written by another job is an error (stale results once passed for a real run); a requeue keeps its job id and resumes.
# Code tarball: bash scripts/build_architecture_push_stage1_code_tarball.sh, upload to $HOME. Supervisor adds --dependency at submission.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_architecture_push_stage1_mass_double_iterations_2026_11_23
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
T=$HOME/architecture_push_stage1_code.tgz
[ -s $T ] || { echo "MISSING $T"; exit 2; }
[ -s $E ] || { echo "MISSING single-law-gravity-control checkpoint $E"; exit 2; }
mkdir -p $W && cd $W && tar xzf $T || { echo "EXTRACT FAILED"; exit 2; }
export PYTHONPATH=$W
for f in scripts/scatter_field.py scripts/scatter_ft_data.py scripts/scatter_mass_curriculum_data.py scripts/train_scatter_mass_curriculum.py scripts/train_scatter_field.py; do
  [ -s $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q "def _massfeat" scripts/scatter_field.py || { echo "STALE scatter_field.py"; exit 2; }
grep -q "def still_falling(val_hist" scripts/train_scatter_mass_curriculum.py || { echo "STALE trainer (no validation plateau guard)"; exit 2; }
md5sum $T scripts/scatter_field.py scripts/train_scatter_mass_curriculum.py $E
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/scatter_mass_curriculum_stage1_double_iterations
C=ckpt_scatter_mass_curriculum_stage1_double_iterations
mkdir -p $O $C
JF=$W/.job_id
if [ -s $O/massfeat_curriculum.json ] && [ "$(cat $JF 2>/dev/null)" != "${SLURM_JOB_ID:-}" ]; then
  echo "STALE RESULT $W/$O/massfeat_curriculum.json not written by job ${SLURM_JOB_ID:-none}; use a new work dir"; exit 3
fi
echo "${SLURM_JOB_ID:-none}" > $JF
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -s $O/massfeat_curriculum.json ]; then
  python scripts/train_scatter_mass_curriculum.py --arm massfeat_curriculum --init-ckpt $E --seed 4738 --stage-iters 1200 --resume --out-dir $O --ckpt-dir $C >> $O/run.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "exit $RC"; tail -20 $O/run.log; exit $RC; }
fi
[ -s $O/massfeat_curriculum.json ] || { echo "MISSING OUTPUT $O/massfeat_curriculum.json"; ls $O; exit 1; }
grep -E '"verdict"|plateau_falling' $O/massfeat_curriculum.json
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
