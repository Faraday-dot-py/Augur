#!/bin/bash
#SBATCH --job-name=augur-merge-audit-into-master-gpu-and-checkpoint-tests
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=02:00:00
#SBATCH --nice=100
#SBATCH --output=/home/adamwebb/augur-merge-audit-into-master-gpu-and-checkpoint-tests-%j.log
# GPU and checkpoint/resume tests on merge commit 1d89cfe (audit bcce19d into master 281942d). Runs every test stage even after a failure, prints per-stage rc and a final
# MERGE_TESTS_ALL_PASS or MERGE_TESTS_FAILED line. Tests only (no experiment), under 1 GPU-h expected. Code tarball: git archive of scratch worktree, upload to $HOME. Usage: sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_merge_tests
rm -rf $W && mkdir -p $W && cd $W && tar xzf $HOME/merge_audit_into_master_code.tgz
export PYTHONPATH=$W
nvidia-smi -L
python -c "import torch,sys; assert torch.cuda.is_available(), 'no cuda'; print(torch.__version__, torch.cuda.get_device_name(0))" || { echo "MERGE_TESTS_FAILED: no cuda"; exit 2; }
FAILS=""
stage() {
  local name=$1; shift
  echo "=== STAGE $name: $*"
  "$@"
  local rc=$?
  echo "=== STAGE $name rc=$rc"
  [ $rc -ne 0 ] && FAILS="$FAILS $name"
}
stage pytest-all python -m pytest tests -q -p no:cacheprovider
stage scatter-tier0-tests python scripts/scatter_tier0_tests.py --device cuda
stage tier0-tiled-closed-loop-tests python scripts/tier0_tiled_closed_loop_tests.py --device cuda
stage tier0-drift-diagnoses-tests python scripts/tier0_drift_diagnoses_tests.py --device cuda
stage scatter-lj-sigterm-resume bash scripts/polaris_scatter_lj_resume_test.sh /tmp/scatter_lj_rt_$$ cuda
grep -q "resumed" /tmp/scatter_lj_rt_$$.log2 || { echo "lj resume: no 'resumed' line"; FAILS="$FAILS scatter-lj-resume-marker"; }
grep -q "^done" /tmp/scatter_lj_rt_$$.log2 || { echo "lj resume: no 'done' line"; FAILS="$FAILS scatter-lj-done-marker"; }
if [ -z "$FAILS" ]; then echo "MERGE_TESTS_ALL_PASS"; else echo "MERGE_TESTS_FAILED:$FAILS"; exit 1; fi
