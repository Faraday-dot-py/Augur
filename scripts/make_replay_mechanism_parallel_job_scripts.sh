#!/bin/bash
# Splits scripts/polaris_replay-mechanism-arms-and-force-error-breakdown.sh into one script per job (breakdowns, five arms, final summary and breakdown).
# Usage: bash scripts/make_replay_mechanism_parallel_job_scripts.sh OUTDIR [AFTEROK_ARM_JOB_IDS_COLON_SEPARATED]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SRC=scripts/polaris_replay-mechanism-arms-and-force-error-breakdown.sh
OUT=$1
DEP=${2:-}
mkdir -p "$OUT"
N=$(grep -n '^stage breakdown-4738-4739' $SRC | cut -d: -f1)
M=$(grep -n '^stage summary' $SRC | cut -d: -f1)
ARMLINES=$(sed -n "$N,$((M-1))p" $SRC)
FIN=$(sed -n "$M,\$p" $SRC | grep -vE "^(echo \"TOTAL_SECONDS|echo ALLDONE)")
HEAD=$(sed -n "1,$((N-1))p" $SRC | sed 's/tar xzf \$HOME\/replay_mechanism_code.tgz/flock $W.lock tar xzf $HOME\/replay_mechanism_code.tgz/')
mk() {
  local name=$1 time=$2 body=$3 dep=${4:-}
  local f=$OUT/polaris_replay-mechanism-$name.sh
  {
    echo "$HEAD" | sed "s/^#SBATCH --job-name=.*/#SBATCH --job-name=augur-replay-mechanism-$name/; s/^#SBATCH --time=.*/#SBATCH --time=$time/; s/^#SBATCH --output=.*/#SBATCH --output=augur-replay-mechanism-$name-%j.log/"
    echo "$body"
    echo 'echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"'
    echo 'echo ALLDONE'
  } > $f
  if [ -n "$dep" ]; then
    sed -i "s/^#SBATCH --requeue/#SBATCH --dependency=afterok:$dep\n#SBATCH --requeue/" $f
  fi
  bash -n $f
  echo $f
}
mk breakdowns-base-vs-replay 03:00:00 "$(echo "$ARMLINES" | grep -E '^(stage breakdown|breakdown )')"
for tag in late-ticks-only no-replay-matched-compute early-ticks-only one-round-of-replay truth-state-replay-second-base; do
  mk arm-$tag 04:00:00 "$(echo "$ARMLINES" | grep -E "^arm replay-mechanism-$tag ")"
done
mk summaries-and-all-arms-breakdown 03:00:00 "$FIN" "$DEP"
