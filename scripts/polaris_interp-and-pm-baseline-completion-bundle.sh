#!/bin/bash
#SBATCH --job-name=augur-interp-and-pm-baseline-completion-bundle
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --nice=100
#SBATCH --output=augur-interp-and-pm-baseline-completion-bundle-%j.log

# No PM-baseline step: row 2 has no missing family (CONFIRMED_PARTIAL_2_OF_3 means Plummer p90 ratio 1.82 < 2.0 on 64 scenes, all three families were run), so a rerun would repeat job 4827.
# Steps: (1) row 1 Yukawa lambda protocol on fresh placement seeds 4744/4745/4746, scored under the literal 4832 line and the tie-aware 4841 line, both unamended;
# (2) CURIOSITY (no verdict): dec0 mesh SAEs of checkpoint 4738 at SAE seeds 4740 and 4741, decoder matching against the existing SAE seeds 4738 and 4739.
# Every step is short (<= 15 min expected) and skipped on rerun if its output exists; SIGTERM stops the current step and requeues.
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
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/yukawa_lambda_protocol.py scripts/law_recovery_followup.py scripts/law_recovery_probe.py scripts/scatter_multi.py scripts/scatter_far_general.py scripts/scatter_far.py scripts/scatter_tiling_eval.py scripts/scatter_tiling.py scripts/scatter_field.py scripts/scatter_scaling_ops.py scripts/train_scatter_field.py scripts/scatter_scaling.py scripts/gravity_sim.py scripts/train_gravity_dynamics.py model/__init__.py model/central_force.py model/token_free.py model/token_graph.py scripts/sae_interp.py scripts/match_sae_seeds.py scripts/match_lowdead_saes.py scripts/score_yukawa_lambda_protocol.py scripts/score_yukawa_lambda_protocol_tie_aware.py results/sae_interp_lowdead_dec0_s4738.npz results/sae_interp_lowdead_dec0_s4738_saeseed4739.npz; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
Y=$R/yukawa_lambda_protocol_fresh_seeds_4744_4745_4746
S=$R/sae_interp_lowdead_dec0_s4738_saeseed
M=$R/sae_interp_sae_seed_matching_4740_4741.txt
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "restart count ${SLURM_RESTART_COUNT:-0}"

FAILED=""
TERMED=""
CHILD=""
trap 'echo "[$(date -Iseconds)] TERM: stopping current step, requeue"; TERMED=1; [ -n "$CHILD" ] && kill -TERM $CHILD 2>/dev/null' TERM

step() {
  NAME=$1; DONE=$2; shift 2
  [ -n "$TERMED" ] && return
  if [ -e $DONE ]; then echo "STEP $NAME skipped, $DONE exists"; return; fi
  if past_deadline; then echo "STEP $NAME skipped, deadline"; DEADLINED=1; return; fi
  dlrun "$@" &
  CHILD=$!
  while kill -0 $CHILD 2>/dev/null; do wait $CHILD; done
  wait $CHILD
  RC=$?
  CHILD=""
  echo "STEP $NAME exit code $RC"
  if past_deadline && [ ! -e $DONE ]; then DEADLINED=1; return; fi
  if [ $RC -ne 0 ] && [ -z "$TERMED" ]; then FAILED="$FAILED $NAME"; fi
}

score() {
  python "$1" $Y.json > "$2.tmp"
  RC=$?
  cat "$2.tmp"
  [ $RC -eq 0 ] && mv "$2.tmp" "$2"
  return $RC
}

match() {
  rc=0
  for new in 4740 4741; do
    for ref in $R/sae_interp_lowdead_dec0_s4738.npz ${S}4739.npz; do
      test -f $S$new.npz || { echo MISSING $S$new.npz; return 2; }
      echo "== matching SAE seed $new against $ref (live features only, random-direction baseline)"
      python scripts/match_lowdead_saes.py $S$new.npz $ref || rc=1
      echo "== matching SAE seed $new against $ref (dead rows included)"
      python scripts/match_sae_seeds.py $S$new.npz $ref || rc=1
    done
  done
  echo "== matching SAE seed 4740 against SAE seed 4741 (live features only, random-direction baseline)"
  python scripts/match_lowdead_saes.py ${S}4740.npz ${S}4741.npz || rc=1
  echo "== matching SAE seed 4740 against SAE seed 4741 (dead rows included)"
  python scripts/match_sae_seeds.py ${S}4740.npz ${S}4741.npz || rc=1
  return $rc
}

match_report() {
  match > $M.tmp
  RC=$?
  cat $M.tmp
  [ $RC -eq 0 ] && mv $M.tmp $M
  return $RC
}

step yukawa_fresh_seeds $Y.json python scripts/yukawa_lambda_protocol.py --n 1024 --seeds 4744,4745,4746 --out $Y.json
step yukawa_score_literal_4832_line $Y.literal.txt score scripts/score_yukawa_lambda_protocol.py $Y.literal.txt
step yukawa_score_tie_aware_4841_line $Y.tie_aware.txt score scripts/score_yukawa_lambda_protocol_tie_aware.py $Y.tie_aware.txt
for SEED in 4740 4741; do
  step sae_dec0_mesh_seed$SEED ${S}${SEED}_sae.pt python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec0 --sae-seed $SEED --parts mesh --mesh-sizes 16,32,64 --scenes 24 --heldout-scenes 6 --out $S$SEED.json
done
step sae_seed_matching $M match_report

if [ -n "${DEADLINED:-}" ] || past_deadline; then
  echo "DEADLINE_STOP: unfinished steps skipped"; ls -l $Y* $S*; exit 4
fi
if [ -n "$TERMED" ]; then
  echo "unfinished, requeue"; scontrol requeue $SLURM_JOB_ID; exit 0
fi
ls -l $Y* $S*
if [ -n "$FAILED" ]; then echo "FAILED STEPS:$FAILED"; exit 1; fi
echo ALLDONE
