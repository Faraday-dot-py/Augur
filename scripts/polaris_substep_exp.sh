#!/bin/bash
#SBATCH --job-name=augur-substep-exp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-substep-exp-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-substep"
export PYTHONPATH=.
OUT=results/substep_exp
mkdir -p $OUT
python tests/test_scatter_substep.py || echo "TESTS FAILED"
LT=$HOME/augur-lt/checkpoints
MODELS="E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt,B=$LT/initB.pt,EW=$LT/losstest/warm_euclid_4000.pt,P0=init"
SMOKE="${SMOKE:-}"
python scripts/scatter_substep_exp.py --models "$MODELS" --out $OUT/acc.json $SMOKE
[ -n "$SMOKE" ] && exit 0
for m in E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt B=$LT/initB.pt EW=$LT/losstest/warm_euclid_4000.pt P0=init; do
  name=${m%%=*}; ck=${m#*=}
  for s in v1:default u2:default,sub2 u4:default,sub4 nf2:default,nf4@2; do
    scheme=${s%%:*}; opt=${s#*:}
    extra=""
    if [ "$name" = P0 ]; then
      extra="--variant ms_kp_pot_v_g128_prior_pp8_full"
    fi
    d=$OUT/cons/${name}_${scheme}
    python scripts/scatter_conservation.py run --ckpt "$ck" --opt "$opt" --steps 1000 --regimes nr100,bh300 --seeds 9100,9200,9300 --truth t4 --out-dir $d $extra
    python scripts/scatter_conservation_summarize.py --dir $d --out-json $d/summary.json --fig-dir $d
  done
done
python scripts/scatter_substep_exp_summarize.py --acc $OUT/acc.json --cons $OUT/cons > $OUT/tables.md
cat $OUT/tables.md
echo ALLDONE
