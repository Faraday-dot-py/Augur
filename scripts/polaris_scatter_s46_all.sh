#!/bin/bash
#SBATCH --job-name=augur-scatter-s46-all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=augur-scatter-s46-all-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_s46"
export PYTHONPATH=.
E="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
B=ms_kp_pot_v_g128
OODM="ood:mass_ratio_2body,mass_dominant,mass_spectrum,mass_scale_equal"
mkdir -p results checkpoints/scatter_s46
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv

run() {
  local name="$1" fix="$2" variant="$3" init="$4" stress="$5" ck
  if [ -f "results/${name}_regress.json" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name $(date +%T)"
  ck="checkpoints/scatter_s46/$name.pt"
  python scripts/train_scatter_ft.py --fix "$fix" --variant "$variant" --init-ckpt "$init" --name "$name" --time-budget 1500 || echo "TRAIN $name FAILED"
  python scripts/scatter_regress.py run --ckpt "$ck" --variant "$variant" --out "results/${name}_regress.json" --battery || echo "REGRESS $name FAILED"
  python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate "results/${name}_regress.json" || true
  for st in $stress; do
    case "$st" in
      ood:*) python scripts/scatter_ood.py --ckpt "$ck" --variant "$variant" --axes "${st#ood:}" --out "results/${name}_ood_${st#ood:}.json" --opt kcache,fastio,cl || echo "STRESS $name $st FAILED" ;;
      *) python scripts/scatter_scaling.py --ckpt "$ck" --variant "$variant" --phase "$st" --pp cell --out "results/${name}_scaling_$st.json" || echo "STRESS $name $st FAILED" ;;
    esac
  done
  echo "DONE $name $(date +%T)"
}

run ctrl ctrl $B "$E" "$OODM,density_sigma,density_plummer,softening_eps r46 n46"
run cu cu ${B}_cu "$E" "r46 n46"
run largen largen $B "$E" "n46 n46big"
run mass mass $B "$E" "$OODM"
run massnorm massnorm ${B}_mnorm "$E" "$OODM"
run masswide masswide $B "checkpoints/scatter_s46/mass.pt" "ood:mass_spectrum,mass_dominant,mass_ratio_2body"
run eps eps ${B}_eps "$E" "ood:softening_eps"
run dense dense $B "$E" "ood:density_sigma,density_plummer"
echo "S46 ALL DONE"
