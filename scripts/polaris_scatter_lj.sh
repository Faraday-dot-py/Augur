#!/bin/bash
#SBATCH --job-name=augur-slj
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=augur-slj-%j.log
# usage: sbatch --export=ARM=ft scripts/polaris_scatter_lj.sh   (ARM = ft: finetune from Exp E; scratch: stock zero-init heads)
# Needs ~/lj_code.tgz (repo files), ~/augur_fkg/kern/E_frozen.pt, ~/augur/results/lj_{gas,liquid,solid,coexist,melt,quench}.npz.
# Requeue-safe: train_scatter_lj.py --resume, eval only runs after training finishes.
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
ARM=${ARM:-ft}
D=$HOME/augur_slj
mkdir -p $D/results $D/ckpt $D/kern
cd $D
tar xzf $HOME/lj_code.tgz
[ -f kern/E_frozen.pt ] || cp -L $HOME/augur_fkg/kern/E_frozen.pt kern/E_frozen.pt
for s in gas liquid solid coexist melt quench; do [ -f results/lj_$s.npz ] || cp $HOME/augur/results/lj_$s.npz results/; done
INIT=""
[ "$ARM" = "ft" ] && INIT="--init-ckpt kern/E_frozen.pt"
if [ "$ARM" = "ft" ] && [ ! -f results/scatter_lj_zeroshot.json ]; then
  python -m scripts.scatter_lj_eval --ckpt kern/E_frozen.pt --zeroshot --tag zeroshot
fi
python -m scripts.train_scatter_lj --tag $ARM --ckpt-dir ckpt $INIT --resume
rc=$?
[ $rc -eq 0 ] || { echo "train exit $rc"; exit $rc; }
python -m scripts.scatter_lj_eval --ckpt ckpt/${ARM}_final.pt --tag $ARM
echo SLJ DONE $ARM
