set -uo pipefail
export PYTHONUNBUFFERED=1
D=$HOME/augur_fkg
mkdir -p $D
if [ ! -e $D/.seeded ]; then cp -rs $HOME/augur_far/. $D/ 2>/dev/null; touch $D/.seeded; fi
cd $D
tar xzf $HOME/far_gen_code.tgz
mkdir -p kern/gen results
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
