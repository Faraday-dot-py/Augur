set -uo pipefail
export PYTHONUNBUFFERED=1
D=$HOME/augur_nfr
mkdir -p $D/kern $D/results $D/ckpt
cd $D
tar xzf $HOME/nfr_code.tgz
[ -f kern/E_frozen.pt ] || cp -L $HOME/augur_fkg/kern/E_frozen.pt kern/E_frozen.pt
export PYTHONPATH=.
