#!/bin/bash
#SBATCH --job-name=augur-hteval-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=02:00:00
#SBATCH --output=augur-hteval-data-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
W="$HOME/augur-hteval"
mkdir -p "$W/results/tiered"
cd "$W"
rm -rf scripts model tests
cp -r "$HOME/augur/scripts" "$HOME/augur/model" "$HOME/augur/tests" "$HOME/augur/conftest.py" .
cp -r "$HOME/augur-hteval-src/scripts/." scripts/
cp -r "$HOME/augur-hteval-src/tests/." tests/
export PYTHONPATH=.
OUT=results/tiered/data
python scripts/tiered_gen.py tier0 --kinds all --shards 2 --scenes 64 --steps 32 --seed 4738 --resume --out $OUT
python scripts/tiered_gen.py tier0 --kinds grav,lj_liquid --shards 1 --scenes 64 --steps 32 --seed 4738 --noise-pos 0.02 --resume --out $OUT/noise
python scripts/tiered_gen.py harvest --shards 2 --windows 32 --steps 16 --seed 4738 --resume --out $OUT
python scripts/tiered_gen.py tier1 --shards 2 --scenes 4 --span 128 --steps 8 --lower range8 --states truth --seed 4738 --resume --out $OUT
python scripts/tiered_gen.py tier1 --shards 1 --scenes 2 --span 256 --n-range 16384 20000 --steps 4 --lower range8 --states truth --seed 4739 --resume --out $OUT/big
du -sh $OUT
