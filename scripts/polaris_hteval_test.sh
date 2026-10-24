#!/bin/bash
#SBATCH --job-name=augur-hteval-test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=augur-hteval-test-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
W="$HOME/augur-hteval-cpu"
mkdir -p "$W"
cd "$W"
rm -rf scripts model tests
cp -r "$HOME/augur/scripts" "$HOME/augur/model" "$HOME/augur/tests" "$HOME/augur/conftest.py" .
cp -r "$HOME/augur-hteval-src/scripts/." scripts/
cp -r "$HOME/augur-hteval-src/tests/." tests/
if [ -n "${OVERLAY:-1}" ]; then
  cp -r "$HOME/augur-hteval-src/extra/scripts/." scripts/
  cp -r "$HOME/augur-hteval-src/extra/model/." model/
fi
export PYTHONPATH=.
python -m pytest tests/test_tiered.py -q --tb=short 2>&1 | tail -80
