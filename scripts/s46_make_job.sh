#!/bin/bash
# usage: s46_make_job.sh NAME [VAR=value ...]  -> ~/polaris-mcp-files/s46/run_NAME.sh (template with env baked in)
set -euo pipefail
name="$1"; shift
d="$HOME/polaris-mcp-files/s46"
mkdir -p "$d"
src="$(dirname "$0")/polaris_scatter_s46.sh"
{
  sed -n '1,/^#SBATCH --output/p' "$src"
  echo "export NAME=$name"
  for kv in "$@"; do echo "export $kv"; done
  sed -n '/^#SBATCH --output/,$p' "$src" | tail -n +2
} > "$d/run_$name.sh"
echo "$d/run_$name.sh"
