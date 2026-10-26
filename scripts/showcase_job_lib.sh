#!/bin/bash
# sourced by polaris_showcase_*.sh. Sets up the code dir and runs one command resumably.
# showcase_run TAG DONE_FILE cmd...: runs cmd in the background so SIGTERM/USR1 (preemption, or --signal before the time limit) is forwarded;
# the python side writes a full-state checkpoint and exits 0 on SIGTERM; if DONE_FILE is missing afterwards the job requeues itself (max 3 restarts).
showcase_setup() {
  export PYTHONUNBUFFERED=1
  mkdir -p "$HOME/augur-showcase"
  if [ ! -d "$HOME/augur-showcase/scripts" ]; then tar -xzf "$HOME/showcase_code.tgz" -C "$HOME/augur-showcase"; fi
  cd "$HOME/augur-showcase"
  export PYTHONPATH=.
  mkdir -p results videos ckpt
  mkdir -p checkpoints/scatter_bh
  [ -e checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt ] || ln -s "$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt" checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
}
showcase_run() {
  local done_file="$1"; shift
  if [ -e "$done_file" ]; then echo "[$(date -Iseconds)] $done_file exists, nothing to do"; return 0; fi
  echo "[$(date -Iseconds)] start: $*"
  "$@" &
  local pid=$!
  trap 'echo "[$(date -Iseconds)] signal, forwarding TERM"; kill -TERM $pid 2>/dev/null' TERM USR1
  wait $pid; local rc=$?
  while kill -0 $pid 2>/dev/null; do wait $pid; rc=$?; done
  trap - TERM USR1
  echo "[$(date -Iseconds)] exit code $rc"
  if [ ! -e "$done_file" ]; then
    echo "[$(date -Iseconds)] not complete"
    if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ] && [ "$rc" -eq 0 ]; then scontrol requeue "$SLURM_JOB_ID"; fi
    return 1
  fi
  return 0
}
