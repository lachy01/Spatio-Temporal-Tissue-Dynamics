#!/bin/bash
# Submit one job as a SLURM array sized to its task list.
#
#   ./hpc/submit.sh cv_penalty          # 40 tasks
#   ./hpc/submit.sh bootstrap 50        # 300 tasks, at most 50 running at once
#
# Prints the job id, so you can chain with --dependency=afterok:<id>.
set -euo pipefail
source "$(dirname "$0")/config.sh"

JOB="${1:?usage: submit.sh <job> [max_concurrent] [array_range]}"
THROTTLE="${2:-50}"
RANGE="${3:-}"          # e.g. "25-25" to run a single task as a real batch job

mix_activate_python
mix_set_threads
cd "$MIX_ROOT"
mkdir -p logs "$MIX_OUT"

N=$(python hpc/worker.py --job "$JOB" --index 0 --count)
ARRAY="${RANGE:-0-$((N-1))}%${THROTTLE}"
echo "job '$JOB': $N tasks, submitting --array=$ARRAY, ${MIX_CPUS} cpus / ${MIX_MEM}MB / ${MIX_TIME} each" >&2

# --account is omitted unless set: Spartan's own examples do not use one.
ACCT=()
[ -n "$MIX_ACCOUNT" ] && ACCT=(--account="$MIX_ACCOUNT")

sbatch "${ACCT[@]}" \
  --partition="$MIX_PARTITION" \
  --time="$MIX_TIME" --mem="$MIX_MEM" --cpus-per-task="$MIX_CPUS" \
  --array="$ARRAY" \
  --export=ALL,MIX_JOB="$JOB",MIX_OUT="$MIX_OUT",MIX_ROOT="$MIX_ROOT" \
  --parsable "$MIX_ROOT/hpc/submit.sbatch"
