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

# Preflight: check a job's prerequisites once, here, rather than discovering a
# missing file identically in all 300 array tasks.
if [ "$JOB" = "bootstrap" ] && [ ! -f "$MIX_OUT/point_estimate.json" ]; then
  cat >&2 <<MSG
error: $MIX_OUT/point_estimate.json not found.

  The bootstrap warm-starts from the point estimate, so it runs after the
  multistart array:

    JID=\$(./hpc/submit.sh multistart)
    sbatch --dependency=afterok:\$JID --wrap="cd $MIX_ROOT && python run/collect_point_estimate.py"
    ./hpc/submit.sh bootstrap 50

  Nothing was submitted.
MSG
  exit 1
fi
if [ ! -f "$(python - <<'PY'
import sys; sys.path.insert(0,"src")
from tissue_dynamics.mixture.data import load_config
print(load_config()["data"]["table_parquet"])
PY
)" ]; then
  echo "error: the table parquet named in config.yaml is missing. Copy it and its" >&2
  echo "       .phi_names.json sidecar into data/ before submitting." >&2
  exit 1
fi

# multistart and bootstrap fit at the chosen penalty; with weight_ridge unset
# they would silently run at 0, which is the specification known to diverge.
# Catch that before burning a full-n array on it.
case "$JOB" in
  multistart|bootstrap)
    WR=$(python - <<'PY'
import sys; sys.path.insert(0,"src")
from tissue_dynamics.mixture.data import load_config
print(load_config()["model"].get("weight_ridge"))
PY
)
    if [ "$WR" = "None" ] || [ -z "$WR" ]; then
      echo "error: model.weight_ridge is unset in config.yaml, so '$JOB' would fit" >&2
      echo "       at weight_ridge=0 -- the setting that diverges. Run cv_penalty," >&2
      echo "       pick the smallest lambda that converged, set it in config.yaml," >&2
      echo "       and resubmit. Nothing was submitted." >&2
      exit 1
    fi
    echo "using weight_ridge=$WR from config.yaml" >&2
    ;;
esac

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
