#!/bin/bash
# Rerun the long downstream analyses on the current production fit, in order.
#
#   caffeinate -i bash run/rerun_downstream.sh
#
# Written for the MisTIC switch (2026-10-13), after the quick analyses
# (estimands, density response, diagnostics, interaction support, the phase
# portrait and division Stage 1) had run. Each step logs to
# logs/rerun_mistic/<step>.log; logs/rerun_mistic/status.txt gets one line per
# step. The division stages follow their dependencies: a failure in Stage 2, 3
# or 4 (a V-test stops a stage with a non-zero exit) skips everything that
# reads its output; Stages 5, 6 (local part) and 7 are independent of each
# other. Stage 5 also needs spatial_analysis's LISA bins. The NB-baseline
# notebook runs last, with errors kept in the notebook so one failing cell
# does not stop the rest.
set -u
cd "$(dirname "$0")/.."
LOG=logs/rerun_mistic
mkdir -p "$LOG"
STATUS="$LOG/status.txt"
echo "started $(date '+%Y-%m-%d %H:%M:%S') at git $(git rev-parse --short HEAD)" >> "$STATUS"

# exit codes go to $LOG/<step>.rc (macOS ships bash 3.2: no associative arrays)
step() {                      # step <name> <command...>
  local name=$1; shift
  local t0 rc
  t0=$(date +%s)
  echo "$(date '+%H:%M:%S') start $name" >> "$STATUS"
  "$@" > "$LOG/$name.log" 2>&1
  rc=$?
  echo "$rc" > "$LOG/$name.rc"
  echo "$(date '+%H:%M:%S') $name exit $rc ($(( ($(date +%s) - t0) / 60 )) min)" >> "$STATUS"
}
skip() {
  echo "skipped" > "$LOG/$1.rc"
  echo "$(date '+%H:%M:%S') $1 skipped: $2" >> "$STATUS"
}
ok() { [ "$(cat "$LOG/$1.rc" 2>/dev/null)" = "0" ]; }

step spatial_analysis python run/spatial_analysis.py
step compare_nb_mixture python run/compare_nb_mixture.py

step division_stage2 python run/division_stage2_nb.py
if ok division_stage2; then step division_stage3 python run/division_stage3_oof.py
else skip division_stage3 "Stage 2 failed"; fi
if ok division_stage3; then step division_stage4 python run/division_stage4_scores.py
else skip division_stage4 "Stage 3 did not complete"; fi
if ok division_stage4; then
  if ok spatial_analysis; then step division_stage5 python run/division_stage5_biology.py
  else skip division_stage5 "spatial_analysis failed (Stage 5 reads its LISA bins)"; fi
  step division_stage6_prepare python run/division_stage6_prepare.py
  step division_stage7 python run/division_stage7_sensitivity.py
else
  for s in division_stage5 division_stage6_prepare division_stage7; do
    skip $s "Stage 4 did not complete"
  done
fi

step nb_baseline_notebook jupyter nbconvert --to notebook --execute --inplace \
  --allow-errors --ExecutePreprocessor.timeout=-1 notebooks/nb_baseline.ipynb

echo "finished $(date '+%Y-%m-%d %H:%M:%S')" >> "$STATUS"
