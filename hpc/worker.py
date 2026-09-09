"""Run one SLURM array task and write its result. Kept deliberately thin.

    python hpc/worker.py --job bootstrap --index $SLURM_ARRAY_TASK_ID

Every task is independent and writes exactly one JSON file, so a partially
completed array is still usable and a failed task can be resubmitted alone.
Runs identically on a laptop, which is how it should be tested before submission.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tissue_dynamics.mixture.data import load_config          # noqa: E402
from tissue_dynamics.mixture.tasks import enumerate_tasks, run_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out", default=os.environ.get("MIX_OUT", "outputs/hpc"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--count", action="store_true",
                    help="print the number of tasks and exit (for --array=0-N)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    tasks = enumerate_tasks(a.job, cfg)
    if a.count:
        print(len(tasks))
        return
    if not 0 <= a.index < len(tasks):
        raise SystemExit(f"index {a.index} out of range for job {a.job} "
                         f"({len(tasks)} tasks)")

    d = os.path.join(a.out, a.job)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{a.index:05d}.json")
    if os.path.exists(path):
        # Idempotent so a partially-failed array can be resubmitted. The hazard
        # is that results from a DIFFERENT model version survive here and get
        # mixed with new ones -- which happened once and silently corrupted a
        # Test 13 sweep. Warn loudly enough to be seen in the log.
        print(f"already done, SKIPPING: {path}")
        print("  ** if the model has changed since this file was written, delete "
              "the directory and resubmit: stale and current fits do not mix. **")
        return

    t0 = time.time()
    res = run_task(tasks[a.index], cfg, out_dir=a.out)
    res["index"] = a.index
    res["elapsed"] = time.time() - t0
    res["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)                       # atomic; no half-written files
    print(f"wrote {path} in {res['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
