"""Run one SLURM array task and write its result. Kept deliberately thin.

    python hpc/worker.py --job bootstrap --index $SLURM_ARRAY_TASK_ID

Every task is independent and writes exactly one JSON file, so a partially
completed array is still usable and a failed task can be resubmitted alone.
Runs identically on a laptop, which is how it should be tested before submission.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tissue_dynamics.mixture.data import load_config          # noqa: E402
from tissue_dynamics.mixture.tasks import enumerate_tasks, run_task  # noqa: E402


FINGERPRINT_SECTIONS = ("data", "state", "model", "oxygen", "optim", "init", "inference")


MISTIC_KEYS = ("technical", "mistic_library_size", "mistic_m")


def fingerprint(cfg):
    """Hash of the configuration sections that change what a fit is.

    The MisTIC keys (2026-10-12) change nothing in an area fit, so under
    `technical: area` they are left out, and results fitted before they existed
    keep their fingerprints.
    """
    secs = {k: cfg.get(k) for k in FINGERPRINT_SECTIONS}
    model = dict(secs.get("model") or {})
    if model.get("technical", "area") == "area":
        for k in MISTIC_KEYS:
            model.pop(k, None)
    secs["model"] = model
    blob = json.dumps(secs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def stale_reasons(existing, task, cfg):
    """Why an existing result does not belong to the current configuration.

    Results written since 2026-10-07 carry a fingerprint of the configuration;
    older ones are checked on the settings they record. Any difference means
    the file came from a different model version and must not be reused.
    """
    mc = cfg["model"]
    out = []
    fp = existing.get("config_fingerprint")
    if fp is not None and fp != fingerprint(cfg):
        out.append("configuration fingerprint differs")
    want = {"rank": int(task.get("rank", mc["rank"])),
            "weight_ridge": float(task.get("lam", mc.get("weight_ridge") or 0.0)),
            "b_penalty": mc.get("b_penalty")}
    for k, v in want.items():
        if existing.get(k) is not None and existing[k] != v:
            out.append(f"{k}: file {existing[k]!r}, config {v!r}")
    width = 16 if cfg["state"].get("neighbourhood", "disc") == "split" else 8
    if existing.get("c") is not None and len(existing["c"]) != width:
        out.append(f"neighbour block width: file {len(existing['c'])}, config {width}")
    # results from before 2026-10-12 carry no "technical" and are area fits
    tech = mc.get("technical", "area")
    if existing.get("technical", "area") != tech and not existing.get("job", "").startswith(
            ("sim_", "division_sim")):
        out.append(f"technical: file {existing.get('technical', 'area')!r}, config {tech!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out", default=os.environ.get(
        "MIX_OUT", f"outputs/hpc_{os.environ['MIX_DATASET']}" if os.environ.get("MIX_DATASET")
        else "outputs/hpc"))
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
        # Idempotent so a partially-failed array can be resubmitted -- but only
        # for results from THIS model version. A result written under another
        # configuration was once skipped silently and reported as current (the
        # rank-2 Test 03 survived a rank-3 resubmission, 2026-10-07), so stale
        # files are moved aside and the task reruns.
        existing = json.load(open(path))
        reasons = stale_reasons(existing, tasks[a.index], cfg)
        if not reasons:
            print(f"already done with the current configuration, SKIPPING: {path}")
            return
        os.makedirs(os.path.join(d, "stale"), exist_ok=True)
        moved = os.path.join(d, "stale", f"{a.index:05d}.{int(time.time())}.json")
        os.replace(path, moved)
        print(f"STALE result moved to {moved} ({'; '.join(reasons)}); rerunning")

    t0 = time.time()
    res = run_task(tasks[a.index], cfg, out_dir=a.out)
    res["index"] = a.index
    res["elapsed"] = time.time() - t0
    res["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    # A fresh result must itself pass the stale check, or the code did not fit
    # what the configuration asks for: the rank-3 Test 03 once reran at rank 2
    # because the simulator ignored model.rank (2026-10-07). Keep the result
    # out of the job's directory and fail the task, so the error shows.
    reasons = stale_reasons(res, tasks[a.index], cfg)
    if reasons:
        bad = os.path.join(d, "mismatch")
        os.makedirs(bad, exist_ok=True)
        with open(os.path.join(bad, f"{a.index:05d}.json"), "w") as fh:
            json.dump(res, fh)
        raise SystemExit(f"result does not match the configuration ({'; '.join(reasons)}); "
                         f"saved to {bad}, not {path}")
    res["config_fingerprint"] = fingerprint(cfg)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)                       # atomic; no half-written files
    print(f"wrote {path} in {res['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
