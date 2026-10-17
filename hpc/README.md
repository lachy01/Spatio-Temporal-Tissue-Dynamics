# Running the mixture on Spartan

Follows the Spartan job-submission docs:
<https://rcs-knowledge-hub.atlassian.net/wiki/spaces/KB/pages/957775887/Submitting+jobs>

Every expensive step of this model is a set of **independent** fits — bootstrap
replicates, CV folds, multi-start restarts, simulation replicates — so each is a
job array: one array index, one fit, one small JSON file. Partial arrays are
usable, a failed task can be resubmitted alone, and re-running a completed index
is a no-op (the worker skips one whose output already exists).

## Task counts

| job | tasks | what it does |
|---|---:|---|
| `cv_penalty` | 40 | `weight_ridge` grid x 5 spatial folds |
| `cv_rank` | 20 | rank r in {1,2,3,5} x 5 folds (spec 6.11) |
| `multistart` | 20 | the 20 restarts of spec 4.4 |
| `bootstrap` | 300 | spatial block bootstrap (spec 7.1) |
| `sim_recovery` | 20 | Test 01 |
| `sim_null_bilinear` | 20 | Test 02 |
| `sim_single_component` | 20 | Test 03 |
| `sim_size_confound` | 20 | Test 07 |
| `division_sim` | 30 | division classifier, Stage 6: G1-G3 x 10 replicates |
| `division_sim_addendum` | 30 | division addendum 1, stage A5: the same draws plus the extended BASE-T grid, TWO-P/TWO-PNB and the section 6 metrics |
| `division_sim_t10` | 30 | Stage 6 rerun on the t10 label variant (BASE-T from the alpha_trim 0.2 truncated null at alpha_B 0.1085) |

## One-time setup

```bash
# copy the repo and data/mixture_cells_major.parquet (357 MB) to $MIX_ROOT
bash hpc/setup_env.sh        # builds .venv-spartan on Python/3.10.4 + foss/2022a
```

Then **smoke-test a single task before submitting the full array** — one bad
path otherwise fails identically 40 or 300 times. Submit it as a normal batch
job on a compute partition rather than running it interactively:

```bash
./hpc/submit.sh cv_penalty 1 25-25        # just array index 25, on sapphire
```

Index 25 is a lambda>0 fold, which is the regime that should fit; index 0 is
lambda=0, which is expected to record a failure rather than converge. Check the
log under `logs/` and the JSON under `outputs/hpc/cv_penalty/` before scaling up.

`sinteractive` is worth having for *debugging* — reading a traceback, poking at
the data — but not for real fits: the interactive partition is capped at 8 cores
and you have to keep the session alive. Everything below runs as batch work.

## Submitting

```bash
./hpc/submit.sh cv_penalty          # 40 tasks
./hpc/submit.sh multistart          # 20 tasks
./hpc/submit.sh bootstrap 50        # 300 tasks, <=50 concurrent
```

The second argument is the array throttle (`%N`), the optional third is an
explicit array range. Spartan documents the throttle as `--array=0-10%4`; 50
concurrent is polite on a shared partition.

`submit.sh` is a convenience — it reads the task count so the `--array` range is
never wrong by hand. If you would rather submit a plain script, `hpc/mixture.slurm`
is self-contained with all `#SBATCH` directives inline:

```bash
sbatch --export=ALL,MIX_JOB=cv_penalty --array=0-39 hpc/mixture.slurm
sbatch --export=ALL,MIX_JOB=bootstrap  --array=0-299%50 hpc/mixture.slurm
```

Array sizes: `cv_penalty` 0-39, `cv_rank` 0-19, `multistart` 0-19,
`bootstrap` 0-299, each `sim_*` 0-19.

### Order matters: bootstrap comes last

`bootstrap` warm-starts from `outputs/hpc/point_estimate.json`, which does not
exist until the `multistart` array has run **and** been collected. Submitting it
early fails every task. `./hpc/submit.sh bootstrap` now refuses to submit in
that case; submitting `hpc/mixture.slurm` directly does not check, so use the
chain:

```bash
JID=$(./hpc/submit.sh multistart)
sbatch --dependency=afterok:$JID \
       --wrap="cd $MIX_ROOT && python run/collect_point_estimate.py"
# then, once that has run:
./hpc/submit.sh bootstrap 50
```

The full order is: `cv_penalty` (choose the penalty) -> set `weight_ridge` in
`config.yaml` -> `cv_rank` -> `multistart` -> `collect_point_estimate.py` ->
`bootstrap`.

## Collecting

```bash
python hpc/gather.py cv_penalty     # held-out loglik per lambda, and the pick
python hpc/gather.py bootstrap      # percentile CIs
```

`gather.py` reports how many tasks are **missing** and how many failed to
converge rather than quietly averaging whatever is present — spec 9.13 forbids
dropping non-converged replicates without recording the rate.

## Division classifier addendum 1, stage A5

```bash
git pull                                   # simulation.py, twostep.py, tasks.py
./hpc/submit.sh division_sim_addendum      # 30 tasks, 2 h each by default
```

Needs, besides the table and `outputs/hpc/point_estimate.json`, the files the
Stage 6 job used: `outputs/division_classifier/{sim_inputs.npz, folds.parquet,
nb_params_nbc_full.json}` (`folds.parquet` was written on Spartan by
`hpc/folds.slurm`, so it is already there). BASE-O-state joins only if
`division_sim_state: true` in config.yaml, which waits on Lachy's A3 decision.
Runtime: Stage 6's `division_sim` tasks took 2-5 min on Spartan; locally, at
n = 20,000, the addendum's extra fits (four nulls, 19 LOG-T fits, TWO-P and
TWO-PNB) roughly doubled a replicate's time. At n = 100,000, expect about
10-15 min per task, so the 2 h default is ample. Copy back
`outputs/hpc/division_sim_addendum/` and run
`python run/division_addendum_a5_gather.py`.

## Stage 6 on the t10 label variant (2026-10-17)

```bash
git pull                                   # simulation.py, tasks.py (job division_sim_t10)
./hpc/submit.sh division_sim_t10           # 30 tasks; the same draws as division_sim
```

Its inputs, `outputs/division_classifier_t10/{sim_inputs.npz, folds.parquet,
nb_params_nbc_full.json}`, are committed, so `git pull` brings them (the folds, NB fits
and allocation model are identical to the original folder's). Each replicate also refits the
truncated null on its simulated counts (about a minute at n = 100,000), so tasks run
a little longer than `division_sim`'s 2-5 min. Copy back
`outputs/hpc/division_sim_t10/` and run
`DIV_VARIANT=t10 python run/division_stage6_gather.py`.

## A second section (ovarian, `MIX_DATASET=ovarian`)

`config_ovarian.yaml` layers the ovarian paths over `config.yaml`; every model
setting (rank 3, lambda 300 scaled to n, MisTIC technical terms, theta = 1) is
shared. With `MIX_DATASET` set, `MIX_OUT` defaults to `outputs/hpc_ovarian`.
Copy `data/ovarian/mixture_cells_ovarian.parquet` and its `.phi_names.json`
first.

```bash
git pull
export MIX_DATASET=ovarian MIX_TIME=0-12:00:00
JID=$(./hpc/submit.sh multistart)
sbatch --dependency=afterok:$JID --export=ALL --wrap="cd $PWD && source hpc/env.sh && python run/collect_point_estimate.py"
# once outputs/hpc_ovarian/point_estimate.json exists:
./hpc/submit.sh bootstrap 50
```

## Spartan-specific notes

- **Default wall time is 10 minutes.** `MIX_TIME` defaults to `0-02:00:00` here;
  every job needs an explicit `--time`.
- **Default partition is `sapphire`**, which is what `MIX_PARTITION` uses.
- **`--account` is omitted** unless `MIX_ACCOUNT` is set — Spartan's own example
  scripts carry no account flag. Set it if your project requires one.
- **`--mem` is in MB** without a suffix. Default here is 16000.
- **`mix_set_threads` is not optional.** Spartan's docs set
  `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}` for multithreaded jobs; JAX needs the
  same told to XLA, or each task grabs every core on the node and array tasks
  sharing a node thrash. This is the usual reason a cluster run comes out slower
  than a laptop.
- Don't run fits on a login node — submit them. `hpc/setup_env.sh` is the
  one thing that belongs on a login node, since it only installs packages.

## Sizing

Measured locally on 4 cores: a `cv_penalty` task (fits 4/5 of 678,885 cells,
2 restarts, 1500 iterations) took **24.5 minutes**. `multistart` and `bootstrap`
fit the *full* sample at up to 2000 iterations and are slower — in a local
rehearsal three restarts had still not converged at a 400-iteration cap, so
budget generously:

| job | suggested `--time` |
|---|---|
| `cv_penalty`, `cv_rank`, `sim_*` | `0-04:00:00` |
| `multistart`, `bootstrap` | `0-12:00:00` |

Peak memory is roughly 2-3 GB (the float64 design is ~380 MB, the rest JAX
workspace), so 16000 MB is comfortable. Confirm both from the first array with
`sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State` and trim.
