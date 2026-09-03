#!/bin/bash
# Self-contained Spartan submission for one mixture job.
#
#   sbatch --export=ALL,MIX_JOB=cv_penalty --array=0-39 hpc/mixture.slurm
#
# or edit MIX_JOB and --array below and just `sbatch hpc/mixture.slurm`.
# Array sizes: cv_penalty 0-39, cv_rank 0-19, multistart 0-19, bootstrap 0-299,
# sim_* 0-19.  `./hpc/submit.sh <job>` sets the range for you from the task list.

#SBATCH --job-name=mixture
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=0-04:00:00
#SBATCH --array=0-39
#SBATCH -o "logs/slurm-%x.%A_%a.out"
#SBATCH -e "logs/slurm-%x.%A_%a.err"
set -euo pipefail

# Job to run; override with --export=ALL,MIX_JOB=...
MIX_JOB="${MIX_JOB:-cv_penalty}"
MIX_ROOT="${MIX_ROOT:-$SLURM_SUBMIT_DIR}"

source "${MIX_ROOT}/hpc/config.sh"
mix_activate_python
mix_set_threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$MIX_ROOT"
mkdir -p logs "$MIX_OUT"
echo "job=${MIX_JOB} task=${SLURM_ARRAY_TASK_ID} cores=${SLURM_JOB_CPUS_PER_NODE} host=$(hostname)"
python hpc/worker.py --job "$MIX_JOB" --index "$SLURM_ARRAY_TASK_ID" --out "$MIX_OUT"
