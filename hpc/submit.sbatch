#!/bin/bash
# Array-task runner. One SLURM array index == one independent fit.
# Submitted by hpc/submit.sh, which sizes the array and sets MIX_JOB.
#SBATCH --job-name=mixture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o "logs/slurm-%x.%A_%a.out"
#SBATCH -e "logs/slurm-%x.%A_%a.err"
set -euo pipefail

source "${MIX_ROOT}/hpc/config.sh"
mix_activate_python
mix_set_threads

# Spartan's documented pattern for multithreaded jobs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "task ${SLURM_ARRAY_TASK_ID} of job ${MIX_JOB} on ${SLURM_JOB_CPUS_PER_NODE} cores"

cd "$MIX_ROOT"
python hpc/worker.py --job "$MIX_JOB" --index "$SLURM_ARRAY_TASK_ID" --out "$MIX_OUT"
