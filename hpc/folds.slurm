#!/bin/bash
# Write the division comparison's stored fold map in the mixture's Spartan
# environment (see run/division_make_folds.py for why it must run here).
#
#   sbatch hpc/folds.slurm
#
# Then copy back outputs/division_classifier/folds.parquet and folds_meta.json.

#SBATCH --job-name=folds
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=0-00:20:00
#SBATCH -o "logs/slurm-%x.%j.out"
#SBATCH -e "logs/slurm-%x.%j.err"
set -euo pipefail

MIX_ROOT="${MIX_ROOT:-$SLURM_SUBMIT_DIR}"
source "${MIX_ROOT}/hpc/config.sh"
mix_activate_python
mix_set_threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$MIX_ROOT"
mkdir -p logs
python run/division_make_folds.py
