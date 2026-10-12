#!/bin/bash
# MisTIC, one spatial tile per array task (run/mistic_run.py).
#
#   sbatch hpc/mistic.slurm                  # all 36 tiles
#   sbatch --array=7 hpc/mistic.slurm        # rerun one tile
#
# Needs data/mistic/inputs/ (run/mistic_prepare.py, or copied from a machine
# that ran it) and the MisTIC env (bash hpc/mistic_setup_env.sh). Each task
# writes data/mistic/tiles/tile_KK/; when all are done, on a login node:
#
#   source hpc/env.sh
#   python run/mistic_gather.py && python run/mistic_covariates.py
#
# Sizing, from a local fit on 4,608 cells (peak 4.3 GB; 38 s import, 64 s per
# epoch on 8 CPU threads): a tile holds ~21,000 cells, and memory grows
# linearly with cells, so ~20-25 GB peak; 64G leaves headroom. MisTIC stops
# early, before n_epochs = 50, once an epoch no longer improves its
# cell-type classifier. On CPU instead of GPU: delete the two GPU lines (or
# pass --partition=sapphire to sbatch and remove --gres) -- the code picks the
# device itself.

#SBATCH --job-name=mistic
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-35
#SBATCH -o "logs/slurm-%x.%A_%a.out"
#SBATCH -e "logs/slurm-%x.%A_%a.err"
set -euo pipefail

MIX_ROOT="${MIX_ROOT:-$SLURM_SUBMIT_DIR}"
source "${MIX_ROOT}/hpc/config.sh"
MISTIC_ENV="${MISTIC_ENV:-$MIX_ROOT/../venvs/mistic}"
if command -v module >/dev/null 2>&1; then
  module purge || true
fi

n="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$n" MKL_NUM_THREADS="$n" OPENBLAS_NUM_THREADS="$n" NUMBA_NUM_THREADS="$n"
export POLARS_MAX_THREADS="$n"

# With a GPU allocated, insist on it: MisTIC would otherwise fall back to the
# CPU without a word if torch cannot reach the device. Without one, use the CPU.
dev=cpu
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && dev=cuda

cd "$MIX_ROOT"
mkdir -p logs
echo "tile=${SLURM_ARRAY_TASK_ID} cores=${n} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} device=${dev}"
"$MISTIC_ENV/bin/python" run/mistic_run.py --tile "$SLURM_ARRAY_TASK_ID" --threads "$n" --device "$dev"
