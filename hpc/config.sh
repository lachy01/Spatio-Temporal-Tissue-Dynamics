#!/bin/bash
# Spartan (University of Melbourne) settings for the mixture array jobs.
# Everything cluster-specific lives here; the sbatch scripts read this and
# nothing else.
#
# Docs followed:
#   https://rcs-knowledge-hub.atlassian.net/wiki/spaces/KB/pages/957775887/Submitting+jobs

# Spartan's default partition is `sapphire`. `interactive` gives instant access
# to <=8 cores / 96 GB for up to 2 days and is the right place to smoke-test a
# single task before submitting an array.
export MIX_PARTITION="${MIX_PARTITION:-sapphire}"

# Spartan's own examples carry no --account, so it is left empty by default and
# simply omitted from sbatch. Set it if your project requires one.
export MIX_ACCOUNT="${MIX_ACCOUNT:-}"

# Spartan's default wall time is 10 MINUTES, which every one of these jobs would
# blow through, so always set it explicitly. Format is D-HH:MM:SS.
export MIX_TIME="${MIX_TIME:-0-02:00:00}"

# --mem is in MB unless suffixed. Spartan's default is 4000MB x cores, which is
# already enough; 16000 gives headroom for the float64 design (~380 MB) plus JAX
# workspace at full n.
export MIX_MEM="${MIX_MEM:-16000}"
export MIX_CPUS="${MIX_CPUS:-4}"

export MIX_ROOT="${MIX_ROOT:-$HOME/Spatio-Temporal-Tissue-Dynamics}"
export MIX_OUT="${MIX_OUT:-$MIX_ROOT/outputs/hpc}"
export MIX_VENV="${MIX_VENV:-$MIX_ROOT/.venv-spartan}"

# Spartan uses lmod. `module purge` first is the documented habit; Python/3.10.4
# is the default under the foss/2022a toolchain. JAX and optax are not site
# modules, so they live in a venv built on top of it (see hpc/setup_env.sh).
mix_activate_python() {
  module purge
  module load foss/2022a
  module load Python/3.10.4
  # shellcheck disable=SC1091
  [ -f "$MIX_VENV/bin/activate" ] && source "$MIX_VENV/bin/activate"
}

# Keep JAX inside its allocation. Spartan's docs set OMP_NUM_THREADS from
# SLURM_CPUS_PER_TASK for multithreaded jobs; JAX needs the same told to XLA, or
# each array task grabs every core on the node and tasks sharing a node thrash.
mix_set_threads() {
  local n="${SLURM_CPUS_PER_TASK:-1}"
  export OMP_NUM_THREADS="${n}"
  export MKL_NUM_THREADS="${n}"
  export OPENBLAS_NUM_THREADS="${n}"
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${n}"
  export JAX_PLATFORMS=cpu
  export PYTHONPATH="$MIX_ROOT/src:${PYTHONPATH:-}"
}
