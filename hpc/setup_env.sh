#!/bin/bash
# One-time environment build on Spartan. Run on a LOGIN node (this is package
# installation, not computation) or in an `sinteractive` session.
#
#   bash hpc/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

module purge
module load foss/2022a
module load Python/3.10.4

python3 -m venv "$MIX_VENV"
# shellcheck disable=SC1091
source "$MIX_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install "jax[cpu]" optax numpy scipy pandas pyarrow patsy scikit-learn

echo
echo "environment built at $MIX_VENV"
python -c "import jax, optax; print('jax', jax.__version__, 'optax', optax.__version__, jax.devices())"
