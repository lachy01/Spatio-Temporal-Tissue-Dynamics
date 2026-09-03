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
# Only what the mixture path imports. The package resolves its exports lazily,
# so torch / scanpy / anndata / matplotlib are NOT needed here -- that is ~2 GB
# of torch avoided. Verified by importing tissue_dynamics.mixture.tasks and
# checking nothing heavy appears in sys.modules.
python -m pip install "jax[cpu]" optax numpy scipy pandas pyarrow patsy scikit-learn pyyaml

echo
echo "environment built at $MIX_VENV"
python -c "import jax, optax; print('jax', jax.__version__, 'optax', optax.__version__, jax.devices())"
PYTHONPATH="$MIX_ROOT/src" python -c "
import tissue_dynamics.mixture.tasks as t
from tissue_dynamics.mixture.data import load_config
print('mixture imports cleanly;', len(t.enumerate_tasks('cv_penalty', load_config())), 'cv_penalty tasks')"
