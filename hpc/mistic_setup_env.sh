#!/bin/bash
# One-time build of the MisTIC environment: a separate conda env with Python
# 3.10 and txMisTIC from PyPI. Kept apart from the mixture venv because
# txMisTIC pins torch 1.13, numpy<2, anndata<0.11 and scanpy 1.10.
#
#   bash hpc/mistic_setup_env.sh
#
# Run on a Spartan login node (package installation, not computation). The env
# is created as a prefix env at $MISTIC_ENV (default ../venvs/mistic, beside
# the mixture venv), so it lives on the project filesystem, not in $HOME.
set -euo pipefail
source "$(dirname "$0")/config.sh"
MISTIC_ENV="${MISTIC_ENV:-$MIX_ROOT/../venvs/mistic}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  # Spartan's conda module; the name differs between software stacks
  module load Miniforge3 2>/dev/null || module load Miniconda3 2>/dev/null \
    || module load Anaconda3 2>/dev/null || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found. Load your site's conda module (module avail conda)" >&2
  echo "       or install Miniforge, then rerun." >&2
  exit 1
fi

conda create -y -p "$MISTIC_ENV" python=3.10
"$MISTIC_ENV/bin/python" -m pip install --upgrade pip
"$MISTIC_ENV/bin/python" -m pip install txMisTIC

# macOS only: scipy's PyPI wheel fails to load on recent macOS (a dyld error
# in scipy.sparse.linalg's PROPACK); conda's build of the same version works.
if ! "$MISTIC_ENV/bin/python" -c "import scipy.sparse.linalg" 2>/dev/null; then
  echo "scipy wheel does not load here; replacing it with conda's build"
  "$MISTIC_ENV/bin/python" -m pip uninstall -y scipy
  conda install -y -p "$MISTIC_ENV" "scipy=1.15" "numpy<2"
fi

echo
echo "MisTIC env at $MISTIC_ENV"
"$MISTIC_ENV/bin/python" -m MisTIC --version
"$MISTIC_ENV/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| GPU visible here:', torch.cuda.is_available())"
