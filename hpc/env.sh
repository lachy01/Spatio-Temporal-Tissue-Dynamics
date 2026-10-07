# Load the mixture's Python environment into the CURRENT shell, for running the
# collect and gather scripts on a Spartan login node:
#
#   source hpc/env.sh
#
# Source it rather than running it with `bash`: a script run with bash sets
# everything in a subshell and leaves your own shell unchanged. Loads the same
# modules and venv as the array jobs (hpc/config.sh), and keeps JAX to one CPU
# thread, as a login node expects.
_mix_here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$_mix_here/config.sh"
mix_activate_python
mix_set_threads
echo "python: $(command -v python) ($(python --version 2>&1)); MIX_ROOT=$MIX_ROOT"
