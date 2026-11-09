#!/bin/bash
# One-time environment setup for the Slurm scripts in this folder.
# Run it once on a login node (compute nodes may have no internet access):
#
#   bash slurm/setup_env.sh            # creates $HOME/venvs/modelsbench
#   VENV=/path/to/venv bash slurm/setup_env.sh
#
# The PyTorch module in releases/2024a (PyTorch/2.6.0-foss-2024a) has no
# CUDA suffix, i.e. it is a CPU build, so the GPU jobs use a venv with the
# official CUDA wheels instead.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/modelsbench}"

module purge
module load releases/2024a
module load Python/3.12.3-GCCcore-13.3.0

python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy imageio tqdm matplotlib

python - <<'PY'
import torch
print("torch", torch.__version__, "CUDA build", torch.version.cuda)
PY
echo "Environment ready: $VENV"
