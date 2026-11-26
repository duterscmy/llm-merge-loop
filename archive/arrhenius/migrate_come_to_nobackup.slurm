#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=migrate-come
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
CONDA_STORE=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda
SOURCE_ENV="$HOME/miniconda3/envs/come"
TARGET_ENV="$CONDA_STORE/envs/come"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd "$PROJECT_DIR"

mkdir -p "$CONDA_STORE/envs" "$CONDA_STORE/pkgs" "$CONDA_STORE/tmp"
export CONDA_PKGS_DIRS="$CONDA_STORE/pkgs"
export TMPDIR="$CONDA_STORE/tmp"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SOURCE_ENV=$SOURCE_ENV"
echo "TARGET_ENV=$TARGET_ENV"

if [[ -e "$TARGET_ENV" ]]; then
  echo "Refusing to overwrite existing target: $TARGET_ENV" >&2
  exit 2
fi

conda create --prefix "$TARGET_ENV" --clone "$SOURCE_ENV" --yes

conda activate "$TARGET_ENV"
python - <<'PY'
import platform
import sys

import torch
import transformers

print("python", sys.version)
print("machine", platform.machine())
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert platform.machine() == "aarch64"
assert torch.cuda.is_available()
PY
conda deactivate

conda list --prefix "$TARGET_ENV" --explicit > "$PROJECT_DIR/env/come-nobackup-explicit.txt"
du -sh "$SOURCE_ENV" "$TARGET_ENV"
echo "Clone and GPU verification completed; the source environment has not been removed."
