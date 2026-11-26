#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=come-storage-config
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
CACHE_ROOT=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/cache

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CACHE_ROOT/pip"
install -m 0644 \
  "$PROJECT_DIR/env/come-activate-storage.sh" \
  "$CONDA_PREFIX/etc/conda/activate.d/come-storage.sh"

conda deactivate
conda activate "$CONDA_ENV"
test "$PIP_CACHE_DIR" = "$CACHE_ROOT/pip"
python -m pip cache dir
