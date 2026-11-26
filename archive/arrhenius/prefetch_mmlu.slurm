#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=come-mmlu-cache
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTHONPATH="$PROJECT_DIR/evaluate:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

hostname
uname -m

python -m experiments.prefetch_mmlu \
  --cache-dir "$HF_DATASETS_CACHE" \
  --marker "$HF_DATASETS_CACHE/cais___mmlu/.complete-57"
