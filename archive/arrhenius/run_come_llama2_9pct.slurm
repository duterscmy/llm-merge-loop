#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=come-llama2-paper
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
BASE_MODEL="$PROJECT_DIR/models/Llama-2-7b-hf"
SAVE_ROOT="$PROJECT_DIR/checkpoints/come-llama2-9pct-paper"
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR/third_party/CoMe"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

test -f "$BASE_MODEL/.conversion-complete"
mkdir -p "$SAVE_ROOT" "$PROJECT_DIR/results/come-llama2-9pct-paper"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python main.py \
  --method concat_merge \
  --model-name "$BASE_MODEL" \
  --target-layers 29 \
  --save-path "$SAVE_ROOT" \
  --calibration-dataset wiki2 \
  --skip-method bi \
  --nsamples 256 \
  --merge-item 2 \
  --wo-repeat \
  --psu 1 \
  --min-max 0.0 \
  --seed 10 \
  --ppl-data wiki2 c4
