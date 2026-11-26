#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=come-mistral-9pct
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
BASE_MODEL="$PROJECT_DIR/models/Mistral-7B-v0.1"
SAVE_ROOT="$PROJECT_DIR/checkpoints/come-mistral-9pct"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$PROJECT_DIR/third_party/CoMe"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$SAVE_ROOT" "$PROJECT_DIR/results/come-mistral-9pct"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

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
  --min-max 0.97 \
  --seed 10 \
  --ppl-data wiki2 c4

echo "Paper-configured Mistral 9% CoMe merge completed."
