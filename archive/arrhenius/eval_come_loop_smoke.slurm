#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=come-loop-smoke
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
BASE_MODEL="$PROJECT_DIR/models/Mistral-7B-v0.1"
MERGED_ROOT="$PROJECT_DIR/checkpoints/come-concat-smoke"
MERGED_MODEL="$MERGED_ROOT/Mistral-7B-v0.1_concat_merge_bi_31_worepeat"
MERGE_INFO="$MERGED_ROOT/Mistral-7B-v0.1_concat_merge_bi_info.json"
OUTPUT="$PROJECT_DIR/results/come-loop-smoke/ppl.json"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -m experiments.eval_come_loop_ppl \
  --base-model "$BASE_MODEL" \
  --merged-model "$MERGED_MODEL" \
  --merge-info "$MERGE_INFO" \
  --output "$OUTPUT" \
  --max-samples 8
