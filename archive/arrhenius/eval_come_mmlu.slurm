#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-4%2
#SBATCH --job-name=come-mmlu
#SBATCH --output=logs/%x-%A_%a.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
BASE_MODEL="$PROJECT_DIR/models/Mistral-7B-v0.1"
MERGED_ROOT="$PROJECT_DIR/checkpoints/come-mistral-9pct"
MERGED_MODEL="$MERGED_ROOT/Mistral-7B-v0.1_concat_merge_bi_29_worepeat"
MERGE_INFO="$MERGED_ROOT/Mistral-7B-v0.1_concat_merge_bi_info.json"
VARIANTS=(base merged merge_aware uniform random)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="$PROJECT_DIR/results/come-mistral-9pct/mmlu-5shot-$VARIANT.json"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR/evaluate:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

test -f "$HF_DATASETS_CACHE/cais___mmlu/.complete-57"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "variant=$VARIANT"

python -m experiments.eval_come_loop_tasks \
  --base-model "$BASE_MODEL" \
  --merged-model "$MERGED_MODEL" \
  --merge-info "$MERGE_INFO" \
  --variant "$VARIANT" \
  --tasks mmlu \
  --num-fewshot 5 \
  --batch-size 8 \
  --bootstrap-iters 1000 \
  --output "$OUTPUT"
