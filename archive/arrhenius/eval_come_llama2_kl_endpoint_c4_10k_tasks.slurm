#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=kl-endpoint-10k-tasks
#SBATCH --output=logs/%x-%A_%a.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
BASE_MODEL="$PROJECT_DIR/models/Llama-2-7b-hf"
DISTILLED_MODEL="$PROJECT_DIR/checkpoints/come-llama2-kl-endpoint-c4-10k"
MERGE_INFO="$PROJECT_DIR/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_info.json"
VARIANTS=(merged merge_aware)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="$PROJECT_DIR/results/come-llama2-kl-endpoint-c4-10k/tasks-zero-shot-$VARIANT.json"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR/evaluate:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "variant=$VARIANT"

python -m experiments.eval_come_loop_tasks \
  --base-model "$BASE_MODEL" \
  --merged-model "$DISTILLED_MODEL" \
  --merge-info "$MERGE_INFO" \
  --variant "$VARIANT" \
  --tasks arc_easy arc_challenge hellaswag openbookqa piqa winogrande \
  --num-fewshot 0 \
  --batch-size 8 \
  --bootstrap-iters 1000 \
  --output "$OUTPUT"
