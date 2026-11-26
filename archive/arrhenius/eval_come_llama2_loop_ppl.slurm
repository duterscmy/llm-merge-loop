#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=llama2-paper-ppl
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
BASE_MODEL="$PROJECT_DIR/models/Llama-2-7b-hf"
MERGED_ROOT="$PROJECT_DIR/checkpoints/come-llama2-9pct-paper"
MERGED_MODEL="$MERGED_ROOT/Llama-2-7b-hf_concat_merge_bi_29_worepeat"
MERGE_INFO="$MERGED_ROOT/Llama-2-7b-hf_concat_merge_bi_info.json"
OUTPUT="$PROJECT_DIR/results/come-llama2-9pct-paper/loop_ppl.json"
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python -m experiments.eval_come_loop_ppl \
  --base-model "$BASE_MODEL" \
  --merged-model "$MERGED_MODEL" \
  --merge-info "$MERGE_INFO" \
  --datasets wiki2 c4 \
  --output "$OUTPUT" \
  --seed 10
