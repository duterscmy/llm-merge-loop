#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=endpoint-c4-1k
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
TEACHER_MODEL="$PROJECT_DIR/models/Llama-2-7b-hf"
STUDENT_MODEL="$PROJECT_DIR/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_29_worepeat"
MERGE_INFO="$PROJECT_DIR/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_info.json"
OUTPUT_DIR="$PROJECT_DIR/checkpoints/come-llama2-endpoint-kd-c4-1k"
METRICS_OUTPUT="$PROJECT_DIR/results/come-llama2-endpoint-kd-c4-1k/train_metrics.json"
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

python -m experiments.distill_come_loop \
  --teacher-model "$TEACHER_MODEL" \
  --student-model "$STUDENT_MODEL" \
  --merge-info "$MERGE_INFO" \
  --output-dir "$OUTPUT_DIR" \
  --metrics-output "$METRICS_OUTPUT" \
  --dataset c4 \
  --sequence-length 512 \
  --max-steps 1000 \
  --validation-blocks 32 \
  --learning-rate 1e-5 \
  --warmup-steps 50 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --seed 10
