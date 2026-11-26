#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=endpoint-kd-smoke
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
TEACHER_MODEL="$PROJECT_DIR/models/Llama-2-7b-hf"
STUDENT_MODEL="$PROJECT_DIR/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_29_worepeat"
MERGE_INFO="$PROJECT_DIR/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_info.json"
METRICS_OUTPUT="$PROJECT_DIR/results/come-llama2-endpoint-kd-pilot/smoke_train_metrics.json"
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
  --output-dir "$PROJECT_DIR/checkpoints/come-llama2-endpoint-kd-smoke-unused" \
  --metrics-output "$METRICS_OUTPUT" \
  --sequence-length 128 \
  --max-steps 1 \
  --validation-blocks 1 \
  --learning-rate 1e-5 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --seed 10 \
  --skip-save
