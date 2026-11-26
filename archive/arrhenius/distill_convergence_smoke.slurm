#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=kd-curve-smoke
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

P=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$P"

export HF_HOME="$P/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python -m experiments.distill_come_loop \
  --teacher-model "$P/models/Llama-2-7b-hf" \
  --student-model "$P/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_29_worepeat" \
  --merge-info "$P/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_info.json" \
  --output-dir "$P/checkpoints/convergence-smoke-unused" \
  --metrics-output "$P/results/distillation-convergence/smoke.json" \
  --dataset c4 --sequence-length 128 --max-steps 2 --validation-blocks 1 \
  --learning-rate 1e-5 --warmup-steps 0 --weight-decay 0.0 --max-grad-norm 1.0 \
  --objective forward_kl_endpoint --temperature 1.0 \
  --endpoint-weight-mode dynamic --endpoint-weight 1.0 \
  --hidden-target-mode endpoint --log-every 1 --eval-every 1 --seed 10 --skip-save
