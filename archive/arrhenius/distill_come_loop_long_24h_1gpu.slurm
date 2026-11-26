#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=come-kd1-24h
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
P=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$P"
export HF_HOME="$P/.cache/huggingface" HF_DATASETS_CACHE="$P/.cache/huggingface/datasets"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
hostname; uname -m; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
torchrun --standalone --nproc_per_node=1 -m experiments.distill_come_loop_long \
  --teacher-model "$P/models/Llama-2-7b-hf" \
  --student-model "$P/checkpoints/come-llama2-kl-endpoint-c4-10k" \
  --merge-info "$P/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_info.json" \
  --run-dir "$P/checkpoints/come-llama2-kl-endpoint-c4-24h-1gpu" \
  --train-files 'en/c4-train.*-of-01024.json.gz' --sequence-length 512 \
  --validation-blocks 64 --max-steps 1000000 --max-runtime-seconds 83700 \
  --eval-interval-seconds 1800 --learning-rate 5e-6 --warmup-steps 5000 \
  --weight-decay 0.0 --max-grad-norm 1.0 --temperature 1.0 \
  --endpoint-weight 0.1 --tokenize-batch-size 64 --log-every 100 --seed 10 --resume auto
