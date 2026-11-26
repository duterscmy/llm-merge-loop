#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --job-name=come-24hbest-tasks
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
P=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$P"
export HF_HOME="$P/.cache/huggingface" HF_DATASETS_CACHE="$P/.cache/huggingface/datasets" TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$P/evaluate:$P${PYTHONPATH:+:$PYTHONPATH}"
python -m experiments.eval_come_loop_tasks \
  --base-model "$P/models/Llama-2-7b-hf" \
  --merged-model "$P/checkpoints/come-llama2-kl-endpoint-c4-10k" \
  --trainer-state "$P/checkpoints/come-llama2-kl-endpoint-c4-24h-1gpu/eval-snapshots/best-step-201400.pt" \
  --merge-info "$P/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_info.json" \
  --variant merge_aware \
  --tasks arc_easy arc_challenge hellaswag openbookqa piqa winogrande \
  --num-fewshot 0 --batch-size 8 --bootstrap-iters 1000 \
  --output "$P/results/come-llama2-kl-endpoint-c4-24h-best-step-201400/tasks-zero-shot-merge_aware.json"
