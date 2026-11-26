#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=aggr-10k-ppl
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
P=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$P"
export HF_HOME="$P/.cache/huggingface" HF_DATASETS_CACHE="$P/.cache/huggingface/datasets" TOKENIZERS_PARALLELISM=false
python -m experiments.eval_come_loop_ppl \
  --base-model "$P/models/Llama-2-7b-hf" \
  --merged-model "$P/checkpoints/come-llama2-aggressive-kl-hidden-c4-10k" \
  --merge-info "$P/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_info.json" \
  --datasets wiki2 c4 --variants merged merge_aware \
  --output "$P/results/come-llama2-aggressive-kl-hidden-c4-10k/loop_ppl.json" --seed 10
