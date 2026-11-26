#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=come-harness-smoke
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
BASE_MODEL="$PROJECT_DIR/models/Mistral-7B-v0.1"
MERGED_ROOT="$PROJECT_DIR/checkpoints/come-mistral-9pct"
MERGED_MODEL="$MERGED_ROOT/Mistral-7B-v0.1_concat_merge_bi_29_worepeat"
MERGE_INFO="$MERGED_ROOT/Mistral-7B-v0.1_concat_merge_bi_info.json"
OUTPUT="$PROJECT_DIR/results/harness-smoke/piqa-merge-aware.json"

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
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
python -c 'import lm_eval, sacrebleu, sqlitedict; print("evaluation imports ok")'
python -c 'from datasets import load_dataset; ds = load_dataset("lighteval/piqa", split="validation"); print("piqa validation", len(ds), ds.column_names)'

python -m experiments.eval_come_loop_tasks \
  --base-model "$BASE_MODEL" \
  --merged-model "$MERGED_MODEL" \
  --merge-info "$MERGE_INFO" \
  --variant merge_aware \
  --tasks piqa \
  --num-fewshot 0 \
  --batch-size 1 \
  --limit 10 \
  --bootstrap-iters 100 \
  --output "$OUTPUT"
