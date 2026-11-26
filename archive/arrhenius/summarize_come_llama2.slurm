#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=llama2-paper-summary
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
cd "$PROJECT_DIR"

python3 -m experiments.summarize_come_llama2 \
  --result-dir results/come-llama2-9pct-paper \
  --output results/come-llama2-9pct-paper/benchmark_summary.json
