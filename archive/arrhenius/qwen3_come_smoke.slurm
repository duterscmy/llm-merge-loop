#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=qwen3-come-smoke
#SBATCH --output=logs/%x-%j.log

set -eo pipefail
root="$SLURM_SUBMIT_DIR"
cd "$root"
test "$(uname -m)" = aarch64
hostname
nvidia-smi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate log-prob
export HF_HOME="$root/.cache/huggingface"
export PIP_CACHE_DIR="$root/.cache/pip"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
deps="$root/.cache/qwen3-come-transformers-4.51.3"
if [[ ! -f "$deps/.ready" ]]; then
    python -m pip install --target "$deps" --no-deps 'transformers==4.51.3' 'tokenizers==0.21.4'
    touch "$deps/.ready"
fi
export PYTHONPATH="$deps:$root:$root/third_party/CoMe"
python -c 'import torch; from models_unit.qwen3.modeling_qwen3 import Qwen3ForCausalLM; assert torch.cuda.is_available()'
base="$root/models/Qwen3-8B-Base"
python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-8B-Base", local_dir=sys.argv[1], allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"], max_workers=4)' "$base"
save="$root/checkpoints/qwen3-come-smoke"
mkdir -p "$save"
cd "$root/third_party/CoMe"
python -u main.py --method concat_merge --model-name "$base" \
    --target-layers 35 --save-path "$save" --calibration-dataset wiki2 \
    --skip-method bi --nsamples 8 --merge-item 2 --wo-repeat --psu 1 --min-max 0.0 --seed 10
cd "$root"
python -m experiments.eval_come_loop_ppl --base-model "$base" \
    --merged-model "$save/Qwen3-8B-Base_concat_merge_bi_35_worepeat" \
    --merge-info "$save/Qwen3-8B-Base_concat_merge_bi_info.json" \
    --datasets wiki2 --variants base merged merge_aware --max-samples 2 \
    --output "$root/results/qwen3-come-smoke/ppl.json"
