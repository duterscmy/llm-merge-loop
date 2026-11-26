#!/usr/bin/env bash
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python -c 'import torch; assert torch.cuda.is_available(), "Run inside a CUDA GPU allocation"'
export HF_HOME="${HF_HOME:-$root/.cache/huggingface}"
export PIP_CACHE_DIR="$root/.cache/pip"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$root:$root/third_party/CoMe${PYTHONPATH:+:$PYTHONPATH}"
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
