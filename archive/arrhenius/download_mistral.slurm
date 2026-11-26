#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=mistral-download
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
MODEL_DIR="$PROJECT_DIR/models/Mistral-7B-v0.1"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

huggingface-cli download mistralai/Mistral-7B-v0.1 \
  --include "*.safetensors" "*.json" "*.model" "README.md" ".gitattributes" \
  --local-dir "$MODEL_DIR"

cd "$PROJECT_DIR/third_party/CoMe"
MODEL_DIR="$MODEL_DIR" python - <<'PY'
import os

import torch

from utils.model_utils import get_llmhander

model_dir = os.environ["MODEL_DIR"]
handler = get_llmhander(model_dir)
handler.model.eval()

inputs = handler.tokenizer("The capital of Sweden is", return_tensors="pt")
inputs = {name: value.to(handler.model.device) for name, value in inputs.items()}
with torch.inference_mode():
    output = handler.model(**inputs, use_cache=False)

print("model_path", model_dir)
print("architecture", handler.model.config.architectures[0])
print("physical_layers", len(handler.layers))
print("parameters", sum(parameter.numel() for parameter in handler.model.parameters()))
print("logits_shape", tuple(output.logits.shape))
print("model_device", handler.model.device)
print("cuda_allocated_gib", round(torch.cuda.memory_allocated() / 1024**3, 3))
PY

echo "Model downloaded and verified at $MODEL_DIR"
