#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=verify-new-come
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
TARGET_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
MODEL_DIR="$PROJECT_DIR/models/Mistral-7B-v0.1"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$TARGET_ENV"
cd "$PROJECT_DIR"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CONDA_PREFIX=$CONDA_PREFIX"
command -v python
python --version
python -m pip check

MODEL_DIR="$MODEL_DIR" python - <<'PY'
import os
import platform
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = os.environ["MODEL_DIR"]
print("python", sys.version)
print("machine", platform.machine())
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

assert platform.machine() == "aarch64"
assert torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()
inputs = tokenizer("The capital of Sweden is", return_tensors="pt").to(model.device)
with torch.inference_mode():
    logits = model(**inputs, use_cache=False).logits
print("logits_shape", tuple(logits.shape))
print("parameters", sum(parameter.numel() for parameter in model.parameters()))
PY

echo "New nobackup come environment passed full verification."
