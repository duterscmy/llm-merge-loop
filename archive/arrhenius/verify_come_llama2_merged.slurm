#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=verify-come-llama2
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
MODEL_DIR="$PROJECT_DIR/checkpoints/come-llama2-9pct-paper/Llama-2-7b-hf_concat_merge_bi_29_worepeat"
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - "$MODEL_DIR" <<'PY'
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map={"": 0},
)

assert model.config.num_hidden_layers == 29
parameter_count = sum(parameter.numel() for parameter in model.parameters())
assert parameter_count == 6_131_265_536, parameter_count

inputs = tokenizer("CoMe distillation smoke test.", return_tensors="pt").to("cuda")
with torch.inference_mode():
    outputs = model(**inputs, use_cache=False)

assert outputs.logits.shape[:2] == inputs.input_ids.shape
assert torch.isfinite(outputs.logits).all()

print(json.dumps({
    "model_dir": model_dir,
    "architecture": model.__class__.__name__,
    "physical_layers": model.config.num_hidden_layers,
    "parameters": parameter_count,
    "dtype": str(next(model.parameters()).dtype),
    "logits_shape": list(outputs.logits.shape),
    "finite_logits": True,
}, indent=2))
PY
