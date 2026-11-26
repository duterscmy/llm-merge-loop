#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=come-concat-smoke
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
BASE_MODEL="$PROJECT_DIR/models/Mistral-7B-v0.1"
SAVE_ROOT="$PROJECT_DIR/checkpoints/come-concat-smoke"
PRUNED_MODEL="$SAVE_ROOT/Mistral-7B-v0.1_concat_merge_bi_31_worepeat"
RESULT_DIR="$PROJECT_DIR/results/come-concat-smoke"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$SAVE_ROOT" "$RESULT_DIR"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "BASE_MODEL=$BASE_MODEL"
echo "PRUNED_MODEL=$PRUNED_MODEL"

cd "$PROJECT_DIR/third_party/CoMe"
python main.py \
  --method concat_merge \
  --model-name "$BASE_MODEL" \
  --target-layers 31 \
  --save-path "$SAVE_ROOT" \
  --calibration-dataset wiki2 \
  --skip-method bi \
  --nsamples 8 \
  --merge-item 2 \
  --wo-repeat \
  --seed 10

PROJECT_DIR="$PROJECT_DIR" \
BASE_MODEL="$BASE_MODEL" \
PRUNED_MODEL="$PRUNED_MODEL" \
RESULT_DIR="$RESULT_DIR" \
python - <<'PY'
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_dir = Path(os.environ["PROJECT_DIR"])
base_model = Path(os.environ["BASE_MODEL"])
pruned_model = Path(os.environ["PRUNED_MODEL"])
result_dir = Path(os.environ["RESULT_DIR"])

model = AutoModelForCausalLM.from_pretrained(
    pruned_model,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(pruned_model)
model.eval()

prompt = "The capital of Sweden is"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {name: value.to(model.device) for name, value in inputs.items()}

with torch.inference_mode():
    output = model(**inputs, use_cache=False)
    generated = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
    )

base_parameters = 7_241_732_096
parameters = sum(parameter.numel() for parameter in model.parameters())
summary = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "base_model": str(base_model),
    "pruned_model": str(pruned_model),
    "method": "concat_merge",
    "calibration_dataset": "wiki2",
    "calibration_samples": 8,
    "seed": 10,
    "architecture": model.config.architectures[0],
    "base_layers": 32,
    "physical_layers": len(model.model.layers),
    "base_parameters": base_parameters,
    "parameters": parameters,
    "parameter_reduction_fraction": 1 - parameters / base_parameters,
    "logits_shape": list(output.logits.shape),
    "prompt": prompt,
    "greedy_completion": tokenizer.decode(generated[0], skip_special_tokens=True),
    "model_device": str(model.device),
    "cuda_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
}

result_path = result_dir / "summary.json"
with result_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"RESULT_PATH={result_path}")
PY

echo "CoMe concat_merge smoke test completed."
