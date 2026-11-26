#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=come-concat-ppl
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
BASE_MODEL="$PROJECT_DIR/models/Mistral-7B-v0.1"
PRUNED_MODEL="$PROJECT_DIR/checkpoints/come-concat-smoke/Mistral-7B-v0.1_concat_merge_bi_31_worepeat"
RESULT_DIR="$PROJECT_DIR/results/come-concat-smoke"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
cd "$PROJECT_DIR/third_party/CoMe"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RESULT_DIR"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

BASE_MODEL="$BASE_MODEL" \
PRUNED_MODEL="$PRUNED_MODEL" \
RESULT_DIR="$RESULT_DIR" \
python - <<'PY'
import gc
import json
import os
from pathlib import Path

import torch

from utils.eval_utils import load_and_eval_ppl
from utils.model_utils import get_llmhander

base_model = Path(os.environ["BASE_MODEL"])
pruned_model = Path(os.environ["PRUNED_MODEL"])
result_dir = Path(os.environ["RESULT_DIR"])


def evaluate(model_path: Path):
    handler = get_llmhander(str(model_path))
    handler.model.eval()
    parameters = sum(parameter.numel() for parameter in handler.model.parameters())
    layers = len(handler.layers)
    ppl = load_and_eval_ppl(
        handler.model,
        dataset="wiki2",
        tokenizer=handler.tokenizer,
    )
    result = {
        "path": str(model_path),
        "layers": layers,
        "parameters": parameters,
        "wiki2_ppl": ppl,
    }
    del handler
    gc.collect()
    torch.cuda.empty_cache()
    return result


base = evaluate(base_model)
pruned = evaluate(pruned_model)
summary = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "dataset": "wikitext-2-raw-v1:test",
    "sequence_length": 2048,
    "base": base,
    "pruned": pruned,
    "parameter_reduction_fraction": 1 - pruned["parameters"] / base["parameters"],
    "ppl_relative_change": pruned["wiki2_ppl"] / base["wiki2_ppl"] - 1,
}

result_path = result_dir / "ppl_comparison.json"
with result_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(json.dumps(summary, indent=2))
print(f"RESULT_PATH={result_path}")
PY

echo "Base/pruned WikiText-2 comparison completed."
