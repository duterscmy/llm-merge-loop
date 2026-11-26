#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=llama2-ms-convert
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
RAW_MODEL_DIR="$PROJECT_DIR/models/Llama-2-7b"
HF_MODEL_DIR="$PROJECT_DIR/models/Llama-2-7b-hf"
MODEL_URL=https://www.modelscope.cn/shakechen/Llama-2-7b.git
CONDA_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
CONVERTER="$CONDA_ENV/lib/python3.11/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/models"

hostname
uname -m

if test -d "$RAW_MODEL_DIR/.git"; then
  git -C "$RAW_MODEL_DIR" lfs checkout consolidated.00.pth tokenizer.model
else
  GIT_LFS_SKIP_SMUDGE=1 git clone "$MODEL_URL" "$RAW_MODEL_DIR"
  git -C "$RAW_MODEL_DIR" lfs pull
fi

git -C "$RAW_MODEL_DIR" lfs status
test -f "$RAW_MODEL_DIR/params.json"
test -f "$RAW_MODEL_DIR/tokenizer.model"

# An OOM-interrupted initial clone can leave the LFS object complete while the
# worktree index is incomplete, in which case `git lfs checkout` is a no-op.
# Recover the file from the content-addressed local object without downloading
# another 13 GB.
if ! test -f "$RAW_MODEL_DIR/consolidated.00.pth"; then
  LFS_OID="$(git -C "$RAW_MODEL_DIR" show HEAD:consolidated.00.pth | sed -n 's/^oid sha256://p')"
  test "${#LFS_OID}" -eq 64
  LFS_OBJECT="$RAW_MODEL_DIR/.git/lfs/objects/${LFS_OID:0:2}/${LFS_OID:2:2}/$LFS_OID"
  test -f "$LFS_OBJECT"
  cp --reflink=auto "$LFS_OBJECT" "$RAW_MODEL_DIR/consolidated.00.pth"
fi

test "$(stat -c %s "$RAW_MODEL_DIR/consolidated.00.pth")" -gt 10000000000

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if ! test -f "$HF_MODEL_DIR/.conversion-complete"; then
  python "$CONVERTER" \
    --input_dir "$RAW_MODEL_DIR" \
    --model_size 7B \
    --llama_version 2 \
    --output_dir "$HF_MODEL_DIR" \
    --safe_serialization
fi

test -f "$HF_MODEL_DIR/config.json"
test -f "$HF_MODEL_DIR/tokenizer.model"
test -f "$HF_MODEL_DIR/model.safetensors.index.json"

HF_MODEL_DIR="$HF_MODEL_DIR" python - <<'PY'
import os
from transformers import AutoConfig, AutoTokenizer

path = os.environ["HF_MODEL_DIR"]
config = AutoConfig.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
assert config.model_type == "llama"
assert config.num_hidden_layers == 32
assert len(tokenizer) >= 32000
print(config)
PY

touch "$RAW_MODEL_DIR/.download-complete"
touch "$HF_MODEL_DIR/.conversion-complete"
du -sh "$RAW_MODEL_DIR" "$HF_MODEL_DIR"
