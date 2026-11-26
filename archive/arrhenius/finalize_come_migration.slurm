#!/usr/bin/env bash
#SBATCH --account=ehpc-reg-2026r01-278-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=finalize-come
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/Sliding-Window-Merging
SOURCE_ENV="$HOME/miniconda3/envs/come"
TARGET_ENV=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
CONDARC_TEMPLATE="$PROJECT_DIR/env/condarc-nobackup.yaml"
CONDARC_TARGET="$HOME/.condarc"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -d "$TARGET_ENV/conda-meta" ]]; then
  echo "Verified target environment is missing: $TARGET_ENV" >&2
  exit 2
fi
if [[ -e "$CONDARC_TARGET" ]]; then
  echo "Refusing to overwrite an existing $CONDARC_TARGET" >&2
  exit 3
fi

install -m 600 "$CONDARC_TEMPLATE" "$CONDARC_TARGET"

conda activate come
echo "Resolved CONDA_PREFIX=$CONDA_PREFIX"
if [[ "$CONDA_PREFIX" != "$TARGET_ENV" ]]; then
  echo "Named activation did not resolve to the nobackup environment" >&2
  exit 4
fi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
conda deactivate

conda env remove --prefix "$SOURCE_ENV" --yes

if [[ -e "$SOURCE_ENV" ]]; then
  echo "Old environment still exists after conda removal" >&2
  exit 5
fi

df -h "$HOME" /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu
du -sh "$HOME/miniconda3" "$TARGET_ENV"
echo "come migration finalized; log-prob and opsa-slime were not modified."
