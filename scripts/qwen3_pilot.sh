#!/usr/bin/env bash
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python -c 'import torch; assert torch.cuda.is_available(), "Run inside a CUDA GPU allocation"'
export HF_HOME="${HF_HOME:-$root/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$root:$root/third_party/CoMe${PYTHONPATH:+:$PYTHONPATH}"
base="$root/models/Qwen3-8B-Base"
merge_root="$root/checkpoints/qwen3-come-33"
merged="$merge_root/Qwen3-8B-Base_concat_merge_bi_33_worepeat"
mapping="$merge_root/Qwen3-8B-Base_concat_merge_bi_info.json"
results="$root/results/qwen3-come-pilot"
packed="$results/packed"
steps=2000
mode="${1:?Use prepare, loop-c4, loop-mixed, or merge-mixed}"

eval_ppl() {
    local output="$1"
    shift
    python -m experiments.eval_come_loop_ppl --base-model "$base" --merged-model "$merged" \
        --merge-info "$mapping" --datasets wiki2 --output "$output" "$@"
}

if [[ "$mode" == prepare ]]; then
    test -s "$root/results/qwen3-come-smoke/ppl.json"
    mkdir -p "$merge_root" "$results"
    if [[ ! -f "$mapping" ]]; then
        cd "$root/third_party/CoMe"
        python -u main.py --method concat_merge --model-name "$base" \
            --target-layers 33 --save-path "$merge_root" --calibration-dataset wiki2 \
            --skip-method bi --nsamples 256 --merge-item 2 --wo-repeat --psu 1 --min-max 0.0 --seed 10
        cd "$root"
    fi
    python -m experiments.prepare_qwen3_recovery --model "$base" --output "$packed" --steps "$steps"
    torchrun --standalone --nproc-per-node=1 -m experiments.distill_come_loop_long \
        --teacher-model "$base" --student-model "$merged" --merge-info "$mapping" \
        --run-dir "$root/checkpoints/training-smoke-${SLURM_JOB_ID:-$$}" --train-blocks-file "$packed/mixed.pt" \
        --dev-blocks-file "$packed/dev.pt" --max-steps 5 --sequence-length 256 \
        --validation-blocks 64 --warmup-steps 0 --max-runtime-seconds 600 --skip-export
    eval_ppl "$results/baselines.json" --variants base merged merge_aware
    exit
fi

extra=()
case "$mode" in
    loop-c4) dataset=c4; variant=merge_aware ;;
    loop-mixed) dataset=mixed; variant=merge_aware ;;
    merge-mixed) dataset=mixed; variant=merged; extra+=(--no-loop) ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
test -s "$results/baselines.json"
run="$root/checkpoints/qwen3-come-pilot/$mode"
torchrun --standalone --nproc-per-node=1 -m experiments.distill_come_loop_long \
    --teacher-model "$base" --student-model "$merged" --merge-info "$mapping" \
    --run-dir "$run" --train-blocks-file "$packed/$dataset.pt" --dev-blocks-file "$packed/dev.pt" \
    --max-steps "$steps" --sequence-length 256 --validation-blocks 64 \
    --warmup-steps 100 --learning-rate 5e-6 --endpoint-weight 0.1 --log-every 50 \
    --eval-every-steps 200 --patience 3 --min-relative-improvement 0.001 \
    --max-runtime-seconds 6300 --skip-export "${extra[@]}"
cp "$run/run_summary.json" "$results/$mode-summary.json"
eval_ppl "$results/$mode-ppl.json" --variants "$variant" --trainer-state "$run/state/best.pt"
