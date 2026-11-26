#!/usr/bin/env python3
"""Combine CoMe zero-shot and MMLU outputs into paper-style rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


VARIANTS = ["base", "merged", "merge_aware", "uniform", "random"]
ZERO_SHOT_METRICS = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "openbookqa": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
}
PAPER = {
    "base": {
        "arc_challenge": 54.01,
        "arc_easy": 79.50,
        "hellaswag": 81.06,
        "openbookqa": 44.00,
        "piqa": 82.05,
        "winogrande": 74.03,
        "mmlu": 62.52,
        "average": 68.17,
    },
    "merged": {
        "arc_challenge": 48.55,
        "arc_easy": 74.37,
        "hellaswag": 76.93,
        "openbookqa": 41.00,
        "piqa": 79.60,
        "winogrande": 72.61,
        "mmlu": 61.52,
        "average": 64.94,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize_variant(result_dir: Path, variant: str) -> dict | None:
    zero_path = result_dir / f"tasks-zero-shot-{variant}.json"
    mmlu_path = result_dir / f"mmlu-5shot-{variant}.json"
    if not zero_path.exists():
        return None

    zero = load(zero_path)
    zero_results = zero["harness"]["results"]
    scores = {
        task: 100.0 * zero_results[task][metric]
        for task, metric in ZERO_SHOT_METRICS.items()
    }
    result = {
        "slurm_job_id_zero_shot": zero.get("slurm_job_id"),
        "physical_layers": zero["physical_layers"],
        "logical_executions": zero["logical_executions"],
        "parameters": zero["parameters"],
        "scores_percent": scores,
    }

    if mmlu_path.exists():
        mmlu = load(mmlu_path)
        subject_results = mmlu["harness"]["results"]
        subject_scores = {
            task: values.get("acc_norm", values["acc"])
            for task, values in subject_results.items()
        }
        mmlu_macro = 100.0 * mean(subject_scores.values())
        counts = mmlu["harness"].get("n_samples", {})
        if counts and set(subject_scores) <= set(counts):
            total = sum(counts[task] for task in subject_scores)
            mmlu_micro = 100.0 * sum(
                subject_scores[task] * counts[task] for task in subject_scores
            ) / total
        else:
            mmlu_micro = None
        scores["mmlu"] = mmlu_macro
        result.update(
            {
                "slurm_job_id_mmlu": mmlu.get("slurm_job_id"),
                "mmlu_subjects": len(subject_scores),
                "mmlu_macro_percent": mmlu_macro,
                "mmlu_micro_percent": mmlu_micro,
                "average_percent": mean(scores.values()),
            }
        )

    if variant in PAPER:
        result["paper_percent"] = PAPER[variant]
        result["delta_from_paper_points"] = {
            key: scores[key] - value
            for key, value in PAPER[variant].items()
            if key in scores
        }
    return result


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    payload = {
        "metric_policy": {
            "zero_shot": "length-normalized accuracy except WinoGrande accuracy",
            "mmlu_primary": "macro-average of 57 subject accuracies",
        },
        "variants": {},
    }
    for variant in VARIANTS:
        summary = summarize_variant(result_dir, variant)
        if summary is not None:
            payload["variants"][variant] = summary

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
