#!/usr/bin/env python3
"""Combine Llama-2 CoMe PPL and zero-shot commonsense outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


VARIANTS = ["base", "merged", "merge_aware", "uniform", "random"]
METRICS = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "openbookqa": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    ppl = json.loads((result_dir / "loop_ppl.json").read_text(encoding="utf-8"))
    summary = {
        "model": "Llama-2-7b",
        "compression": "CoMe concat_merge, 32 to 29 physical layers (9.01%)",
        "merge_configuration": {
            "paper_aligned": True,
            "calibration_dataset": "WikiText-2",
            "calibration_samples": 256,
            "seed": 10,
            "merge_item": 2,
            "p": 1,
            "rho_minimum_retention": 0.0,
            "post_training": False,
        },
        "mmlu_included": False,
        "variants": {},
    }
    for variant in VARIANTS:
        task_path = result_dir / f"tasks-zero-shot-{variant}.json"
        tasks = json.loads(task_path.read_text(encoding="utf-8"))
        raw = tasks["harness"]["results"]
        scores = {task: 100.0 * raw[task][metric] for task, metric in METRICS.items()}
        ppl_result = ppl["variants"][variant]
        summary["variants"][variant] = {
            "physical_layers": ppl_result["physical_layers"],
            "logical_executions": ppl_result["logical_executions"],
            "parameters": ppl_result["parameters"],
            "wiki2_ppl": ppl_result["wiki2_ppl"],
            "c4_ppl": ppl_result["c4_ppl"],
            "commonsense_percent": scores,
            "commonsense_average_percent": mean(scores.values()),
            "slurm_job_id_ppl": ppl.get("slurm_job_id"),
            "slurm_job_id_tasks": tasks.get("slurm_job_id"),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
