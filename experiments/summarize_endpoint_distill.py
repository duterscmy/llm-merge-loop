#!/usr/bin/env python3
"""Summarize endpoint-distillation training, PPL, and commonsense results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


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
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--pilot-ppl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    training = json.loads((result_dir / "train_metrics.json").read_text(encoding="utf-8"))
    ppl = json.loads((result_dir / "loop_ppl.json").read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    pilot_ppl = (
        json.loads(Path(args.pilot_ppl).read_text(encoding="utf-8"))
        if args.pilot_ppl
        else None
    )

    validation_before = training["validation_before"]
    validation_after = training["validation_after"]
    before = validation_before.get("endpoint_loss", validation_before.get("mean"))
    after = validation_after.get("endpoint_loss", validation_after.get("mean"))
    summary = {
        "model": "Llama-2-7b",
        "method": (
            "CoMe 32-to-29 plus shared-layer loop, forward-logit-KL plus "
            "normalized group-endpoint hidden-state distillation"
            if training["objective"] == "forward_kl_endpoint"
            else "CoMe 32-to-29 plus shared-layer loop, dense group-endpoint distillation"
        ),
        "training": {
            "dataset": training["dataset"],
            "sequence_length": training["sequence_length"],
            "steps": training["max_steps"],
            "tokens": training.get(
                "training_tokens", training["sequence_length"] * training["max_steps"]
            ),
            "objective": training["objective"],
            "temperature": training.get("temperature"),
            "endpoint_weight_mode": training.get("endpoint_weight_mode"),
            "endpoint_weight": training.get("endpoint_weight"),
            "learning_rate": training["learning_rate"],
            "warmup_steps": training["warmup_steps"],
            "trainable_parameters": training["trainable_parameters"],
            "training_seconds": training.get("training_seconds"),
            "training_tokens_per_second": training.get("training_tokens_per_second"),
            "validation_total_loss_before": validation_before.get("loss"),
            "validation_total_loss_after": validation_after.get("loss"),
            "validation_kl_loss_before": validation_before.get("kl_loss"),
            "validation_kl_loss_after": validation_after.get("kl_loss"),
            "validation_endpoint_loss_before": before,
            "validation_endpoint_loss_after": after,
            "validation_relative_reduction_percent": 100.0 * (before - after) / before,
            "slurm_job_id": training["slurm_job_id"],
        },
        "mmlu_included": False,
        "variants": {},
    }

    for variant in ("merged", "merge_aware"):
        task_payload = json.loads(
            (result_dir / f"tasks-zero-shot-{variant}.json").read_text(encoding="utf-8")
        )
        raw = task_payload["harness"]["results"]
        scores = {task: 100.0 * raw[task][metric] for task, metric in METRICS.items()}
        current_ppl = ppl["variants"][variant]
        old = baseline["variants"][variant]
        payload = {
            "physical_layers": current_ppl["physical_layers"],
            "logical_executions": current_ppl["logical_executions"],
            "parameters": current_ppl["parameters"],
            "wiki2_ppl": current_ppl["wiki2_ppl"],
            "c4_ppl": current_ppl["c4_ppl"],
            "commonsense_percent": scores,
            "commonsense_average_percent": mean(scores.values()),
            "delta_from_untrained": {
                "wiki2_ppl": current_ppl["wiki2_ppl"] - old["wiki2_ppl"],
                "c4_ppl": current_ppl["c4_ppl"] - old["c4_ppl"],
                "commonsense_average_points": mean(scores.values())
                - old["commonsense_average_percent"],
            },
            "slurm_job_id_ppl": ppl["slurm_job_id"],
            "slurm_job_id_tasks": task_payload["slurm_job_id"],
        }
        if pilot_ppl is not None:
            payload["pilot_40_step_ppl"] = {
                "wiki2": pilot_ppl["variants"][variant]["wiki2_ppl"],
                "c4": pilot_ppl["variants"][variant]["c4_ppl"],
            }
        summary["variants"][variant] = payload

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
