#!/usr/bin/env python
"""Evaluate a dense or CoMe-merged model, optionally with a loop schedule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from loop import (
    build_merge_aware_plan,
    build_random_control_plan,
    build_uniform_control_plan,
    final_merge_mapping,
    temporary_execution_plan,
)
from experiments.distill_come_loop import merged_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--merged-model")
    parser.add_argument("--merge-info")
    parser.add_argument("--trainer-state")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["base", "merged", "merge_aware", "uniform", "random"],
    )
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_model(path: str, trainer_state: str | None = None, mapping=None):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    model.config.use_cache = False
    if trainer_state:
        state = torch.load(trainer_state, map_location="cpu", weights_only=False)
        for group in merged_groups(mapping):
            index = group["physical_index"]
            model.model.layers[index].load_state_dict(state["trainable"][str(index)], strict=True)
        model._trainer_state_metadata = {
            "path": trainer_state,
            "step": state["step"],
            "best": state["best"],
        }
    return model


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate(model, tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    # Import only after the compute-node runtime is active.  The vendored
    # harness accepts an instantiated HF model, which lets loop variants share
    # module objects without serializing a fake 32-layer checkpoint.
    from lm_eval import evaluator
    from lm_eval.models.gpt2 import HFLM

    adapter = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
    )
    return evaluator.simple_evaluate(
        model=adapter,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device="cuda",
        no_cache=True,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
    )


def main() -> None:
    args = parse_args()
    if args.tasks == ["mmlu"]:
        from lm_eval.tasks.hendrycks_test import SUBJECTS

        args.tasks = [f"hendrycksTest-{subject}" for subject in SUBJECTS]
    if args.variant == "base":
        model_path = args.base_model
        plan = None
        mapping = None
    else:
        if not args.merged_model or not args.merge_info:
            raise ValueError("Merged variants require --merged-model and --merge-info")
        model_path = args.merged_model
        with open(args.merge_info, encoding="utf-8") as handle:
            mapping = final_merge_mapping(json.load(handle))
        plan_builders = {
            "merged": lambda: list(range(len(mapping))),
            "merge_aware": lambda: build_merge_aware_plan(mapping),
            "uniform": lambda: build_uniform_control_plan(mapping),
            "random": lambda: build_random_control_plan(mapping, args.seed),
        }
        plan = plan_builders[args.variant]()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = load_model(model_path, args.trainer_state, mapping)
    physical_parameters = parameter_count(model)
    physical_layers = len(model.model.layers)

    if plan is None:
        results = evaluate(model, tokenizer, args)
        logical_executions = physical_layers
    else:
        with temporary_execution_plan(model, plan):
            if parameter_count(model) != physical_parameters:
                raise AssertionError("Loop schedule changed the physical parameter count")
            results = evaluate(model, tokenizer, args)
        logical_executions = len(plan)

    payload = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "variant": args.variant,
        "model_path": model_path,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "seed": args.seed,
        "physical_layers": physical_layers,
        "logical_executions": logical_executions,
        "parameters": physical_parameters,
        "execution_plan": plan,
        "trainer_state": getattr(model, "_trainer_state_metadata", None),
        "harness": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"RESULT_PATH={output}", flush=True)


if __name__ == "__main__":
    main()
