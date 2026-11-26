#!/usr/bin/env python
"""Compare CoMe, merge-aware looping, and equal-compute loop controls."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
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
    parser.add_argument("--merged-model", required=True)
    parser.add_argument("--trainer-state")
    parser.add_argument("--merge-info", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wiki2"],
        choices=["wiki2", "c4"],
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "merged", "merge_aware", "uniform", "random"],
        choices=["base", "merged", "merge_aware", "uniform", "random"],
    )
    return parser.parse_args()


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def wiki2_blocks(tokenizer, sequence_length: int, max_samples: int | None) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encoded = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids
    samples = encoded.numel() // sequence_length
    if max_samples is not None:
        samples = min(samples, max_samples)
    if samples < 1:
        raise ValueError("WikiText-2 produced no complete evaluation blocks")
    return encoded[:, : samples * sequence_length].reshape(samples, sequence_length)


def c4_blocks(tokenizer, sequence_length: int, max_samples: int | None) -> torch.Tensor:
    dataset = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    # Match CoMe's evaluation protocol: concatenate the first 1,100 validation
    # documents and evaluate at most 256 complete 2,048-token blocks.
    encoded = tokenizer(" ".join(dataset[:1100]["text"]), return_tensors="pt").input_ids
    samples = min(256, encoded.numel() // sequence_length)
    if max_samples is not None:
        samples = min(samples, max_samples)
    if samples < 1:
        raise ValueError("C4 produced no complete evaluation blocks")
    return encoded[:, : samples * sequence_length].reshape(samples, sequence_length)


def evaluation_blocks(tokenizer, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    builders = {"wiki2": wiki2_blocks, "c4": c4_blocks}
    return {
        dataset: builders[dataset](tokenizer, args.sequence_length, args.max_samples)
        for dataset in args.datasets
    }


@torch.inference_mode()
def perplexity(model, blocks: torch.Tensor) -> float:
    negative_log_likelihoods = []
    for index, block in enumerate(blocks):
        input_ids = block.unsqueeze(0).to(model.device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
        )
        negative_log_likelihoods.append(loss.float() * input_ids[:, 1:].numel())
        running = torch.stack(negative_log_likelihoods).sum() / (
            (index + 1) * (input_ids.size(1) - 1)
        )
        print(f"block={index + 1}/{len(blocks)} running_ppl={torch.exp(running).item():.6f}", flush=True)
    mean_nll = torch.stack(negative_log_likelihoods).sum() / (
        len(blocks) * (blocks.size(1) - 1)
    )
    return math.exp(mean_nll.item())


def load_model(path: str, trainer_state: str | None = None, mapping=None):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
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


def evaluate_base(path: str, blocks_by_dataset: dict[str, torch.Tensor]) -> dict[str, Any]:
    model = load_model(path)
    result = {
        "path": path,
        "physical_layers": len(model.model.layers),
        "logical_executions": len(model.model.layers),
        "parameters": parameter_count(model),
    }
    for dataset, blocks in blocks_by_dataset.items():
        print(f"=== variant=base dataset={dataset} ===", flush=True)
        result[f"{dataset}_ppl"] = perplexity(model, blocks)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    with open(args.merge_info, encoding="utf-8") as handle:
        metadata = json.load(handle)
    mapping = final_merge_mapping(metadata)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    blocks_by_dataset = evaluation_blocks(tokenizer, args)
    results: dict[str, Any] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "datasets": args.datasets,
        "sequence_length": args.sequence_length,
        "samples": {name: len(blocks) for name, blocks in blocks_by_dataset.items()},
        "seed": args.seed,
        "merge_mapping": mapping,
        "variants": {},
    }

    if "base" in args.variants:
        results["variants"]["base"] = evaluate_base(args.base_model, blocks_by_dataset)

    merged_variants = [name for name in args.variants if name != "base"]
    if merged_variants:
        model = load_model(args.merged_model, args.trainer_state, mapping)
        physical_layers = len(model.model.layers)
        physical_parameters = parameter_count(model)
        plans = {
            "merged": list(range(physical_layers)),
            "merge_aware": build_merge_aware_plan(mapping),
            "uniform": build_uniform_control_plan(mapping),
            "random": build_random_control_plan(mapping, args.seed),
        }

        if physical_layers != len(mapping):
            raise ValueError(
                f"Merged checkpoint has {physical_layers} layers but mapping has {len(mapping)} entries"
            )

        for name in merged_variants:
            plan = plans[name]
            print(f"=== variant={name} plan={plan} ===", flush=True)
            with temporary_execution_plan(model, plan):
                parameters_with_plan = parameter_count(model)
                if parameters_with_plan != physical_parameters:
                    raise AssertionError("Logical looping changed the physical parameter count")
                ppls = {}
                for dataset, blocks in blocks_by_dataset.items():
                    print(f"=== variant={name} dataset={dataset} ===", flush=True)
                    ppls[f"{dataset}_ppl"] = perplexity(model, blocks)
            results["variants"][name] = {
                "path": args.merged_model,
                "physical_layers": physical_layers,
                "logical_executions": len(plan),
                "parameters": physical_parameters,
                "execution_plan": plan,
                "trainer_state": getattr(model, "_trainer_state_metadata", None),
                **ppls,
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    print(f"RESULT_PATH={output}", flush=True)


if __name__ == "__main__":
    main()
