#!/usr/bin/env python
"""Distill dense group endpoints into a parameter-shared CoMe loop model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from loop import build_merge_aware_plan, final_merge_mapping, temporary_execution_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--merge-info", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--dataset", choices=["wikitext", "c4"], default="wikitext")
    parser.add_argument(
        "--c4-data-file",
        default="en/c4-train.00000-of-01024.json.gz",
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--validation-blocks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--objective",
        choices=["endpoint", "forward_kl", "forward_kl_endpoint"],
        default="endpoint",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--endpoint-weight-mode",
        choices=["fixed", "dynamic"],
        default="dynamic",
        help="Dynamic mode matches the endpoint-loss magnitude to the KL magnitude.",
    )
    parser.add_argument(
        "--endpoint-weight",
        type=float,
        default=1.0,
        help="Multiplier applied after optional dynamic magnitude balancing.",
    )
    parser.add_argument(
        "--hidden-target-mode",
        choices=["endpoint", "per_iteration"],
        default="endpoint",
        help=(
            "endpoint matches only the dense group endpoint; per_iteration "
            "matches every shared-layer execution to its corresponding dense layer."
        ),
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate the fixed validation split every N steps; 0 disables periodic evaluation.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--skip-save", action="store_true")
    return parser.parse_args()


def flatten_leaves(item: Any) -> list[int]:
    if isinstance(item, list):
        leaves: list[int] = []
        for child in item:
            leaves.extend(flatten_leaves(child))
        return leaves
    if not isinstance(item, int):
        raise TypeError(f"Unexpected merge-map item: {item!r}")
    return [item]


def merged_groups(mapping: list[Any]) -> list[dict[str, Any]]:
    groups = []
    for physical_index, item in enumerate(mapping):
        original_layers = flatten_leaves(item)
        if len(original_layers) > 1:
            groups.append(
                {
                    "physical_index": physical_index,
                    "original_layers": original_layers,
                    # hidden_states[0] is the embedding output, hence +1.
                    "teacher_hidden_index": max(original_layers) + 1,
                }
            )
    if not groups:
        raise ValueError("The CoMe mapping contains no merged layer groups")
    return groups


def wikitext_blocks(tokenizer, sequence_length: int, seed: int) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    input_ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids[0]
    block_count = input_ids.numel() // sequence_length
    if block_count < 2:
        raise ValueError("WikiText-2 did not produce enough complete training blocks")
    blocks = input_ids[: block_count * sequence_length].reshape(block_count, sequence_length)
    generator = torch.Generator().manual_seed(seed)
    return blocks[torch.randperm(block_count, generator=generator)]


def c4_blocks(
    tokenizer,
    sequence_length: int,
    block_count: int,
    seed: int,
    data_file: str,
) -> torch.Tensor:
    dataset = load_dataset(
        "allenai/c4",
        data_files={"train": data_file},
        split="train",
    )
    if not len(dataset):
        raise ValueError("C4 shard is empty")

    needed_tokens = block_count * sequence_length
    token_ids: list[int] = []
    start = random.Random(seed).randrange(len(dataset))
    for offset in range(len(dataset)):
        row = dataset[(start + offset) % len(dataset)]
        token_ids.extend(tokenizer(row["text"], add_special_tokens=False).input_ids)
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
        if len(token_ids) >= needed_tokens:
            break
    if len(token_ids) < needed_tokens:
        raise ValueError(f"C4 shard produced {len(token_ids)} tokens; need {needed_tokens}")
    return torch.tensor(token_ids[:needed_tokens], dtype=torch.long).reshape(
        block_count, sequence_length
    )


def normalized_hidden_mse(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    hidden_size = student_hidden.shape[-1]
    student_normalized = F.layer_norm(student_hidden.float(), (hidden_size,))
    teacher_normalized = F.layer_norm(teacher_hidden.float(), (hidden_size,))
    return F.mse_loss(student_normalized, teacher_normalized)


def endpoint_loss(
    student_hidden_states,
    teacher_hidden_states,
    groups,
    hidden_target_mode: str = "endpoint",
) -> tuple[torch.Tensor, list[float]]:
    losses = []
    for group in groups:
        if hidden_target_mode == "endpoint":
            hidden_indices = [group["teacher_hidden_index"]]
        else:
            # With the merge-aware 32-execution plan, hidden_states[k + 1]
            # is the output of the shared physical layer at its logical dense
            # depth k.  Thus 23 -> a and 24 -> a*2 are supervised separately.
            hidden_indices = [layer + 1 for layer in group["original_layers"]]
        group_losses = [
            normalized_hidden_mse(
                student_hidden_states[hidden_index], teacher_hidden_states[hidden_index]
            )
            for hidden_index in hidden_indices
        ]
        losses.append(torch.stack(group_losses).mean())
    total = torch.stack(losses).mean()
    return total, [loss.detach().item() for loss in losses]


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Token-mean KL(p_teacher || p_student), evaluated over the full vocabulary."""
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    return per_token.mean() * (temperature**2)


def combined_loss(
    student_outputs,
    teacher_outputs,
    groups,
    objective: str,
    temperature: float,
    endpoint_weight_mode: str,
    endpoint_weight: float,
    hidden_target_mode: str = "endpoint",
) -> tuple[torch.Tensor, dict[str, Any]]:
    hidden_loss, group_losses = endpoint_loss(
        student_outputs.hidden_states,
        teacher_outputs.hidden_states,
        groups,
        hidden_target_mode,
    )
    kl_loss = None
    dynamic_scale = 1.0
    if objective != "endpoint":
        kl_loss = forward_kl_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            temperature,
        )

    if objective == "endpoint":
        total = hidden_loss
    elif objective == "forward_kl":
        total = kl_loss
    else:
        if endpoint_weight_mode == "dynamic":
            dynamic_scale = float(
                (kl_loss.detach() / hidden_loss.detach().clamp_min(1e-8)).item()
            )
        total = kl_loss + endpoint_weight * dynamic_scale * hidden_loss

    details = {
        "loss": total.detach().item(),
        "kl_loss": None if kl_loss is None else kl_loss.detach().item(),
        "endpoint_loss": hidden_loss.detach().item(),
        "endpoint_scale": dynamic_scale,
        "weighted_endpoint_loss": (
            endpoint_weight * dynamic_scale * hidden_loss.detach().item()
        ),
        "group_losses": group_losses,
    }
    return total, details


@torch.no_grad()
def evaluate_losses(
    teacher,
    student,
    blocks,
    groups,
    objective,
    temperature,
    endpoint_weight_mode,
    endpoint_weight,
    hidden_target_mode,
) -> dict[str, Any]:
    student_was_training = student.training
    student.eval()
    metrics = []
    per_group = [[] for _ in groups]
    nll_sum = 0.0
    nll_tokens = 0
    for block in blocks:
        input_ids = block.unsqueeze(0).to("cuda", non_blocking=True)
        teacher_outputs = teacher(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        student_outputs = student(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        _, details = combined_loss(
            student_outputs,
            teacher_outputs,
            groups,
            objective,
            temperature,
            endpoint_weight_mode,
            endpoint_weight,
            hidden_target_mode,
        )
        metrics.append(details)
        shifted_logits = student_outputs.logits[:, :-1].float().reshape(
            -1, student_outputs.logits.shape[-1]
        )
        shifted_labels = input_ids[:, 1:].reshape(-1)
        nll_sum += F.cross_entropy(
            shifted_logits,
            shifted_labels,
            reduction="sum",
        ).item()
        nll_tokens += shifted_labels.numel()
        for index, group_loss in enumerate(details["group_losses"]):
            per_group[index].append(group_loss)
    if student_was_training:
        student.train()

    def mean_metric(name: str):
        values = [metric[name] for metric in metrics if metric[name] is not None]
        return None if not values else sum(values) / len(values)

    mean_nll = nll_sum / nll_tokens
    return {
        "nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "ppl_tokens": nll_tokens,
        "loss": mean_metric("loss"),
        "kl_loss": mean_metric("kl_loss"),
        "endpoint_loss": mean_metric("endpoint_loss"),
        "endpoint_scale": mean_metric("endpoint_scale"),
        "weighted_endpoint_loss": mean_metric("weighted_endpoint_loss"),
        "per_group": {
            "+".join(map(str, group["original_layers"])): sum(losses) / len(losses)
            for group, losses in zip(groups, per_group)
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.validation_blocks < 1 or args.log_every < 1:
        raise ValueError("max_steps, validation_blocks, and log_every must be positive")
    if args.eval_every < 0:
        raise ValueError("eval_every must be non-negative")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.endpoint_weight < 0:
        raise ValueError("endpoint_weight must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.merge_info, encoding="utf-8") as handle:
        metadata = json.load(handle)
    mapping = final_merge_mapping(metadata)
    plan = build_merge_aware_plan(mapping)
    groups = merged_groups(mapping)

    needed_blocks = args.validation_blocks + args.max_steps
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    if args.dataset == "wikitext":
        blocks = wikitext_blocks(tokenizer, args.sequence_length, args.seed)
    else:
        blocks = c4_blocks(
            tokenizer,
            args.sequence_length,
            needed_blocks,
            args.seed,
            args.c4_data_file,
        )
    if len(blocks) < needed_blocks:
        raise ValueError(f"Need {needed_blocks} blocks but only found {len(blocks)}")
    validation_blocks = blocks[: args.validation_blocks]
    training_blocks = blocks[args.validation_blocks : needed_blocks]

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
    }
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **load_kwargs).eval()
    student = AutoModelForCausalLM.from_pretrained(args.student_model, **load_kwargs)
    teacher.config.use_cache = False
    student.config.use_cache = False

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    physical_layers = student.model.layers
    if len(physical_layers) != len(mapping):
        raise ValueError(
            f"Student has {len(physical_layers)} physical layers but mapping has {len(mapping)} entries"
        )
    for group in groups:
        for parameter in physical_layers[group["physical_index"]].parameters():
            parameter.requires_grad_(True)

    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    print(
        json.dumps(
            {
                "teacher": args.teacher_model,
                "student": args.student_model,
                "merge_mapping": mapping,
                "execution_plan": plan,
                "endpoint_groups": groups,
                "trainable_parameters": trainable_count,
                "sequence_length": args.sequence_length,
                "max_steps": args.max_steps,
                "training_tokens": args.max_steps * args.sequence_length,
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "objective": args.objective,
                "temperature": args.temperature,
                "endpoint_weight_mode": args.endpoint_weight_mode,
                "endpoint_weight": args.endpoint_weight,
                "hidden_target_mode": args.hidden_target_mode,
                "eval_every": args.eval_every,
            },
            indent=2,
        ),
        flush=True,
    )

    step_metrics = []
    validation_curve = []
    with temporary_execution_plan(student, plan):
        before = evaluate_losses(
            teacher,
            student,
            validation_blocks,
            groups,
            args.objective,
            args.temperature,
            args.endpoint_weight_mode,
            args.endpoint_weight,
            args.hidden_target_mode,
        )
        validation_curve.append({"step": 0, "training_tokens": 0, **before})
        print(f"VALIDATION_BEFORE={json.dumps(before, sort_keys=True)}", flush=True)

        student.train()
        train_started = time.perf_counter()
        interval_started = train_started
        last_logged_step = 0
        for step, block in enumerate(training_blocks, start=1):
            input_ids = block.unsqueeze(0).to("cuda", non_blocking=True)
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            student_outputs = student(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            loss, loss_details = combined_loss(
                student_outputs,
                teacher_outputs,
                groups,
                args.objective,
                args.temperature,
                args.endpoint_weight_mode,
                args.endpoint_weight,
                args.hidden_target_mode,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            metric = {
                "step": step,
                **loss_details,
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if step % args.log_every == 0 or step == 1 or step == args.max_steps:
                torch.cuda.synchronize()
                now = time.perf_counter()
                interval_steps = step - last_logged_step
                metric.update(
                    {
                        "elapsed_seconds": now - train_started,
                        "interval_seconds": now - interval_started,
                        "tokens_per_second": (
                            interval_steps * args.sequence_length / (now - interval_started)
                        ),
                    }
                )
                interval_started = now
                last_logged_step = step
                step_metrics.append(metric)
                print(f"TRAIN={json.dumps(metric)}", flush=True)

            if (
                args.eval_every > 0
                and step % args.eval_every == 0
                and step < args.max_steps
            ):
                periodic = evaluate_losses(
                    teacher,
                    student,
                    validation_blocks,
                    groups,
                    args.objective,
                    args.temperature,
                    args.endpoint_weight_mode,
                    args.endpoint_weight,
                    args.hidden_target_mode,
                )
                curve_point = {
                    "step": step,
                    "training_tokens": step * args.sequence_length,
                    **periodic,
                }
                validation_curve.append(curve_point)
                print(
                    f"PERIODIC_VALIDATION={json.dumps(curve_point, sort_keys=True)}",
                    flush=True,
                )

        training_seconds = time.perf_counter() - train_started
        after = evaluate_losses(
            teacher,
            student,
            validation_blocks,
            groups,
            args.objective,
            args.temperature,
            args.endpoint_weight_mode,
            args.endpoint_weight,
            args.hidden_target_mode,
        )
        validation_curve.append(
            {
                "step": args.max_steps,
                "training_tokens": args.max_steps * args.sequence_length,
                **after,
            }
        )
        print(f"VALIDATION_AFTER={json.dumps(after, sort_keys=True)}", flush=True)

    output_dir = Path(args.output_dir)
    if not args.skip_save:
        student.config.use_cache = True
        output_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(output_dir)

    metrics = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "objective": args.objective,
        "teacher_model": args.teacher_model,
        "initial_student_model": args.student_model,
        "output_dir": None if args.skip_save else str(output_dir),
        "dataset": (
            "wikitext-2-raw-v1:train"
            if args.dataset == "wikitext"
            else f"allenai/c4:{args.c4_data_file}"
        ),
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "max_steps": args.max_steps,
        "training_tokens": args.max_steps * args.sequence_length,
        "training_seconds": training_seconds,
        "training_tokens_per_second": (
            args.max_steps * args.sequence_length / training_seconds
        ),
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "endpoint_weight_mode": args.endpoint_weight_mode,
        "endpoint_weight": args.endpoint_weight,
        "hidden_target_mode": args.hidden_target_mode,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "merge_mapping": mapping,
        "execution_plan": plan,
        "endpoint_groups": groups,
        "trainable_parameters": trainable_count,
        "validation_before": before,
        "validation_after": after,
        "validation_curve": validation_curve,
        "steps": step_metrics,
    }
    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    if not args.skip_save:
        print(f"MODEL_PATH={output_dir}", flush=True)
    print(f"METRICS_PATH={metrics_output}", flush=True)


if __name__ == "__main__":
    main()
