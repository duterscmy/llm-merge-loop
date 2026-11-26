#!/usr/bin/env python
"""Resumable multi-GPU KL + endpoint distillation for the CoMe loop model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from experiments.distill_come_loop import merged_groups
from loop import build_merge_aware_plan, final_merge_mapping, temporary_execution_plan


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--student-model", required=True)
    p.add_argument("--merge-info", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--train-files", default="en/c4-train.*-of-01024.json.gz")
    p.add_argument("--validation-file", default="en/c4-validation.00000-of-00008.json.gz")
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--validation-blocks", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=1_000_000)
    p.add_argument("--max-runtime-seconds", type=int, default=83_700)
    p.add_argument("--eval-interval-seconds", type=int, default=1_800)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--endpoint-weight", type=float, default=0.1)
    p.add_argument("--tokenize-batch-size", type=int, default=64)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--resume", choices=["auto", "never", "required"], default="auto")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--train-blocks-file", type=Path)
    p.add_argument("--dev-blocks-file", type=Path)
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--eval-every-steps", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-relative-improvement", type=float, default=0.001)
    return p.parse_args()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class PackedStream:
    def __init__(self, dataset, tokenizer, length, tokenize_batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.length = length
        self.tokenize_batch_size = tokenize_batch_size
        self.epoch = 0
        self.buffer = []
        self.iterator = None

    def _iterator(self):
        if self.iterator is None:
            self.iterator = iter(self.dataset)
        return self.iterator

    def next_block(self):
        while len(self.buffer) < self.length:
            texts = []
            while len(texts) < self.tokenize_batch_size:
                try:
                    texts.append(next(self._iterator())["text"])
                except StopIteration:
                    self.epoch += 1
                    self.dataset.set_epoch(self.epoch)
                    self.iterator = iter(self.dataset)
            encoded = self.tokenizer(texts, add_special_tokens=False).input_ids
            eos = self.tokenizer.eos_token_id
            for ids in encoded:
                self.buffer.extend(ids)
                if eos is not None:
                    self.buffer.append(eos)
        block = self.buffer[: self.length]
        del self.buffer[: self.length]
        return torch.tensor(block, dtype=torch.long)

    def state_dict(self):
        return {"dataset": self.dataset.state_dict(), "epoch": self.epoch, "buffer": self.buffer}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.dataset.set_epoch(self.epoch)
        self.dataset.load_state_dict(state["dataset"])
        self.buffer = list(state["buffer"])
        self.iterator = None


def rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    # The checkpoint itself may be loaded with map_location=cuda so model and
    # optimizer tensors resume efficiently. RNG APIs nevertheless require CPU
    # ByteTensors, including the source state passed to the CUDA generator.
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())


def trainable_state(modules):
    return {
        str(index): {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}
        for index, module in modules.items()
    }


def load_trainable_state(modules, state):
    for index, module in modules.items():
        module.load_state_dict(state[str(index)], strict=True)


def endpoint_tensor(student_hidden, teacher_hidden, groups):
    losses = []
    for group in groups:
        teacher_index = group["teacher_hidden_index"]
        student_index = group.get("student_hidden_index", teacher_index)
        size = student_hidden[student_index].shape[-1]
        losses.append(F.mse_loss(F.layer_norm(student_hidden[student_index].float(), (size,)),
                                 F.layer_norm(teacher_hidden[teacher_index].float(), (size,))))
    return torch.stack(losses)


def kl_tensor(student_logits, teacher_logits, temperature):
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(s, t, reduction="none", log_target=True).sum(-1).mean() * temperature**2


def build_dev(tokenizer, args, rank, device):
    run_dir = Path(args.run_dir)
    cache = run_dir / "fixed_dev_tokens.pt"
    if rank == 0:
        if args.dev_blocks_file:
            blocks = torch.load(args.dev_blocks_file, map_location="cpu", weights_only=True)
        elif cache.exists():
            blocks = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            ds = load_dataset("allenai/c4", data_files={"validation": args.validation_file},
                              split="validation", streaming=True)
            stream = PackedStream(ds, tokenizer, args.sequence_length, args.tokenize_batch_size)
            blocks = torch.stack([stream.next_block() for _ in range(args.validation_blocks)])
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blocks, cache)
    else:
        blocks = torch.empty((args.validation_blocks, args.sequence_length), dtype=torch.long)
    if blocks.shape != (args.validation_blocks, args.sequence_length):
        raise ValueError("Unexpected development block shape")
    blocks = blocks.to(device)
    dist.broadcast(blocks, src=0)
    return blocks


class FixedBlockStream:
    def __init__(self, path, rank, world, length, steps):
        blocks = torch.load(path, map_location="cpu", weights_only=True)
        if blocks.ndim != 2 or blocks.shape[1] != length or len(blocks) < steps * world:
            raise ValueError("Insufficient fixed training blocks for requested token budget")
        self.blocks = blocks[rank::world]
        self.position = 0

    def next_block(self):
        block = self.blocks[self.position]
        self.position += 1
        return block

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


@torch.no_grad()
def evaluate(teacher, student, blocks, groups, temperature, rank, world, device):
    student_was_training = student.training
    student.eval()
    # student CE, teacher CE, token count, KL*positions, positions, endpoint sums, blocks
    stats = torch.zeros(7 + len(groups), dtype=torch.float64, device=device)
    for i in range(rank, len(blocks), world):
        x = blocks[i].unsqueeze(0)
        t = teacher(input_ids=x, output_hidden_states=True, use_cache=False, return_dict=True)
        s = student(input_ids=x, output_hidden_states=True, use_cache=False, return_dict=True)
        targets = x[:, 1:].reshape(-1)
        stats[0] += F.cross_entropy(s.logits[:, :-1].float().reshape(-1, s.logits.size(-1)), targets, reduction="sum")
        stats[1] += F.cross_entropy(t.logits[:, :-1].float().reshape(-1, t.logits.size(-1)), targets, reduction="sum")
        stats[2] += targets.numel()
        positions = s.logits.shape[1]
        stats[3] += kl_tensor(s.logits, t.logits, temperature) * positions
        stats[4] += positions
        endpoints = endpoint_tensor(s.hidden_states, t.hidden_states, groups)
        stats[5] += endpoints.mean()
        stats[6] += 1
        stats[7:] += endpoints.double()
    dist.all_reduce(stats)
    if student_was_training:
        student.train()
    ce = (stats[0] / stats[2]).item()
    teacher_ce = (stats[1] / stats[2]).item()
    return {
        "ppl": math.exp(min(ce, 30.0)), "ce": ce,
        "teacher_ppl": math.exp(min(teacher_ce, 30.0)), "teacher_ce": teacher_ce,
        "kl": (stats[3] / stats[4]).item(), "endpoint": (stats[5] / stats[6]).item(),
        "per_group": {"+".join(map(str, g["original_layers"])): (stats[7+j] / stats[6]).item()
                      for j, g in enumerate(groups)},
        "blocks": int(stats[6].item()),
    }


def save_checkpoint(path, modules, optimizer, scheduler, stream, step, best, history,
                    elapsed, args, rank, world):
    local = {"stream": stream.state_dict(), "rng": rng_state()}
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        state = {
            "version": 1, "step": step, "best": best, "history": history,
            "elapsed_seconds": elapsed, "world_size": world,
            "trainable": trainable_state(modules), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rank_states": gathered, "args": vars(args),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
    dist.barrier()


def export_model(student, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    student.config.use_cache = True
    student.save_pretrained(path, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(path)


def main():
    args = args_parser()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    run_dir = Path(args.run_dir)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    with open(args.merge_info, encoding="utf-8") as f:
        mapping = final_merge_mapping(json.load(f))
    plan, groups = build_merge_aware_plan(mapping), merged_groups(mapping)
    if args.no_loop:
        plan = list(range(len(mapping)))
        for group in groups:
            group["student_hidden_index"] = group["physical_index"] + 1
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    dev = build_dev(tokenizer, args, rank, device)
    if args.train_blocks_file:
        stream = FixedBlockStream(args.train_blocks_file, rank, world, args.sequence_length, args.max_steps)
    else:
        train_ds = load_dataset("allenai/c4", data_files={"train": args.train_files},
                                split="train", streaming=True).shuffle(seed=args.seed, buffer_size=1)
        train_ds = split_dataset_by_node(train_ds, rank=rank, world_size=world)
        stream = PackedStream(train_ds, tokenizer, args.sequence_length, args.tokenize_batch_size)

    load_kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **load_kwargs).to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(args.student_model, **load_kwargs).to(device)
    teacher.config.use_cache = student.config.use_cache = False
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    physical = student.model.layers
    modules = {g["physical_index"]: physical[g["physical_index"]] for g in groups}
    for module in modules.values():
        module.requires_grad_(True)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    checkpoint_dir = run_dir / "state"
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    step, elapsed_before, history = 0, 0.0, []
    best = {"ppl": float("inf"), "step": -1}
    resume_path = last_path if last_path.exists() else None
    if args.resume == "required" and resume_path is None:
        raise FileNotFoundError(last_path)
    if args.resume != "never" and resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["world_size"] != world:
            raise ValueError(f"Resume requires world_size={state['world_size']}, got {world}")
        load_trainable_state(modules, state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stream.load_state_dict(state["rank_states"][rank]["stream"])
        restore_rng(state["rank_states"][rank]["rng"])
        step, best = state["step"], state["best"]
        elapsed_before, history = state["elapsed_seconds"], state["history"]

    config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "world_size": world, "global_batch_size": world,
              "tokens_per_step": world * args.sequence_length, "mapping": mapping,
              "execution_plan": plan, "groups": groups,
              "trainable_parameters": sum(p.numel() for p in params),
              "initial_step": step, "resume_path": str(resume_path) if resume_path else None}
    if rank == 0:
        atomic_json(run_dir / "run_config.json", config)
        print("CONFIG=" + json.dumps(config), flush=True)

    start = time.monotonic()
    last_eval = start
    stop_reason = "budget"
    with temporary_execution_plan(student, plan):
        ddp = DDP(student, device_ids=[local_rank], broadcast_buffers=False,
                  gradient_as_bucket_view=True, find_unused_parameters=False)
        metrics = evaluate(teacher, student, dev, groups, args.temperature, rank, world, device)
        metrics.update(step=step, elapsed_seconds=elapsed_before, tokens=step * world * args.sequence_length)
        if rank == 0:
            print("DEV=" + json.dumps(metrics), flush=True)
        improved = metrics["ppl"] < best["ppl"]
        if improved:
            best = {"ppl": metrics["ppl"], "step": step}
        history.append(metrics)
        save_checkpoint(last_path, modules, optimizer, scheduler, stream, step, best, history,
                        elapsed_before, args, rank, world)
        if improved:
            save_checkpoint(best_path, modules, optimizer, scheduler, stream, step, best, history,
                            elapsed_before, args, rank, world)

        plateau_reference = history[0]["ppl"]
        stale_checks = 0
        previous_step = history[0]["step"]
        for point in history[1:]:
            if point["step"] == previous_step:
                continue
            previous_step = point["step"]
            if point["ppl"] < plateau_reference * (1 - args.min_relative_improvement):
                plateau_reference, stale_checks = point["ppl"], 0
            else:
                stale_checks += 1
        student.train()
        while (step < args.max_steps and time.monotonic() - start < args.max_runtime_seconds
               and (not args.patience or stale_checks < args.patience)):
            x = stream.next_block().unsqueeze(0).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                t = teacher(input_ids=x, output_hidden_states=True, use_cache=False, return_dict=True)
            s = ddp(input_ids=x, output_hidden_states=True, use_cache=False, return_dict=True)
            kl = kl_tensor(s.logits, t.logits, args.temperature)
            hidden = endpoint_tensor(s.hidden_states, t.hidden_states, groups).mean()
            scale = kl.detach() / hidden.detach().clamp_min(1e-8)
            loss = kl + args.endpoint_weight * scale * hidden
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            if not torch.isfinite(grad):
                raise FloatingPointError(f"Non-finite gradient at step {step + 1}")
            optimizer.step()
            scheduler.step()
            step += 1
            do_periodic_check = step % args.log_every == 0
            if step == 1 or do_periodic_check:
                values = torch.tensor([loss.detach(), kl.detach(), hidden.detach(), grad.detach()], device=device)
                dist.all_reduce(values)
                values /= world
                if rank == 0:
                    now = time.monotonic()
                    total_elapsed = elapsed_before + now - start
                    print("TRAIN=" + json.dumps({"step": step, "loss": values[0].item(),
                          "kl": values[1].item(), "endpoint": values[2].item(),
                          "grad_norm": values[3].item(), "lr": scheduler.get_last_lr()[0],
                          "elapsed_seconds": total_elapsed,
                          "tokens": step * world * args.sequence_length,
                          "tokens_per_second": ((step * world * args.sequence_length) /
                                                max(total_elapsed, 1e-6))}), flush=True)
            # All ranks must enter evaluation at the same DDP sequence number.  Only
            # rank 0 owns the wall-clock decision and broadcasts it at synchronized
            # logging boundaries; independent local clocks can otherwise interleave
            # evaluation all-reduces with gradient all-reduces and deadlock NCCL.
            do_eval = torch.zeros(1, dtype=torch.uint8, device=device)
            step_check = args.eval_every_steps and step % args.eval_every_steps == 0
            if step_check or (do_periodic_check and not args.eval_every_steps):
                if rank == 0 and (step_check or time.monotonic() - last_eval >= args.eval_interval_seconds):
                    do_eval.fill_(1)
                dist.broadcast(do_eval, src=0)
            if do_eval.item():
                metrics = evaluate(teacher, student, dev, groups, args.temperature, rank, world, device)
                elapsed = elapsed_before + time.monotonic() - start
                metrics.update(step=step, elapsed_seconds=elapsed, tokens=step * world * args.sequence_length)
                improved = metrics["ppl"] < best["ppl"]
                if improved:
                    best = {"ppl": metrics["ppl"], "step": step}
                history.append(metrics)
                if rank == 0:
                    print("DEV=" + json.dumps(metrics), flush=True)
                save_checkpoint(last_path, modules, optimizer, scheduler, stream, step, best, history,
                                elapsed, args, rank, world)
                if improved:
                    save_checkpoint(best_path, modules, optimizer, scheduler, stream, step, best, history,
                                    elapsed, args, rank, world)
                last_eval = time.monotonic()
                if metrics["ppl"] < plateau_reference * (1 - args.min_relative_improvement):
                    plateau_reference, stale_checks = metrics["ppl"], 0
                else:
                    stale_checks += 1
                if args.patience and stale_checks >= args.patience:
                    stop_reason = "dev_ppl_plateau"
                    break

        metrics = evaluate(teacher, student, dev, groups, args.temperature, rank, world, device)
        elapsed = elapsed_before + time.monotonic() - start
        metrics.update(step=step, elapsed_seconds=elapsed, tokens=step * world * args.sequence_length)
        improved = metrics["ppl"] < best["ppl"]
        if improved:
            best = {"ppl": metrics["ppl"], "step": step}
        history.append(metrics)
        if rank == 0:
            print("DEV_FINAL=" + json.dumps(metrics), flush=True)
        save_checkpoint(last_path, modules, optimizer, scheduler, stream, step, best, history,
                        elapsed, args, rank, world)
        if improved:
            save_checkpoint(best_path, modules, optimizer, scheduler, stream, step, best, history,
                            elapsed, args, rank, world)
        del ddp

    if rank == 0:
        if not args.skip_export:
            export_model(student, tokenizer, run_dir / "last_model")
            best_state = torch.load(best_path, map_location=device, weights_only=False)
            load_trainable_state(modules, best_state["trainable"])
            export_model(student, tokenizer, run_dir / "best_model")
        summary = {"status": "complete", "stop_reason": stop_reason, "last_step": step, "best": best,
                   "elapsed_seconds": elapsed, "training_tokens": step * world * args.sequence_length,
                   "last_dev": metrics, "history": history,
                   "last_state": str(last_path), "best_state": str(best_path)}
        atomic_json(run_dir / "run_summary.json", summary)
        print("SUMMARY=" + json.dumps(summary), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
