"""Prepare paired, bounded C4 and C4/Wiki-Rewrite recovery streams."""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_url
from transformers import AutoTokenizer


def shingles(text):
    words = re.findall(r"\w+", text.casefold())
    return zip(*(words[offset:] for offset in range(12)))


def pack(rows, tokenizer, count, length, excluded):
    blocks, buffer = [], []
    rejected = 0
    for row in rows:
        text = row["text"]
        if not text.strip():
            continue
        if any(key in excluded for key in shingles(text)):
            rejected += 1
            continue
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        buffer.append(tokenizer.eos_token_id)
        usable = min(len(buffer) // length, count - len(blocks))
        blocks.extend(buffer[i * length:(i + 1) * length] for i in range(usable))
        del buffer[:usable * length]
        if len(blocks) == count:
            return torch.tensor(blocks, dtype=torch.long), rejected
    raise RuntimeError(f"Only obtained {len(blocks)} of {count} requested blocks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--length", type=int, default=256)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 5:
        parser.error("--steps must be a positive multiple of five")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
    excluded = {key for split in ("validation", "test") for row in wiki[split]
                for key in shingles(row["text"])}
    c4 = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                      split="train", streaming=True).shuffle(seed=10, buffer_size=128)
    repo = "nvidia/Nemotron-Pretraining-Specialized-v1"
    api = HfApi()
    revision = api.dataset_info(repo).sha
    files = sorted(path for path in api.list_repo_files(repo, repo_type="dataset", revision=revision)
                   if path.startswith("Nemotron-Pretraining-Wiki-Rewrite/") and path.endswith(".parquet"))
    if not files:
        raise RuntimeError("No Wiki-Rewrite parquet shards found")
    wiki_rows = load_dataset("parquet", data_files=hf_hub_url(repo, files[0], repo_type="dataset",
                             revision=revision), split="train", streaming=True)
    c4_blocks, rejected_c4 = pack(c4, tokenizer, args.steps, args.length, excluded)
    wiki_blocks, rejected_wiki = pack(wiki_rows, tokenizer, args.steps // 5, args.length, excluded)
    dev, _ = pack(wiki["validation"], tokenizer, 64, args.length, set())
    mixed = c4_blocks.clone()
    mixed[4::5] = wiki_blocks
    args.output.mkdir(parents=True, exist_ok=True)
    for name, blocks in (("c4", c4_blocks), ("mixed", mixed), ("dev", dev)):
        torch.save(blocks, args.output / f"{name}.pt")
    manifest = {"seed": 10, "steps": args.steps, "length": args.length,
                "wiki_repository": repo, "wiki_revision": revision,
                "wiki_license_dataset_card": ["CC BY-SA 4.0", "GFDL 1.3"],
                "wiki_shard": files[0], "wiki_tokens": wiki_blocks.numel(),
                "rejected_documents": {"c4": rejected_c4, "wiki": rejected_wiki},
                "overlap_filter": "exact normalized 12-word matches within WikiText-2 dev/test rows"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
