#!/usr/bin/env python
"""Populate every cais/mmlu subject in a shared Hugging Face cache."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--attempts", type=int, default=5)
    return parser.parse_args()


def cached(cache_dir: Path, subject: str) -> bool:
    subject_dir = cache_dir / "cais___mmlu" / subject
    return subject_dir.is_dir() and any(subject_dir.rglob("dataset_info.json"))


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    marker = Path(args.marker)

    # Keep the canonical list in the vendored harness so prefetch and evaluation
    # cannot silently disagree about which MMLU subjects are required.
    from lm_eval.tasks.hendrycks_test import SUBJECTS

    missing = [subject for subject in SUBJECTS if not cached(cache_dir, subject)]
    print(f"MMLU cache: {len(SUBJECTS) - len(missing)}/{len(SUBJECTS)} complete")
    print(f"Missing subjects: {missing}", flush=True)

    for subject in missing:
        for attempt in range(1, args.attempts + 1):
            try:
                print(f"Downloading {subject} (attempt {attempt}/{args.attempts})", flush=True)
                load_dataset("cais/mmlu", subject, cache_dir=str(cache_dir))
                break
            except Exception:
                if attempt == args.attempts:
                    raise
                delay = 30 * attempt
                print(f"Retrying {subject} after {delay}s", flush=True)
                time.sleep(delay)

    still_missing = [subject for subject in SUBJECTS if not cached(cache_dir, subject)]
    if still_missing:
        raise RuntimeError(f"MMLU cache is incomplete: {still_missing}")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("57/57 cais/mmlu subjects cached\n", encoding="utf-8")
    print(f"MMLU cache complete; marker={marker}", flush=True)


if __name__ == "__main__":
    main()
