"""Build logical layer schedules from CoMe's saved merge metadata."""

from __future__ import annotations

import random
from typing import Any, Iterable


def leaf_count(item: Any) -> int:
    """Count original layers represented by a (possibly nested) CoMe node."""
    if isinstance(item, list):
        return sum(leaf_count(child) for child in item)
    return 1


def final_merge_mapping(metadata: dict[str, Any]) -> list[Any]:
    """Return the final physical-to-original layer mapping in a CoMe info file."""
    history = metadata.get("merge_info")
    if not isinstance(history, list) or not history:
        raise ValueError("CoMe metadata has no non-empty 'merge_info' history")

    mapping = history[-1]
    if not isinstance(mapping, list):
        raise ValueError("Final CoMe merge mapping must be a list")
    return mapping


def _expand_counts(repeat_counts: Iterable[int]) -> list[int]:
    plan: list[int] = []
    for physical_index, count in enumerate(repeat_counts):
        if count < 1:
            raise ValueError("Every physical layer must execute at least once")
        plan.extend([physical_index] * count)
    return plan


def build_merge_aware_plan(mapping: list[Any]) -> list[int]:
    """Repeat each merged physical layer once per original layer it represents."""
    return _expand_counts(leaf_count(group) for group in mapping)


def _control_candidates(mapping: list[Any]) -> list[int]:
    singleton_indices = [index for index, group in enumerate(mapping) if leaf_count(group) == 1]
    return singleton_indices or list(range(len(mapping)))


def _extra_executions(mapping: list[Any]) -> int:
    return sum(leaf_count(group) for group in mapping) - len(mapping)


def build_random_control_plan(mapping: list[Any], seed: int) -> list[int]:
    """Spend the same extra executions on deterministic random non-merged layers."""
    repeat_counts = [1] * len(mapping)
    candidates = _control_candidates(mapping)
    rng = random.Random(seed)
    extras = _extra_executions(mapping)

    shuffled: list[int] = []
    while len(shuffled) < extras:
        cycle = candidates.copy()
        rng.shuffle(cycle)
        shuffled.extend(cycle)
    for physical_index in shuffled[:extras]:
        repeat_counts[physical_index] += 1
    return _expand_counts(repeat_counts)


def build_uniform_control_plan(mapping: list[Any]) -> list[int]:
    """Spend the same extra executions at uniformly spaced non-merged layers."""
    repeat_counts = [1] * len(mapping)
    candidates = _control_candidates(mapping)
    extras = _extra_executions(mapping)

    for extra_index in range(extras):
        position = int((extra_index + 0.5) * len(candidates) / extras) if extras else 0
        position = min(position, len(candidates) - 1)
        repeat_counts[candidates[position]] += 1
    return _expand_counts(repeat_counts)
