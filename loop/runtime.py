"""Temporarily install a logical layer schedule without copying parameters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

from torch import nn


@contextmanager
def temporary_execution_plan(model, plan: Sequence[int]) -> Iterator[None]:
    """Run physical layers according to ``plan`` and restore the model afterwards.

    Repeated indices refer to the exact same module object. KV caching is disabled
    because a shared attention module has one physical ``layer_idx`` but can occupy
    multiple logical depths.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a decoder model exposing model.layers")

    physical_layers = model.model.layers
    physical_count = len(physical_layers)
    if not plan:
        raise ValueError("Execution plan cannot be empty")
    if min(plan) < 0 or max(plan) >= physical_count:
        raise IndexError(f"Execution plan references a layer outside [0, {physical_count})")

    old_num_hidden_layers = model.config.num_hidden_layers
    old_use_cache = model.config.use_cache
    logical_layers = nn.ModuleList([physical_layers[index] for index in plan])

    model.model.layers = logical_layers
    model.config.num_hidden_layers = len(plan)
    model.config.use_cache = False
    try:
        yield
    finally:
        model.model.layers = physical_layers
        model.config.num_hidden_layers = old_num_hidden_layers
        model.config.use_cache = old_use_cache
