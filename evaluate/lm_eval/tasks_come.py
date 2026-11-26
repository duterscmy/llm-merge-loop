"""Compatibility alias for the CoMe-focused task registry."""

from lm_eval.tasks import ALL_TASKS, TASK_REGISTRY, get_task, get_task_dict

__all__ = ["ALL_TASKS", "TASK_REGISTRY", "get_task", "get_task_dict"]
