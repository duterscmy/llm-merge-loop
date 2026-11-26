"""Minimal task registry for reproducing the CoMe Mistral evaluation table."""

from pprint import pprint
from typing import List, Union

import lm_eval.base

from . import arc
from . import hellaswag
from . import hendrycks_test
from . import openbookqa
from . import piqa
from . import winogrande


TASK_REGISTRY = {
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    "hellaswag": hellaswag.HellaSwag,
    "openbookqa": openbookqa.OpenBookQA,
    "piqa": piqa.PiQA,
    "winogrande": winogrande.Winogrande,
    **hendrycks_test.create_all_tasks(),
}

ALL_TASKS = sorted(TASK_REGISTRY)


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict).isdisjoint(task_name_from_object_dict)
    return {**task_name_dict, **task_name_from_object_dict}
