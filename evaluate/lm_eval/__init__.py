"""Compatibility entry point for the vendored legacy evaluation harness.

The repository contains both the original harness used by
``eval_zeroshot_acc.py`` and a partial newer FLAP harness.  Importing the
partial FLAP entry point makes even the legacy CLI depend on missing newer
modules, so expose the matching legacy evaluator here.
"""

from .evaluator import evaluate, simple_evaluate
