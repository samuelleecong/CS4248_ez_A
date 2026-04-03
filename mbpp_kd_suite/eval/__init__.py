from __future__ import annotations


def evaluate_config(*args, **kwargs):
    from .engine import evaluate_config as _evaluate_config

    return _evaluate_config(*args, **kwargs)

__all__ = ["evaluate_config"]
