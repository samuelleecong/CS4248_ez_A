from __future__ import annotations

import random

import numpy as np
import torch

from .config import TrainConfig
from .constants import MPS_EVAL_BATCH_CAP, MPS_TRAIN_BATCH_CAP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def apply_device_runtime_optimizations(cfg: TrainConfig, device: torch.device) -> None:
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")

    if not cfg.optimize_for_mps:
        return

    if device.type != "mps":
        print("--optimize-for-mps was set, but no MPS device is available. Using default runtime settings.")
        return

    adjusted_batch_size = min(cfg.batch_size, MPS_TRAIN_BATCH_CAP)
    adjusted_eval_batch_size = min(cfg.eval_batch_size, MPS_EVAL_BATCH_CAP)
    if adjusted_batch_size != cfg.batch_size or adjusted_eval_batch_size != cfg.eval_batch_size:
        print(
            "Applying MPS-safe batch caps: "
            f"batch_size {cfg.batch_size} -> {adjusted_batch_size}, "
            f"eval_batch_size {cfg.eval_batch_size} -> {adjusted_eval_batch_size}"
        )
    else:
        print("MPS optimization flag enabled; current batch sizes are already within the safe caps.")
    cfg.batch_size = adjusted_batch_size
    cfg.eval_batch_size = adjusted_eval_batch_size
