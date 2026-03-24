from __future__ import annotations

from typing import Any

import torch

from mbpp_kd_suite.metrics import paired_ranking_metrics
from mbpp_kd_suite.runtime import pick_device, set_seed
from .data_adapters import get_dataset_adapter
from .model_adapters import load_model_adapter
from .profiler import StageProfiler
from .types import EvalConfig


def evaluate_config(cfg: EvalConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    profiler = StageProfiler(device=device)

    model_adapter = profiler.profile(
        "model_load",
        lambda: load_model_adapter(
            model_source=cfg.model_source,
            model_name_or_path=cfg.model_name_or_path,
            checkpoint_format=cfg.checkpoint_format,
            max_query_length=cfg.max_query_length,
            max_code_length=cfg.max_code_length,
            batch_size=cfg.batch_size,
            device=device,
        ),
    )

    corpus = profiler.profile(
        "dataset_load",
        lambda: get_dataset_adapter(cfg.dataset_name, path=cfg.dataset_path).load(),
    )
    split_records = corpus.get_split(cfg.split)
    if not split_records:
        raise ValueError(f"Requested split '{cfg.split}' is empty for dataset {cfg.dataset_name}")

    queries = [record.query for record in split_records]
    codes = [record.code for record in split_records]

    query_embs = profiler.profile("query_encode", lambda: model_adapter.encode_queries(queries))
    code_embs = profiler.profile("code_encode", lambda: model_adapter.encode_codes(codes))
    score_matrix = profiler.profile("similarity_retrieval", lambda: query_embs @ code_embs.T)
    metrics = profiler.profile(
        "metric_aggregation",
        lambda: paired_ranking_metrics(score_matrix.numpy(), ks=cfg.ks),
    )
    profiling = profiler.finalize()

    return {
        "dataset": {
            "name": cfg.dataset_name,
            "path": cfg.dataset_path,
            "counts": corpus.counts(),
        },
        "model": {
            **model_adapter.metadata(),
            "source": cfg.model_source,
            "checkpoint_format": cfg.checkpoint_format or "auto",
        },
        "split": cfg.split,
        "metrics": metrics,
        "profiling": profiling,
        "counts": {
            "queries": len(queries),
            "codes": len(codes),
        },
        "config": {
            "ks": list(cfg.ks),
            "max_query_length": cfg.max_query_length,
            "max_code_length": cfg.max_code_length,
            "batch_size": cfg.batch_size,
            "device": str(device),
            "seed": cfg.seed,
        },
    }


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return pick_device()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device request: {requested}")
