#!/usr/bin/env python3
"""Run MBPP retrieval experiments and persist results artifacts.

This script is the source of truth for reproducible runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

# This runner is expected to execute in constrained/network-restricted envs.
# Force cache-only behavior to avoid long hangs on remote retries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


K_VALUES = (1, 5, 10, 20)
METRIC_COLUMNS = [
    "run_id",
    "timestamp",
    "method",
    "stage",
    "technique",
    "model_name",
    "protocol",
    "train_split",
    "eval_split",
    "config_id",
    "mrr",
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@20",
    "map@10",
    "ndcg@10",
    "precision@10",
    "mean_rank",
    "runtime_sec",
    "status",
    "ranks_file",
    "error",
]
TRAINING_COLUMNS = [
    "run_id",
    "timestamp",
    "stage",
    "technique",
    "model_name",
    "config_id",
    "train_split",
    "epoch",
    "avg_loss",
    "total_steps",
    "warmup_steps",
    "runtime_sec",
    "status",
    "error",
]
COMPARISON_COLUMNS = [
    "run_id",
    "timestamp",
    "comparison",
    "protocol",
    "metric",
    "base_method",
    "base_stage",
    "base_model",
    "compare_method",
    "compare_stage",
    "compare_model",
    "base_value",
    "compare_value",
    "delta",
    "ci_low",
    "ci_high",
    "n_bootstrap",
    "status",
    "notes",
]
DEFAULT_MODEL_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
]


@dataclass(frozen=True)
class SweepConfig:
    config_id: str
    epochs: int
    batch_size: int
    lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class HardNegativeConfig:
    config_id: str
    epochs: int
    batch_size: int
    lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    triplet_margin: float = 0.2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root (missing pyproject.toml).")


def append_row(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df_old = pd.read_csv(path, keep_default_na=False)
    else:
        df_old = pd.DataFrame(columns=columns)

    row_clean = {k: row.get(k, "") for k in columns}
    if df_old.empty:
        df_new = pd.DataFrame([row_clean], columns=columns)
    else:
        df_new = pd.concat([df_old, pd.DataFrame([row_clean])], ignore_index=True)
    df_new.to_csv(path, index=False)


def load_df_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, keep_default_na=False)
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df[columns]
    return pd.DataFrame(columns=columns)


def dataframe_to_markdown_compat(df: pd.DataFrame) -> str:
    """Render Markdown table without requiring optional tabulate dependency."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        if df.empty:
            return ""

        def format_cell(value: Any) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            return text.replace("|", r"\|").replace("\n", "<br>")

        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        divider = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(format_cell(row[c]) for c in cols) + " |")
        return "\n".join([header, divider] + rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "mps":
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_texts_for_model(model_name: str, queries: list[str], codes: list[str]) -> tuple[list[str], list[str]]:
    name = model_name.lower()
    if "e5" in name:
        return [f"query: {q}" for q in queries], [f"passage: {c}" for c in codes]
    if "bge" in name:
        prompt = "Represent this sentence for searching relevant code snippets: "
        return [prompt + q for q in queries], codes
    return queries, codes


def compute_ranks_from_embeddings(query_embeddings: np.ndarray, code_embeddings: np.ndarray) -> np.ndarray:
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    code_embeddings = np.asarray(code_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(code_embeddings.shape[1])
    index.add(code_embeddings)
    _, neighbors = index.search(query_embeddings, code_embeddings.shape[0])
    ranks = np.empty(query_embeddings.shape[0], dtype=np.int32)
    for i, retrieved in enumerate(neighbors):
        ranks[i] = int(np.where(retrieved == i)[0][0]) + 1
    return ranks


def compute_metrics_from_ranks(ranks: np.ndarray, k_values: tuple[int, ...] = K_VALUES) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int32)
    metrics: dict[str, float] = {
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    for k in k_values:
        hits = (ranks <= k).astype(np.float32)
        metrics[f"recall@{k}"] = float(np.mean(hits))
        metrics[f"precision@{k}"] = float(np.mean(hits / k))
        ap_k = np.where(ranks <= k, 1.0 / ranks, 0.0)
        ndcg_k = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
        metrics[f"map@{k}"] = float(np.mean(ap_k))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_k))
    return metrics


def exact_random_metrics(num_docs: int, k_values: tuple[int, ...] = K_VALUES) -> dict[str, float]:
    ranks = np.arange(1, num_docs + 1, dtype=np.float64)
    probs = np.ones(num_docs, dtype=np.float64) / num_docs
    out: dict[str, float] = {
        "mrr": float(np.sum((1.0 / ranks) * probs)),
        "mean_rank": float(np.sum(ranks * probs)),
    }
    for k in k_values:
        hits = (ranks <= k).astype(np.float64)
        out[f"recall@{k}"] = float(np.sum(hits * probs))
        out[f"precision@{k}"] = float(np.sum((hits / k) * probs))
        out[f"map@{k}"] = float(np.sum(np.where(ranks <= k, 1.0 / ranks, 0.0) * probs))
        out[f"ndcg@{k}"] = float(np.sum(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0) * probs))
    return out


def evaluate_tfidf(queries: list[str], codes: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), lowercase=True)
    code_mat = vectorizer.fit_transform(codes)
    query_mat = vectorizer.transform(queries)
    sim = (query_mat @ code_mat.T).toarray()
    order = np.argsort(-sim, axis=1)
    ranks = np.empty(sim.shape[0], dtype=np.int32)
    for i in range(sim.shape[0]):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    return ranks, compute_metrics_from_ranks(ranks)


def evaluate_dense(
    model: SentenceTransformer,
    model_name: str,
    queries: list[str],
    codes: list[str],
    batch_size: int = 16,
) -> tuple[np.ndarray, dict[str, float]]:
    q_fmt, c_fmt = format_texts_for_model(model_name, queries, codes)
    q_emb = model.encode(
        q_fmt,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    c_emb = model.encode(
        c_fmt,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    ranks = compute_ranks_from_embeddings(q_emb, c_emb)
    return ranks, compute_metrics_from_ranks(ranks)


def build_examples(pairs: list[dict[str, Any]]) -> list[InputExample]:
    return [InputExample(texts=[p["query"], p["code"]]) for p in pairs]


def run_training_loop(
    model: SentenceTransformer,
    train_examples: list[InputExample],
    loss_fn: torch.nn.Module,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    desc_prefix: str,
) -> dict[str, Any]:
    del device  # model.to(device) done by caller; keep signature explicit.

    dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, drop_last=True)
    dataloader.collate_fn = model.smart_batching_collate

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(dataloader))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    epoch_rows: list[dict[str, Any]] = []
    train_start = time.perf_counter()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"{desc_prefix} Epoch {epoch + 1}/{epochs}")
        for features, labels in pbar:
            optimizer.zero_grad()
            loss_value = loss_fn(features, labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss_value.item())
            pbar.set_postfix(loss=f"{loss_value.item():.4f}")

        avg_loss = running_loss / max(1, len(dataloader))
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            }
        )
    train_runtime = time.perf_counter() - train_start
    return {
        "epoch_rows": epoch_rows,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "runtime_sec": train_runtime,
    }


def finetune_with_mnr(
    base_model_name: str,
    train_examples: list[InputExample],
    device: str,
    cfg: SweepConfig,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    model = SentenceTransformer(base_model_name, local_files_only=True)
    model.to(device)
    loss_fn = losses.MultipleNegativesRankingLoss(model=model)
    stats = run_training_loop(
        model=model,
        train_examples=train_examples,
        loss_fn=loss_fn,
        device=device,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        desc_prefix="MNR",
    )
    return model, stats


def mine_hard_negatives(
    model: SentenceTransformer,
    model_name: str,
    pairs: list[dict[str, Any]],
    batch_size: int = 16,
) -> list[InputExample]:
    queries = [p["query"] for p in pairs]
    codes = [p["code"] for p in pairs]
    q_fmt, c_fmt = format_texts_for_model(model_name, queries, codes)

    q_emb = model.encode(
        q_fmt,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    c_emb = model.encode(
        c_fmt,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(c_emb.shape[1])
    index.add(c_emb)
    _, neighbors = index.search(q_emb, min(10, len(codes)))

    rng = np.random.default_rng(42)
    hard_examples: list[InputExample] = []
    n = len(codes)
    for i, row in enumerate(neighbors):
        neg_idx = None
        for idx in row:
            if idx != i:
                neg_idx = int(idx)
                break
        if neg_idx is None:
            neg_idx = int(rng.integers(0, n - 1))
            if neg_idx >= i:
                neg_idx += 1
            neg_idx = min(neg_idx, n - 1)
        hard_examples.append(InputExample(texts=[queries[i], codes[i], codes[neg_idx]]))
    return hard_examples


def finetune_with_hard_negatives(
    starting_model_path: str | Path,
    model_name_for_formatting: str,
    train_pairs: list[dict[str, Any]],
    device: str,
    cfg: HardNegativeConfig,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    model = SentenceTransformer(str(starting_model_path), local_files_only=True)
    model.to(device)
    hard_examples = mine_hard_negatives(model, model_name_for_formatting, train_pairs)

    loss_fn = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=cfg.triplet_margin,
    )
    stats = run_training_loop(
        model=model,
        train_examples=hard_examples,
        loss_fn=loss_fn,
        device=device,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        desc_prefix="HardNeg",
    )
    stats["num_hard_examples"] = len(hard_examples)
    return model, stats


def rank_file_relpath(
    method: str,
    stage: str,
    technique: str,
    model_name: str,
    protocol: str,
    config_id: str,
) -> str:
    signature = "|".join([method, stage, technique, model_name, protocol, config_id])
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"ranks/{safe_slug(method)}__{safe_slug(stage)}__{safe_slug(protocol)}__{digest}.npy"


def save_ranks(run_dir: Path, rel_path: str, ranks: np.ndarray) -> Path:
    out = run_dir / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, ranks.astype(np.int32))
    return out


def load_ranks_if_exists(run_dir: Path, rel_path: str) -> np.ndarray | None:
    p = run_dir / rel_path
    if p.exists():
        return np.load(p)
    return None


def bootstrap_diff_ci(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    if ranks_a.shape != ranks_b.shape:
        raise ValueError("Rank arrays for bootstrap must have identical shape.")

    rng = np.random.default_rng(seed)
    n = ranks_a.shape[0]
    diffs = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a = ranks_a[idx]
        b = ranks_b[idx]
        if metric == "mrr":
            va = float(np.mean(1.0 / a))
            vb = float(np.mean(1.0 / b))
        elif metric == "recall@10":
            va = float(np.mean(a <= 10))
            vb = float(np.mean(b <= 10))
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        diffs[i] = va - vb

    delta = float(np.mean(diffs))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return delta, float(ci_low), float(ci_high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MBPP retrieval experiment matrix.")
    parser.add_argument(
        "--output-dir",
        default="experiments/kai/results",
        help="Output root relative to project root.",
    )
    parser.add_argument("--run-id", default=None, help="Run identifier; default is timestamp.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-matrix", action="store_true", help="Run full experiment matrix.")
    parser.add_argument("--fast-smoke", action="store_true", help="Quick schema smoke test.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing artifacts.")
    parser.add_argument(
        "--finetune-all-pretrained",
        action="store_true",
        help="Fine-tune all pretrained candidates (not just selected backbone).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.full_matrix and args.fast_smoke:
        print("Cannot use --full-matrix and --fast-smoke together.", file=sys.stderr)
        return 2

    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path.parent)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("mbpp_run_%Y%m%dT%H%M%SZ")
    output_root = (project_root / args.output_dir).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = run_dir / "metrics"
    reports_dir = run_dir / "reports"
    metadata_dir = run_dir / "metadata"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "metrics_all.csv"
    training_path = metrics_dir / "training_stats.csv"
    comparisons_path = metrics_dir / "comparisons.csv"
    summary_md_path = reports_dir / "summary.md"
    summary_txt_path = reports_dir / "summary.txt"
    metadata_path = metadata_dir / "run_metadata.json"
    failures_path = logs_dir / "failures.log"

    existing_metrics = load_df_or_empty(metrics_path, METRIC_COLUMNS)
    done_success_keys: set[tuple[str, ...]] = set()
    for _, row in existing_metrics.iterrows():
        if str(row.get("status", "")) != "success":
            continue
        key = (
            str(row["method"]),
            str(row["stage"]),
            str(row["technique"]),
            str(row["model_name"]),
            str(row["protocol"]),
            str(row["train_split"]),
            str(row["eval_split"]),
            str(row["config_id"]),
        )
        done_success_keys.add(key)

    def step_key(
        method: str,
        stage: str,
        technique: str,
        model_name: str,
        protocol: str,
        train_split: str,
        eval_split: str,
        config_id: str,
    ) -> tuple[str, ...]:
        return (method, stage, technique, model_name, protocol, train_split, eval_split, config_id)

    def log_failure(message: str) -> None:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_path.open("a", encoding="utf-8") as f:
            f.write(f"[{utc_now_iso()}] {message}\n")

    seed_everything(args.seed)
    resolved_device = resolve_device(args.device)
    start_time = time.perf_counter()

    if args.fast_smoke:
        model_candidates = DEFAULT_MODEL_CANDIDATES[:2]
    else:
        model_candidates = list(DEFAULT_MODEL_CANDIDATES)

    sweep_configs = [
        SweepConfig("sweep_e1_b16_lr2e5", epochs=1, batch_size=16, lr=2e-5, warmup_ratio=0.10, weight_decay=0.01),
        SweepConfig("sweep_e2_b16_lr2e5", epochs=2, batch_size=16, lr=2e-5, warmup_ratio=0.10, weight_decay=0.01),
        SweepConfig("sweep_e1_b32_lr1e5", epochs=1, batch_size=32, lr=1e-5, warmup_ratio=0.06, weight_decay=0.01),
    ]
    if args.fast_smoke:
        sweep_configs = sweep_configs[:1]

    run_pretrained = True
    run_sweep = args.full_matrix and not args.fast_smoke
    run_final = args.full_matrix and not args.fast_smoke
    run_hardneg = args.full_matrix and not args.fast_smoke

    hardneg_cfg_template = HardNegativeConfig(
        config_id="hardneg_e1_triplet",
        epochs=1,
        batch_size=16,
        lr=1e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        max_grad_norm=1.0,
        triplet_margin=0.2,
    )

    metadata = {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "device_arg": args.device,
        "resolved_device": resolved_device,
        "seed": args.seed,
        "args": vars(args),
        "model_candidates": model_candidates,
        "sweep_configs": [asdict(c) for c in sweep_configs],
        "hardneg_template": asdict(hardneg_cfg_template),
        "k_values": list(K_VALUES),
        "run_flags": {
            "pretrained": run_pretrained,
            "sweep": run_sweep,
            "final": run_final,
            "hardneg": run_hardneg,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Loading MBPP dataset...")
    ds = load_dataset("google-research-datasets/mbpp", download_mode="reuse_dataset_if_exists")

    def extract_pairs(split_name: str) -> list[dict[str, Any]]:
        rows = []
        for r in ds[split_name]:
            q = str(r["text"]).strip()
            c = str(r["code"]).strip()
            if q and c:
                rows.append({"query": q, "code": c, "task_id": int(r["task_id"]), "split": split_name})
        rows.sort(key=lambda x: x["task_id"])
        return rows

    train_pairs = extract_pairs("train")
    val_pairs = extract_pairs("validation")
    test_pairs = extract_pairs("test")
    prompt_pairs = extract_pairs("prompt")

    train_tune_pairs = train_pairs + prompt_pairs
    final_train_pairs = train_pairs + val_pairs + prompt_pairs
    full_pairs = train_pairs + val_pairs + test_pairs + prompt_pairs

    protocols = {
        "heldout_test": {
            "queries": [p["query"] for p in test_pairs],
            "codes": [p["code"] for p in test_pairs],
            "eval_split": "test",
        },
        "full_corpus": {
            "queries": [p["query"] for p in full_pairs],
            "codes": [p["code"] for p in full_pairs],
            "eval_split": "all",
        },
        "tune_validation": {
            "queries": [p["query"] for p in val_pairs],
            "codes": [p["code"] for p in val_pairs],
            "eval_split": "validation",
        },
    }

    def write_metric_row(
        *,
        method: str,
        stage: str,
        technique: str,
        model_name: str,
        protocol: str,
        train_split: str,
        eval_split: str,
        config_id: str,
        metrics: dict[str, float] | None,
        runtime_sec: float,
        status: str,
        ranks_relpath: str = "",
        error: str = "",
    ) -> None:
        row = {
            "run_id": run_id,
            "timestamp": utc_now_iso(),
            "method": method,
            "stage": stage,
            "technique": technique,
            "model_name": model_name,
            "protocol": protocol,
            "train_split": train_split,
            "eval_split": eval_split,
            "config_id": config_id,
            "runtime_sec": runtime_sec,
            "status": status,
            "ranks_file": ranks_relpath,
            "error": error,
        }
        for col in ["mrr", "recall@1", "recall@5", "recall@10", "recall@20", "map@10", "ndcg@10", "precision@10", "mean_rank"]:
            row[col] = float(metrics[col]) if (metrics and col in metrics) else np.nan

        append_row(metrics_path, row, METRIC_COLUMNS)
        if status == "success":
            done_success_keys.add(step_key(method, stage, technique, model_name, protocol, train_split, eval_split, config_id))

    def write_training_rows(
        *,
        stage: str,
        technique: str,
        model_name: str,
        config_id: str,
        train_split: str,
        stats: dict[str, Any],
        status: str = "success",
        error: str = "",
    ) -> None:
        for epoch_row in stats.get("epoch_rows", []):
            row = {
                "run_id": run_id,
                "timestamp": utc_now_iso(),
                "stage": stage,
                "technique": technique,
                "model_name": model_name,
                "config_id": config_id,
                "train_split": train_split,
                "epoch": epoch_row["epoch"],
                "avg_loss": epoch_row["avg_loss"],
                "total_steps": stats.get("total_steps", epoch_row.get("total_steps", 0)),
                "warmup_steps": stats.get("warmup_steps", epoch_row.get("warmup_steps", 0)),
                "runtime_sec": stats.get("runtime_sec", np.nan),
                "status": status,
                "error": error,
            }
            append_row(training_path, row, TRAINING_COLUMNS)

    # 1) Baselines: random + tfidf for heldout_test and full_corpus
    for protocol_name in ("heldout_test", "full_corpus"):
        pool = protocols[protocol_name]
        eval_split = str(pool["eval_split"])
        num_docs = len(pool["codes"])

        key_random = step_key("baseline", "baseline", "random", "random", protocol_name, "n/a", eval_split, "baseline_random")
        if not (args.resume and key_random in done_success_keys):
            start = time.perf_counter()
            try:
                random_metrics = exact_random_metrics(num_docs=num_docs, k_values=K_VALUES)
                write_metric_row(
                    method="baseline",
                    stage="baseline",
                    technique="random",
                    model_name="random",
                    protocol=protocol_name,
                    train_split="n/a",
                    eval_split=eval_split,
                    config_id="baseline_random",
                    metrics=random_metrics,
                    runtime_sec=time.perf_counter() - start,
                    status="success",
                )
            except Exception as exc:  # pragma: no cover
                tb = traceback.format_exc(limit=5)
                log_failure(f"random baseline failed ({protocol_name}): {exc}\n{tb}")
                write_metric_row(
                    method="baseline",
                    stage="baseline",
                    technique="random",
                    model_name="random",
                    protocol=protocol_name,
                    train_split="n/a",
                    eval_split=eval_split,
                    config_id="baseline_random",
                    metrics=None,
                    runtime_sec=time.perf_counter() - start,
                    status="failed",
                    error=str(exc),
                )

        key_tfidf = step_key("baseline", "baseline", "tfidf", "tfidf_lexical", protocol_name, "n/a", eval_split, "baseline_tfidf")
        tfidf_rank_rel = rank_file_relpath("baseline", "baseline", "tfidf", "tfidf_lexical", protocol_name, "baseline_tfidf")
        tfidf_rank_exists = (run_dir / tfidf_rank_rel).exists()
        if not (args.resume and key_tfidf in done_success_keys and tfidf_rank_exists):
            start = time.perf_counter()
            try:
                ranks, tfidf_metrics = evaluate_tfidf(pool["queries"], pool["codes"])
                save_ranks(run_dir, tfidf_rank_rel, ranks)
                write_metric_row(
                    method="baseline",
                    stage="baseline",
                    technique="tfidf",
                    model_name="tfidf_lexical",
                    protocol=protocol_name,
                    train_split="n/a",
                    eval_split=eval_split,
                    config_id="baseline_tfidf",
                    metrics=tfidf_metrics,
                    runtime_sec=time.perf_counter() - start,
                    status="success",
                    ranks_relpath=tfidf_rank_rel,
                )
            except Exception as exc:
                tb = traceback.format_exc(limit=5)
                log_failure(f"tfidf baseline failed ({protocol_name}): {exc}\n{tb}")
                write_metric_row(
                    method="baseline",
                    stage="baseline",
                    technique="tfidf",
                    model_name="tfidf_lexical",
                    protocol=protocol_name,
                    train_split="n/a",
                    eval_split=eval_split,
                    config_id="baseline_tfidf",
                    metrics=None,
                    runtime_sec=time.perf_counter() - start,
                    status="failed",
                    error=str(exc),
                )

    # 2) Pretrained dense matrix
    if run_pretrained:
        for model_name in model_candidates:
            for protocol_name in ("heldout_test", "full_corpus"):
                pool = protocols[protocol_name]
                eval_split = str(pool["eval_split"])
                config_id = "zero_shot"
                key = step_key("pretrained", "pretrained", "zero_shot", model_name, protocol_name, "n/a", eval_split, config_id)
                rank_rel = rank_file_relpath("pretrained", "pretrained", "zero_shot", model_name, protocol_name, config_id)
                rank_exists = (run_dir / rank_rel).exists()
                if args.resume and key in done_success_keys and rank_exists:
                    continue

                print(f"[pretrained] Evaluating {model_name} on {protocol_name} ...")
                start = time.perf_counter()
                try:
                    model = SentenceTransformer(model_name, local_files_only=True)
                    model.to(resolved_device)
                    ranks, metrics = evaluate_dense(model, model_name, pool["queries"], pool["codes"])
                    save_ranks(run_dir, rank_rel, ranks)
                    write_metric_row(
                        method="pretrained",
                        stage="pretrained",
                        technique="zero_shot",
                        model_name=model_name,
                        protocol=protocol_name,
                        train_split="n/a",
                        eval_split=eval_split,
                        config_id=config_id,
                        metrics=metrics,
                        runtime_sec=time.perf_counter() - start,
                        status="success",
                        ranks_relpath=rank_rel,
                    )
                except Exception as exc:
                    tb = traceback.format_exc(limit=5)
                    log_failure(f"pretrained eval failed ({model_name}, {protocol_name}): {exc}\n{tb}")
                    write_metric_row(
                        method="pretrained",
                        stage="pretrained",
                        technique="zero_shot",
                        model_name=model_name,
                        protocol=protocol_name,
                        train_split="n/a",
                        eval_split=eval_split,
                        config_id=config_id,
                        metrics=None,
                        runtime_sec=time.perf_counter() - start,
                        status="failed",
                        error=str(exc),
                    )

    # Determine selected backbone from heldout_test pretrained rows.
    metrics_df = load_df_or_empty(metrics_path, METRIC_COLUMNS)
    pretrained_test = metrics_df[
        (metrics_df["method"] == "pretrained")
        & (metrics_df["stage"] == "pretrained")
        & (metrics_df["protocol"] == "heldout_test")
        & (metrics_df["status"] == "success")
    ].copy()
    if pretrained_test.empty:
        print("No successful pretrained test rows found; cannot continue to fine-tuning.", file=sys.stderr)
        return 1
    pretrained_test["mrr"] = pd.to_numeric(pretrained_test["mrr"], errors="coerce")
    pretrained_test["recall@10"] = pd.to_numeric(pretrained_test["recall@10"], errors="coerce")
    pretrained_test = pretrained_test.sort_values(by=["mrr", "recall@10"], ascending=[False, False])
    selected_model_name = str(pretrained_test.iloc[0]["model_name"])
    print(f"Selected backbone: {selected_model_name}")

    # 3) Hyperparameter sweep on train+prompt, evaluate on validation.
    train_tune_examples = build_examples(train_tune_pairs)
    sweep_checkpoints_dir = run_dir / "checkpoints" / "sweep"
    sweep_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if run_sweep:
        for cfg in sweep_configs:
            key = step_key(
                "finetune",
                "sweep_mnr",
                "mnr",
                selected_model_name,
                "tune_validation",
                "train+prompt",
                "validation",
                cfg.config_id,
            )
            rank_rel = rank_file_relpath("finetune", "sweep_mnr", "mnr", selected_model_name, "tune_validation", cfg.config_id)
            rank_exists = (run_dir / rank_rel).exists()
            if args.resume and key in done_success_keys and rank_exists:
                continue

            print(f"[sweep] Running {cfg.config_id} ...")
            start = time.perf_counter()
            ckpt_dir = sweep_checkpoints_dir / safe_slug(cfg.config_id)
            try:
                model, stats = finetune_with_mnr(
                    base_model_name=selected_model_name,
                    train_examples=train_tune_examples,
                    device=resolved_device,
                    cfg=cfg,
                )
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(ckpt_dir))
                write_training_rows(
                    stage="sweep_mnr",
                    technique="mnr",
                    model_name=selected_model_name,
                    config_id=cfg.config_id,
                    train_split="train+prompt",
                    stats=stats,
                )

                val_pool = protocols["tune_validation"]
                ranks, metrics = evaluate_dense(model, selected_model_name, val_pool["queries"], val_pool["codes"])
                save_ranks(run_dir, rank_rel, ranks)
                write_metric_row(
                    method="finetune",
                    stage="sweep_mnr",
                    technique="mnr",
                    model_name=selected_model_name,
                    protocol="tune_validation",
                    train_split="train+prompt",
                    eval_split="validation",
                    config_id=cfg.config_id,
                    metrics=metrics,
                    runtime_sec=time.perf_counter() - start,
                    status="success",
                    ranks_relpath=rank_rel,
                )
            except Exception as exc:
                tb = traceback.format_exc(limit=5)
                log_failure(f"sweep failed ({cfg.config_id}): {exc}\n{tb}")
                write_metric_row(
                    method="finetune",
                    stage="sweep_mnr",
                    technique="mnr",
                    model_name=selected_model_name,
                    protocol="tune_validation",
                    train_split="train+prompt",
                    eval_split="validation",
                    config_id=cfg.config_id,
                    metrics=None,
                    runtime_sec=time.perf_counter() - start,
                    status="failed",
                    error=str(exc),
                )

    # Select best config from sweep rows (or fallback if sweep disabled).
    metrics_df = load_df_or_empty(metrics_path, METRIC_COLUMNS)
    sweep_rows = metrics_df[
        (metrics_df["method"] == "finetune")
        & (metrics_df["stage"] == "sweep_mnr")
        & (metrics_df["protocol"] == "tune_validation")
        & (metrics_df["status"] == "success")
    ].copy()
    if not sweep_rows.empty:
        sweep_rows["mrr"] = pd.to_numeric(sweep_rows["mrr"], errors="coerce")
        sweep_rows["recall@10"] = pd.to_numeric(sweep_rows["recall@10"], errors="coerce")
        sweep_rows = sweep_rows.sort_values(by=["mrr", "recall@10"], ascending=[False, False])
        best_config_id = str(sweep_rows.iloc[0]["config_id"])
        cfg_map = {cfg.config_id: cfg for cfg in sweep_configs}
        best_cfg = cfg_map.get(best_config_id, sweep_configs[0])
    else:
        best_cfg = sweep_configs[0]
        best_config_id = best_cfg.config_id
    print(f"Best config: {best_config_id}")

    # 4) Final standard fine-tune on train+validation+prompt.
    final_train_examples = build_examples(final_train_pairs)
    if args.finetune_all_pretrained:
        finetune_model_names = list(pretrained_test["model_name"].drop_duplicates())
    else:
        finetune_model_names = [selected_model_name]
    print(f"Final fine-tune targets: {finetune_model_names}")

    if run_final:
        for target_model_name in finetune_model_names:
            standard_ckpt = run_dir / "checkpoints" / "final_standard_mnr" / safe_slug(target_model_name)
            final_model = None

            train_key = step_key(
                "finetune",
                "final_standard",
                "mnr",
                target_model_name,
                "train_only",
                "train+validation+prompt",
                "n/a",
                best_cfg.config_id,
            )
            if not (args.resume and train_key in done_success_keys and standard_ckpt.exists()):
                print(f"[final] Training standard MNR model ({target_model_name}) ...")
                t0 = time.perf_counter()
                try:
                    final_model, stats = finetune_with_mnr(
                        base_model_name=target_model_name,
                        train_examples=final_train_examples,
                        device=resolved_device,
                        cfg=best_cfg,
                    )
                    standard_ckpt.mkdir(parents=True, exist_ok=True)
                    final_model.save(str(standard_ckpt))
                    write_training_rows(
                        stage="final_standard",
                        technique="mnr",
                        model_name=target_model_name,
                        config_id=best_cfg.config_id,
                        train_split="train+validation+prompt",
                        stats=stats,
                    )
                    write_metric_row(
                        method="finetune",
                        stage="final_standard",
                        technique="mnr",
                        model_name=target_model_name,
                        protocol="train_only",
                        train_split="train+validation+prompt",
                        eval_split="n/a",
                        config_id=best_cfg.config_id,
                        metrics={"mrr": np.nan},
                        runtime_sec=time.perf_counter() - t0,
                        status="success",
                    )
                except Exception as exc:
                    tb = traceback.format_exc(limit=5)
                    log_failure(f"final standard training failed ({target_model_name}): {exc}\n{tb}")
                    write_metric_row(
                        method="finetune",
                        stage="final_standard",
                        technique="mnr",
                        model_name=target_model_name,
                        protocol="train_only",
                        train_split="train+validation+prompt",
                        eval_split="n/a",
                        config_id=best_cfg.config_id,
                        metrics=None,
                        runtime_sec=time.perf_counter() - t0,
                        status="failed",
                        error=str(exc),
                    )
            if final_model is None and standard_ckpt.exists():
                final_model = SentenceTransformer(str(standard_ckpt), local_files_only=True)
                final_model.to(resolved_device)

            # Evaluate standard model on heldout_test and full_corpus.
            if final_model is not None:
                for protocol_name in ("heldout_test", "full_corpus"):
                    pool = protocols[protocol_name]
                    eval_split = str(pool["eval_split"])
                    key = step_key(
                        "finetune",
                        "final_standard",
                        "mnr",
                        target_model_name,
                        protocol_name,
                        "train+validation+prompt",
                        eval_split,
                        best_cfg.config_id,
                    )
                    rank_rel = rank_file_relpath(
                        "finetune",
                        "final_standard",
                        "mnr",
                        target_model_name,
                        protocol_name,
                        best_cfg.config_id,
                    )
                    rank_exists = (run_dir / rank_rel).exists()
                    if args.resume and key in done_success_keys and rank_exists:
                        continue

                    t0 = time.perf_counter()
                    try:
                        ranks, metrics = evaluate_dense(final_model, target_model_name, pool["queries"], pool["codes"])
                        save_ranks(run_dir, rank_rel, ranks)
                        write_metric_row(
                            method="finetune",
                            stage="final_standard",
                            technique="mnr",
                            model_name=target_model_name,
                            protocol=protocol_name,
                            train_split="train+validation+prompt",
                            eval_split=eval_split,
                            config_id=best_cfg.config_id,
                            metrics=metrics,
                            runtime_sec=time.perf_counter() - t0,
                            status="success",
                            ranks_relpath=rank_rel,
                        )
                    except Exception as exc:
                        tb = traceback.format_exc(limit=5)
                        log_failure(f"final standard eval failed ({target_model_name}, {protocol_name}): {exc}\n{tb}")
                        write_metric_row(
                            method="finetune",
                            stage="final_standard",
                            technique="mnr",
                            model_name=target_model_name,
                            protocol=protocol_name,
                            train_split="train+validation+prompt",
                            eval_split=eval_split,
                            config_id=best_cfg.config_id,
                            metrics=None,
                            runtime_sec=time.perf_counter() - t0,
                            status="failed",
                            error=str(exc),
                        )

    # 5) Hard-negative stage from final standard checkpoints.
    if run_hardneg:
        for target_model_name in finetune_model_names:
            standard_ckpt = run_dir / "checkpoints" / "final_standard_mnr" / safe_slug(target_model_name)
            if not standard_ckpt.exists():
                continue

            hardneg_ckpt = run_dir / "checkpoints" / "final_hardneg_triplet" / safe_slug(target_model_name)
            hard_cfg = HardNegativeConfig(
                config_id=hardneg_cfg_template.config_id,
                epochs=hardneg_cfg_template.epochs,
                batch_size=best_cfg.batch_size,
                lr=min(best_cfg.lr, hardneg_cfg_template.lr),
                warmup_ratio=hardneg_cfg_template.warmup_ratio,
                weight_decay=best_cfg.weight_decay,
                max_grad_norm=hardneg_cfg_template.max_grad_norm,
                triplet_margin=hardneg_cfg_template.triplet_margin,
            )

            train_key = step_key(
                "finetune",
                "final_hardneg",
                "hard_negative_triplet",
                target_model_name,
                "train_only",
                "train+validation+prompt",
                "n/a",
                hard_cfg.config_id,
            )
            hard_model = None
            if not (args.resume and train_key in done_success_keys and hardneg_ckpt.exists()):
                print(f"[hardneg] Training hard-negative model ({target_model_name}) ...")
                t0 = time.perf_counter()
                try:
                    hard_model, stats = finetune_with_hard_negatives(
                        starting_model_path=standard_ckpt,
                        model_name_for_formatting=target_model_name,
                        train_pairs=final_train_pairs,
                        device=resolved_device,
                        cfg=hard_cfg,
                    )
                    hardneg_ckpt.mkdir(parents=True, exist_ok=True)
                    hard_model.save(str(hardneg_ckpt))
                    write_training_rows(
                        stage="final_hardneg",
                        technique="hard_negative_triplet",
                        model_name=target_model_name,
                        config_id=hard_cfg.config_id,
                        train_split="train+validation+prompt",
                        stats=stats,
                    )
                    write_metric_row(
                        method="finetune",
                        stage="final_hardneg",
                        technique="hard_negative_triplet",
                        model_name=target_model_name,
                        protocol="train_only",
                        train_split="train+validation+prompt",
                        eval_split="n/a",
                        config_id=hard_cfg.config_id,
                        metrics={"mrr": np.nan},
                        runtime_sec=time.perf_counter() - t0,
                        status="success",
                    )
                except Exception as exc:
                    tb = traceback.format_exc(limit=5)
                    log_failure(f"hard-negative training failed ({target_model_name}): {exc}\n{tb}")
                    write_metric_row(
                        method="finetune",
                        stage="final_hardneg",
                        technique="hard_negative_triplet",
                        model_name=target_model_name,
                        protocol="train_only",
                        train_split="train+validation+prompt",
                        eval_split="n/a",
                        config_id=hard_cfg.config_id,
                        metrics=None,
                        runtime_sec=time.perf_counter() - t0,
                        status="failed",
                        error=str(exc),
                    )

            if hard_model is None and hardneg_ckpt.exists():
                hard_model = SentenceTransformer(str(hardneg_ckpt), local_files_only=True)
                hard_model.to(resolved_device)

            if hard_model is not None:
                for protocol_name in ("heldout_test", "full_corpus"):
                    pool = protocols[protocol_name]
                    eval_split = str(pool["eval_split"])
                    key = step_key(
                        "finetune",
                        "final_hardneg",
                        "hard_negative_triplet",
                        target_model_name,
                        protocol_name,
                        "train+validation+prompt",
                        eval_split,
                        hard_cfg.config_id,
                    )
                    rank_rel = rank_file_relpath(
                        "finetune",
                        "final_hardneg",
                        "hard_negative_triplet",
                        target_model_name,
                        protocol_name,
                        hard_cfg.config_id,
                    )
                    rank_exists = (run_dir / rank_rel).exists()
                    if args.resume and key in done_success_keys and rank_exists:
                        continue

                    t0 = time.perf_counter()
                    try:
                        ranks, metrics = evaluate_dense(hard_model, target_model_name, pool["queries"], pool["codes"])
                        save_ranks(run_dir, rank_rel, ranks)
                        write_metric_row(
                            method="finetune",
                            stage="final_hardneg",
                            technique="hard_negative_triplet",
                            model_name=target_model_name,
                            protocol=protocol_name,
                            train_split="train+validation+prompt",
                            eval_split=eval_split,
                            config_id=hard_cfg.config_id,
                            metrics=metrics,
                            runtime_sec=time.perf_counter() - t0,
                            status="success",
                            ranks_relpath=rank_rel,
                        )
                    except Exception as exc:
                        tb = traceback.format_exc(limit=5)
                        log_failure(f"hard-negative eval failed ({target_model_name}, {protocol_name}): {exc}\n{tb}")
                        write_metric_row(
                            method="finetune",
                            stage="final_hardneg",
                            technique="hard_negative_triplet",
                            model_name=target_model_name,
                            protocol=protocol_name,
                            train_split="train+validation+prompt",
                            eval_split=eval_split,
                            config_id=hard_cfg.config_id,
                            metrics=None,
                            runtime_sec=time.perf_counter() - t0,
                            status="failed",
                            error=str(exc),
                        )

    # 6) Comparisons and summary.
    metrics_df = load_df_or_empty(metrics_path, METRIC_COLUMNS)
    for col in ["mrr", "recall@10", "recall@1", "recall@5", "recall@20", "map@10", "ndcg@10", "precision@10", "mean_rank", "runtime_sec"]:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    def pick_row(
        method: str,
        stage: str,
        technique: str,
        model_name: str,
        protocol: str,
        status: str = "success",
    ) -> pd.Series | None:
        filt = (
            (metrics_df["method"] == method)
            & (metrics_df["stage"] == stage)
            & (metrics_df["technique"] == technique)
            & (metrics_df["model_name"] == model_name)
            & (metrics_df["protocol"] == protocol)
            & (metrics_df["status"] == status)
        )
        rows = metrics_df[filt]
        if rows.empty:
            return None
        return rows.sort_values("timestamp").iloc[-1]

    # Determine best pretrained on heldout_test by mrr/recall@10.
    pre_test = metrics_df[
        (metrics_df["method"] == "pretrained")
        & (metrics_df["stage"] == "pretrained")
        & (metrics_df["protocol"] == "heldout_test")
        & (metrics_df["status"] == "success")
    ].copy()
    if not pre_test.empty:
        pre_test = pre_test.sort_values(by=["mrr", "recall@10"], ascending=[False, False])
        best_pre_row = pre_test.iloc[0]
    else:
        best_pre_row = None

    tfidf_row = pick_row("baseline", "baseline", "tfidf", "tfidf_lexical", "heldout_test")
    std_row = metrics_df[
        (metrics_df["method"] == "finetune")
        & (metrics_df["stage"] == "final_standard")
        & (metrics_df["model_name"] == selected_model_name)
        & (metrics_df["protocol"] == "heldout_test")
        & (metrics_df["status"] == "success")
    ]
    std_row = std_row.sort_values("timestamp").iloc[-1] if not std_row.empty else None
    hard_row = metrics_df[
        (metrics_df["method"] == "finetune")
        & (metrics_df["stage"] == "final_hardneg")
        & (metrics_df["model_name"] == selected_model_name)
        & (metrics_df["protocol"] == "heldout_test")
        & (metrics_df["status"] == "success")
    ]
    hard_row = hard_row.sort_values("timestamp").iloc[-1] if not hard_row.empty else None

    comparison_specs = []
    if best_pre_row is not None and tfidf_row is not None:
        comparison_specs.append(("best_pretrained_vs_tfidf", best_pre_row, tfidf_row))
    if std_row is not None and best_pre_row is not None:
        comparison_specs.append(("final_standard_vs_best_pretrained", std_row, best_pre_row))
    if hard_row is not None and std_row is not None:
        comparison_specs.append(("final_hardneg_vs_final_standard", hard_row, std_row))

    # Recreate comparisons.csv on each run for consistency.
    if comparisons_path.exists():
        comparisons_path.unlink()

    for cmp_name, row_a, row_b in comparison_specs:
        for metric in ("mrr", "recall@10"):
            base_value = float(row_b[metric])
            compare_value = float(row_a[metric])
            delta_agg = compare_value - base_value

            ci_low = np.nan
            ci_high = np.nan
            status = "success"
            notes = ""
            try:
                ranks_a = load_ranks_if_exists(run_dir, str(row_a["ranks_file"]))
                ranks_b = load_ranks_if_exists(run_dir, str(row_b["ranks_file"]))
                if ranks_a is None or ranks_b is None:
                    status = "partial"
                    notes = "Missing rank files; bootstrap CI unavailable."
                else:
                    delta_boot, ci_low, ci_high = bootstrap_diff_ci(
                        ranks_a=ranks_a,
                        ranks_b=ranks_b,
                        metric=metric,
                        n_bootstrap=2000,
                        seed=args.seed,
                    )
                    # Keep aggregated delta as primary; include bootstrap center in notes.
                    notes = f"bootstrap_delta_mean={delta_boot:.6f}"
            except Exception as exc:
                status = "failed"
                notes = f"Bootstrap failed: {exc}"
                log_failure(f"bootstrap comparison failed ({cmp_name}, {metric}): {exc}")

            cmp_row = {
                "run_id": run_id,
                "timestamp": utc_now_iso(),
                "comparison": cmp_name,
                "protocol": "heldout_test",
                "metric": metric,
                "base_method": str(row_b["method"]),
                "base_stage": str(row_b["stage"]),
                "base_model": str(row_b["model_name"]),
                "compare_method": str(row_a["method"]),
                "compare_stage": str(row_a["stage"]),
                "compare_model": str(row_a["model_name"]),
                "base_value": base_value,
                "compare_value": compare_value,
                "delta": delta_agg,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_bootstrap": 2000,
                "status": status,
                "notes": notes,
            }
            append_row(comparisons_path, cmp_row, COMPARISON_COLUMNS)

    # Summary outputs
    metrics_success = metrics_df[metrics_df["status"] == "success"].copy()
    heldout_ranked = metrics_success[metrics_success["protocol"] == "heldout_test"].sort_values(
        by=["mrr", "recall@10"], ascending=[False, False]
    )
    full_ranked = metrics_success[metrics_success["protocol"] == "full_corpus"].sort_values(
        by=["mrr", "recall@10"], ascending=[False, False]
    )
    comps_df = load_df_or_empty(comparisons_path, COMPARISON_COLUMNS)

    md_lines = [
        f"# MBPP Retrieval Results: `{run_id}`",
        "",
        "## Run Metadata",
        f"- Started: {metadata['started_at']}",
        f"- Finished: {utc_now_iso()}",
        f"- Python: `{metadata['python_version'].splitlines()[0]}`",
        f"- Torch: `{metadata['torch_version']}`",
        f"- Device: `{resolved_device}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Held-out Test Ranking (Primary)",
        dataframe_to_markdown_compat(heldout_ranked[[
            "method",
            "stage",
            "technique",
            "model_name",
            "mrr",
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@20",
            "map@10",
            "ndcg@10",
            "precision@10",
        ]]) if not heldout_ranked.empty else "_No held-out test rows._",
        "",
        "## Full-corpus Diagnostic Ranking",
        dataframe_to_markdown_compat(full_ranked[[
            "method",
            "stage",
            "technique",
            "model_name",
            "mrr",
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@20",
            "map@10",
            "ndcg@10",
            "precision@10",
        ]]) if not full_ranked.empty else "_No full-corpus rows._",
        "",
        "## Before/After Comparisons",
        dataframe_to_markdown_compat(comps_df) if not comps_df.empty else "_No comparison rows available._",
        "",
        "## Notes",
        "- Primary metrics are reported on held-out test.",
        "- Full-corpus results are diagnostic only.",
        "- F1 is intentionally excluded from primary reporting (single relevant doc per query).",
    ]
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    txt_lines = [
        f"MBPP Retrieval Summary | run_id={run_id}",
        f"device={resolved_device} | seed={args.seed}",
        "",
        "Top held-out test methods:",
    ]
    if not heldout_ranked.empty:
        for _, row in heldout_ranked.head(10).iterrows():
            txt_lines.append(
                f"- {row['method']}::{row['stage']}::{row['model_name']} "
                f"| mrr={row['mrr']:.6f} recall@10={row['recall@10']:.6f} map@10={row['map@10']:.6f}"
            )
    else:
        txt_lines.append("- none")
    txt_lines.append("")
    txt_lines.append("Comparisons:")
    if not comps_df.empty:
        for _, row in comps_df.iterrows():
            txt_lines.append(
                f"- {row['comparison']} {row['metric']}: "
                f"delta={row['delta']:.6f} ci=[{row['ci_low']}, {row['ci_high']}] status={row['status']}"
            )
    else:
        txt_lines.append("- none")
    summary_txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    metadata["finished_at"] = utc_now_iso()
    metadata["elapsed_sec"] = time.perf_counter() - start_time
    metadata["selected_model_name"] = selected_model_name
    metadata["finetune_model_names"] = finetune_model_names
    metadata["finetune_all_pretrained"] = bool(args.finetune_all_pretrained)
    metadata["best_config_id"] = best_config_id
    metadata["artifacts"] = {
        "metrics_all_csv": str(metrics_path),
        "training_stats_csv": str(training_path),
        "comparisons_csv": str(comparisons_path),
        "summary_md": str(summary_md_path),
        "summary_txt": str(summary_txt_path),
        "failures_log": str(failures_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Run complete: {run_dir}")
    print(f"- metrics: {metrics_path}")
    print(f"- training: {training_path}")
    print(f"- comparisons: {comparisons_path}")
    print(f"- summary.md: {summary_md_path}")
    print(f"- summary.txt: {summary_txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
