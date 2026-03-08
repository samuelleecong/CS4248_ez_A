#!/usr/bin/env python3
"""Standalone knowledge-distillation fine-tuning for MBPP code search.

Trains a student SentenceTransformer to match a teacher's per-batch similarity
distributions (KL divergence) while retaining the MNR retrieval objective.
Both student and teacher are specified explicitly, so any model pair can be tested
without running the full experiment matrix.

Prerequisites:
  1. Export teacher targets once:
       python experiments/kai/scripts/export_teacher_targets.py \\
           --teacher <teacher_model> --split all

  2. Run KD:
       python experiments/kai/scripts/run_kd_experiments.py \\
           --student sentence-transformers/all-MiniLM-L6-v2 \\
           --teacher-targets experiments/kai/artifacts/kd_targets/<file>.npz \\
           --run-id my_kd_run

Output mirrors run_mbpp_experiments.py (same CSV schema) so results from both
scripts can be merged and plotted together.

Loss per batch:
  total = alpha * KL( softmax(S_student / T) || softmax(S_teacher / T) )
        + (1 - alpha) * CrossEntropy( S_student * 20, diagonal_labels )

  where S_* is the (B, B) pairwise cosine-similarity matrix for the batch,
  T is the softmax temperature, and alpha weights KD vs task loss.
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
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Constants — identical to run_mbpp_experiments.py for CSV compatibility
# ---------------------------------------------------------------------------

K_VALUES = (1, 5, 10, 20)

METRIC_COLUMNS = [
    "run_id", "timestamp", "method", "stage", "technique", "model_name",
    "protocol", "train_split", "eval_split", "config_id",
    "mrr", "recall@1", "recall@5", "recall@10", "recall@20",
    "map@10", "ndcg@10", "precision@10", "mean_rank",
    "runtime_sec", "status", "ranks_file", "error",
]
TRAINING_COLUMNS = [
    "run_id", "timestamp", "stage", "technique", "model_name", "config_id",
    "train_split", "epoch", "avg_loss", "total_steps", "warmup_steps",
    "runtime_sec", "status", "error",
]
COMPARISON_COLUMNS = [
    "run_id", "timestamp", "comparison", "protocol", "metric",
    "base_method", "base_stage", "base_model",
    "compare_method", "compare_stage", "compare_model",
    "base_value", "compare_value", "delta", "ci_low", "ci_high",
    "n_bootstrap", "status", "notes",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KDConfig:
    """Hyperparameters for one KD training run.

    Loss = alpha * KL(softmax(S_student/T) || softmax(S_teacher/T))
         + (1 - alpha) * MNR_CrossEntropy(S_student)

    alpha=0 degenerates to pure MNR; alpha=1 to pure distillation.
    """
    config_id: str
    epochs: int
    batch_size: int
    lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float = 1.0
    alpha: float = 0.5      # weight on KD loss
    temperature: float = 4.0  # softmax temperature for similarity distributions


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root (missing pyproject.toml).")


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


def format_texts_for_model(
    model_name: str,
    queries: list[str],
    codes: list[str],
) -> tuple[list[str], list[str]]:
    name = model_name.lower()
    if "e5" in name:
        return [f"query: {q}" for q in queries], [f"passage: {c}" for c in codes]
    if "bge" in name:
        prompt = "Represent this sentence for searching relevant code snippets: "
        return [prompt + q for q in queries], codes
    return queries, codes


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
    try:
        return df.to_markdown(index=False)
    except Exception:
        if df.empty:
            return ""
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        divider = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df.iterrows():
            def fmt(v: Any) -> str:
                if pd.isna(v):
                    return ""
                return (f"{v:.6f}" if isinstance(v, float) else str(v)).replace("|", r"\|")
            rows.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
        return "\n".join([header, divider] + rows)


def compute_ranks_from_embeddings(
    query_emb: np.ndarray,
    code_emb: np.ndarray,
) -> np.ndarray:
    query_emb = np.asarray(query_emb, dtype=np.float32)
    code_emb = np.asarray(code_emb, dtype=np.float32)
    index = faiss.IndexFlatIP(code_emb.shape[1])
    index.add(code_emb)
    _, neighbors = index.search(query_emb, code_emb.shape[0])
    ranks = np.empty(query_emb.shape[0], dtype=np.int32)
    for i, retrieved in enumerate(neighbors):
        ranks[i] = int(np.where(retrieved == i)[0][0]) + 1
    return ranks


def compute_metrics_from_ranks(
    ranks: np.ndarray,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, float]:
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


def evaluate_dense(
    model: SentenceTransformer,
    model_name: str,
    queries: list[str],
    codes: list[str],
    batch_size: int = 16,
) -> tuple[np.ndarray, dict[str, float]]:
    q_fmt, c_fmt = format_texts_for_model(model_name, queries, codes)
    q_emb = model.encode(
        q_fmt, batch_size=batch_size,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    )
    c_emb = model.encode(
        c_fmt, batch_size=batch_size,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    )
    ranks = compute_ranks_from_embeddings(q_emb, c_emb)
    return ranks, compute_metrics_from_ranks(ranks)


def rank_file_relpath(
    method: str, stage: str, technique: str,
    model_name: str, protocol: str, config_id: str,
) -> str:
    sig = "|".join([method, stage, technique, model_name, protocol, config_id])
    digest = hashlib.sha1(sig.encode()).hexdigest()[:12]
    return f"ranks/{safe_slug(method)}__{safe_slug(stage)}__{safe_slug(protocol)}__{digest}.npy"


def save_ranks(run_dir: Path, rel_path: str, ranks: np.ndarray) -> None:
    out = run_dir / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, ranks.astype(np.int32))


def load_ranks_if_exists(run_dir: Path, rel_path: str) -> np.ndarray | None:
    p = run_dir / rel_path
    return np.load(p) if p.exists() else None


def bootstrap_diff_ci(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    if ranks_a.shape != ranks_b.shape:
        raise ValueError("Rank arrays must have identical shape for bootstrap.")
    rng = np.random.default_rng(seed)
    n = ranks_a.shape[0]
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a, b = ranks_a[idx], ranks_b[idx]
        if metric == "mrr":
            va, vb = float(np.mean(1.0 / a)), float(np.mean(1.0 / b))
        elif metric == "recall@10":
            va, vb = float(np.mean(a <= 10)), float(np.mean(b <= 10))
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        diffs[i] = va - vb
    delta = float(np.mean(diffs))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return delta, float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# Teacher targets
# ---------------------------------------------------------------------------

def load_teacher_targets(
    targets_path: Path,
) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    """Load pre-exported teacher embeddings.

    Returns:
        task_id_to_idx : {task_id -> row index in embedding arrays}
        query_emb      : float32 (N, D_teacher)
        code_emb       : float32 (N, D_teacher)
    """
    data = np.load(targets_path)
    task_ids = data["task_ids"]
    q_emb = data["query_emb"].astype(np.float32)
    c_emb = data["code_emb"].astype(np.float32)
    task_id_to_idx: dict[int, int] = {int(tid): i for i, tid in enumerate(task_ids)}
    return task_id_to_idx, q_emb, c_emb


# ---------------------------------------------------------------------------
# KD training
# ---------------------------------------------------------------------------

def _encode_batch_with_grad(
    model: SentenceTransformer,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Tokenize and encode texts, keeping the computation graph for backprop."""
    features = model.tokenize(texts)
    features = {k: v.to(device) for k, v in features.items()}
    out = model(features)
    emb = out["sentence_embedding"]
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def finetune_with_kd(
    student_model_name: str,
    train_pairs: list[dict[str, Any]],
    teacher_task_id_map: dict[int, int],
    teacher_q_emb: np.ndarray,
    teacher_c_emb: np.ndarray,
    device: str,
    cfg: KDConfig,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    """Fine-tune student with KL-divergence knowledge distillation.

    Per-batch loss:
      KD   = KL( softmax(S_student/T) || softmax(S_teacher/T) )
      Task = CrossEntropy( S_student * 20, diagonal_labels )    [MNR]
      Total = alpha * KD + (1 - alpha) * Task

    Similarity matrices are compared — not raw vectors — so teacher and student
    can have different embedding dimensions.
    """
    model = SentenceTransformer(student_model_name)
    model.to(device)

    covered = [p for p in train_pairs if p["task_id"] in teacher_task_id_map]
    missing = len(train_pairs) - len(covered)
    if missing:
        print(f"  [kd] Warning: {missing}/{len(train_pairs)} pairs have no teacher target, skipping them.")
    if not covered:
        raise ValueError(
            "No training pairs have matching teacher targets. "
            "Re-run export_teacher_targets.py with --split all."
        )

    dev = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(covered) // cfg.batch_size)
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    mnr_scale = 20.0
    rng = random.Random(42)

    epoch_rows: list[dict[str, Any]] = []
    train_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        indices = list(range(len(covered)))
        rng.shuffle(indices)
        batches = [
            indices[i : i + cfg.batch_size]
            for i in range(0, len(indices) - cfg.batch_size + 1, cfg.batch_size)
        ]

        model.train()
        running_loss = 0.0
        pbar = tqdm(batches, desc=f"KD Epoch {epoch + 1}/{cfg.epochs}")

        for batch_idx_list in pbar:
            batch = [covered[i] for i in batch_idx_list]
            queries = [p["query"] for p in batch]
            codes = [p["code"] for p in batch]
            t_idxs = [teacher_task_id_map[p["task_id"]] for p in batch]

            t_q = torch.from_numpy(teacher_q_emb[t_idxs]).to(dev)
            t_c = torch.from_numpy(teacher_c_emb[t_idxs]).to(dev)

            optimizer.zero_grad()
            s_q = _encode_batch_with_grad(model, queries, dev)
            s_c = _encode_batch_with_grad(model, codes, dev)
            B = len(batch)

            # MNR task loss
            mnr_scores = torch.mm(s_q, s_c.T) * mnr_scale
            labels = torch.arange(B, device=dev)
            mnr_loss = torch.nn.functional.cross_entropy(mnr_scores, labels)

            # KD loss: KL on per-query soft similarity distributions
            teacher_sims = torch.mm(t_q, t_c.T)
            teacher_probs = torch.nn.functional.softmax(teacher_sims / cfg.temperature, dim=1)
            student_sims = torch.mm(s_q, s_c.T)
            student_log_probs = torch.nn.functional.log_softmax(student_sims / cfg.temperature, dim=1)
            kl_loss = torch.nn.functional.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean"
            )

            loss = cfg.alpha * kl_loss + (1.0 - cfg.alpha) * mnr_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                kl=f"{kl_loss.item():.4f}",
                mnr=f"{mnr_loss.item():.4f}",
            )

        avg_loss = running_loss / max(1, len(batches))
        epoch_rows.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        })

    return model, {
        "epoch_rows": epoch_rows,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "runtime_sec": time.perf_counter() - train_start,
        "n_covered_pairs": len(covered),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KD fine-tuning for MBPP code search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Models
    parser.add_argument(
        "--student",
        required=True,
        metavar="MODEL",
        help="Student SentenceTransformer model name (must be cached locally).",
    )
    parser.add_argument(
        "--teacher-targets",
        required=True,
        metavar="PATH",
        help="Path to teacher targets .npz produced by export_teacher_targets.py.",
    )

    # Run identity
    parser.add_argument("--run-id", default=None,
                        help="Run identifier. Defaults to a UTC timestamp.")
    parser.add_argument("--output-dir", default="experiments/kai/results",
                        help="Output root relative to project root.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Training hyperparameters (single-config mode)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # KD-specific hyperparameters
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight on KD loss. 0 = pure MNR, 1 = pure distillation.")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Softmax temperature for similarity distributions.")

    # Sweep / run modes
    parser.add_argument("--full-matrix", action="store_true",
                        help="Sweep 4 configs (alpha x epochs) instead of using CLI hyperparams.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps already recorded as success in metrics_all.csv.")
    parser.add_argument("--fast-smoke", action="store_true",
                        help="Encode only (no training). Quick schema and import check.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path.parent)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("kd_run_%Y%m%dT%H%M%SZ")
    output_root = (project_root / args.output_dir).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = run_dir / "metrics"
    reports_dir = run_dir / "reports"
    metadata_dir = run_dir / "metadata"
    logs_dir = run_dir / "logs"
    for d in (metrics_dir, reports_dir, metadata_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "metrics_all.csv"
    training_path = metrics_dir / "training_stats.csv"
    comparisons_path = metrics_dir / "comparisons.csv"
    summary_md_path = reports_dir / "summary.md"
    summary_txt_path = reports_dir / "summary.txt"
    metadata_path = metadata_dir / "run_metadata.json"
    failures_path = logs_dir / "failures.log"

    # Resume: collect already-completed step keys.
    existing_metrics = load_df_or_empty(metrics_path, METRIC_COLUMNS)
    done_keys: set[tuple[str, ...]] = set()
    for _, row in existing_metrics.iterrows():
        if str(row.get("status", "")) == "success":
            done_keys.add((
                str(row["method"]), str(row["stage"]), str(row["technique"]),
                str(row["model_name"]), str(row["protocol"]),
                str(row["train_split"]), str(row["eval_split"]), str(row["config_id"]),
            ))

    def step_key(method, stage, technique, model_name, protocol, train_split, eval_split, config_id):
        return (method, stage, technique, model_name, protocol, train_split, eval_split, config_id)

    def log_failure(msg: str) -> None:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_path.open("a", encoding="utf-8") as f:
            f.write(f"[{utc_now_iso()}] {msg}\n")

    seed_everything(args.seed)
    resolved_device = resolve_device(args.device)
    start_time = time.perf_counter()

    # Resolve teacher metadata.
    teacher_targets_path = Path(args.teacher_targets).resolve()
    if not teacher_targets_path.exists():
        print(f"Teacher targets not found: {teacher_targets_path}", file=sys.stderr)
        return 1

    meta_json = teacher_targets_path.with_name(teacher_targets_path.stem + "_meta.json")
    teacher_model_name = str(args.teacher_targets)  # fallback display name
    if meta_json.exists():
        try:
            tmeta = json.loads(meta_json.read_text(encoding="utf-8"))
            teacher_model_name = tmeta.get("model_name", teacher_model_name)
        except Exception:
            pass

    # Build KD configs.
    if args.full_matrix:
        kd_configs = [
            KDConfig("kd_e1_b16_lr2e5_a05_t4", 1, args.batch_size, args.lr, args.warmup_ratio, args.weight_decay, args.max_grad_norm, alpha=0.5, temperature=4.0),
            KDConfig("kd_e2_b16_lr2e5_a05_t4", 2, args.batch_size, args.lr, args.warmup_ratio, args.weight_decay, args.max_grad_norm, alpha=0.5, temperature=4.0),
            KDConfig("kd_e1_b16_lr2e5_a03_t4", 1, args.batch_size, args.lr, args.warmup_ratio, args.weight_decay, args.max_grad_norm, alpha=0.3, temperature=4.0),
            KDConfig("kd_e1_b16_lr2e5_a07_t4", 1, args.batch_size, args.lr, args.warmup_ratio, args.weight_decay, args.max_grad_norm, alpha=0.7, temperature=4.0),
        ]
    else:
        config_id = (
            f"kd_e{args.epochs}_b{args.batch_size}"
            f"_lr{args.lr:.0e}_a{args.alpha:.0f}_t{args.temperature:.0f}"
        ).replace("+", "").replace("-0", "")
        kd_configs = [
            KDConfig(
                config_id=config_id,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                warmup_ratio=args.warmup_ratio,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                alpha=args.alpha,
                temperature=args.temperature,
            )
        ]

    metadata = {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "script": "run_kd_experiments.py",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "device_arg": args.device,
        "resolved_device": resolved_device,
        "seed": args.seed,
        "student_model": args.student,
        "teacher_model": teacher_model_name,
        "teacher_targets_path": str(teacher_targets_path),
        "kd_configs": [asdict(c) for c in kd_configs],
        "args": vars(args),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("Loading MBPP dataset...")
    ds = load_dataset("google-research-datasets/mbpp", download_mode="reuse_dataset_if_exists")

    def extract_pairs(split_name: str) -> list[dict[str, Any]]:
        rows = []
        for r in ds[split_name]:
            q, c = str(r["text"]).strip(), str(r["code"]).strip()
            if q and c:
                rows.append({"query": q, "code": c, "task_id": int(r["task_id"]), "split": split_name})
        rows.sort(key=lambda x: x["task_id"])
        return rows

    train_pairs = extract_pairs("train")
    val_pairs = extract_pairs("validation")
    test_pairs = extract_pairs("test")
    prompt_pairs = extract_pairs("prompt")
    final_train_pairs = train_pairs + val_pairs + prompt_pairs
    full_pairs = train_pairs + val_pairs + test_pairs + prompt_pairs

    protocols = {
        "heldout_test": {
            "queries": [p["query"] for p in test_pairs],
            "codes":   [p["code"]  for p in test_pairs],
            "eval_split": "test",
        },
        "full_corpus": {
            "queries": [p["query"] for p in full_pairs],
            "codes":   [p["code"]  for p in full_pairs],
            "eval_split": "all",
        },
    }

    # ------------------------------------------------------------------
    # Helpers: write rows
    # ------------------------------------------------------------------
    def write_metric_row(
        *, method, stage, technique, model_name, protocol,
        train_split, eval_split, config_id,
        metrics: dict[str, float] | None, runtime_sec: float,
        status: str, ranks_relpath: str = "", error: str = "",
    ) -> None:
        row: dict[str, Any] = {
            "run_id": run_id, "timestamp": utc_now_iso(),
            "method": method, "stage": stage, "technique": technique,
            "model_name": model_name, "protocol": protocol,
            "train_split": train_split, "eval_split": eval_split,
            "config_id": config_id,
            "runtime_sec": runtime_sec, "status": status,
            "ranks_file": ranks_relpath, "error": error,
        }
        for col in ["mrr", "recall@1", "recall@5", "recall@10", "recall@20",
                    "map@10", "ndcg@10", "precision@10", "mean_rank"]:
            row[col] = float(metrics[col]) if (metrics and col in metrics) else np.nan
        append_row(metrics_path, row, METRIC_COLUMNS)
        if status == "success":
            done_keys.add(step_key(method, stage, technique, model_name, protocol,
                                   train_split, eval_split, config_id))

    def write_training_rows(
        *, stage, technique, model_name, config_id, train_split,
        stats: dict[str, Any], status: str = "success", error: str = "",
    ) -> None:
        for epoch_row in stats.get("epoch_rows", []):
            row = {
                "run_id": run_id, "timestamp": utc_now_iso(),
                "stage": stage, "technique": technique,
                "model_name": model_name, "config_id": config_id,
                "train_split": train_split,
                "epoch": epoch_row["epoch"],
                "avg_loss": epoch_row["avg_loss"],
                "total_steps": stats.get("total_steps", 0),
                "warmup_steps": stats.get("warmup_steps", 0),
                "runtime_sec": stats.get("runtime_sec", np.nan),
                "status": status, "error": error,
            }
            append_row(training_path, row, TRAINING_COLUMNS)

    # ------------------------------------------------------------------
    # Stage 1: Student zero-shot baseline
    # ------------------------------------------------------------------
    print(f"Student zero-shot: {args.student}")
    student_zs_model = None
    for protocol_name, pool in protocols.items():
        eval_split = str(pool["eval_split"])
        key = step_key("pretrained", "pretrained", "zero_shot",
                       args.student, protocol_name, "n/a", eval_split, "zero_shot")
        rank_rel = rank_file_relpath("pretrained", "pretrained", "zero_shot",
                                     args.student, protocol_name, "zero_shot")
        rank_exists = (run_dir / rank_rel).exists()
        if args.resume and key in done_keys and rank_exists:
            print(f"  [resume] zero_shot {protocol_name} skipped")
            continue

        t0 = time.perf_counter()
        try:
            if student_zs_model is None:
                student_zs_model = SentenceTransformer(args.student)
                student_zs_model.to(resolved_device)
            ranks, metrics = evaluate_dense(
                student_zs_model, args.student, pool["queries"], pool["codes"]
            )
            save_ranks(run_dir, rank_rel, ranks)
            write_metric_row(
                method="pretrained", stage="pretrained", technique="zero_shot",
                model_name=args.student, protocol=protocol_name,
                train_split="n/a", eval_split=eval_split, config_id="zero_shot",
                metrics=metrics, runtime_sec=time.perf_counter() - t0, status="success",
                ranks_relpath=rank_rel,
            )
            print(f"  {protocol_name}: MRR={metrics['mrr']:.4f}  R@10={metrics['recall@10']:.4f}")
        except Exception as exc:
            tb = traceback.format_exc(limit=5)
            log_failure(f"zero_shot eval failed ({protocol_name}): {exc}\n{tb}")
            write_metric_row(
                method="pretrained", stage="pretrained", technique="zero_shot",
                model_name=args.student, protocol=protocol_name,
                train_split="n/a", eval_split=eval_split, config_id="zero_shot",
                metrics=None, runtime_sec=time.perf_counter() - t0, status="failed",
                error=str(exc),
            )

    if args.fast_smoke:
        print("fast-smoke: stopping after zero-shot eval.")
        return 0

    # ------------------------------------------------------------------
    # Stage 2: KD fine-tuning
    # ------------------------------------------------------------------
    print(f"Loading teacher targets from {teacher_targets_path} ...")
    teacher_task_id_map, teacher_q_emb, teacher_c_emb = load_teacher_targets(teacher_targets_path)
    print(f"  {len(teacher_task_id_map)} task_ids loaded, embed_dim={teacher_q_emb.shape[1]}")

    for cfg in kd_configs:
        kd_ckpt = run_dir / "checkpoints" / "kd_mnr" / safe_slug(args.student) / cfg.config_id
        kd_model = None

        train_key = step_key("kd", "kd_mnr", "kd_kl", args.student, "train_only",
                             "train+validation+prompt", "n/a", cfg.config_id)

        if not (args.resume and train_key in done_keys and kd_ckpt.exists()):
            print(f"[kd] Training {cfg.config_id}  (alpha={cfg.alpha}, T={cfg.temperature}, epochs={cfg.epochs}) ...")
            t0 = time.perf_counter()
            try:
                kd_model, stats = finetune_with_kd(
                    student_model_name=args.student,
                    train_pairs=final_train_pairs,
                    teacher_task_id_map=teacher_task_id_map,
                    teacher_q_emb=teacher_q_emb,
                    teacher_c_emb=teacher_c_emb,
                    device=resolved_device,
                    cfg=cfg,
                )
                kd_ckpt.mkdir(parents=True, exist_ok=True)
                kd_model.save(str(kd_ckpt))
                write_training_rows(
                    stage="kd_mnr", technique="kd_kl",
                    model_name=args.student, config_id=cfg.config_id,
                    train_split="train+validation+prompt", stats=stats,
                )
                write_metric_row(
                    method="kd", stage="kd_mnr", technique="kd_kl",
                    model_name=args.student, protocol="train_only",
                    train_split="train+validation+prompt", eval_split="n/a",
                    config_id=cfg.config_id,
                    metrics={"mrr": np.nan},
                    runtime_sec=time.perf_counter() - t0, status="success",
                )
            except Exception as exc:
                tb = traceback.format_exc(limit=5)
                log_failure(f"kd training failed ({cfg.config_id}): {exc}\n{tb}")
                write_metric_row(
                    method="kd", stage="kd_mnr", technique="kd_kl",
                    model_name=args.student, protocol="train_only",
                    train_split="train+validation+prompt", eval_split="n/a",
                    config_id=cfg.config_id, metrics=None,
                    runtime_sec=time.perf_counter() - t0, status="failed",
                    error=str(exc),
                )
                continue

        if kd_model is None and kd_ckpt.exists():
            kd_model = SentenceTransformer(str(kd_ckpt))
            kd_model.to(resolved_device)

        if kd_model is None:
            continue

        for protocol_name, pool in protocols.items():
            eval_split = str(pool["eval_split"])
            key = step_key("kd", "kd_mnr", "kd_kl", args.student, protocol_name,
                           "train+validation+prompt", eval_split, cfg.config_id)
            rank_rel = rank_file_relpath("kd", "kd_mnr", "kd_kl",
                                         args.student, protocol_name, cfg.config_id)
            rank_exists = (run_dir / rank_rel).exists()
            if args.resume and key in done_keys and rank_exists:
                print(f"  [resume] kd eval {protocol_name} {cfg.config_id} skipped")
                continue

            t0 = time.perf_counter()
            try:
                ranks, metrics = evaluate_dense(kd_model, args.student, pool["queries"], pool["codes"])
                save_ranks(run_dir, rank_rel, ranks)
                write_metric_row(
                    method="kd", stage="kd_mnr", technique="kd_kl",
                    model_name=args.student, protocol=protocol_name,
                    train_split="train+validation+prompt", eval_split=eval_split,
                    config_id=cfg.config_id, metrics=metrics,
                    runtime_sec=time.perf_counter() - t0, status="success",
                    ranks_relpath=rank_rel,
                )
                print(f"  {protocol_name} [{cfg.config_id}]: MRR={metrics['mrr']:.4f}  R@10={metrics['recall@10']:.4f}")
            except Exception as exc:
                tb = traceback.format_exc(limit=5)
                log_failure(f"kd eval failed ({cfg.config_id}, {protocol_name}): {exc}\n{tb}")
                write_metric_row(
                    method="kd", stage="kd_mnr", technique="kd_kl",
                    model_name=args.student, protocol=protocol_name,
                    train_split="train+validation+prompt", eval_split=eval_split,
                    config_id=cfg.config_id, metrics=None,
                    runtime_sec=time.perf_counter() - t0, status="failed",
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Stage 3: Comparisons (KD vs zero-shot student, best KD config)
    # ------------------------------------------------------------------
    metrics_df = load_df_or_empty(metrics_path, METRIC_COLUMNS)
    for col in ["mrr", "recall@10", "recall@1", "recall@5", "recall@20",
                "map@10", "ndcg@10", "precision@10", "mean_rank", "runtime_sec"]:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    zs_row = metrics_df[
        (metrics_df["method"] == "pretrained") &
        (metrics_df["stage"] == "pretrained") &
        (metrics_df["model_name"] == args.student) &
        (metrics_df["protocol"] == "heldout_test") &
        (metrics_df["status"] == "success")
    ]
    zs_row = zs_row.sort_values("timestamp").iloc[-1] if not zs_row.empty else None

    kd_rows = metrics_df[
        (metrics_df["method"] == "kd") &
        (metrics_df["stage"] == "kd_mnr") &
        (metrics_df["model_name"] == args.student) &
        (metrics_df["protocol"] == "heldout_test") &
        (metrics_df["status"] == "success")
    ].copy()
    best_kd_row = None
    if not kd_rows.empty:
        kd_rows = kd_rows.sort_values("mrr", ascending=False)
        best_kd_row = kd_rows.iloc[0]

    comparison_specs = []
    if best_kd_row is not None and zs_row is not None:
        comparison_specs.append(("kd_vs_student_zero_shot", best_kd_row, zs_row))

    if comparisons_path.exists():
        comparisons_path.unlink()

    for cmp_name, row_a, row_b in comparison_specs:
        for metric in ("mrr", "recall@10"):
            base_value = float(row_b[metric])
            compare_value = float(row_a[metric])
            delta_agg = compare_value - base_value
            ci_low = ci_high = np.nan
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
                        ranks_a, ranks_b, metric, n_bootstrap=2000, seed=args.seed
                    )
                    notes = f"bootstrap_delta_mean={delta_boot:.6f}"
            except Exception as exc:
                status = "failed"
                notes = f"Bootstrap failed: {exc}"
                log_failure(f"bootstrap failed ({cmp_name}, {metric}): {exc}")

            append_row(comparisons_path, {
                "run_id": run_id, "timestamp": utc_now_iso(),
                "comparison": cmp_name, "protocol": "heldout_test", "metric": metric,
                "base_method": str(row_b["method"]), "base_stage": str(row_b["stage"]),
                "base_model": str(row_b["model_name"]),
                "compare_method": str(row_a["method"]), "compare_stage": str(row_a["stage"]),
                "compare_model": str(row_a["model_name"]),
                "base_value": base_value, "compare_value": compare_value,
                "delta": delta_agg, "ci_low": ci_low, "ci_high": ci_high,
                "n_bootstrap": 2000, "status": status, "notes": notes,
            }, COMPARISON_COLUMNS)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    metrics_success = metrics_df[metrics_df["status"] == "success"].copy()
    heldout_ranked = metrics_success[metrics_success["protocol"] == "heldout_test"].sort_values(
        ["mrr", "recall@10"], ascending=False
    )
    full_ranked = metrics_success[metrics_success["protocol"] == "full_corpus"].sort_values(
        ["mrr", "recall@10"], ascending=False
    )
    comps_df = load_df_or_empty(comparisons_path, COMPARISON_COLUMNS)

    display_cols = ["method", "stage", "technique", "model_name", "config_id",
                    "mrr", "recall@1", "recall@5", "recall@10", "recall@20", "map@10", "ndcg@10"]

    md_lines = [
        f"# MBPP KD Results: `{run_id}`",
        "",
        "## Run Info",
        f"- Student : `{args.student}`",
        f"- Teacher : `{teacher_model_name}`",
        f"- Device  : `{resolved_device}`",
        f"- Seed    : `{args.seed}`",
        f"- Started : {metadata['started_at']}",
        f"- Finished: {utc_now_iso()}",
        "",
        "## Held-out Test Ranking",
        dataframe_to_markdown_compat(heldout_ranked[display_cols]) if not heldout_ranked.empty else "_No rows._",
        "",
        "## Full-corpus Ranking",
        dataframe_to_markdown_compat(full_ranked[display_cols]) if not full_ranked.empty else "_No rows._",
        "",
        "## Comparisons",
        dataframe_to_markdown_compat(comps_df) if not comps_df.empty else "_No comparisons._",
    ]
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    txt_lines = [
        f"MBPP KD Summary | run_id={run_id}",
        f"student={args.student}  teacher={teacher_model_name}",
        f"device={resolved_device}  seed={args.seed}",
        "",
        "Held-out test results:",
    ]
    for _, row in heldout_ranked.head(10).iterrows():
        txt_lines.append(
            f"  {row['method']}::{row['stage']}::{row.get('config_id','')} "
            f"mrr={row['mrr']:.4f}  recall@10={row['recall@10']:.4f}"
        )
    txt_lines.append("")
    txt_lines.append("Comparisons:")
    for _, row in comps_df.iterrows():
        txt_lines.append(
            f"  {row['comparison']} {row['metric']}: "
            f"delta={row['delta']:.4f}  ci=[{row['ci_low']}, {row['ci_high']}]  status={row['status']}"
        )
    summary_txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    metadata["finished_at"] = utc_now_iso()
    metadata["elapsed_sec"] = time.perf_counter() - start_time
    metadata["artifacts"] = {
        "metrics_all_csv": str(metrics_path),
        "training_stats_csv": str(training_path),
        "comparisons_csv": str(comparisons_path),
        "summary_md": str(summary_md_path),
        "summary_txt": str(summary_txt_path),
        "failures_log": str(failures_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nRun complete: {run_dir}")
    print(f"  metrics   : {metrics_path}")
    print(f"  training  : {training_path}")
    print(f"  comparisons: {comparisons_path}")
    print(f"  summary   : {summary_txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
