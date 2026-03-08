#!/usr/bin/env python3
"""Export teacher model embeddings for knowledge distillation.

Encodes all MBPP query-code pairs with a teacher SentenceTransformer and saves
the resulting embeddings as a compressed .npz file that run_mbpp_experiments.py
consumes during KD fine-tuning (--teacher-targets flag).

Output layout (relative to <output-dir>/kd_targets/):
  <model_slug>_<split>.npz
    task_ids   : int32  (N,)   – MBPP task IDs, sorted ascending
    query_emb  : float32 (N, D) – L2-normalised query embeddings
    code_emb   : float32 (N, D) – L2-normalised code embeddings

  <model_slug>_<split>_meta.json
    model_name, split, n_examples, embed_dim, batch_size, device, timestamp, runtime_sec

Usage:
  python export_teacher_targets.py \\
      --teacher BAAI/bge-base-en-v1.5 \\
      --split all \\
      --output-dir experiments/kai/artifacts \\
      --batch-size 32 \\
      --device auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm  # noqa: F401  (imported for side-effect progress bars)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ALL_SPLITS = ("train", "validation", "test", "prompt")


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root (missing pyproject.toml).")


def safe_slug(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")


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


def extract_split_pairs(ds: dict, split_name: str) -> list[dict]:
    rows = []
    for r in ds[split_name]:
        q = str(r["text"]).strip()
        c = str(r["code"]).strip()
        if q and c:
            rows.append({"query": q, "code": c, "task_id": int(r["task_id"]), "split": split_name})
    rows.sort(key=lambda x: x["task_id"])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export teacher SentenceTransformer embeddings for KD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--teacher",
        required=True,
        metavar="MODEL_NAME",
        help="Teacher SentenceTransformer model name (must be cached locally).",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=[*ALL_SPLITS, "all"],
        help="Which MBPP split(s) to encode. 'all' encodes all 974 examples.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/kai/artifacts",
        metavar="DIR",
        help="Root output dir relative to project root. Targets go under <DIR>/kd_targets/.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="'auto' picks CUDA > MPS > CPU.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .npz even if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent)
    output_root = (project_root / args.output_dir / "kd_targets").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(args.device)
    model_slug = safe_slug(args.teacher)
    npz_path = output_root / f"{model_slug}_{args.split}.npz"
    meta_path = output_root / f"{model_slug}_{args.split}_meta.json"

    if npz_path.exists() and not args.force:
        print(f"Targets already exist at {npz_path}. Use --force to overwrite.")
        return 0

    print(f"Device        : {resolved_device}")
    print(f"Teacher       : {args.teacher}")
    print(f"Split         : {args.split}")
    print(f"Output        : {npz_path}")

    print("Loading MBPP dataset...")
    ds = load_dataset("google-research-datasets/mbpp", download_mode="reuse_dataset_if_exists")

    if args.split == "all":
        pairs: list[dict] = []
        for sname in ALL_SPLITS:
            pairs.extend(extract_split_pairs(ds, sname))
        pairs.sort(key=lambda x: x["task_id"])
    else:
        pairs = extract_split_pairs(ds, args.split)

    if not pairs:
        print("No pairs extracted. Aborting.", file=sys.stderr)
        return 1

    print(f"Loaded {len(pairs)} pairs.")

    print(f"Loading teacher model: {args.teacher}")
    teacher = SentenceTransformer(args.teacher)
    teacher.to(resolved_device)
    teacher.eval()

    queries = [p["query"] for p in pairs]
    codes = [p["code"] for p in pairs]
    task_ids = np.array([p["task_id"] for p in pairs], dtype=np.int32)

    q_fmt, c_fmt = format_texts_for_model(args.teacher, queries, codes)

    t0 = time.perf_counter()

    print("Encoding queries...")
    with torch.no_grad():
        q_emb = teacher.encode(
            q_fmt,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

    print("Encoding codes...")
    with torch.no_grad():
        c_emb = teacher.encode(
            c_fmt,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

    elapsed = time.perf_counter() - t0
    assert q_emb.shape == c_emb.shape, f"Shape mismatch: {q_emb.shape} vs {c_emb.shape}"
    print(f"Encoding done in {elapsed:.1f}s  |  shape={q_emb.shape}")

    print(f"Saving targets → {npz_path}")
    np.savez_compressed(
        npz_path,
        task_ids=task_ids,
        query_emb=q_emb,
        code_emb=c_emb,
    )

    meta = {
        "model_name": args.teacher,
        "split": args.split,
        "n_examples": int(len(pairs)),
        "embed_dim": int(q_emb.shape[1]),
        "batch_size": args.batch_size,
        "device": resolved_device,
        "normalized": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": elapsed,
        "npz_path": str(npz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata      → {meta_path}")
    print(f"Done. {len(pairs)} examples, embed_dim={q_emb.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
