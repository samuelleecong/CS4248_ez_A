from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval.data_adapters import get_dataset_adapter
from eval.model_adapters import load_model_adapter
from mbpp_kd_suite.metrics import paired_ranks
from mbpp_kd_suite.runtime import pick_device, set_seed


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_source: str
    model_name_or_path: str
    checkpoint_format: str = "auto"
    hf_subfolder: str | None = None


def parse_model_spec(raw: str) -> ModelSpec:
    parts = raw.split("|", 4)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid --model-spec '{raw}'. Expected label|source|model_name_or_path|checkpoint_format|hf_subfolder"
        )
    if len(parts) == 3:
        label, source, model_name_or_path = parts
        checkpoint_format = "auto"
        hf_subfolder = None
    elif len(parts) == 4:
        label, source, model_name_or_path, checkpoint_format = parts
        hf_subfolder = None
    else:
        label, source, model_name_or_path, checkpoint_format, hf_subfolder = parts
    return ModelSpec(
        label=label.strip(),
        model_source=source.strip(),
        model_name_or_path=model_name_or_path.strip(),
        checkpoint_format=checkpoint_format.strip() or "auto",
        hf_subfolder=(hf_subfolder.strip() or None) if hf_subfolder is not None else None,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export per-query paired retrieval ranks for multiple models.")
    parser.add_argument("--dataset-name", choices=("mbpp", "codesearchnet"), required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-query-length", type=int, default=160)
    parser.add_argument("--max-code-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Repeated model spec in the form label|source|model_name_or_path|checkpoint_format",
    )
    parser.add_argument("--teacher-label", default=None)
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--kd-label", default=None)
    return parser


def resolve_device(requested: str):
    import torch

    normalized = requested.strip().lower()
    if normalized == "auto":
        return pick_device()
    return torch.device(normalized)


def truncate_records(records: list[Any], max_queries: int | None) -> list[Any]:
    if max_queries is None or len(records) <= max_queries:
        return records
    return records[:max_queries]


def bucketize_rows(
    per_query_rows: list[dict[str, Any]],
    teacher_label: str | None,
    baseline_label: str | None,
    kd_label: str | None,
) -> list[dict[str, Any]]:
    if not teacher_label or not baseline_label or not kd_label:
        return []

    buckets = {
        "kd_helps": [],
        "kd_hurts": [],
        "student_rank1": [],
        "teacher_wins_kd_not_transfer": [],
    }
    teacher_key = f"{teacher_label}_rank"
    baseline_key = f"{baseline_label}_rank"
    kd_key = f"{kd_label}_rank"

    for row in per_query_rows:
        teacher_rank = int(row[teacher_key])
        baseline_rank = int(row[baseline_key])
        kd_rank = int(row[kd_key])
        if kd_rank < baseline_rank:
            buckets["kd_helps"].append(row)
        elif kd_rank > baseline_rank:
            buckets["kd_hurts"].append(row)
        if baseline_rank == 1:
            buckets["student_rank1"].append(row)
        if teacher_rank < baseline_rank and kd_rank >= baseline_rank:
            buckets["teacher_wins_kd_not_transfer"].append(row)

    summaries: list[dict[str, Any]] = []
    for bucket_name, bucket_rows in buckets.items():
        if not bucket_rows:
            summaries.append(
                {
                    "bucket": bucket_name,
                    "count": 0,
                    "avg_teacher_rank": "",
                    "avg_baseline_rank": "",
                    "avg_kd_rank": "",
                    "example_query": "",
                }
            )
            continue
        summaries.append(
            {
                "bucket": bucket_name,
                "count": len(bucket_rows),
                "avg_teacher_rank": f"{np.mean([int(row[teacher_key]) for row in bucket_rows]):.2f}",
                "avg_baseline_rank": f"{np.mean([int(row[baseline_key]) for row in bucket_rows]):.2f}",
                "avg_kd_rank": f"{np.mean([int(row[kd_key]) for row in bucket_rows]):.2f}",
                "example_query": bucket_rows[0]["query"],
            }
        )
    return summaries


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if len(args.model_spec) < 1:
        raise ValueError("At least one --model-spec is required")

    set_seed(args.seed)
    device = resolve_device(args.device)
    model_specs = [parse_model_spec(raw) for raw in args.model_spec]
    corpus = get_dataset_adapter(args.dataset_name, path=args.dataset_path).load()
    split_records = truncate_records(corpus.get_split(args.split), args.max_queries)
    queries = [record.query for record in split_records]
    codes = [record.code for record in split_records]

    rank_map: dict[str, np.ndarray] = {}
    for spec in model_specs:
        adapter = load_model_adapter(
            model_source=spec.model_source,
            model_name_or_path=spec.model_name_or_path,
            hf_subfolder=spec.hf_subfolder,
            checkpoint_format=spec.checkpoint_format,
            max_query_length=args.max_query_length,
            max_code_length=args.max_code_length,
            batch_size=args.batch_size,
            device=device,
        )
        query_embs = adapter.encode_queries(queries)
        code_embs = adapter.encode_codes(codes)
        score_matrix = (query_embs @ code_embs.T).numpy()
        rank_map[spec.label] = paired_ranks(score_matrix)

    per_query_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(split_records):
        row: dict[str, Any] = {
            "query_index": idx,
            "record_id": record.id,
            "dataset": args.dataset_name,
            "split": args.split,
            "query_tokens": len(record.query.split()),
            "code_tokens": len(record.code.split()),
            "query": record.query,
        }
        for spec in model_specs:
            row[f"{spec.label}_rank"] = int(rank_map[spec.label][idx])
        per_query_rows.append(row)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    per_query_path = prefix.with_name(prefix.name + "_per_query.csv")
    bucket_path = prefix.with_name(prefix.name + "_bucket_summary.csv")
    write_csv(per_query_path, per_query_rows)
    bucket_rows = bucketize_rows(
        per_query_rows=per_query_rows,
        teacher_label=args.teacher_label,
        baseline_label=args.baseline_label,
        kd_label=args.kd_label,
    )
    if bucket_rows:
        write_csv(bucket_path, bucket_rows)
    else:
        bucket_path.write_text("", encoding="utf-8")

    print(f"Wrote per-query ranks to {per_query_path}")
    if bucket_rows:
        print(f"Wrote bucket summary to {bucket_path}")
    else:
        print("Skipped bucket summary because teacher/baseline/kd labels were not fully provided")


if __name__ == "__main__":
    main()
