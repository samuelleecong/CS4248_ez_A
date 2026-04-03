from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from mbpp_kd_suite.data import load_retrieval_dataset, taco_row_to_pair

from .types import RetrievalCorpus, RetrievalRecord


def load_mbpp_ood_corpus(path: str | None, split_seed: int) -> tuple[RetrievalCorpus, dict[str, list[str]]]:
    rows = _load_mbpp_rows(path)
    records = _rows_to_unique_mbpp_records(rows)
    return _deterministic_partition(records, split_seed=split_seed, source_dataset="mbpp_sanitized_ood")


def load_taco_retrieval_corpus(
    dataset_name: str,
    dataset_path: str | None,
    split_seed: int,
) -> RetrievalCorpus:
    if dataset_path:
        return _load_local_taco_corpus(Path(dataset_path).expanduser(), split_seed=split_seed)
    dataset = load_retrieval_dataset(dataset_name=dataset_name, taco_val_size=1000, seed=split_seed)
    return RetrievalCorpus(
        train=_pairs_to_records(dataset["train"], split="train", source_dataset=dataset_name),
        validation=_pairs_to_records(dataset["validation"], split="validation", source_dataset=dataset_name),
        test=_pairs_to_records(dataset["test"], split="test", source_dataset=dataset_name),
    )


def _load_mbpp_rows(path: str | None) -> list[dict[str, Any]]:
    if path:
        candidate = Path(path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"MBPP dataset path does not exist: {candidate}")
        return _load_local_mbpp_rows(candidate)

    dataset: DatasetDict = load_dataset("mbpp", "sanitized")
    rows: list[dict[str, Any]] = []
    for split_name in dataset.keys():
        rows.extend(dict(row) for row in dataset[split_name])
    return rows


def _load_local_mbpp_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for filename in (
            "train.jsonl",
            "train.json",
            "validation.jsonl",
            "validation.json",
            "valid.jsonl",
            "valid.json",
            "test.jsonl",
            "test.json",
            "mbpp.jsonl",
            "sanitized-mbpp.json",
        ):
            candidate = path / filename
            if candidate.exists():
                rows.extend(_load_json_records(candidate))
        if rows:
            return rows
        raise FileNotFoundError(f"No supported MBPP files found under {path}")
    return _load_json_records(path)


def _rows_to_unique_mbpp_records(rows: list[dict[str, Any]]) -> list[RetrievalRecord]:
    deduped: dict[str, RetrievalRecord] = {}
    for index, row in enumerate(rows):
        query = str(row.get("text") or row.get("prompt") or "").strip()
        code = str(row.get("code") or "").strip()
        if not query or not code:
            continue
        task_id = row.get("task_id")
        record_id = str(task_id if task_id is not None else index)
        deduped[record_id] = RetrievalRecord(
            id=record_id,
            query=query,
            code=code,
            split="pool",
            source_dataset="mbpp_sanitized_ood",
        )
    if len(deduped) < 3:
        raise ValueError("MBPP OOD pool must contain at least 3 valid query/code pairs")
    return [deduped[key] for key in sorted(deduped.keys(), key=_sortable_id)]


def _load_local_taco_corpus(path: Path, split_seed: int) -> RetrievalCorpus:
    if not path.exists():
        raise FileNotFoundError(f"TACO dataset path does not exist: {path}")
    if path.is_dir():
        split_records = _maybe_load_local_taco_split_dir(path)
        if split_records is not None:
            return split_records
        for filename in ("taco.jsonl", "taco.json", "train.jsonl", "train.json"):
            candidate = path / filename
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No supported TACO files found under {path}")
    rows = _load_json_records(path)
    records = _rows_to_taco_records(rows, split="pool", source_dataset="taco_local")
    corpus, _ = _deterministic_partition(records, split_seed=split_seed, source_dataset="taco_local")
    return corpus


def _maybe_load_local_taco_split_dir(path: Path) -> RetrievalCorpus | None:
    split_map = {
        "train": [path / "train.jsonl", path / "train.json"],
        "validation": [path / "validation.jsonl", path / "validation.json", path / "valid.jsonl", path / "valid.json"],
        "test": [path / "test.jsonl", path / "test.json"],
    }
    resolved: dict[str, list[dict[str, Any]]] = {}
    for split_name, candidates in split_map.items():
        for candidate in candidates:
            if candidate.exists():
                resolved[split_name] = _load_json_records(candidate)
                break
    if len(resolved) != 3:
        return None
    return RetrievalCorpus(
        train=_rows_to_taco_records(resolved["train"], split="train", source_dataset="taco_local"),
        validation=_rows_to_taco_records(resolved["validation"], split="validation", source_dataset="taco_local"),
        test=_rows_to_taco_records(resolved["test"], split="test", source_dataset="taco_local"),
    )


def _rows_to_taco_records(rows: list[dict[str, Any]], split: str, source_dataset: str) -> list[RetrievalRecord]:
    records: list[RetrievalRecord] = []
    for index, row in enumerate(rows):
        pair = taco_row_to_pair(row)
        if pair is None:
            continue
        record_id = str(row.get("task_id") or row.get("question_id") or row.get("id") or f"{split}:{index}")
        records.append(
            RetrievalRecord(
                id=record_id,
                query=pair[0],
                code=pair[1],
                split=split,
                source_dataset=source_dataset,
            )
        )
    if not records:
        raise ValueError(f"No valid TACO retrieval pairs found for split '{split}'")
    return records


def _pairs_to_records(split_rows: Any, split: str, source_dataset: str) -> list[RetrievalRecord]:
    return [
        RetrievalRecord(
            id=f"{split}:{index}",
            query=str(row["text"]).strip(),
            code=str(row["code"]).strip(),
            split=split,
            source_dataset=source_dataset,
        )
        for index, row in enumerate(split_rows)
        if str(row["text"]).strip() and str(row["code"]).strip()
    ]


def _deterministic_partition(
    records: list[RetrievalRecord],
    *,
    split_seed: int,
    source_dataset: str,
) -> tuple[RetrievalCorpus, dict[str, list[str]]]:
    if len(records) < 3:
        raise ValueError(f"{source_dataset} requires at least 3 valid records to build deterministic splits")

    ordered = list(records)
    rng = random.Random(split_seed)
    rng.shuffle(ordered)

    total = len(ordered)
    test_size = max(1, total // 5)
    validation_size = max(1, total // 10)
    train_size = total - test_size - validation_size
    if train_size < 1:
        raise ValueError(f"{source_dataset} did not yield enough records after deterministic partitioning")

    train_rows = _with_split(ordered[:train_size], "train")
    validation_rows = _with_split(ordered[train_size : train_size + validation_size], "validation")
    test_rows = _with_split(ordered[train_size + validation_size :], "test")
    id_manifest = {
        "train": [record.id for record in train_rows],
        "validation": [record.id for record in validation_rows],
        "test": [record.id for record in test_rows],
    }
    return RetrievalCorpus(train=train_rows, validation=validation_rows, test=test_rows), id_manifest


def _with_split(records: list[RetrievalRecord], split: str) -> list[RetrievalRecord]:
    return [
        RetrievalRecord(
            id=record.id,
            query=record.query,
            code=record.code,
            split=split,
            source_dataset=record.source_dataset,
        )
        for record in records
    ]


def _sortable_id(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):09d}")
    except ValueError:
        return (1, value)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            flattened: list[dict[str, Any]] = []
            for value in payload.values():
                if isinstance(value, list):
                    flattened.extend(row for row in value if isinstance(row, dict))
            if flattened:
                return flattened
        raise ValueError(f"Unsupported JSON structure in {path}")

    opener = gzip.open if path.suffix.endswith("gz") else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows
