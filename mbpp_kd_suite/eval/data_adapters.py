from __future__ import annotations

import gzip
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from datasets import DatasetDict, load_dataset

from mbpp_kd_suite.data import load_mbpp_dataset
from .types import RetrievalCorpus, RetrievalRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MBPP_PATH = PROJECT_ROOT / "datasets" / "mbpp" / "mbpp.jsonl"
DEFAULT_CODESARCHNET_PATH = PROJECT_ROOT / "datasets" / "CodeSearchNet" / "resources" / "data" / "python" / "final" / "jsonl"
_CODESARCHNET_REMOTE_CANDIDATES = (
    ("code_search_net", "python"),
    ("code_x_glue_tc_nl_code_search_adv", "python"),
)


class DatasetAdapter(ABC):
    def __init__(self, path: str | None = None) -> None:
        self.path = path

    @abstractmethod
    def load(self) -> RetrievalCorpus:
        raise NotImplementedError


class MBPPAdapter(DatasetAdapter):
    def load(self) -> RetrievalCorpus:
        local_path = self._resolve_local_path()
        if local_path is not None:
            return self._load_local(local_path)
        return self._load_remote()

    def _resolve_local_path(self) -> Path | None:
        if self.path:
            candidate = Path(self.path).expanduser()
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"MBPP dataset path does not exist: {candidate}")
        if DEFAULT_MBPP_PATH.exists():
            return DEFAULT_MBPP_PATH
        return None

    def _load_local(self, path: Path) -> RetrievalCorpus:
        if path.is_dir():
            split_records = self._maybe_load_split_directory(path)
            if split_records is not None:
                return split_records
            for candidate_name in ("mbpp.jsonl", "sanitized-mbpp.json"):
                candidate = path / candidate_name
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(f"No supported MBPP files found under {path}")

        rows = _load_json_records(path)
        records = [_mbpp_row_to_record(row, idx, split="unsplit") for idx, row in enumerate(rows)]
        records = _filter_nonempty(records)
        return _split_unsplit_records(records, source_dataset="mbpp")

    def _maybe_load_split_directory(self, path: Path) -> RetrievalCorpus | None:
        split_map = {
            "train": [path / "train.jsonl", path / "train.json"],
            "validation": [path / "validation.jsonl", path / "validation.json", path / "valid.jsonl", path / "valid.json"],
            "test": [path / "test.jsonl", path / "test.json"],
        }
        found: dict[str, list[dict[str, Any]]] = {}
        for split, candidates in split_map.items():
            for candidate in candidates:
                if candidate.exists():
                    found[split] = _load_json_records(candidate)
                    break
        if len(found) != 3:
            return None
        return RetrievalCorpus(
            train=_filter_nonempty([_mbpp_row_to_record(row, idx, split="train") for idx, row in enumerate(found["train"])]),
            validation=_filter_nonempty([
                _mbpp_row_to_record(row, idx, split="validation") for idx, row in enumerate(found["validation"])
            ]),
            test=_filter_nonempty([_mbpp_row_to_record(row, idx, split="test") for idx, row in enumerate(found["test"])]),
        )

    def _load_remote(self) -> RetrievalCorpus:
        dataset: DatasetDict = load_mbpp_dataset("google-research-datasets/mbpp")
        return RetrievalCorpus(
            train=_filter_nonempty([_mbpp_row_to_record(row, idx, split="train") for idx, row in enumerate(dataset["train"])]),
            validation=_filter_nonempty([
                _mbpp_row_to_record(row, idx, split="validation") for idx, row in enumerate(dataset["validation"])
            ]),
            test=_filter_nonempty([_mbpp_row_to_record(row, idx, split="test") for idx, row in enumerate(dataset["test"])]),
        )


class CodeSearchNetAdapter(DatasetAdapter):
    def load(self) -> RetrievalCorpus:
        local_root = self._resolve_local_root()
        if local_root is not None:
            corpus = self._load_local(local_root)
            if corpus is not None:
                return corpus
            if self.path:
                raise FileNotFoundError(
                    f"CodeSearchNet path {local_root} does not contain python/final/jsonl train/valid/test splits"
                )
        return self._load_remote()

    def _resolve_local_root(self) -> Path | None:
        if self.path:
            candidate = Path(self.path).expanduser()
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"CodeSearchNet dataset path does not exist: {candidate}")
        if DEFAULT_CODESARCHNET_PATH.exists():
            return DEFAULT_CODESARCHNET_PATH
        return None

    def _load_local(self, root: Path) -> RetrievalCorpus | None:
        jsonl_root = root
        if (root / "python" / "final" / "jsonl").exists():
            jsonl_root = root / "python" / "final" / "jsonl"
        split_dirs = {
            "train": jsonl_root / "train",
            "validation": jsonl_root / "valid",
            "test": jsonl_root / "test",
        }
        if not all(path.exists() for path in split_dirs.values()):
            return None

        def load_split(split_name: str, split_dir: Path) -> list[RetrievalRecord]:
            rows: list[dict[str, Any]] = []
            file_paths = sorted(split_dir.glob("*.jsonl")) + sorted(split_dir.glob("*.jsonl.gz"))
            if not file_paths:
                file_paths = sorted(split_dir.glob("*.gz"))
            for file_path in file_paths:
                rows.extend(_load_json_records(file_path))
            return _filter_nonempty(
                [_codesearchnet_row_to_record(row, idx, split=split_name) for idx, row in enumerate(rows)]
            )

        return RetrievalCorpus(
            train=load_split("train", split_dirs["train"]),
            validation=load_split("validation", split_dirs["validation"]),
            test=load_split("test", split_dirs["test"]),
        )

    def _load_remote(self) -> RetrievalCorpus:
        errors: list[str] = []
        for dataset_name, config_name in _CODESARCHNET_REMOTE_CANDIDATES:
            try:
                dataset = load_dataset(dataset_name, config_name)
                return RetrievalCorpus(
                    train=_load_codesearchnet_hf_split(dataset, ("train",), "train"),
                    validation=_load_codesearchnet_hf_split(dataset, ("validation", "valid"), "validation"),
                    test=_load_codesearchnet_hf_split(dataset, ("test",), "test"),
                )
            except Exception as exc:  # pragma: no cover - network / remote schema dependent
                errors.append(f"{dataset_name}/{config_name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "Unable to load CodeSearchNet from local path or remote source. Tried local-first and remote fallbacks. "
            + " | ".join(errors)
        )


def get_dataset_adapter(name: str, path: str | None = None) -> DatasetAdapter:
    normalized = name.strip().lower()
    if normalized in {"mbpp", "google-research-datasets/mbpp"}:
        return MBPPAdapter(path=path)
    if normalized in {"codesearchnet", "code_search_net", "codesearchnet_python"}:
        return CodeSearchNetAdapter(path=path)
    raise ValueError(f"Unsupported dataset adapter: {name}")


def _load_codesearchnet_hf_split(dataset: DatasetDict, candidate_names: tuple[str, ...], split_name: str) -> list[RetrievalRecord]:
    for candidate in candidate_names:
        if candidate in dataset:
            return _filter_nonempty(
                [_codesearchnet_row_to_record(row, idx, split=split_name) for idx, row in enumerate(dataset[candidate])]
            )
    raise ValueError(f"CodeSearchNet dataset is missing split for {split_name}: expected one of {candidate_names}")


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
    records: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _mbpp_row_to_record(row: dict[str, Any], idx: int, split: str) -> RetrievalRecord:
    query = str(row.get("text") or row.get("prompt") or "").strip()
    code = str(row.get("code") or "").strip()
    task_id = row.get("task_id")
    record_id = str(task_id if task_id is not None else idx)
    return RetrievalRecord(
        id=record_id,
        query=query,
        code=code,
        split=split,
        source_dataset="mbpp",
    )


def _codesearchnet_row_to_record(row: dict[str, Any], idx: int, split: str) -> RetrievalRecord:
    query = str(
        row.get("docstring")
        or row.get("doc")
        or row.get("func_documentation_string")
        or ""
    ).strip()
    if not query:
        for token_key in ("docstring_tokens", "func_documentation_tokens"):
            tokens = row.get(token_key)
            if isinstance(tokens, list):
                query = " ".join(str(token) for token in tokens if str(token).strip()).strip()
                if query:
                    break
    code = str(
        row.get("code")
        or row.get("original_string")
        or row.get("whole_func_string")
        or row.get("func_code_string")
        or ""
    ).strip()
    identity_parts = [
        str(row.get("repo") or row.get("repository_name") or "").strip(),
        str(row.get("path") or row.get("func_path_in_repository") or "").strip(),
        str(row.get("func_name") or "").strip(),
        str(row.get("url") or row.get("func_code_url") or "").strip(),
    ]
    stable_id = "::".join(part for part in identity_parts if part)
    if not stable_id:
        stable_id = f"{split}:{idx}"
    return RetrievalRecord(
        id=stable_id,
        query=query,
        code=code,
        split=split,
        source_dataset="codesearchnet",
    )


def _filter_nonempty(records: Iterable[RetrievalRecord]) -> list[RetrievalRecord]:
    return [record for record in records if record.query.strip() and record.code.strip()]


def _split_unsplit_records(records: list[RetrievalRecord], source_dataset: str) -> RetrievalCorpus:
    if len(records) < 3:
        raise ValueError(f"{source_dataset} local file must contain at least 3 valid records to create train/validation/test")
    ordered = sorted(records, key=lambda record: record.id)
    total = len(ordered)
    test_size = max(1, total // 10)
    val_size = max(1, total // 10)
    train_end = total - (val_size + test_size)
    if train_end < 1:
        raise ValueError(f"{source_dataset} local file does not have enough records after deterministic split")

    def relabel(rows: list[RetrievalRecord], split_name: str) -> list[RetrievalRecord]:
        return [
            RetrievalRecord(
                id=row.id,
                query=row.query,
                code=row.code,
                split=split_name,
                source_dataset=row.source_dataset,
            )
            for row in rows
        ]

    return RetrievalCorpus(
        train=relabel(ordered[:train_end], "train"),
        validation=relabel(ordered[train_end : train_end + val_size], "validation"),
        test=relabel(ordered[train_end + val_size :], "test"),
    )
