from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

from .config import RetrievalSplit, RetrievalSplits
from .constants import CSN_DATASET_NAMES, TACO_DATASET_NAMES
from .modeling import ModelEncodingSpec, format_texts_for_role


class QueryOnlyDataset(TorchDataset):
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[int, str]:
        return idx, self.queries[idx]


class PairedTextDataset(TorchDataset):
    def __init__(self, queries: list[str], codes: list[str]) -> None:
        if len(queries) != len(codes):
            raise ValueError("Queries and codes must have the same length.")
        self.queries = queries
        self.codes = codes

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.queries[idx], self.codes[idx]


def make_query_dataloader(
    queries: list[str],
    tokenizer: AutoTokenizer,
    encoding_spec: ModelEncodingSpec,
    batch_size: int,
    max_query_length: int,
    shuffle: bool,
) -> DataLoader:
    dataset = QueryOnlyDataset(queries)

    def collate_fn(batch: list[tuple[int, str]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        idxs, query_texts = zip(*batch)
        tokenized = tokenizer(
            format_texts_for_role(list(query_texts), text_role="query", encoding_spec=encoding_spec),
            max_length=max_query_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return torch.tensor(idxs, dtype=torch.long), tokenized

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=collate_fn,
    )


def make_pair_dataloader(
    queries: list[str],
    codes: list[str],
    tokenizer: AutoTokenizer,
    encoding_spec: ModelEncodingSpec,
    batch_size: int,
    max_query_length: int,
    max_code_length: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PairedTextDataset(queries=queries, codes=codes)

    def collate_fn(batch: list[tuple[str, str]]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        query_texts, code_texts = zip(*batch)
        query_tokens = tokenizer(
            format_texts_for_role(list(query_texts), text_role="query", encoding_spec=encoding_spec),
            max_length=max_query_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        code_tokens = tokenizer(
            format_texts_for_role(list(code_texts), text_role="document", encoding_spec=encoding_spec),
            max_length=max_code_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return query_tokens, code_tokens

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=collate_fn,
    )


def extract_split_fields(split: Any) -> RetrievalSplit:
    return RetrievalSplit(
        queries=[row["text"] for row in split],
        codes=[row["code"] for row in split],
    )


def dataset_dict_to_splits(dataset: DatasetDict) -> RetrievalSplits:
    return RetrievalSplits(
        train=extract_split_fields(dataset["train"]),
        validation=extract_split_fields(dataset["validation"]),
        test=extract_split_fields(dataset["test"]),
    )


def load_cached_mbpp_dataset() -> DatasetDict | None:
    cache_roots = [
        Path.cwd() / ".hf_cache" / "datasets" / "google-research-datasets___mbpp" / "full" / "0.0.0",
        Path.home() / ".cache" / "huggingface" / "datasets" / "google-research-datasets___mbpp" / "full" / "0.0.0",
    ]

    for root in cache_roots:
        if not root.exists():
            continue
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir():
                continue
            train_file = candidate / "mbpp-train.arrow"
            val_file = candidate / "mbpp-validation.arrow"
            test_file = candidate / "mbpp-test.arrow"
            if train_file.exists() and val_file.exists() and test_file.exists():
                return DatasetDict(
                    {
                        "train": HFDataset.from_file(str(train_file)),
                        "validation": HFDataset.from_file(str(val_file)),
                        "test": HFDataset.from_file(str(test_file)),
                    }
                )
    return None


def load_mbpp_dataset(dataset_name: str) -> DatasetDict:
    if dataset_name == "google-research-datasets/mbpp":
        cached = load_cached_mbpp_dataset()
        if cached is not None:
            return cached
    return load_dataset(dataset_name)


def pairs_to_dataset(pairs: list[tuple[str, str]]) -> HFDataset:
    return HFDataset.from_dict(
        {
            "text": [query for query, _ in pairs],
            "code": [code for _, code in pairs],
        }
    )


def non_empty_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def parse_taco_solutions(raw_solutions: Any) -> list[str]:
    if isinstance(raw_solutions, list):
        return [solution.strip() for solution in raw_solutions if isinstance(solution, str) and solution.strip()]
    if not isinstance(raw_solutions, str) or not raw_solutions.strip():
        return []

    try:
        parsed = json.loads(raw_solutions)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []
    return [solution.strip() for solution in parsed if isinstance(solution, str) and solution.strip()]


def taco_row_to_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    question = non_empty_text(row.get("question"))
    if not question:
        return None

    starter_code = non_empty_text(row.get("starter_code"))
    solutions = parse_taco_solutions(row.get("solutions"))
    if not solutions:
        return None

    query = question
    if starter_code:
        query = f"{question}\n\nStarter code:\n{starter_code}"
    return query, solutions[0]


def make_taco_retrieval_dataset(dataset_name: str, taco_val_size: int, seed: int) -> DatasetDict:
    # BAAI/TACO uses a legacy loading script no longer supported by datasets>=3.
    # Automatically fall back to the Parquet mirror.
    effective_name = dataset_name
    if dataset_name == "BAAI/TACO":
        try:
            raw_dataset = load_dataset(dataset_name, trust_remote_code=True)
        except (RuntimeError, TypeError):
            print("BAAI/TACO loading script not supported, falling back to BEE-spoke-data/TACO-hf")
            effective_name = "BEE-spoke-data/TACO-hf"
            raw_dataset = load_dataset(effective_name)
    else:
        raw_dataset = load_dataset(effective_name)
    if "train" not in raw_dataset or "test" not in raw_dataset:
        raise ValueError(f"TACO dataset {dataset_name} must provide train and test splits.")

    train_pairs = [pair for row in raw_dataset["train"] if (pair := taco_row_to_pair(row)) is not None]
    test_pairs = [pair for row in raw_dataset["test"] if (pair := taco_row_to_pair(row)) is not None]
    if len(train_pairs) <= 1 or not test_pairs:
        raise ValueError(f"TACO dataset {dataset_name} did not yield enough retrieval pairs.")

    rng = random.Random(seed)
    shuffled_indices = list(range(len(train_pairs)))
    rng.shuffle(shuffled_indices)

    val_size = min(taco_val_size, max(1, len(train_pairs) // 10))
    if len(train_pairs) - val_size < 1:
        raise ValueError("TACO train split is too small after validation holdout.")

    val_indices = set(shuffled_indices[:val_size])
    final_train_pairs = [pair for idx, pair in enumerate(train_pairs) if idx not in val_indices]
    final_val_pairs = [pair for idx, pair in enumerate(train_pairs) if idx in val_indices]

    return DatasetDict(
        {
            "train": pairs_to_dataset(final_train_pairs),
            "validation": pairs_to_dataset(final_val_pairs),
            "test": pairs_to_dataset(test_pairs),
        }
    )


def make_csn_retrieval_dataset(dataset_name: str) -> DatasetDict:
    """Load CodeSearchNet Python split and normalise to {text, code} schema.

    CSN ships with predefined train/validation/test splits.  Rows with an
    empty docstring are filtered out because they cannot serve as queries.
    """
    raw = load_dataset(dataset_name, "python")
    for split_name in ("train", "validation", "test"):
        if split_name not in raw:
            raise ValueError(f"CodeSearchNet dataset is missing the '{split_name}' split.")

    def _to_pairs(split: Any) -> list[tuple[str, str]]:
        pairs = []
        for row in split:
            doc = row.get("func_documentation_string", "")
            if not isinstance(doc, str):
                continue
            doc = doc.strip()
            if not doc:
                continue
            pairs.append((doc, row["whole_func_string"]))
        return pairs

    return DatasetDict(
        {
            "train": pairs_to_dataset(_to_pairs(raw["train"])),
            "validation": pairs_to_dataset(_to_pairs(raw["validation"])),
            "test": pairs_to_dataset(_to_pairs(raw["test"])),
        }
    )


def load_retrieval_dataset(dataset_name: str, taco_val_size: int, seed: int) -> DatasetDict:
    if dataset_name in TACO_DATASET_NAMES:
        return make_taco_retrieval_dataset(
            dataset_name=dataset_name,
            taco_val_size=taco_val_size,
            seed=seed,
        )
    if dataset_name in CSN_DATASET_NAMES:
        return make_csn_retrieval_dataset(dataset_name)
    return load_mbpp_dataset(dataset_name)
