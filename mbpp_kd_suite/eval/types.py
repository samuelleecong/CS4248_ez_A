from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalRecord:
    id: str
    query: str
    code: str
    split: str
    source_dataset: str


@dataclass(frozen=True)
class RetrievalCorpus:
    train: list[RetrievalRecord]
    validation: list[RetrievalRecord]
    test: list[RetrievalRecord]

    def get_split(self, split: str) -> list[RetrievalRecord]:
        if split == "validation":
            return self.validation
        if split == "test":
            return self.test
        if split == "train":
            return self.train
        raise ValueError(f"Unsupported split: {split}")

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


@dataclass(frozen=True)
class EvalConfig:
    dataset_name: str
    dataset_path: str | None
    model_source: str
    model_name_or_path: str
    checkpoint_format: str | None
    split: str
    ks: tuple[int, ...]
    max_query_length: int
    max_code_length: int
    batch_size: int
    device: str
    output_dir: str
    seed: int
