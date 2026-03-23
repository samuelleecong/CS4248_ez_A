from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import resolve_output_root
from .data import dataset_dict_to_splits, load_retrieval_dataset
from .metrics import reciprocal_rank_metrics
from .runtime import apply_device_runtime_optimizations, maybe_empty_device_cache, pick_device, set_seed


@dataclass
class CrossEncoderTrainConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    dataset_name: str = "google-research-datasets/mbpp"
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    negatives_per_query: int = 4
    max_length: int = 384
    seed: int = 42
    output_dir: str = "cross_encoder_teacher"
    taco_val_size: int = 1000
    optimize_for_mps: bool = False
    max_train_queries: int | None = None
    max_eval_queries: int | None = None
    save_model: bool = True


class LabeledPairDataset(Dataset):
    def __init__(self, examples: list[tuple[str, str, float]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        return self.examples[idx]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a cross-encoder reranker as a teacher on MBPP-style retrieval data."
    )
    parser.add_argument("--model-name", default=CrossEncoderTrainConfig.model_name)
    parser.add_argument("--dataset-name", default=CrossEncoderTrainConfig.dataset_name)
    parser.add_argument("--batch-size", type=int, default=CrossEncoderTrainConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=CrossEncoderTrainConfig.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=CrossEncoderTrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=CrossEncoderTrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=CrossEncoderTrainConfig.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=CrossEncoderTrainConfig.warmup_ratio)
    parser.add_argument("--negatives-per-query", type=int, default=CrossEncoderTrainConfig.negatives_per_query)
    parser.add_argument("--max-length", type=int, default=CrossEncoderTrainConfig.max_length)
    parser.add_argument("--seed", type=int, default=CrossEncoderTrainConfig.seed)
    parser.add_argument("--output-dir", default=CrossEncoderTrainConfig.output_dir)
    parser.add_argument("--taco-val-size", type=int, default=CrossEncoderTrainConfig.taco_val_size)
    parser.add_argument("--optimize-for-mps", action="store_true")
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--no-save-model", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> CrossEncoderTrainConfig:
    return CrossEncoderTrainConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        negatives_per_query=args.negatives_per_query,
        max_length=args.max_length,
        seed=args.seed,
        output_dir=args.output_dir,
        taco_val_size=args.taco_val_size,
        optimize_for_mps=args.optimize_for_mps,
        max_train_queries=args.max_train_queries,
        max_eval_queries=args.max_eval_queries,
        save_model=not args.no_save_model,
    )


def build_run_dir(cfg: CrossEncoderTrainConfig) -> tuple[Path, Path]:
    output_root = resolve_output_root(cfg.output_dir)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return output_root, run_dir


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def maybe_truncate_queries(
    queries: list[str],
    codes: list[str],
    limit: int | None,
) -> tuple[list[str], list[str]]:
    if limit is None or limit >= len(queries):
        return queries, codes
    return queries[:limit], codes[:limit]


def make_train_examples(
    queries: list[str],
    codes: list[str],
    negatives_per_query: int,
    seed: int,
) -> list[tuple[str, str, float]]:
    if negatives_per_query < 1:
        raise ValueError("--negatives-per-query must be at least 1.")

    rng = torch.Generator().manual_seed(seed)
    n = len(queries)
    if n < 2:
        raise ValueError("Need at least 2 training pairs to sample negatives.")

    examples: list[tuple[str, str, float]] = []
    for idx, query in enumerate(queries):
        examples.append((query, codes[idx], 1.0))

        candidates = torch.arange(n)
        mask = candidates != idx
        negative_pool = candidates[mask]
        sample_count = min(negatives_per_query, negative_pool.numel())
        perm = negative_pool[torch.randperm(negative_pool.numel(), generator=rng)[:sample_count]]
        for neg_idx in perm.tolist():
            examples.append((query, codes[neg_idx], 0.0))
    return examples


def make_train_loader(
    examples: list[tuple[str, str, float]],
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    dataset = LabeledPairDataset(examples)

    def collate_fn(batch: list[tuple[str, str, float]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        queries = [query for query, _, _ in batch]
        docs = [doc for _, doc, _ in batch]
        labels = torch.tensor([label for _, _, label in batch], dtype=torch.float32)
        tokens = tokenizer(
            queries,
            docs,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokens, labels

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def score_all_pairs(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    queries: list[str],
    docs: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    model.eval()
    all_rows: list[torch.Tensor] = []
    for query in tqdm(queries, desc=desc, leave=False):
        row_scores: list[torch.Tensor] = []
        for start in range(0, len(docs), batch_size):
            doc_batch = docs[start : start + batch_size]
            tokens = tokenizer(
                [query] * len(doc_batch),
                doc_batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = to_device(tokens, device)
            logits = model(**tokens).logits.squeeze(-1)
            row_scores.append(logits.detach().cpu())
        all_rows.append(torch.cat(row_scores, dim=0))
        maybe_empty_device_cache(device)
    return torch.stack(all_rows, dim=0)


@torch.no_grad()
def evaluate_split(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    queries: list[str],
    docs: list[str],
    cfg: CrossEncoderTrainConfig,
    device: torch.device,
    split_name: str,
) -> dict[str, float]:
    queries, docs = maybe_truncate_queries(queries, docs, cfg.max_eval_queries)
    score_matrix = score_all_pairs(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        docs=docs,
        batch_size=cfg.eval_batch_size,
        max_length=cfg.max_length,
        device=device,
        desc=f"cross_{split_name}",
    )
    metrics = reciprocal_rank_metrics(score_matrix.numpy())
    metrics["num_queries"] = float(len(queries))
    metrics["num_docs"] = float(len(docs))
    return metrics


def run(cfg: CrossEncoderTrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = pick_device()
    runtime_cfg = type("RuntimeCfg", (), {
        "batch_size": cfg.batch_size,
        "eval_batch_size": cfg.eval_batch_size,
        "optimize_for_mps": cfg.optimize_for_mps,
    })()
    apply_device_runtime_optimizations(cfg=runtime_cfg, device=device)
    cfg.batch_size = runtime_cfg.batch_size
    cfg.eval_batch_size = runtime_cfg.eval_batch_size

    output_root, run_dir = build_run_dir(cfg)
    write_json(run_dir / "config.json", {**asdict(cfg), "resolved_output_dir": str(output_root)})

    print(f"Using device: {device}")
    print(f"Loading retrieval dataset: {cfg.dataset_name}")
    dataset = load_retrieval_dataset(
        dataset_name=cfg.dataset_name,
        taco_val_size=cfg.taco_val_size,
        seed=cfg.seed,
    )
    data = dataset_dict_to_splits(dataset)
    train_queries, train_codes = maybe_truncate_queries(
        data.train.queries,
        data.train.codes,
        cfg.max_train_queries,
    )
    print(
        "Dataset splits -> "
        f"train: {len(train_queries)}, val: {len(data.validation.queries)}, test: {len(data.test.queries)}"
    )

    print(f"Loading cross-encoder teacher: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    ).to(device)

    zero_shot = {
        "validation": evaluate_split(
            model=model,
            tokenizer=tokenizer,
            queries=data.validation.queries,
            docs=data.validation.codes,
            cfg=cfg,
            device=device,
            split_name="val_zero_shot",
        ),
        "test": evaluate_split(
            model=model,
            tokenizer=tokenizer,
            queries=data.test.queries,
            docs=data.test.codes,
            cfg=cfg,
            device=device,
            split_name="test_zero_shot",
        ),
    }
    print(
        "Zero-shot test -> "
        f"MRR={zero_shot['test']['MRR']:.4f} "
        f"R@1={zero_shot['test']['Recall@1']:.4f} "
        f"R@10={zero_shot['test']['Recall@10']:.4f}"
    )

    train_examples = make_train_examples(
        queries=train_queries,
        codes=train_codes,
        negatives_per_query=cfg.negatives_per_query,
        seed=cfg.seed,
    )
    train_loader = make_train_loader(
        examples=train_examples,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(len(train_loader) * cfg.epochs, 1)
    warmup_steps = int(math.ceil(total_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_mrr = -math.inf
    best_state_dict: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"cross-encoder epoch {epoch}/{cfg.epochs}", leave=False)
        for tokenized, labels in pbar:
            tokenized = to_device(tokenized, device)
            labels = labels.to(device)

            logits = model(**tokenized).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            steps += 1
            loss_sum += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        maybe_empty_device_cache(device)
        val_metrics = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            queries=data.validation.queries,
            docs=data.validation.codes,
            cfg=cfg,
            device=device,
            split_name=f"val_epoch_{epoch}",
        )
        epoch_stats = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(steps, 1),
            "val_MRR": val_metrics["MRR"],
            "val_Recall@1": val_metrics["Recall@1"],
            "val_Recall@10": val_metrics["Recall@10"],
        }
        history.append(epoch_stats)

        if val_metrics["MRR"] > best_val_mrr:
            best_val_mrr = val_metrics["MRR"]
            best_state_dict = copy.deepcopy(model.state_dict())

    assert best_state_dict is not None
    model.load_state_dict(best_state_dict)
    model.eval()

    finetuned = {
        "validation": evaluate_split(
            model=model,
            tokenizer=tokenizer,
            queries=data.validation.queries,
            docs=data.validation.codes,
            cfg=cfg,
            device=device,
            split_name="val_best",
        ),
        "test": evaluate_split(
            model=model,
            tokenizer=tokenizer,
            queries=data.test.queries,
            docs=data.test.codes,
            cfg=cfg,
            device=device,
            split_name="test_best",
        ),
    }

    summary = {
        "teacher_type": "cross_encoder_reranker",
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "zero_shot": zero_shot,
        "best_finetuned": finetuned,
        "improvement": {
            "validation_mrr_delta": finetuned["validation"]["MRR"] - zero_shot["validation"]["MRR"],
            "test_mrr_delta": finetuned["test"]["MRR"] - zero_shot["test"]["MRR"],
        },
    }

    write_json(run_dir / "history.json", history)
    write_json(run_dir / "results_summary.json", summary)

    if cfg.save_model:
        model_dir = run_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    print(
        "Best finetuned test -> "
        f"MRR={finetuned['test']['MRR']:.4f} "
        f"R@1={finetuned['test']['Recall@1']:.4f} "
        f"R@10={finetuned['test']['Recall@10']:.4f}"
    )
    print(f"Artifacts saved to: {run_dir}")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(config_from_args(args))
