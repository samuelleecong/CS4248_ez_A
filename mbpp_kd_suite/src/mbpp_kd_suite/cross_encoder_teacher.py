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
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import resolve_output_root
from .data import dataset_dict_to_splits, load_retrieval_dataset
from .metrics import evaluate_symmetric_backbone, reciprocal_rank_metrics
from .modeling import encode_texts_backbone, infer_model_encoding_spec
from .runtime import (
    apply_device_runtime_optimizations,
    maybe_empty_device_cache,
    pick_device,
    set_seed,
)


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
    negatives_per_query: int = 6
    negative_strategy: str = "mixed"
    negative_miner_model: str = "sentence-transformers/all-mpnet-base-v2"
    hard_negative_pool_size: int = 20
    train_objective: str = "combined"
    pair_bce_weight: float = 0.25
    max_length: int = 512
    miner_batch_size: int = 64
    miner_max_query_length: int = 160
    miner_max_code_length: int = 256
    baseline_bi_encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    shortlist_train_top_k: int | None = None
    shortlist_train_model: str | None = None
    seed: int = 42
    output_dir: str = "cross_encoder_teacher"
    taco_val_size: int = 1000
    optimize_for_mps: bool = False
    max_train_queries: int | None = None
    max_eval_queries: int | None = None
    save_model: bool = True
    compare_to_baseline: bool = True


@dataclass(frozen=True)
class GroupedQueryExample:
    query: str
    positive_doc: str
    negative_docs: tuple[str, ...]


class GroupedQueryDataset(Dataset):
    def __init__(self, examples: list[GroupedQueryExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> GroupedQueryExample:
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
    parser.add_argument(
        "--negative-strategy",
        choices=("random", "hard", "mixed"),
        default=CrossEncoderTrainConfig.negative_strategy,
    )
    parser.add_argument("--negative-miner-model", default=CrossEncoderTrainConfig.negative_miner_model)
    parser.add_argument("--hard-negative-pool-size", type=int, default=CrossEncoderTrainConfig.hard_negative_pool_size)
    parser.add_argument(
        "--train-objective",
        choices=("bce", "group_softmax", "combined"),
        default=CrossEncoderTrainConfig.train_objective,
    )
    parser.add_argument("--pair-bce-weight", type=float, default=CrossEncoderTrainConfig.pair_bce_weight)
    parser.add_argument("--max-length", type=int, default=CrossEncoderTrainConfig.max_length)
    parser.add_argument("--miner-batch-size", type=int, default=CrossEncoderTrainConfig.miner_batch_size)
    parser.add_argument("--miner-max-query-length", type=int, default=CrossEncoderTrainConfig.miner_max_query_length)
    parser.add_argument("--miner-max-code-length", type=int, default=CrossEncoderTrainConfig.miner_max_code_length)
    parser.add_argument("--baseline-bi-encoder-model", default=CrossEncoderTrainConfig.baseline_bi_encoder_model)
    parser.add_argument(
        "--shortlist-train-top-k",
        type=int,
        default=None,
        help="If set, build reranker training negatives only from the bi-encoder top-k shortlist.",
    )
    parser.add_argument(
        "--shortlist-train-model",
        default=None,
        help="Bi-encoder used to build shortlist negatives. Defaults to --baseline-bi-encoder-model.",
    )
    parser.add_argument("--seed", type=int, default=CrossEncoderTrainConfig.seed)
    parser.add_argument("--output-dir", default=CrossEncoderTrainConfig.output_dir)
    parser.add_argument("--taco-val-size", type=int, default=CrossEncoderTrainConfig.taco_val_size)
    parser.add_argument("--optimize-for-mps", action="store_true")
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--skip-baseline-compare", action="store_true")
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
        negative_strategy=args.negative_strategy,
        negative_miner_model=args.negative_miner_model,
        hard_negative_pool_size=args.hard_negative_pool_size,
        train_objective=args.train_objective,
        pair_bce_weight=args.pair_bce_weight,
        max_length=args.max_length,
        miner_batch_size=args.miner_batch_size,
        miner_max_query_length=args.miner_max_query_length,
        miner_max_code_length=args.miner_max_code_length,
        baseline_bi_encoder_model=args.baseline_bi_encoder_model,
        shortlist_train_top_k=args.shortlist_train_top_k,
        shortlist_train_model=args.shortlist_train_model,
        seed=args.seed,
        output_dir=args.output_dir,
        taco_val_size=args.taco_val_size,
        optimize_for_mps=args.optimize_for_mps,
        max_train_queries=args.max_train_queries,
        max_eval_queries=args.max_eval_queries,
        save_model=not args.no_save_model,
        compare_to_baseline=not args.skip_baseline_compare,
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


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _resolve_negative_miner_model(cfg: CrossEncoderTrainConfig) -> str:
    return cfg.shortlist_train_model or cfg.baseline_bi_encoder_model


@torch.no_grad()
def mine_hard_negative_indices(
    queries: list[str],
    codes: list[str],
    cfg: CrossEncoderTrainConfig,
    device: torch.device,
    model_name: str | None = None,
    pool_size: int | None = None,
) -> list[list[int]]:
    if len(queries) != len(codes):
        raise ValueError("Queries and codes must be aligned for hard negative mining.")
    if len(queries) < 2:
        return [[] for _ in queries]

    miner_model_name = model_name or cfg.negative_miner_model
    print(f"Mining hard negatives with bi-encoder: {miner_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(miner_model_name)
    model = AutoModel.from_pretrained(miner_model_name).to(device)
    encoding_spec = infer_model_encoding_spec(
        miner_model_name,
        getattr(model.config, "_name_or_path", None),
        getattr(tokenizer, "name_or_path", None),
    )
    model.eval()

    query_embs = encode_texts_backbone(
        model=model,
        tokenizer=tokenizer,
        texts=queries,
        text_role="query",
        encoding_spec=encoding_spec,
        max_length=cfg.miner_max_query_length,
        batch_size=cfg.miner_batch_size,
        device=device,
        desc="mine_q",
    )
    doc_embs = encode_texts_backbone(
        model=model,
        tokenizer=tokenizer,
        texts=codes,
        text_role="document",
        encoding_spec=encoding_spec,
        max_length=cfg.miner_max_code_length,
        batch_size=cfg.miner_batch_size,
        device=device,
        desc="mine_d",
    )
    del model
    maybe_empty_device_cache(device)

    requested_pool_size = pool_size if pool_size is not None else cfg.hard_negative_pool_size
    top_k = min(requested_pool_size, len(codes) - 1)
    if top_k <= 0:
        return [[] for _ in queries]

    hard_pools: list[list[int]] = []
    doc_embs_t = doc_embs.T
    for start in tqdm(range(0, len(queries), cfg.miner_batch_size), desc="mine_topk", leave=False):
        query_batch = query_embs[start : start + cfg.miner_batch_size]
        scores = query_batch @ doc_embs_t
        row_indices = torch.arange(start, start + query_batch.size(0))
        scores[torch.arange(query_batch.size(0)), row_indices] = -1e9
        top_indices = scores.topk(top_k, dim=-1).indices.tolist()
        hard_pools.extend(top_indices)
    return hard_pools


def build_grouped_examples(
    queries: list[str],
    codes: list[str],
    cfg: CrossEncoderTrainConfig,
    hard_negative_pools: list[list[int]] | None,
) -> tuple[list[GroupedQueryExample], dict[str, Any]]:
    if len(queries) != len(codes):
        raise ValueError("Queries and codes must have the same length.")
    if len(queries) < 2:
        raise ValueError("Need at least 2 training pairs to build negative groups.")

    rng = torch.Generator().manual_seed(cfg.seed)
    total_docs = len(codes)
    actual_negatives = min(cfg.negatives_per_query, total_docs - 1)
    if actual_negatives <= 0:
        raise ValueError("Need at least one negative candidate per query.")

    examples: list[GroupedQueryExample] = []
    hard_counts: list[int] = []
    random_counts: list[int] = []

    for idx, query in enumerate(queries):
        hard_pool = hard_negative_pools[idx] if hard_negative_pools is not None else []
        hard_pool = [candidate for candidate in _dedupe_preserve_order(hard_pool) if candidate != idx]
        hard_pool_set = set(hard_pool)

        if cfg.negative_strategy == "hard":
            hard_target = actual_negatives
        elif cfg.negative_strategy == "mixed":
            hard_target = min(actual_negatives, max(1, math.ceil(actual_negatives * 0.75)))
        else:
            hard_target = 0

        selected: list[int] = hard_pool[:hard_target]
        selected_set = set(selected)

        remaining_needed = actual_negatives - len(selected)
        if remaining_needed > 0:
            candidates = [candidate for candidate in range(total_docs) if candidate != idx and candidate not in selected_set]
            if remaining_needed > len(candidates):
                raise ValueError("Not enough unique negatives to complete the training groups.")
            perm = torch.randperm(len(candidates), generator=rng)[:remaining_needed].tolist()
            selected.extend(candidates[pos] for pos in perm)

        selected = _dedupe_preserve_order(selected)
        if len(selected) != actual_negatives:
            raise ValueError("Negative sampling did not produce a fixed group size.")

        negatives = tuple(codes[neg_idx] for neg_idx in selected)
        examples.append(
            GroupedQueryExample(
                query=query,
                positive_doc=codes[idx],
                negative_docs=negatives,
            )
        )
        hard_used = sum(1 for neg_idx in selected if neg_idx in hard_pool_set)
        hard_counts.append(hard_used)
        random_counts.append(actual_negatives - hard_used)

    summary = {
        "queries": float(len(examples)),
        "actual_negatives_per_query": float(actual_negatives),
        "avg_hard_negatives_per_query": float(sum(hard_counts) / len(hard_counts)),
        "avg_random_negatives_per_query": float(sum(random_counts) / len(random_counts)),
        "negative_strategy": cfg.negative_strategy,
    }
    return examples, summary


def make_train_loader(
    examples: list[GroupedQueryExample],
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    dataset = GroupedQueryDataset(examples)
    group_size = 1 + len(examples[0].negative_docs)

    def collate_fn(batch: list[GroupedQueryExample]) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        flat_queries: list[str] = []
        flat_docs: list[str] = []
        pair_labels: list[float] = []

        for example in batch:
            docs = [example.positive_doc, *example.negative_docs]
            flat_queries.extend([example.query] * len(docs))
            flat_docs.extend(docs)
            pair_labels.extend([1.0] + [0.0] * (len(docs) - 1))

        tokens = tokenizer(
            flat_queries,
            flat_docs,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(pair_labels, dtype=torch.float32)
        return tokens, labels, group_size

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


def compute_training_loss(
    logits: torch.Tensor,
    pair_labels: torch.Tensor,
    group_size: int,
    objective: str,
    pair_bce_weight: float,
) -> tuple[torch.Tensor, float, float]:
    group_logits = logits.view(-1, group_size)
    group_targets = torch.zeros(group_logits.size(0), dtype=torch.long, device=logits.device)
    group_loss = F.cross_entropy(group_logits, group_targets)
    pair_loss = F.binary_cross_entropy_with_logits(logits, pair_labels)

    if objective == "bce":
        total_loss = pair_loss
    elif objective == "group_softmax":
        total_loss = group_loss
    elif objective == "combined":
        total_loss = group_loss + pair_bce_weight * pair_loss
    else:
        raise ValueError(f"Unsupported train objective: {objective}")
    return total_loss, float(group_loss.item()), float(pair_loss.item())


def evaluate_bi_encoder_baseline(
    cfg: CrossEncoderTrainConfig,
    data: Any,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    val_queries, val_codes = maybe_truncate_queries(data.validation.queries, data.validation.codes, cfg.max_eval_queries)
    test_queries, test_codes = maybe_truncate_queries(data.test.queries, data.test.codes, cfg.max_eval_queries)
    return evaluate_symmetric_backbone(
        model_name=cfg.baseline_bi_encoder_model,
        val_queries=val_queries,
        val_codes=val_codes,
        test_queries=test_queries,
        test_codes=test_codes,
        max_query_length=cfg.miner_max_query_length,
        max_code_length=cfg.miner_max_code_length,
        eval_batch_size=cfg.miner_batch_size,
        device=device,
    )


def run(cfg: CrossEncoderTrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = pick_device()
    runtime_cfg = type(
        "RuntimeCfg",
        (),
        {
            "batch_size": cfg.batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "optimize_for_mps": cfg.optimize_for_mps,
        },
    )()
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

    if cfg.shortlist_train_top_k is not None and cfg.shortlist_train_top_k <= 0:
        raise ValueError("--shortlist-train-top-k must be positive when provided.")
    if cfg.shortlist_train_top_k is not None and cfg.negative_strategy != "hard":
        raise ValueError("--shortlist-train-top-k currently requires --negative-strategy hard.")
    if cfg.shortlist_train_top_k is not None and cfg.negatives_per_query > cfg.shortlist_train_top_k:
        raise ValueError("--negatives-per-query cannot exceed --shortlist-train-top-k.")

    baseline_metrics: dict[str, Any] | None = None
    if cfg.compare_to_baseline:
        print(f"Evaluating bi-encoder baseline teacher: {cfg.baseline_bi_encoder_model}")
        baseline_metrics = evaluate_bi_encoder_baseline(cfg=cfg, data=data, device=device)
        print(
            "Baseline bi-encoder test -> "
            f"MRR={baseline_metrics['test']['MRR']:.4f} "
            f"R@1={baseline_metrics['test']['Recall@1']:.4f} "
            f"R@10={baseline_metrics['test']['Recall@10']:.4f}"
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
        "Zero-shot cross-encoder test -> "
        f"MRR={zero_shot['test']['MRR']:.4f} "
        f"R@1={zero_shot['test']['Recall@1']:.4f} "
        f"R@10={zero_shot['test']['Recall@10']:.4f}"
    )

    hard_negative_pools: list[list[int]] | None = None
    if cfg.shortlist_train_top_k is not None:
        shortlist_model = _resolve_negative_miner_model(cfg)
        print(
            "Building training negatives from bi-encoder shortlist -> "
            f"model={shortlist_model} top_k={cfg.shortlist_train_top_k}"
        )
        hard_negative_pools = mine_hard_negative_indices(
            queries=train_queries,
            codes=train_codes,
            cfg=cfg,
            device=device,
            model_name=shortlist_model,
            pool_size=cfg.shortlist_train_top_k,
        )
    elif cfg.negative_strategy in {"hard", "mixed"}:
        hard_negative_pools = mine_hard_negative_indices(
            queries=train_queries,
            codes=train_codes,
            cfg=cfg,
            device=device,
        )

    train_examples, negative_sampling = build_grouped_examples(
        queries=train_queries,
        codes=train_codes,
        cfg=cfg,
        hard_negative_pools=hard_negative_pools,
    )
    write_json(run_dir / "negative_sampling.json", negative_sampling)

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
        group_sum = 0.0
        pair_sum = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"cross-encoder epoch {epoch}/{cfg.epochs}", leave=False)
        for tokenized, pair_labels, group_size in pbar:
            tokenized = to_device(tokenized, device)
            pair_labels = pair_labels.to(device)

            logits = model(**tokenized).logits.squeeze(-1)
            total_loss, group_loss, pair_loss = compute_training_loss(
                logits=logits,
                pair_labels=pair_labels,
                group_size=group_size,
                objective=cfg.train_objective,
                pair_bce_weight=cfg.pair_bce_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            steps += 1
            loss_sum += float(total_loss.item())
            group_sum += group_loss
            pair_sum += pair_loss
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", group=f"{group_loss:.4f}")

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
            "train_group_loss": group_sum / max(steps, 1),
            "train_pair_bce": pair_sum / max(steps, 1),
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

    comparisons: dict[str, Any] = {}
    if baseline_metrics is not None:
        comparisons["zero_shot_vs_baseline"] = {
            "validation_mrr_delta": zero_shot["validation"]["MRR"] - baseline_metrics["validation"]["MRR"],
            "test_mrr_delta": zero_shot["test"]["MRR"] - baseline_metrics["test"]["MRR"],
        }
        comparisons["finetuned_vs_baseline"] = {
            "validation_mrr_delta": finetuned["validation"]["MRR"] - baseline_metrics["validation"]["MRR"],
            "test_mrr_delta": finetuned["test"]["MRR"] - baseline_metrics["test"]["MRR"],
        }

    summary = {
        "teacher_type": "cross_encoder_reranker",
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "training_design": {
            "train_objective": cfg.train_objective,
            "pair_bce_weight": cfg.pair_bce_weight,
            "negative_strategy": cfg.negative_strategy,
            "negative_miner_model": cfg.negative_miner_model if cfg.negative_strategy in {"hard", "mixed"} else None,
            "hard_negative_pool_size": cfg.hard_negative_pool_size,
            "negatives_per_query": cfg.negatives_per_query,
            "max_length": cfg.max_length,
            "shortlist_train_top_k": cfg.shortlist_train_top_k,
            "shortlist_train_model": _resolve_negative_miner_model(cfg) if cfg.shortlist_train_top_k is not None else None,
        },
        "negative_sampling": negative_sampling,
        "baseline_bi_encoder": baseline_metrics,
        "zero_shot": zero_shot,
        "best_finetuned": finetuned,
        "comparisons": comparisons,
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
        "Best finetuned cross-encoder test -> "
        f"MRR={finetuned['test']['MRR']:.4f} "
        f"R@1={finetuned['test']['Recall@1']:.4f} "
        f"R@10={finetuned['test']['Recall@10']:.4f}"
    )
    if baseline_metrics is not None:
        delta = comparisons["finetuned_vs_baseline"]["test_mrr_delta"]
        print(f"Finetuned cross-encoder vs baseline bi-encoder (test MRR delta): {delta:+.4f}")
    print(f"Artifacts saved to: {run_dir}")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(config_from_args(args))
