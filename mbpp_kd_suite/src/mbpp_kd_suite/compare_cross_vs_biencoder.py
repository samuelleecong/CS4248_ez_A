from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import resolve_output_root
from .cross_encoder_teacher import evaluate_split as evaluate_cross_split
from .data import dataset_dict_to_splits, load_retrieval_dataset
from .metrics import reciprocal_rank_metrics
from .runtime import pick_device, set_seed


@dataclass
class CompareConfig:
    cross_encoder_model: str
    bi_encoder_model: str
    dataset_name: str = "google-research-datasets/mbpp"
    protocol: str = "heldout_test"
    eval_batch_size: int = 32
    bi_encoder_batch_size: int = 64
    max_length: int = 512
    max_eval_queries: int | None = None
    taco_val_size: int = 1000
    seed: int = 42
    output_dir: str = "cross_vs_biencoder_compare"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fairly compare a cross-encoder reranker against a bi-encoder on the same MBPP-style candidate pool."
    )
    parser.add_argument("--cross-encoder-model", required=True, help="HF model name or local path for the cross-encoder.")
    parser.add_argument("--bi-encoder-model", required=True, help="HF model name or local SentenceTransformer checkpoint path.")
    parser.add_argument("--dataset-name", default=CompareConfig.dataset_name)
    parser.add_argument("--protocol", choices=("heldout_test", "full_corpus"), default=CompareConfig.protocol)
    parser.add_argument("--eval-batch-size", type=int, default=CompareConfig.eval_batch_size)
    parser.add_argument("--bi-encoder-batch-size", type=int, default=CompareConfig.bi_encoder_batch_size)
    parser.add_argument("--max-length", type=int, default=CompareConfig.max_length)
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--taco-val-size", type=int, default=CompareConfig.taco_val_size)
    parser.add_argument("--seed", type=int, default=CompareConfig.seed)
    parser.add_argument("--output-dir", default=CompareConfig.output_dir)
    return parser


def config_from_args(args: argparse.Namespace) -> CompareConfig:
    return CompareConfig(
        cross_encoder_model=args.cross_encoder_model,
        bi_encoder_model=args.bi_encoder_model,
        dataset_name=args.dataset_name,
        protocol=args.protocol,
        eval_batch_size=args.eval_batch_size,
        bi_encoder_batch_size=args.bi_encoder_batch_size,
        max_length=args.max_length,
        max_eval_queries=args.max_eval_queries,
        taco_val_size=args.taco_val_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def build_run_dir(cfg: CompareConfig) -> tuple[Path, Path]:
    output_root = resolve_output_root(cfg.output_dir)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return output_root, run_dir


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def extract_mbpp_prompt_pairs() -> tuple[list[str], list[str]]:
    ds = load_dataset("google-research-datasets/mbpp", download_mode="reuse_dataset_if_exists")
    prompt_rows = []
    for row in ds["prompt"]:
        query = str(row["text"]).strip()
        code = str(row["code"]).strip()
        if query and code:
            prompt_rows.append((int(row["task_id"]), query, code))
    prompt_rows.sort(key=lambda item: item[0])
    return [row[1] for row in prompt_rows], [row[2] for row in prompt_rows]


def build_protocol_pool(data: Any, protocol: str) -> tuple[list[str], list[str], str]:
    train_queries = list(data.train.queries)
    train_codes = list(data.train.codes)
    val_queries = list(data.validation.queries)
    val_codes = list(data.validation.codes)
    test_queries = list(data.test.queries)
    test_codes = list(data.test.codes)

    if protocol == "heldout_test":
        return test_queries, test_codes, "test"
    if protocol == "full_corpus":
        prompt_queries, prompt_codes = extract_mbpp_prompt_pairs()
        return (
            train_queries + val_queries + test_queries + prompt_queries,
            train_codes + val_codes + test_codes + prompt_codes,
            "all",
        )
    raise ValueError(f"Unsupported protocol: {protocol}")


def maybe_truncate(queries: list[str], codes: list[str], limit: int | None) -> tuple[list[str], list[str]]:
    if limit is None or limit >= len(queries):
        return queries, codes
    return queries[:limit], codes[:limit]


def evaluate_bi_encoder(
    model_name_or_path: str,
    queries: list[str],
    codes: list[str],
    batch_size: int,
) -> dict[str, float]:
    model = SentenceTransformer(model_name_or_path, local_files_only=False)
    q_emb = model.encode(
        queries,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    c_emb = model.encode(
        codes,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    scores = q_emb @ c_emb.T
    return reciprocal_rank_metrics(scores)


def run(cfg: CompareConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = pick_device()
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
    queries, codes, eval_split = build_protocol_pool(data, cfg.protocol)
    queries, codes = maybe_truncate(queries, codes, cfg.max_eval_queries)
    print(f"Protocol: {cfg.protocol} | queries={len(queries)} docs={len(codes)} eval_split={eval_split}")

    print(f"Loading cross-encoder: {cfg.cross_encoder_model}")
    cross_tokenizer = AutoTokenizer.from_pretrained(cfg.cross_encoder_model)
    cross_model = AutoModelForSequenceClassification.from_pretrained(cfg.cross_encoder_model).to(device)
    cross_start = time.perf_counter()
    cross_metrics = evaluate_cross_split(
        model=cross_model,
        tokenizer=cross_tokenizer,
        queries=queries,
        docs=codes,
        cfg=type(
            "CrossEvalCfg",
            (),
            {
                "max_eval_queries": None,
                "eval_batch_size": cfg.eval_batch_size,
                "max_length": cfg.max_length,
            },
        )(),
        device=device,
        split_name=f"{cfg.protocol}_cross",
    )
    cross_runtime = time.perf_counter() - cross_start

    print(f"Loading bi-encoder: {cfg.bi_encoder_model}")
    bi_start = time.perf_counter()
    bi_metrics = evaluate_bi_encoder(
        model_name_or_path=cfg.bi_encoder_model,
        queries=queries,
        codes=codes,
        batch_size=cfg.bi_encoder_batch_size,
    )
    bi_runtime = time.perf_counter() - bi_start

    summary = {
        "dataset_name": cfg.dataset_name,
        "protocol": cfg.protocol,
        "eval_split": eval_split,
        "num_queries": len(queries),
        "num_docs": len(codes),
        "cross_encoder": {
            "model": cfg.cross_encoder_model,
            "metrics": cross_metrics,
            "runtime_sec": cross_runtime,
        },
        "bi_encoder": {
            "model": cfg.bi_encoder_model,
            "metrics": bi_metrics,
            "runtime_sec": bi_runtime,
        },
        "delta_cross_minus_bi": {
            "MRR": cross_metrics["MRR"] - bi_metrics["MRR"],
            "Recall@1": cross_metrics["Recall@1"] - bi_metrics["Recall@1"],
            "Recall@5": cross_metrics["Recall@5"] - bi_metrics["Recall@5"],
            "Recall@10": cross_metrics["Recall@10"] - bi_metrics["Recall@10"],
            "MedianRank": cross_metrics["MedianRank"] - bi_metrics["MedianRank"],
        },
    }
    write_json(run_dir / "results_summary.json", summary)

    print(
        "Cross-encoder -> "
        f"MRR={cross_metrics['MRR']:.4f} "
        f"R@1={cross_metrics['Recall@1']:.4f} "
        f"R@10={cross_metrics['Recall@10']:.4f} "
        f"({cross_runtime:.1f}s)"
    )
    print(
        "Bi-encoder     -> "
        f"MRR={bi_metrics['MRR']:.4f} "
        f"R@1={bi_metrics['Recall@1']:.4f} "
        f"R@10={bi_metrics['Recall@10']:.4f} "
        f"({bi_runtime:.1f}s)"
    )
    print(
        "Delta (cross - bi) -> "
        f"MRR={summary['delta_cross_minus_bi']['MRR']:+.4f} "
        f"R@1={summary['delta_cross_minus_bi']['Recall@1']:+.4f} "
        f"R@10={summary['delta_cross_minus_bi']['Recall@10']:+.4f}"
    )
    print(f"Artifacts saved to: {run_dir}")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(config_from_args(args))
