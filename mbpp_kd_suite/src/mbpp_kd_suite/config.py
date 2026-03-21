from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .constants import ARTIFACT_ROOT, METHOD_ORDER


@dataclass
class TrainConfig:
    teacher_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    student_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dataset_name: str = "code_search_net"
    methods: tuple[str, ...] = tuple(METHOD_ORDER)
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 8
    lr: float = 2e-5
    weight_decay: float = 1e-2
    temperature: float = 0.05
    distill_weight: float = 1.0
    align_weight: float = 1.0
    pair_weight: float = 1.0
    relation_weight: float = 1.0
    pair_hard_negatives: int = 4
    dark_negatives: int = 4
    dark_mix_ratio: float = 0.65
    hpd_dim: int = 128
    max_query_length: int = 160
    max_code_length: int = 256
    seed: int = 42
    output_dir: str = "runs"
    projection_init: str = "none"
    eval_mode: str = "symmetric"
    taco_val_size: int = 1000
    save_models: bool = False
    run_direct_baselines: bool = True
    run_finetuned_teacher: bool = True
    extra_baseline_models: tuple[str, ...] = ()
    run_diagnostics: bool = True
    optimize_for_mps: bool = False


@dataclass
class DistillTargets:
    name: str
    train_query: torch.Tensor
    train_doc: torch.Tensor
    val_query: torch.Tensor
    val_doc: torch.Tensor
    test_query: torch.Tensor
    test_doc: torch.Tensor

    def split(self, split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = "val" if split_name == "validation" else split_name
        return getattr(self, f"{prefix}_query"), getattr(self, f"{prefix}_doc")

    @property
    def hidden_size(self) -> int:
        return int(self.train_query.shape[1])


@dataclass(frozen=True)
class RetrievalSplit:
    queries: list[str]
    codes: list[str]

    def __post_init__(self) -> None:
        if len(self.queries) != len(self.codes):
            raise ValueError("Queries and codes must have the same length.")


@dataclass(frozen=True)
class RetrievalSplits:
    train: RetrievalSplit
    validation: RetrievalSplit
    test: RetrievalSplit

    def items(self) -> tuple[tuple[str, RetrievalSplit], ...]:
        return (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        )


def resolve_output_root(output_dir: str) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    if output_dir.startswith("./") or output_dir.startswith("../"):
        return output_path
    if output_path.parts and output_path.parts[0] == ARTIFACT_ROOT.name:
        return output_path
    return ARTIFACT_ROOT / output_path


def parse_methods(methods_arg: str) -> tuple[str, ...]:
    methods = tuple(method.strip() for method in methods_arg.split(",") if method.strip())
    if not methods:
        raise ValueError("At least one method must be specified.")
    unknown = sorted(set(methods) - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MBPP benchmark suite for embedding-focused KD methods"
    )
    parser.add_argument("--teacher-model", default=TrainConfig.teacher_model)
    parser.add_argument("--student-model", default=TrainConfig.student_model)
    parser.add_argument("--dataset-name", default=TrainConfig.dataset_name)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TrainConfig.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--distill-weight", type=float, default=TrainConfig.distill_weight)
    parser.add_argument("--align-weight", type=float, default=TrainConfig.align_weight)
    parser.add_argument("--pair-weight", type=float, default=TrainConfig.pair_weight)
    parser.add_argument("--relation-weight", type=float, default=TrainConfig.relation_weight)
    parser.add_argument("--pair-hard-negatives", type=int, default=TrainConfig.pair_hard_negatives)
    parser.add_argument("--dark-negatives", type=int, default=TrainConfig.dark_negatives)
    parser.add_argument("--dark-mix-ratio", type=float, default=TrainConfig.dark_mix_ratio)
    parser.add_argument("--hpd-dim", type=int, default=TrainConfig.hpd_dim)
    parser.add_argument("--max-query-length", type=int, default=TrainConfig.max_query_length)
    parser.add_argument("--max-code-length", type=int, default=TrainConfig.max_code_length)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument(
        "--projection-init",
        choices=("none", "least_squares_queries", "least_squares_both"),
        default=TrainConfig.projection_init,
    )
    parser.add_argument(
        "--eval-mode",
        choices=("asymmetric", "symmetric"),
        default=TrainConfig.eval_mode,
    )
    parser.add_argument("--taco-val-size", type=int, default=TrainConfig.taco_val_size)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--skip-direct-baselines", action="store_true")
    parser.add_argument("--skip-finetuned-teacher", action="store_true")
    parser.add_argument(
        "--extra-baseline-models",
        default="",
        help="Comma-separated list of extra model names to evaluate zero-shot and fine-tuned",
    )
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--optimize-for-mps", action="store_true")
    return parser


def parse_extra_baseline_models(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(m.strip() for m in raw.split(",") if m.strip())


def train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        dataset_name=args.dataset_name,
        methods=parse_methods(args.methods),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        distill_weight=args.distill_weight,
        align_weight=args.align_weight,
        pair_weight=args.pair_weight,
        relation_weight=args.relation_weight,
        pair_hard_negatives=args.pair_hard_negatives,
        dark_negatives=args.dark_negatives,
        dark_mix_ratio=args.dark_mix_ratio,
        hpd_dim=args.hpd_dim,
        max_query_length=args.max_query_length,
        max_code_length=args.max_code_length,
        seed=args.seed,
        output_dir=args.output_dir,
        projection_init=args.projection_init,
        eval_mode=args.eval_mode,
        taco_val_size=args.taco_val_size,
        save_models=args.save_models,
        run_direct_baselines=not args.skip_direct_baselines,
        run_finetuned_teacher=not args.skip_finetuned_teacher,
        extra_baseline_models=parse_extra_baseline_models(args.extra_baseline_models),
        run_diagnostics=not args.skip_diagnostics,
        optimize_for_mps=args.optimize_for_mps,
    )
