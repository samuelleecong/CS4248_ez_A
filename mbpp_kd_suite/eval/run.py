from __future__ import annotations

import argparse

from .engine import evaluate_config
from .reporting import write_eval_report
from .types import EvalConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone retrieval evaluator for mbpp_kd_suite checkpoints")
    parser.add_argument("--dataset-name", choices=("mbpp", "codesearchnet"), required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--model-source", choices=("hf", "local"), required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--checkpoint-format", choices=("auto", "hf_dir", "suite_student_dir"), default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--ks", default="1,5,10")
    parser.add_argument("--max-query-length", type=int, default=160)
    parser.add_argument("--max-code-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_ks(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values:
        raise ValueError("At least one k value must be provided")
    if any(value <= 0 for value in values):
        raise ValueError("All k values must be positive")
    return values


def eval_config_from_args(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        model_source=args.model_source,
        model_name_or_path=args.model_name_or_path,
        checkpoint_format=args.checkpoint_format,
        split=args.split,
        ks=parse_ks(args.ks),
        max_query_length=args.max_query_length,
        max_code_length=args.max_code_length,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = eval_config_from_args(args)
    result = evaluate_config(cfg)
    run_dir = write_eval_report(result, cfg)
    print("=== Eval Summary ===")
    print(f"dataset={cfg.dataset_name} split={cfg.split} model={cfg.model_name_or_path}")
    print(
        f"MRR={result['metrics']['MRR']:.4f} "
        f"R@1={result['metrics'].get('Recall@1', 0.0):.4f} "
        f"R@5={result['metrics'].get('Recall@5', 0.0):.4f} "
        f"R@10={result['metrics'].get('Recall@10', 0.0):.4f}"
    )
    print(f"artifacts={run_dir}")


if __name__ == "__main__":
    main()
