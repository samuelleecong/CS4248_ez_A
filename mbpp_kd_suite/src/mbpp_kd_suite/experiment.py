from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer

from .config import DistillTargets, RetrievalSplits, TrainConfig, build_arg_parser, resolve_output_root, train_config_from_args
from .constants import FINETUNED_TEACHER_NAME, METHOD_ORDER, PAPER_SPECS
from .data import dataset_dict_to_splits, load_retrieval_dataset
from .metrics import evaluate_symmetric_backbone, score_metrics_from_embeddings, summarize_analysis
from .modeling import encode_texts_backbone, infer_model_encoding_spec
from .runtime import apply_device_runtime_optimizations, pick_device, set_seed
from .training import make_method_targets, train_student


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_run_dir(cfg: TrainConfig) -> tuple[Path, Path]:
    output_root = resolve_output_root(cfg.output_dir)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return output_root, run_dir


def _write_run_metadata(run_dir: Path, cfg: TrainConfig, output_root: Path) -> None:
    config_payload = asdict(cfg)
    config_payload["methods"] = list(cfg.methods)
    config_payload["extra_baseline_models"] = list(cfg.extra_baseline_models)
    config_payload["resolved_output_dir"] = str(output_root)
    _write_json(run_dir / "config.json", config_payload)
    _write_json(run_dir / "paper_registry.json", [asdict(spec) for spec in PAPER_SPECS])


def _load_text_splits(cfg: TrainConfig) -> RetrievalSplits:
    dataset = load_retrieval_dataset(
        dataset_name=cfg.dataset_name,
        taco_val_size=cfg.taco_val_size,
        seed=cfg.seed,
    )
    return dataset_dict_to_splits(dataset)


def _load_teacher_backbone(
    cfg: TrainConfig,
    device: Any,
) -> tuple[AutoTokenizer, AutoModel, Any]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model)
    model = AutoModel.from_pretrained(cfg.teacher_model).to(device)
    encoding_spec = infer_model_encoding_spec(
        cfg.teacher_model,
        getattr(model.config, "_name_or_path", None),
        getattr(tokenizer, "name_or_path", None),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, encoding_spec


def _encode_teacher_targets(
    cfg: TrainConfig,
    teacher_model: AutoModel,
    teacher_tokenizer: AutoTokenizer,
    teacher_encoding_spec: Any,
    data: RetrievalSplits,
    device: Any,
) -> DistillTargets:
    split_prefixes = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    encoded: dict[str, Any] = {}

    for split_name, split in data.items():
        prefix = split_prefixes[split_name]
        encoded[f"{prefix}_query"] = encode_texts_backbone(
            model=teacher_model,
            tokenizer=teacher_tokenizer,
            texts=split.queries,
            text_role="query",
            encoding_spec=teacher_encoding_spec,
            max_length=cfg.max_query_length,
            batch_size=cfg.eval_batch_size,
            device=device,
            desc=f"teacher_{prefix}_q",
        )
        encoded[f"{prefix}_doc"] = encode_texts_backbone(
            model=teacher_model,
            tokenizer=teacher_tokenizer,
            texts=split.codes,
            text_role="document",
            encoding_spec=teacher_encoding_spec,
            max_length=cfg.max_code_length,
            batch_size=cfg.eval_batch_size,
            device=device,
            desc=f"teacher_{prefix}_d",
        )

    return DistillTargets(name="teacher_full", **encoded)


def _model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def _print_final_metrics(result: dict[str, Any]) -> None:
    print("\n=== Final Test Metrics ===")
    fixed_order = ["direct_big_teacher", "direct_small_student", FINETUNED_TEACHER_NAME, *METHOD_ORDER]
    extra_keys = sorted(k for k in result if k not in fixed_order and k != "analysis")
    for run_name in [*fixed_order, *extra_keys]:
        if run_name not in result:
            continue
        metric = result[run_name]["test"]
        print(
            f"{run_name:>35} | MRR={metric['MRR']:.4f} | R@1={metric['Recall@1']:.4f} | "
            f"R@5={metric['Recall@5']:.4f} | R@10={metric['Recall@10']:.4f}"
        )


def run(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = pick_device()
    apply_device_runtime_optimizations(cfg=cfg, device=device)

    output_root, run_dir = _build_run_dir(cfg)
    _write_run_metadata(run_dir=run_dir, cfg=cfg, output_root=output_root)

    tb_log_dir = run_dir / "tensorboard"
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs: {tb_log_dir}")

    print(f"Using device: {device}")
    print(f"Loading retrieval dataset: {cfg.dataset_name}")
    data = _load_text_splits(cfg)
    print(
        "Dataset splits -> "
        f"train: {len(data.train.queries)}, val: {len(data.validation.queries)}, test: {len(data.test.queries)}"
    )

    print(f"Loading teacher model: {cfg.teacher_model}")
    teacher_tokenizer, teacher_model, teacher_encoding_spec = _load_teacher_backbone(cfg=cfg, device=device)

    print("Precomputing teacher embeddings...")
    teacher_targets = _encode_teacher_targets(
        cfg=cfg,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_encoding_spec=teacher_encoding_spec,
        data=data,
        device=device,
    )

    result: dict[str, Any] = {
        "direct_big_teacher": {
            "validation": score_metrics_from_embeddings(teacher_targets.val_query, teacher_targets.val_doc),
            "test": score_metrics_from_embeddings(teacher_targets.test_query, teacher_targets.test_doc),
        }
    }
    method_targets = make_method_targets(cfg=cfg, full_teacher_targets=teacher_targets)

    print("Running direct baseline: big teacher model (zero-shot)")
    if cfg.run_direct_baselines:
        print("Running direct baseline: small student model (zero-shot)")
        result["direct_small_student"] = evaluate_symmetric_backbone(
            model_name=cfg.student_model,
            val_queries=data.validation.queries,
            val_codes=data.validation.codes,
            test_queries=data.test.queries,
            test_codes=data.test.codes,
            max_query_length=cfg.max_query_length,
            max_code_length=cfg.max_code_length,
            eval_batch_size=cfg.eval_batch_size,
            device=device,
        )

    if cfg.run_finetuned_teacher:
        print(f"Training finetuned teacher baseline: {cfg.teacher_model}")
        ft_teacher_metrics, _, _ = train_student(
            name=FINETUNED_TEACHER_NAME,
            cfg=cfg,
            run_dir=run_dir,
            device=device,
            data=data,
            targets=teacher_targets,
            full_teacher_targets=teacher_targets,
            model_name=cfg.teacher_model,
            supervised=True,
            tb_writer=tb_writer,
        )
        result[FINETUNED_TEACHER_NAME] = ft_teacher_metrics

    for extra_model in cfg.extra_baseline_models:
        slug = _model_slug(extra_model)

        direct_name = f"direct_{slug}"
        print(f"Running direct baseline: {extra_model} (zero-shot)")
        result[direct_name] = evaluate_symmetric_backbone(
            model_name=extra_model,
            val_queries=data.validation.queries,
            val_codes=data.validation.codes,
            test_queries=data.test.queries,
            test_codes=data.test.codes,
            max_query_length=cfg.max_query_length,
            max_code_length=cfg.max_code_length,
            eval_batch_size=cfg.eval_batch_size,
            device=device,
        )

        ft_name = f"finetuned_{slug}"
        print(f"Training finetuned baseline: {extra_model}")
        ft_extra_metrics, _, _ = train_student(
            name=ft_name,
            cfg=cfg,
            run_dir=run_dir,
            device=device,
            data=data,
            targets=teacher_targets,
            full_teacher_targets=teacher_targets,
            model_name=extra_model,
            supervised=True,
            tb_writer=tb_writer,
        )
        result[ft_name] = ft_extra_metrics

    for method_name in cfg.methods:
        print(f"Training method: {method_name}")
        method_metrics, _, _ = train_student(
            name=method_name,
            cfg=cfg,
            run_dir=run_dir,
            device=device,
            data=data,
            targets=method_targets[method_name],
            full_teacher_targets=teacher_targets,
            tb_writer=tb_writer,
        )
        result[method_name] = method_metrics

    analysis = summarize_analysis(result)
    result["analysis"] = analysis

    _write_json(run_dir / "results_summary.json", result)
    _write_json(run_dir / "diagnostics_summary.json", analysis)

    _print_final_metrics(result)
    print("\n=== Diagnostics ===")
    for model_name, values in analysis.items():
        print(f"{model_name}: {values}")

    tb_writer.close()
    print(f"\nArtifacts saved to: {run_dir}")
    print(f"TensorBoard logs: {tb_log_dir}")
    return result


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(train_config_from_args(args))
