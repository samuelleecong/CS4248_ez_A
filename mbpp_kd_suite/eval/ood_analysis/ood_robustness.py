from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mbpp_kd_suite.metrics import paired_ranking_metrics, paired_ranks
from mbpp_kd_suite.runtime import maybe_empty_device_cache, pick_device, set_seed

from .ood_data import load_mbpp_ood_corpus, load_taco_retrieval_corpus
from .perturbations import LEXICAL_PROBE_TIERS, PERTURBATION_TIERS, perturb_queries
from ..reporting import resolve_eval_output_root


DEFAULT_TACO_DATASET = "BEE-spoke-data/TACO-hf"
TASK_CHOICES = ("mbpp_ood", "taco_robustness", "all")
HFEncoderAdapter = None


@dataclass(frozen=True)
class WorkflowConfig:
    models: tuple[str, ...]
    task: str
    mbpp_dataset_path: str | None
    taco_dataset_name: str
    taco_dataset_path: str | None
    split: str
    split_seed: int
    perturbation_tier: str
    lexical_map_path: str | None
    ks: tuple[int, ...]
    max_query_length: int
    max_code_length: int
    batch_size: int
    device: str
    output_dir: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MBPP OOD evaluation and TACO query-perturbation robustness experiments with HF-hosted models."
    )
    parser.add_argument("--model", action="append", default=[], help="Repeatable Hugging Face model ID.")
    parser.add_argument("--models-file", default=None, help="Optional newline-delimited model ID file.")
    parser.add_argument("--task", choices=TASK_CHOICES, default="all")
    parser.add_argument("--mbpp-dataset-path", default=None)
    parser.add_argument("--taco-dataset-name", default=DEFAULT_TACO_DATASET)
    parser.add_argument("--taco-dataset-path", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--perturbation-tier",
        choices=(*PERTURBATION_TIERS, "all"),
        default="all",
        help="Only applies to the TACO robustness task; MBPP OOD always runs clean.",
    )
    parser.add_argument("--lexical-map-path", default=None)
    parser.add_argument("--ks", default="1,5,10")
    parser.add_argument("--max-query-length", type=int, default=160)
    parser.add_argument("--max-code-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-dir", default="runs/ood_analysis")
    return parser


def parse_ks(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values:
        raise ValueError("At least one k value must be provided")
    if any(value <= 0 for value in values):
        raise ValueError("All k values must be positive")
    return values


def load_models(model_args: list[str], models_file: str | None) -> tuple[str, ...]:
    models = [value.strip() for value in model_args if value.strip()]
    if models_file:
        file_path = Path(models_file).expanduser()
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model in seen:
            continue
        deduped.append(model)
        seen.add(model)
    if not deduped:
        raise ValueError("Provide at least one --model or --models-file entry")
    return tuple(deduped)


def workflow_config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    return WorkflowConfig(
        models=load_models(args.model, args.models_file),
        task=args.task,
        mbpp_dataset_path=args.mbpp_dataset_path,
        taco_dataset_name=args.taco_dataset_name,
        taco_dataset_path=args.taco_dataset_path,
        split=args.split,
        split_seed=args.split_seed,
        perturbation_tier=args.perturbation_tier,
        lexical_map_path=args.lexical_map_path,
        ks=parse_ks(args.ks),
        max_query_length=args.max_query_length,
        max_code_length=args.max_code_length,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = workflow_config_from_args(args)
    run_dir = run_workflow(cfg)
    print("=== OOD / Robustness Summary ===")
    print(f"models={len(cfg.models)} task={cfg.task} split={cfg.split} seed={cfg.split_seed}")
    print(f"artifacts={run_dir}")


def run_workflow(cfg: WorkflowConfig) -> Path:
    set_seed(cfg.split_seed)
    device = _resolve_device(cfg.device)
    output_root = resolve_eval_output_root(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    lexical_probe_rows: list[dict[str, Any]] = []
    selected_ids: dict[str, list[str]] = {}
    task_payloads: list[dict[str, Any]] = []

    tasks = _resolve_tasks(cfg.task)
    _log_progress(
        "Starting evaluation "
        f"(models={len(cfg.models)}, tasks={len(tasks)}, split={cfg.split}, batch_size={cfg.batch_size}, device={device})"
    )
    for task_index, task_name in enumerate(tasks, start=1):
        _log_progress(f"[Task {task_index}/{len(tasks)}] Loading task context for '{task_name}'")
        task_context = _load_task_context(task_name, cfg)
        _log_progress(
            f"[Task {task_index}/{len(tasks)}] Loaded {len(task_context['records'])} records "
            f"from {task_context['dataset_name']} with tiers={','.join(task_context['tiers'])}"
        )
        task_payloads.append(task_context["payload"])
        if "selected_ids" in task_context:
            selected_ids[task_name] = task_context["selected_ids"]
        tier_queries = _build_tier_queries(
            task_context["queries"],
            task_context["tiers"],
            cfg.split_seed,
            lexical_map_path=cfg.lexical_map_path,
        )

        for model_index, model_name in enumerate(cfg.models, start=1):
            adapter_cls = HFEncoderAdapter
            if adapter_cls is None:
                from ..model_adapters import HFEncoderAdapter as adapter_cls

            adapter = None
            code_embeddings = None
            model_started = time.perf_counter()
            try:
                _log_progress(
                    f"[Task {task_index}/{len(tasks)}][Model {model_index}/{len(cfg.models)}] "
                    f"Loading {model_name}"
                )
                adapter = adapter_cls(
                    model_name_or_path=model_name,
                    max_query_length=cfg.max_query_length,
                    max_code_length=cfg.max_code_length,
                    batch_size=cfg.batch_size,
                    device=device,
                )
                _log_progress(
                    f"[Task {task_index}/{len(tasks)}][Model {model_index}/{len(cfg.models)}] "
                    "Encoding code corpus"
                )
                code_embeddings = adapter.encode_codes(task_context["codes"])

                for tier_index, tier in enumerate(task_context["tiers"], start=1):
                    perturbed_queries = tier_queries[tier]
                    started = time.perf_counter()
                    _log_progress(
                        f"[Task {task_index}/{len(tasks)}][Model {model_index}/{len(cfg.models)}]"
                        f"[Tier {tier_index}/{len(task_context['tiers'])}] Running {tier}"
                    )
                    query_embeddings = adapter.encode_queries(perturbed_queries)
                    score_matrix = (query_embeddings @ code_embeddings.T).detach().cpu().numpy()
                    ranks = paired_ranks(score_matrix)
                    metrics = paired_ranking_metrics(score_matrix, ks=cfg.ks)
                    metrics["MeanRank"] = float(np.mean(ranks))
                    metrics["MedianRank"] = float(np.median(ranks))
                    top_indices = np.argmax(score_matrix, axis=1)
                    runtime_sec = time.perf_counter() - started
                    _log_progress(
                        f"[Task {task_index}/{len(tasks)}][Model {model_index}/{len(cfg.models)}]"
                        f"[Tier {tier_index}/{len(task_context['tiers'])}] "
                        f"Done {tier}: MRR={float(metrics['MRR']):.4f}, R@10={float(metrics.get('Recall@10', np.nan)):.4f}, "
                        f"time={runtime_sec:.2f}s"
                    )

                    metrics_rows.append(
                        {
                            "model_name": model_name,
                            "task": task_name,
                            "dataset_name": task_context["dataset_name"],
                            "split": cfg.split,
                            "perturbation_tier": tier,
                            "mrr": float(metrics["MRR"]),
                            "recall@1": float(metrics.get("Recall@1", np.nan)),
                            "recall@5": float(metrics.get("Recall@5", np.nan)),
                            "recall@10": float(metrics.get("Recall@10", np.nan)),
                            "mean_rank": float(metrics["MeanRank"]),
                            "median_rank": float(metrics["MedianRank"]),
                            "runtime_sec": runtime_sec,
                            "query_count": len(task_context["records"]),
                            "code_count": len(task_context["records"]),
                        }
                    )

                    for row_index, (record, perturbed_query, rank) in enumerate(zip(task_context["records"], perturbed_queries, ranks)):
                        per_query_rows.append(
                            {
                                "model_name": model_name,
                                "task": task_name,
                                "dataset_name": task_context["dataset_name"],
                                "split": cfg.split,
                                "perturbation_tier": tier,
                                "record_id": record.id,
                                "original_query": record.query,
                                "perturbed_query": perturbed_query,
                                "code": record.code,
                                "rank": int(rank),
                                "reciprocal_rank": float(1.0 / rank),
                            }
                        )
                        if tier in LEXICAL_PROBE_TIERS:
                            lexical_probe_rows.append(
                                {
                                    "model_name": model_name,
                                    "task": task_name,
                                    "dataset_name": task_context["dataset_name"],
                                    "split": cfg.split,
                                    "perturbation_tier": tier,
                                    "record_id": record.id,
                                    "original_query": record.query,
                                    "perturbed_query": perturbed_query,
                                    "gold_rank": int(rank),
                                    "reciprocal_rank": float(1.0 / rank),
                                    "top_retrieved_rank_index": int(top_indices[row_index]),
                                    "top_retrieved_code_summary": _summarize_code(task_context["codes"][int(top_indices[row_index])]),
                                    "top_retrieved_matches_gold": bool(int(top_indices[row_index]) == row_index),
                                }
                            )

                    del query_embeddings, score_matrix, ranks, metrics, top_indices
                    maybe_empty_device_cache(device)
            finally:
                if code_embeddings is not None:
                    del code_embeddings
                if adapter is not None:
                    del adapter
                maybe_empty_device_cache(device)
                _log_progress(
                    f"[Task {task_index}/{len(tasks)}][Model {model_index}/{len(cfg.models)}] "
                    f"Finished {model_name} in {time.perf_counter() - model_started:.2f}s"
                )

    _attach_clean_deltas(metrics_rows)
    _write_csv(run_dir / "metrics.csv", metrics_rows)
    _write_csv(run_dir / "per_query_results.csv", per_query_rows)
    _write_csv(run_dir / "lexical_probe_results.csv", lexical_probe_rows)
    example_rows = _build_example_cases(metrics_rows, per_query_rows, lexical_probe_rows)
    _write_csv(run_dir / "example_cases.csv", example_rows)
    (run_dir / "selected_ids.json").write_text(json.dumps(selected_ids, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    summary_payload = {
        "run_dir": str(run_dir),
        "tasks": task_payloads,
        "models": list(cfg.models),
        "metrics": metrics_rows,
        "lexical_probe_results_count": len(lexical_probe_rows),
        "example_cases_count": len(example_rows),
        "selected_ids": selected_ids,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(_build_summary_markdown(cfg, metrics_rows), encoding="utf-8")
    (run_dir / "analysis_summary.md").write_text(
        _build_analysis_summary(cfg, metrics_rows, example_rows),
        encoding="utf-8",
    )
    _log_progress(f"Completed evaluation. Artifacts written to {run_dir}")
    return run_dir


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return pick_device()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device request: {requested}")


def _resolve_tasks(task: str) -> tuple[str, ...]:
    if task == "all":
        return ("mbpp_ood", "taco_robustness")
    return (task,)


def _load_task_context(task_name: str, cfg: WorkflowConfig) -> dict[str, Any]:
    if task_name == "mbpp_ood":
        corpus, selected_ids = load_mbpp_ood_corpus(cfg.mbpp_dataset_path, split_seed=cfg.split_seed)
        records = corpus.get_split(cfg.split)
        return {
            "dataset_name": "mbpp_sanitized_ood",
            "records": records,
            "queries": [record.query for record in records],
            "codes": [record.code for record in records],
            "tiers": ("clean",),
            "selected_ids": selected_ids[cfg.split],
            "payload": {
                "task": task_name,
                "dataset_name": "mbpp_sanitized_ood",
                "split": cfg.split,
                "count": len(records),
            },
        }

    if task_name == "taco_robustness":
        corpus = load_taco_retrieval_corpus(
            dataset_name=cfg.taco_dataset_name,
            dataset_path=cfg.taco_dataset_path,
            split_seed=cfg.split_seed,
            split=cfg.split,
        )
        records = corpus.get_split(cfg.split)
        return {
            "dataset_name": cfg.taco_dataset_name if not cfg.taco_dataset_path else "taco_local",
            "records": records,
            "queries": [record.query for record in records],
            "codes": [record.code for record in records],
            "tiers": _resolve_tiers(cfg.perturbation_tier),
            "payload": {
                "task": task_name,
                "dataset_name": cfg.taco_dataset_name if not cfg.taco_dataset_path else "taco_local",
                "split": cfg.split,
                "count": len(records),
            },
        }

    raise ValueError(f"Unsupported task: {task_name}")


def _resolve_tiers(tier: str) -> tuple[str, ...]:
    if tier == "all":
        return PERTURBATION_TIERS
    if tier == "clean":
        return ("clean",)
    return ("clean", tier)


def _attach_clean_deltas(metrics_rows: list[dict[str, Any]]) -> None:
    clean_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in metrics_rows:
        if row["perturbation_tier"] != "clean":
            continue
        clean_index[(row["model_name"], row["task"], row["dataset_name"], row["split"])] = row

    for row in metrics_rows:
        baseline = clean_index.get((row["model_name"], row["task"], row["dataset_name"], row["split"]))
        if baseline is None:
            row["delta_mrr_vs_clean"] = 0.0
            row["delta_recall10_vs_clean"] = 0.0
            continue
        row["delta_mrr_vs_clean"] = float(row["mrr"] - baseline["mrr"])
        row["delta_recall10_vs_clean"] = float(row["recall@10"] - baseline["recall@10"])


def _build_tier_queries(
    queries: list[str],
    tiers: tuple[str, ...],
    seed: int,
    lexical_map_path: str | None,
) -> dict[str, list[str]]:
    cached: dict[str, list[str]] = {}
    for tier in tiers:
        started = time.perf_counter()
        _log_progress(f"[Perturbations] Preparing tier '{tier}' for {len(queries)} queries")
        cached[tier] = queries if tier == "clean" else perturb_queries(
            queries,
            tier=tier,
            seed=seed,
            lexical_map_path=lexical_map_path,
        )
        _log_progress(f"[Perturbations] Prepared tier '{tier}' in {time.perf_counter() - started:.2f}s")
    return cached


def _summarize_code(code: str, limit: int = 120) -> str:
    first_line = next((line.strip() for line in code.splitlines() if line.strip()), "")
    if len(first_line) <= limit:
        return first_line
    return first_line[: limit - 3] + "..."


def _build_example_cases(
    metrics_rows: list[dict[str, Any]],
    per_query_rows: list[dict[str, Any]],
    lexical_probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    clean_index = {
        (row["model_name"], row["task"], row["dataset_name"], row["record_id"]): row
        for row in per_query_rows
        if row["perturbation_tier"] == "clean"
    }

    by_probe: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in lexical_probe_rows:
        by_probe.setdefault((row["model_name"], row["perturbation_tier"]), []).append(row)

    for (model_name, tier), rows in by_probe.items():
        enriched: list[dict[str, Any]] = []
        for row in rows:
            clean_row = clean_index.get((row["model_name"], row["task"], row["dataset_name"], row["record_id"]))
            clean_rank = int(clean_row["rank"]) if clean_row is not None else int(row["gold_rank"])
            delta_rank = int(row["gold_rank"]) - clean_rank
            enriched.append({**row, "clean_rank": clean_rank, "delta_rank": delta_rank})
        if not enriched:
            continue
        worst = max(enriched, key=lambda item: (item["delta_rank"], item["gold_rank"]))
        best = min(enriched, key=lambda item: (item["delta_rank"], item["gold_rank"]))
        examples.append(
            {
                "case_type": "keyword_failure",
                "model_name": model_name,
                "task": worst["task"],
                "dataset_name": worst["dataset_name"],
                "perturbation_tier": tier,
                "record_id": worst["record_id"],
                "original_query": worst["original_query"],
                "perturbed_query": worst["perturbed_query"],
                "clean_rank": worst["clean_rank"],
                "perturbed_rank": worst["gold_rank"],
                "delta_rank": worst["delta_rank"],
                "top_retrieved_code_summary": worst["top_retrieved_code_summary"],
            }
        )
        examples.append(
            {
                "case_type": "keyword_success",
                "model_name": model_name,
                "task": best["task"],
                "dataset_name": best["dataset_name"],
                "perturbation_tier": tier,
                "record_id": best["record_id"],
                "original_query": best["original_query"],
                "perturbed_query": best["perturbed_query"],
                "clean_rank": best["clean_rank"],
                "perturbed_rank": best["gold_rank"],
                "delta_rank": best["delta_rank"],
                "top_retrieved_code_summary": best["top_retrieved_code_summary"],
            }
        )

    mbpp_clean = [row for row in per_query_rows if row["task"] == "mbpp_ood" and row["perturbation_tier"] == "clean"]
    for model_name in sorted({row["model_name"] for row in mbpp_clean}):
        rows = [row for row in mbpp_clean if row["model_name"] == model_name]
        if not rows:
            continue
        best = min(rows, key=lambda item: int(item["rank"]))
        worst = max(rows, key=lambda item: int(item["rank"]))
        for case_type, row in (("mbpp_ood_success", best), ("mbpp_ood_failure", worst)):
            examples.append(
                {
                    "case_type": case_type,
                    "model_name": model_name,
                    "task": row["task"],
                    "dataset_name": row["dataset_name"],
                    "perturbation_tier": row["perturbation_tier"],
                    "record_id": row["record_id"],
                    "original_query": row["original_query"],
                    "perturbed_query": row["perturbed_query"],
                    "clean_rank": int(row["rank"]),
                    "perturbed_rank": int(row["rank"]),
                    "delta_rank": 0,
                    "top_retrieved_code_summary": _summarize_code(row["code"]),
                }
            )
    return examples


def _build_analysis_summary(
    cfg: WorkflowConfig,
    metrics_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Analysis Summary",
        "",
        f"- Models evaluated: `{len(cfg.models)}`",
        f"- Tasks: `{cfg.task}`",
        f"- Split: `{cfg.split}`",
        "",
        "## OOD Snapshot",
        "",
    ]
    mbpp_rows = [row for row in metrics_rows if row["task"] == "mbpp_ood" and row["perturbation_tier"] == "clean"]
    for row in sorted(mbpp_rows, key=lambda item: float(item["mrr"]), reverse=True)[:5]:
        lines.append(f"- `{row['model_name']}` MBPP MRR={float(row['mrr']):.4f}, R@10={float(row['recall@10']):.4f}")

    lines.extend(["", "## Keyword Probe Snapshot", ""])
    lexical_rows = [row for row in metrics_rows if row["perturbation_tier"] in LEXICAL_PROBE_TIERS]
    for row in sorted(lexical_rows, key=lambda item: float(item["delta_mrr_vs_clean"]))[:8]:
        lines.append(
            f"- `{row['model_name']}` `{row['perturbation_tier']}` dMRR={float(row['delta_mrr_vs_clean']):.4f}, "
            f"dR@10={float(row['delta_recall10_vs_clean']):.4f}"
        )

    lines.extend(["", "## Representative Cases", ""])
    for row in example_rows[:12]:
        lines.append(
            f"- `{row['case_type']}` `{row['model_name']}` record `{row['record_id']}`: "
            f"clean_rank={row['clean_rank']} perturbed_rank={row['perturbed_rank']}"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary_markdown(cfg: WorkflowConfig, metrics_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# OOD And Robustness Evaluation",
        "",
        f"- Models: `{len(cfg.models)}`",
        f"- Task selection: `{cfg.task}`",
        f"- Split: `{cfg.split}`",
        f"- Split seed: `{cfg.split_seed}`",
        "",
        "## Metrics",
        "",
        "| Model | Task | Dataset | Tier | MRR | R@1 | R@5 | R@10 | Mean Rank | dMRR vs clean | dR@10 vs clean |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_rows:
        lines.append(
            "| {model} | {task} | {dataset} | {tier} | {mrr:.4f} | {r1:.4f} | {r5:.4f} | {r10:.4f} | "
            "{mean_rank:.2f} | {dmrr:.4f} | {dr10:.4f} |".format(
                model=row["model_name"],
                task=row["task"],
                dataset=row["dataset_name"],
                tier=row["perturbation_tier"],
                mrr=float(row["mrr"]),
                r1=float(row["recall@1"]),
                r5=float(row["recall@5"]),
                r10=float(row["recall@10"]),
                mean_rank=float(row["mean_rank"]),
                dmrr=float(row["delta_mrr_vs_clean"]),
                dr10=float(row["delta_recall10_vs_clean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics.csv`: per-model, per-task, per-tier summary metrics",
            "- `per_query_results.csv`: original/perturbed queries with paired ranks",
            "- `selected_ids.json`: MBPP OOD split record IDs",
            "- `summary.json`: machine-readable run summary",
        ]
    )
    return "\n".join(lines) + "\n"


def _make_run_dir(output_root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = output_root / timestamp
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def _log_progress(message: str) -> None:
    print(f"[ood_robustness] {message}", flush=True)


if __name__ == "__main__":
    main()
