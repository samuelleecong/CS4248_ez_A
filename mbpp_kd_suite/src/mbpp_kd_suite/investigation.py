from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi


SUITE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SUITE_ROOT.parent
DOCS_ROOT = SUITE_ROOT / "docs"
EXPERIMENT_LOG = DOCS_ROOT / "EXPERIMENT_LOG.md"
PROJECT_STATUS = DOCS_ROOT / "PROJECT_STATUS.md"
EVAL_RUN_INDEX = SUITE_ROOT / "eval" / "runs" / "run_index.csv"
OUTPUT_EXPERIMENT_INVENTORY = DOCS_ROOT / "kd_experiment_inventory.csv"
OUTPUT_CHECKPOINT_MANIFEST = DOCS_ROOT / "kd_checkpoint_manifest.csv"
OUTPUT_DATASET_STATS = DOCS_ROOT / "kd_dataset_stats.csv"

RUN_HEADING_RE = re.compile(r"^## Run: `(?P<timestamp>[^`]+)` \(`(?P<bucket>[^`]+)`")
TABLE_ROW_RE = re.compile(r"^\|\s*(?P<model>[^|]+?)\s*\|\s*(?P<mrr>[^|]+?)\s*\|\s*(?P<r1>[^|]+?)\s*\|\s*(?P<r5>[^|]+?)\s*\|\s*(?P<r10>[^|]+?)\s*\|$")
OPTION_RE = re.compile(r"^--(?P<name>[a-z0-9\-]+)$")
DEFAULT_TEACHER = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_STUDENT = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATASET = "google-research-datasets/mbpp"
HF_REPO_DEFAULT = "cs4248-nlp/cs4248-model-weights"
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


@dataclass
class InventoryRow:
    source: str
    evidence_class: str
    status: str
    dataset: str
    split: str
    teacher_model: str
    student_model: str
    method: str
    training_budget: str
    mrr: str
    recall_at_1: str
    recall_at_5: str
    recall_at_10: str
    ndcg_at_10: str
    map_at_10: str
    artifact_location: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "evidence_class": self.evidence_class,
            "status": self.status,
            "dataset": self.dataset,
            "split": self.split,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "method": self.method,
            "training_budget": self.training_budget,
            "MRR": self.mrr,
            "Recall@1": self.recall_at_1,
            "Recall@5": self.recall_at_5,
            "Recall@10": self.recall_at_10,
            "nDCG@10": self.ndcg_at_10,
            "MAP@10": self.map_at_10,
            "artifact_location": self.artifact_location,
            "notes": self.notes,
        }


@dataclass
class CheckpointRow:
    local_path: str
    hf_repo_id: str
    dataset: str
    teacher_model: str
    student_model: str
    method: str
    seed: str
    checkpoint_tag: str
    selected_for_report: str
    status: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        return {
            "local_path": self.local_path,
            "hf_repo_id": self.hf_repo_id,
            "dataset": self.dataset,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "method": self.method,
            "seed": self.seed,
            "checkpoint_tag": self.checkpoint_tag,
            "selected_for_report": self.selected_for_report,
            "status": self.status,
            "notes": self.notes,
        }


def compact_dataset_name(value: str) -> str:
    if not value:
        return "-"
    normalized = value.strip().lower()
    if normalized in {"google-research-datasets/mbpp", "mbpp"}:
        return "mbpp"
    if normalized in {"code_search_net", "codesearchnet", "codesearchnet_python", "code_search_net/python"}:
        return "codesearchnet"
    if normalized in {"baai/taco", "bee-spoke-data/taco-hf", "taco"}:
        return "taco"
    return value.rsplit("/", 1)[-1]


def safe_metric(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def normalized_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def median_value(values: Sequence[int | float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0].keys()))
        writer.writeheader()
        writer.writerows(row_list)


def resolve_doc_artifact(path_text: str) -> Path | None:
    candidates = [
        SUITE_ROOT / path_text,
        SUITE_ROOT / "artifacts" / path_text,
        SUITE_ROOT / "artifacts" / "legacy" / path_text,
        PROJECT_ROOT / path_text,
        PROJECT_ROOT / "artifacts" / path_text,
        PROJECT_ROOT / "artifacts" / "legacy" / path_text,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def parse_command_flags(block: str) -> dict[str, str]:
    flattened = " ".join(line.strip().rstrip("\\") for line in block.splitlines() if line.strip())
    tokens = shlex.split(flattened)
    flags: dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        match = OPTION_RE.match(token)
        if not match:
            idx += 1
            continue
        key = match.group("name")
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            flags[key] = tokens[idx + 1]
            idx += 2
        else:
            flags[key] = "true"
            idx += 1
    return flags


def parse_metric_table(lines: list[str], start_index: int) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    idx = start_index
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("|"):
            break
        if "---" in line or "Model" in line:
            idx += 1
            continue
        match = TABLE_ROW_RE.match(line)
        if match:
            rows.append(
                {
                    "model": match.group("model").strip().strip("`") ,
                    "MRR": safe_metric(match.group("mrr")),
                    "Recall@1": safe_metric(match.group("r1")),
                    "Recall@5": safe_metric(match.group("r5")),
                    "Recall@10": safe_metric(match.group("r10")),
                }
            )
        idx += 1
    return rows, idx


def method_to_models(method: str, teacher_model: str, student_model: str) -> tuple[str, str, str]:
    normalized = method.strip()
    if normalized == "direct_big_teacher":
        return normalized, teacher_model, "-"
    if normalized in {"direct_small_student", "supervised_student"}:
        return normalized, teacher_model, student_model
    if normalized == "finetuned_teacher":
        return normalized, teacher_model, teacher_model
    if normalized.startswith("finetuned_"):
        model_name = normalized.removeprefix("finetuned_")
        return normalized, teacher_model, model_name
    if normalized.startswith("direct_"):
        model_name = normalized.removeprefix("direct_")
        return normalized, model_name, "-"
    return normalized, teacher_model, student_model


def collect_doc_runs() -> list[InventoryRow]:
    if not EXPERIMENT_LOG.exists():
        return []
    text = EXPERIMENT_LOG.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[InventoryRow] = []
    idx = 0
    current_timestamp = ""
    current_bucket = ""
    while idx < len(lines):
        heading = RUN_HEADING_RE.match(lines[idx])
        if not heading:
            idx += 1
            continue
        current_timestamp = heading.group("timestamp")
        current_bucket = heading.group("bucket")
        idx += 1

        command_block = ""
        teacher_model = DEFAULT_TEACHER
        student_model = DEFAULT_STUDENT
        dataset_name = DEFAULT_DATASET
        training_budget_parts: list[str] = []
        artifact_paths: list[str] = []
        metric_rows_pending: list[dict[str, str]] = []

        while idx < len(lines) and not lines[idx].startswith("## "):
            line = lines[idx]
            if line.startswith("```bash"):
                block_lines: list[str] = []
                idx += 1
                while idx < len(lines) and not lines[idx].startswith("```"):
                    block_lines.append(lines[idx])
                    idx += 1
                command_block = "\n".join(block_lines)
                flags = parse_command_flags(command_block)
                teacher_model = flags.get("teacher-model", DEFAULT_TEACHER)
                student_model = flags.get("student-model", DEFAULT_STUDENT)
                dataset_name = flags.get("dataset-name", DEFAULT_DATASET)
                for key in ("epochs", "batch-size", "eval-batch-size", "seed"):
                    if key in flags:
                        training_budget_parts.append(f"{key}={flags[key]}")
            elif line.startswith("| Model "):
                metric_rows_pending, idx = parse_metric_table(lines, idx)
                continue
            elif line.startswith("Artifacts:"):
                idx += 1
                while idx < len(lines) and lines[idx].startswith("- "):
                    artifact_paths.append(lines[idx].removeprefix("- ").strip().strip("`") )
                    idx += 1
                continue
            idx += 1

        resolved_artifact = next(
            (resolved for path in artifact_paths if (resolved := resolve_doc_artifact(path)) is not None),
            None,
        )
        artifact_location = "; ".join(artifact_paths) if artifact_paths else current_bucket
        status = "verified" if resolved_artifact else ("missing_artifact" if artifact_paths else "doc-only")
        evidence_class = "checkpoint-verified" if status == "verified" else "doc-only"
        for metric_row in metric_rows_pending:
            method, row_teacher, row_student = method_to_models(
                metric_row["model"],
                teacher_model=teacher_model,
                student_model=student_model,
            )
            rows.append(
                InventoryRow(
                    source=f"docs/EXPERIMENT_LOG.md:{current_timestamp}",
                    evidence_class=evidence_class,
                    status=status,
                    dataset=compact_dataset_name(dataset_name),
                    split="test",
                    teacher_model=row_teacher,
                    student_model=row_student,
                    method=method,
                    training_budget=", ".join(training_budget_parts),
                    mrr=metric_row["MRR"],
                    recall_at_1=metric_row["Recall@1"],
                    recall_at_5=metric_row["Recall@5"],
                    recall_at_10=metric_row["Recall@10"],
                    ndcg_at_10="",
                    map_at_10="",
                    artifact_location=str(resolved_artifact or artifact_location),
                    notes=current_bucket,
                )
            )
    return rows


def collect_eval_runs() -> list[InventoryRow]:
    if not EVAL_RUN_INDEX.exists():
        return []
    rows: list[InventoryRow] = []
    with EVAL_RUN_INDEX.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            model_label = record.get("model_label", "")
            method = "raw" if record.get("model_source") == "hf" else model_label.split("__", 1)[0]
            rows.append(
                InventoryRow(
                    source="eval/runs/run_index.csv",
                    evidence_class="checkpoint-verified",
                    status="verified",
                    dataset=compact_dataset_name(record.get("dataset", "")),
                    split=record.get("split", ""),
                    teacher_model="-",
                    student_model=record.get("model_name_or_path", ""),
                    method=method,
                    training_budget="seed=" + (record.get("seed") or "42") if record.get("seed") else "",
                    mrr=safe_metric(record.get("MRR")),
                    recall_at_1=safe_metric(record.get("Recall@1")),
                    recall_at_5=safe_metric(record.get("Recall@5")),
                    recall_at_10=safe_metric(record.get("Recall@10")),
                    ndcg_at_10=safe_metric(record.get("nDCG@10")),
                    map_at_10=safe_metric(record.get("MAP@10")),
                    artifact_location=record.get("run_dir", ""),
                    notes=model_label,
                )
            )
    return rows


def collect_teacher_trials() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for metrics_path in sorted((SUITE_ROOT / "teacher_trials").glob("*/direct_baseline_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_metrics = payload.get("test", {})
        rows.append(
            InventoryRow(
                source=str(metrics_path.relative_to(SUITE_ROOT)),
                evidence_class="checkpoint-verified",
                status="verified",
                dataset="mbpp",
                split="test",
                teacher_model=payload.get("teacher_model", ""),
                student_model="-",
                method="raw_teacher_trial",
                training_budget="",
                mrr=safe_metric(test_metrics.get("MRR")),
                recall_at_1=safe_metric(test_metrics.get("Recall@1")),
                recall_at_5=safe_metric(test_metrics.get("Recall@5")),
                recall_at_10=safe_metric(test_metrics.get("Recall@10")),
                ndcg_at_10="",
                map_at_10="",
                artifact_location=str(metrics_path.resolve()),
                notes=metrics_path.parent.name,
            )
        )
    return rows


def _artifact_roots() -> list[Path]:
    roots = [SUITE_ROOT / "artifacts", PROJECT_ROOT / "artifacts"]
    return [root for root in roots if root.exists()]


def collect_local_metric_artifacts() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for root in _artifact_roots():
        for metrics_path in sorted(root.glob("**/metrics.json")):
            if metrics_path.parts[-2] == "eval":
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            history_path = metrics_path.with_name("history.json")
            run_dir = metrics_path.parent.parent
            config_path = run_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            method_name = metrics_path.parent.name
            model_name = payload.get("model_name", "")
            teacher_model = config.get("teacher_model", DEFAULT_TEACHER)
            student_model = config.get("student_model", model_name or DEFAULT_STUDENT)
            method, row_teacher, row_student = method_to_models(method_name, teacher_model, student_model)
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
            budget_parts = []
            for key in ("epochs", "batch_size", "eval_batch_size", "seed"):
                if key in config:
                    budget_parts.append(f"{key}={config[key]}")
            dataset_guess = compact_dataset_name(config.get("dataset_name", "mbpp" if "finetune_comparison" in str(metrics_path) else ""))
            rows.append(
                InventoryRow(
                    source=str(metrics_path.relative_to(PROJECT_ROOT)),
                    evidence_class="checkpoint-verified",
                    status="verified",
                    dataset=dataset_guess,
                    split="test",
                    teacher_model=row_teacher,
                    student_model=row_student,
                    method=method,
                    training_budget=", ".join(budget_parts),
                    mrr=safe_metric(payload.get("test", {}).get("MRR")),
                    recall_at_1=safe_metric(payload.get("test", {}).get("Recall@1")),
                    recall_at_5=safe_metric(payload.get("test", {}).get("Recall@5")),
                    recall_at_10=safe_metric(payload.get("test", {}).get("Recall@10")),
                    ndcg_at_10=safe_metric(payload.get("test", {}).get("nDCG@10")),
                    map_at_10=safe_metric(payload.get("test", {}).get("MAP@10")),
                    artifact_location=str(metrics_path.resolve()),
                    notes=f"history_rows={len(history)}",
                )
            )
    return rows


def collect_hf_checkpoint_rows(repo_id: str) -> list[CheckpointRow]:
    api = HfApi()
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        return [
            CheckpointRow(
                local_path="",
                hf_repo_id=repo_id,
                dataset="",
                teacher_model="",
                student_model="",
                method="",
                seed="",
                checkpoint_tag="",
                selected_for_report="no",
                status="error",
                notes=f"HF listing failed: {type(exc).__name__}: {exc}",
            )
        ]

    grouped: dict[str, list[str]] = {}
    for file_path in repo_files:
        if "/" not in file_path:
            continue
        prefix = file_path.split("/", 1)[0]
        grouped.setdefault(prefix, []).append(file_path)

    rows: list[CheckpointRow] = []
    for prefix, files in sorted(grouped.items()):
        rows.append(
            CheckpointRow(
                local_path="",
                hf_repo_id=f"{repo_id}/{prefix}",
                dataset="",
                teacher_model="",
                student_model="",
                method="",
                seed="",
                checkpoint_tag=prefix,
                selected_for_report="no",
                status="verified",
                notes=f"repo_files={len(files)}",
            )
        )
    return rows


def _looks_like_hf_checkpoint(path: Path) -> bool:
    return (path / "config.json").exists() and (path / "model.safetensors").exists()


def _looks_like_suite_checkpoint(path: Path) -> bool:
    return (path / "model" / "backbone").exists() and (path / "model" / "tokenizer").exists()


def collect_local_checkpoint_rows() -> list[CheckpointRow]:
    rows: list[CheckpointRow] = []

    for root in _artifact_roots():
        for model_dir in sorted(root.glob("**/model")):
            if not (model_dir / "backbone").exists() or not (model_dir / "tokenizer").exists():
                continue
            run_dir = model_dir.parent.parent
            config_path = run_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            method = model_dir.parent.name
            rows.append(
                CheckpointRow(
                    local_path=str(model_dir.parent.resolve()),
                    hf_repo_id="",
                    dataset=compact_dataset_name(config.get("dataset_name", "")),
                    teacher_model=config.get("teacher_model", ""),
                    student_model=config.get("student_model", ""),
                    method=method,
                    seed=str(config.get("seed", "")),
                    checkpoint_tag="best-val",
                    selected_for_report="yes" if method in {"embed_distill", "score_distill", "supervised_student"} else "no",
                    status="verified",
                    notes="suite_student_dir",
                )
            )

    kai_checkpoint_root = PROJECT_ROOT / "experiments" / "kai" / "results" / "mbpp_full_matrix" / "checkpoints"
    if kai_checkpoint_root.exists():
        for checkpoint_dir in sorted(kai_checkpoint_root.glob("**/*")):
            if not checkpoint_dir.is_dir() or not _looks_like_hf_checkpoint(checkpoint_dir):
                continue
            parent_name = checkpoint_dir.parent.name
            rows.append(
                CheckpointRow(
                    local_path=str(checkpoint_dir.resolve()),
                    hf_repo_id="",
                    dataset="mbpp",
                    teacher_model="",
                    student_model=checkpoint_dir.name.replace("__", "/"),
                    method=parent_name,
                    seed="",
                    checkpoint_tag=checkpoint_dir.name,
                    selected_for_report="yes" if parent_name in {"final_standard_mnr", "final_hardneg_triplet"} else "no",
                    status="verified",
                    notes="external_sentence_transformer_dir",
                )
            )
    return rows


def dedupe_inventory_rows(rows: list[InventoryRow]) -> list[InventoryRow]:
    deduped: dict[tuple[str, ...], InventoryRow] = {}
    priority = {"checkpoint-verified": 2, "doc-only": 1}
    for row in rows:
        key = (
            row.dataset,
            row.split,
            row.teacher_model,
            row.student_model,
            row.method,
            row.training_budget,
            row.mrr,
            row.recall_at_1,
            row.recall_at_5,
            row.recall_at_10,
        )
        existing = deduped.get(key)
        if existing is None or priority.get(row.evidence_class, 0) > priority.get(existing.evidence_class, 0):
            deduped[key] = row
    return sorted(
        deduped.values(),
        key=lambda row: (row.dataset, row.method, row.mrr, row.source),
        reverse=True,
    )


def compute_dataset_stats() -> list[dict[str, str]]:
    from eval.data_adapters import get_dataset_adapter

    rows: list[dict[str, str]] = []
    for dataset_name in ("mbpp", "codesearchnet"):
        corpus = get_dataset_adapter(dataset_name).load()
        for split_name in ("train", "validation", "test"):
            split_records = corpus.get_split(split_name)
            if not split_records:
                continue
            query_lengths = [len(record.query.split()) for record in split_records]
            code_lengths = [len(record.code.split()) for record in split_records]
            overlap_rates: list[float] = []
            jaccard_scores: list[float] = []
            for record in split_records:
                query_tokens = normalized_tokens(record.query)
                code_tokens = normalized_tokens(record.code)
                overlap = query_tokens & code_tokens
                union = query_tokens | code_tokens
                overlap_rates.append(len(overlap) / len(query_tokens) if query_tokens else 0.0)
                jaccard_scores.append(len(overlap) / len(union) if union else 0.0)
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "count": str(len(split_records)),
                    "avg_query_tokens": f"{sum(query_lengths) / len(query_lengths):.2f}",
                    "avg_code_tokens": f"{sum(code_lengths) / len(code_lengths):.2f}",
                    "median_query_tokens": f"{median_value(query_lengths):.2f}",
                    "median_code_tokens": f"{median_value(code_lengths):.2f}",
                    "avg_query_token_overlap_rate": f"{sum(overlap_rates) / len(overlap_rates):.4f}",
                    "median_query_token_overlap_rate": f"{median_value(overlap_rates):.4f}",
                    "avg_query_code_jaccard": f"{sum(jaccard_scores) / len(jaccard_scores):.4f}",
                    "median_query_code_jaccard": f"{median_value(jaccard_scores):.4f}",
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the KD investigation manifests used by the internal report.")
    parser.add_argument("--hf-repo-id", default=HF_REPO_DEFAULT)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--experiment-output", default=str(OUTPUT_EXPERIMENT_INVENTORY))
    parser.add_argument("--checkpoint-output", default=str(OUTPUT_CHECKPOINT_MANIFEST))
    parser.add_argument("--dataset-stats-output", default=str(OUTPUT_DATASET_STATS))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    inventory_rows = dedupe_inventory_rows(
        [
            *collect_doc_runs(),
            *collect_eval_runs(),
            *collect_teacher_trials(),
            *collect_local_metric_artifacts(),
        ]
    )
    checkpoint_rows = collect_local_checkpoint_rows()
    if not args.skip_hf:
        checkpoint_rows.extend(collect_hf_checkpoint_rows(args.hf_repo_id))

    experiment_output = Path(args.experiment_output)
    checkpoint_output = Path(args.checkpoint_output)
    dataset_output = Path(args.dataset_stats_output)
    experiment_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.parent.mkdir(parents=True, exist_ok=True)

    write_csv(experiment_output, [row.to_dict() for row in inventory_rows])
    write_csv(checkpoint_output, [row.to_dict() for row in checkpoint_rows])
    write_csv(dataset_output, compute_dataset_stats())

    print(f"Wrote experiment inventory to {experiment_output}")
    print(f"Wrote checkpoint manifest to {checkpoint_output}")
    print(f"Wrote dataset stats to {dataset_output}")
    print(f"inventory_rows={len(inventory_rows)} checkpoint_rows={len(checkpoint_rows)}")


if __name__ == "__main__":
    main()
