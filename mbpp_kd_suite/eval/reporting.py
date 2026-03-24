from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .types import EvalConfig

EVAL_ROOT = Path(__file__).resolve().parent


def write_eval_report(result: dict[str, Any], cfg: EvalConfig) -> Path:
    output_root = resolve_eval_output_root(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    run_dir = _build_run_dir(output_root=output_root, result=result, cfg=cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    result_payload = dict(result)
    result_payload["run"] = {
        "timestamp": run_timestamp,
        "run_dir": str(run_dir),
        "relative_run_dir": str(run_dir.relative_to(output_root)),
    }

    _write_json(run_dir / "metrics.json", result_payload)
    _write_json(run_dir / "config.json", asdict(cfg))
    _write_metrics_csv(run_dir / "metrics.csv", result_payload)
    _write_profiling_csv(run_dir / "profiling.csv", result_payload["profiling"]["stages"])
    _write_summary_md(run_dir / "summary.md", result_payload, cfg)
    _plot_metrics_overview(run_dir / "metrics_overview.png", result_payload["metrics"], cfg.ks)
    _plot_runtime_memory(run_dir / "runtime_memory.png", result_payload["profiling"]["stages"])
    _refresh_run_indices(output_root)
    return run_dir


def resolve_eval_output_root(output_dir: str) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    if output_dir.startswith("./") or output_dir.startswith("../"):
        return output_path
    if output_path.parts and output_path.parts[0] == "eval":
        return output_path
    return EVAL_ROOT / output_path


def _build_run_dir(output_root: Path, result: dict[str, Any], cfg: EvalConfig) -> Path:
    dataset_dir = output_root / _slugify(cfg.dataset_name) / cfg.split
    dataset_dir.mkdir(parents=True, exist_ok=True)

    model_part = _slugify(_model_label(result, cfg))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = dataset_dir / f"{timestamp}_{model_part}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = dataset_dir / f"{base.name}_{suffix:02d}"
        suffix += 1
    return candidate


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "run"


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _metrics_row(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "run_timestamp": result["run"]["timestamp"],
        "run_dir": result["run"]["run_dir"],
        "relative_run_dir": result["run"]["relative_run_dir"],
        "dataset": result["dataset"]["name"],
        "split": result["split"],
        "model_source": result["model"]["source"],
        "model_label": _model_label(result),
        "model_name_or_path": result["model"].get("model_name_or_path"),
        "checkpoint_format": result["model"].get("checkpoint_format"),
        "queries": result["counts"]["queries"],
        "codes": result["counts"]["codes"],
        "total_duration_sec": result["profiling"]["total_duration_sec"],
        "peak_memory_bytes": result["profiling"]["peak_memory_bytes"],
    }
    row.update(result["metrics"])
    return row


def _write_metrics_csv(path: Path, result: dict[str, Any]) -> None:
    row = _metrics_row(result)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_profiling_csv(path: Path, stages: list[dict[str, float | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "duration_sec", "peak_memory_bytes"])
        writer.writeheader()
        for row in stages:
            writer.writerow(row)


def _write_summary_md(path: Path, result: dict[str, Any], cfg: EvalConfig) -> None:
    lines = [
        "# Eval Run Summary",
        "",
        f"- Timestamp: `{result['run']['timestamp']}`",
        f"- Dataset: `{result['dataset']['name']}`",
        f"- Split: `{result['split']}`",
        f"- Model source: `{result['model']['source']}`",
        f"- Model label: `{_model_label(result, cfg)}`",
        f"- Model path/name: `{result['model'].get('model_name_or_path', cfg.model_name_or_path)}`",
        f"- Query count: `{result['counts']['queries']}`",
        f"- Code count: `{result['counts']['codes']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in result["metrics"].items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Profiling",
        "",
        "| Stage | Seconds | Peak Memory (MB) |",
        "| --- | ---: | ---: |",
    ])
    for stage in result["profiling"]["stages"]:
        memory_mb = float(stage["peak_memory_bytes"]) / (1024.0 * 1024.0)
        lines.append(
            f"| {stage['stage']} | {float(stage['duration_sec']):.6f} | {memory_mb:.2f} |"
        )

    lines.extend([
        "",
        "## Files",
        "",
        "- `metrics.csv`: flat machine-readable summary",
        "- `profiling.csv`: per-stage timing and memory",
        "- `metrics.json`: nested full result payload",
        "- `config.json`: resolved evaluation config",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metrics_overview(path: Path, metrics: dict[str, float], ks: tuple[int, ...]) -> None:
    focus_keys = ["MRR", *[f"Recall@{k}" for k in ks]]
    max_k = max(ks)
    for metric_name in (f"nDCG@{max_k}", f"MAP@{max_k}"):
        if metric_name in metrics:
            focus_keys.append(metric_name)
    values = [float(metrics[key]) for key in focus_keys if key in metrics]
    labels = [key for key in focus_keys if key in metrics]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics Overview")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_runtime_memory(path: Path, stages: list[dict[str, float | str]]) -> None:
    labels = [str(row["stage"]) for row in stages]
    runtimes = [float(row["duration_sec"]) for row in stages]
    memory_mb = [float(row["peak_memory_bytes"]) / (1024.0 * 1024.0) for row in stages]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(labels, runtimes, color="#59A14F")
    axes[0].set_title("Runtime by Stage")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(labels, memory_mb, color="#E15759")
    axes[1].set_title("Peak Memory by Stage")
    axes[1].set_ylabel("MB")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _refresh_run_indices(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_root.glob("**/metrics.csv")):
        if metrics_path.name != "metrics.csv":
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
        if row is None:
            continue
        row["relative_run_dir"] = str(metrics_path.parent.relative_to(output_root))
        rows.append(row)

    rows.sort(key=lambda row: row.get("run_timestamp", ""), reverse=True)
    _write_run_index_csv(output_root / "run_index.csv", rows)
    _write_run_index_md(output_root / "run_index.md", rows)


def _write_run_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_index_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Eval Run Index",
        "",
        "| Timestamp | Dataset | Split | Model | MRR | Recall@1 | Recall@5 | Recall@10 | Total Seconds | Run Dir |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_timestamp} | {dataset} | {split} | {model} | {mrr} | {r1} | {r5} | {r10} | {sec} | `{run_dir}` |".format(
                run_timestamp=row.get("run_timestamp", ""),
                dataset=row.get("dataset", ""),
                split=row.get("split", ""),
                model=row.get("model_label") or _model_label_from_string(row.get("model_name_or_path", "")),
                mrr=_fmt_float(row.get("MRR")),
                r1=_fmt_float(row.get("Recall@1")),
                r5=_fmt_float(row.get("Recall@5")),
                r10=_fmt_float(row.get("Recall@10")),
                sec=_fmt_float(row.get("total_duration_sec")),
                run_dir=row.get("relative_run_dir", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _model_label(result: dict[str, Any], cfg: EvalConfig | None = None) -> str:
    raw = str(result["model"].get("model_name_or_path") or (cfg.model_name_or_path if cfg else "model"))
    return _model_label_from_string(raw, source=str(result["model"].get("source", "")))


def _model_label_from_string(raw: str, source: str = "") -> str:
    value = raw.strip()
    if not value:
        return "model"
    if source == "local" or value.startswith(("/", "./", "../", "~")):
        path = Path(value).expanduser()
        parts = [part for part in path.parts if part not in (path.anchor, "")]
        if len(parts) >= 2:
            return f"{parts[-2]}__{parts[-1]}"
        if parts:
            return parts[-1]
    if "/" in value:
        return value.strip("/").replace("/", "__")
    return value
