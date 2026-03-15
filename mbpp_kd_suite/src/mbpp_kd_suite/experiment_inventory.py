from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
SKIP_ROOT_DIRS = {
    ".git",
    ".hf_cache",
    ".venv",
    "docs",
    "papers",
    "src",
    "__pycache__",
}


def discover_run_dirs(root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        rel_parts = current_path.relative_to(root).parts
        if rel_parts and rel_parts[0] in SKIP_ROOT_DIRS:
            dirs[:] = []
            continue

        dirs[:] = [
            name
            for name in dirs
            if name not in {"__pycache__", ".git", ".hf_cache", ".venv"}
        ]

        if RUN_DIR_RE.fullmatch(current_path.name) and "config.json" in files:
            run_dirs.append(current_path)
            dirs[:] = []
    return sorted(run_dirs, reverse=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_model_name(value: str | None) -> str:
    if not value:
        return "-"
    if "models--" in value and "/snapshots/" in value:
        fragment = value.split("models--", 1)[1].split("/snapshots/", 1)[0]
        return fragment.replace("--", "/")
    return value


def compact_dataset_name(value: str | None) -> str:
    if not value:
        return "-"
    return value.rsplit("/", 1)[-1]


def methods_label(methods: list[str]) -> str:
    if not methods:
        return "-"
    if len(methods) <= 3:
        return ",".join(methods)
    return f"{methods[0]},{methods[1]},{methods[2]} +{len(methods) - 3}"


def collect_run_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(root):
        config_path = run_dir / "config.json"
        config = load_json(config_path)
        group_path = run_dir.parent.relative_to(root)
        rows.append(
            {
                "timestamp": run_dir.name,
                "group": str(group_path),
                "dataset": compact_dataset_name(config.get("dataset_name")),
                "methods": list(config.get("methods", [])),
                "teacher_model": compact_model_name(config.get("teacher_model")),
                "student_model": compact_model_name(config.get("student_model")),
                "save_models": bool(config.get("save_models", False)),
                "path": str(run_dir.relative_to(root)),
                "resolved_output_dir": config.get("resolved_output_dir", str(group_path)),
            }
        )
    return sorted(rows, key=lambda row: (row["timestamp"], row["group"]), reverse=True)


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No experiment runs found.")
        return

    headers = ("timestamp", "group", "dataset", "methods", "path")
    widths = {
        "timestamp": 15,
        "group": max(len("group"), min(36, max(len(row["group"]) for row in rows))),
        "dataset": max(len("dataset"), min(18, max(len(row["dataset"]) for row in rows))),
        "methods": max(len("methods"), min(28, max(len(methods_label(row["methods"])) for row in rows))),
    }

    print(
        f"{headers[0]:<{widths['timestamp']}} "
        f"{headers[1]:<{widths['group']}} "
        f"{headers[2]:<{widths['dataset']}} "
        f"{headers[3]:<{widths['methods']}} path"
    )
    print(
        f"{'-' * widths['timestamp']} "
        f"{'-' * widths['group']} "
        f"{'-' * widths['dataset']} "
        f"{'-' * widths['methods']} ----"
    )

    for row in rows:
        print(
            f"{row['timestamp']:<{widths['timestamp']}} "
            f"{row['group']:<{widths['group']}} "
            f"{row['dataset']:<{widths['dataset']}} "
            f"{methods_label(row['methods']):<{widths['methods']}} "
            f"{row['path']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List MBPP KD Suite experiment runs and where they are saved."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Suite root to scan. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the run inventory as JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = collect_run_records(root)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print_table(rows)


if __name__ == "__main__":
    main()
