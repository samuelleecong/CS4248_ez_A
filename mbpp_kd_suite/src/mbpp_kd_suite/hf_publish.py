from __future__ import annotations

import argparse
import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "cs4248-nlp/cs4248-model-weights"
DEFAULT_MANIFEST_PATH = "docs/kd_checkpoint_manifest.csv"


@dataclass
class PublishRow:
    raw: dict[str, str]
    local_path: Path
    repo_id: str
    repo_subpath: str
    commit_message: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish selected checkpoints from the KD manifest to a Hugging Face model repo.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--only-publish-yes", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    return parser


def _slugify(value: str) -> str:
    cleaned = value.strip().lower()
    for src, dst in {
        "/": "-",
        " ": "-",
        "__": "-",
        ".": "-",
    }.items():
        cleaned = cleaned.replace(src, dst)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _default_repo_subpath(row: dict[str, str]) -> str:
    dataset = row["dataset"].strip().lower()
    split = "python" if dataset == "codesearchnet" else "default"
    teacher = row.get("teacher_model", "").strip() or "none"
    student = row.get("student_model", "").strip() or "none"
    method = row["method"].strip()
    seed = row.get("seed", "").strip() or "na"
    checkpoint_tag = row.get("checkpoint_tag", "").strip() or "best-val"
    return "/".join(
        [
            "checkpoints",
            dataset,
            split,
            f"teacher_{_slugify(teacher)}",
            f"student_{_slugify(student)}",
            method,
            f"seed{seed}",
            checkpoint_tag,
        ]
    )


def _default_commit_message(row: dict[str, str], repo_subpath: str) -> str:
    dataset = row["dataset"].strip().lower()
    method = row["method"].strip()
    student = row.get("student_model", "").strip() or "none"
    return f"Add {dataset} {method} checkpoint for {student} at {repo_subpath}"


def _load_manifest_rows(manifest_path: Path, repo_id: str, only_publish_yes: bool) -> tuple[list[str], list[dict[str, str]], list[PublishRow]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    required = ["local_path", "dataset", "teacher_model", "student_model", "method", "seed", "checkpoint_tag"]
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    for extra in ["publish_to_hf", "hf_repo_subpath", "hf_commit_message", "hf_published_revision", "hf_repo_id"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
            for row in rows:
                row.setdefault(extra, "")

    publish_rows: list[PublishRow] = []
    for row in rows:
        if only_publish_yes and row.get("publish_to_hf", "").strip().lower() != "yes":
            continue
        local_path = Path(row["local_path"]).expanduser().resolve()
        if not local_path.exists() or not local_path.is_dir():
            raise FileNotFoundError(f"Local checkpoint directory does not exist: {local_path}")
        effective_repo_id = row.get("hf_repo_id", "").strip() or repo_id
        repo_subpath = row.get("hf_repo_subpath", "").strip() or _default_repo_subpath(row)
        commit_message = row.get("hf_commit_message", "").strip() or _default_commit_message(row, repo_subpath)
        publish_rows.append(
            PublishRow(
                raw=row,
                local_path=local_path,
                repo_id=effective_repo_id,
                repo_subpath=repo_subpath,
                commit_message=commit_message,
            )
        )
    return fieldnames, rows, publish_rows


def _existing_repo_paths(api: HfApi, repo_id: str, repo_type: str) -> set[str]:
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    return {s.rfilename for s in info.siblings}


def _render_repo_readme(rows: list[PublishRow], repo_id: str) -> str:
    lines = [
        "---",
        "tags:",
        "- sentence-transformers",
        "- text-retrieval",
        "- code-search",
        "- knowledge-distillation",
        "library_name: transformers",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Shared model-checkpoint repository for CS4248 code retrieval experiments.",
        "",
        "## Layout",
        "",
        "Checkpoints are grouped by dataset, teacher, student, method, seed, and selection tag.",
        "",
        "```",
        "checkpoints/<dataset>/<subset>/teacher_<teacher>/student_<student>/<method>/seed<seed>/<checkpoint_tag>/",
        "```",
        "",
        "## Published checkpoints",
        "",
        "| Dataset | Teacher | Student | Method | Seed | Repo path |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        row = item.raw
        lines.append(
            f"| {row['dataset']} | {row.get('teacher_model', '') or '-'} | {row.get('student_model', '') or '-'} | {row['method']} | {row.get('seed', '') or '-'} | `{item.repo_subpath}` |"
        )
    lines.append("")
    lines.append("See `manifests/published_checkpoints.csv` for machine-readable metadata.")
    return "\n".join(lines) + "\n"


def _render_published_markdown(rows: list[PublishRow]) -> str:
    lines = [
        "# Published Checkpoints",
        "",
        "| Dataset | Teacher | Student | Method | Seed | Checkpoint tag | Repo subpath |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        row = item.raw
        lines.append(
            f"| {row['dataset']} | {row.get('teacher_model', '') or '-'} | {row.get('student_model', '') or '-'} | {row['method']} | {row.get('seed', '') or '-'} | {row.get('checkpoint_tag', '') or '-'} | `{item.repo_subpath}` |"
        )
    return "\n".join(lines) + "\n"


def _write_local_manifest(manifest_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_publication_report(rows: list[PublishRow], revisions: dict[str, str], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "hf_publication_report.csv"
    md_path = output_root / "hf_publication_report.md"
    fieldnames = [
        "repo_id",
        "repo_subpath",
        "dataset",
        "teacher_model",
        "student_model",
        "method",
        "seed",
        "checkpoint_tag",
        "revision",
        "local_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            row = item.raw
            writer.writerow(
                {
                    "repo_id": item.repo_id,
                    "repo_subpath": item.repo_subpath,
                    "dataset": row["dataset"],
                    "teacher_model": row.get("teacher_model", ""),
                    "student_model": row.get("student_model", ""),
                    "method": row["method"],
                    "seed": row.get("seed", ""),
                    "checkpoint_tag": row.get("checkpoint_tag", ""),
                    "revision": revisions.get(item.repo_subpath, ""),
                    "local_path": str(item.local_path),
                }
            )
    lines = [
        "# HF Publication Report",
        "",
        "| Repo path | Dataset | Teacher | Student | Method | Revision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        row = item.raw
        lines.append(
            f"| `{item.repo_subpath}` | {row['dataset']} | {row.get('teacher_model', '') or '-'} | {row.get('student_model', '') or '-'} | {row['method']} | {revisions.get(item.repo_subpath, '') or '-'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    fieldnames, rows, publish_rows = _load_manifest_rows(
        manifest_path=manifest_path,
        repo_id=args.repo_id,
        only_publish_yes=args.only_publish_yes,
    )
    if not publish_rows:
        print("No manifest rows selected for publication.")
        return

    revisions: dict[str, str] = {}
    if args.dry_run:
        for item in publish_rows:
            print(f"DRY RUN: {item.local_path} -> hf://{item.repo_id}/{item.repo_subpath} :: {item.commit_message}")
        return

    api = HfApi()
    repo_id = publish_rows[0].repo_id
    api.create_repo(repo_id=repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    existing_paths = _existing_repo_paths(api, repo_id=repo_id, repo_type=args.repo_type)

    for item in publish_rows:
        path_prefix = f"{item.repo_subpath.rstrip('/')}/"
        if args.skip_existing and any(path == item.repo_subpath or path.startswith(path_prefix) for path in existing_paths):
            print(f"SKIP existing: hf://{item.repo_id}/{item.repo_subpath}")
            continue
        commit_info = api.upload_folder(
            repo_id=item.repo_id,
            repo_type=args.repo_type,
            folder_path=str(item.local_path),
            path_in_repo=item.repo_subpath,
            commit_message=item.commit_message,
        )
        revisions[item.repo_subpath] = getattr(commit_info, "oid", "") or ""
        item.raw["hf_repo_id"] = item.repo_id
        item.raw["hf_repo_subpath"] = item.repo_subpath
        item.raw["hf_commit_message"] = item.commit_message
        item.raw["hf_published_revision"] = revisions[item.repo_subpath]
        print(f"Uploaded {item.local_path} -> hf://{item.repo_id}/{item.repo_subpath}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        manifests_dir = tmp_root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        (tmp_root / "README.md").write_text(_render_repo_readme(publish_rows, repo_id), encoding="utf-8")
        with (manifests_dir / "published_checkpoints.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "repo_id",
                    "repo_subpath",
                    "dataset",
                    "teacher_model",
                    "student_model",
                    "method",
                    "seed",
                    "checkpoint_tag",
                    "revision",
                ],
            )
            writer.writeheader()
            for item in publish_rows:
                row = item.raw
                writer.writerow(
                    {
                        "repo_id": item.repo_id,
                        "repo_subpath": item.repo_subpath,
                        "dataset": row["dataset"],
                        "teacher_model": row.get("teacher_model", ""),
                        "student_model": row.get("student_model", ""),
                        "method": row["method"],
                        "seed": row.get("seed", ""),
                        "checkpoint_tag": row.get("checkpoint_tag", ""),
                        "revision": revisions.get(item.repo_subpath, row.get("hf_published_revision", "")),
                    }
                )
        (manifests_dir / "published_checkpoints.md").write_text(_render_published_markdown(publish_rows), encoding="utf-8")
        api.upload_folder(
            repo_id=repo_id,
            repo_type=args.repo_type,
            folder_path=str(tmp_root),
            path_in_repo="",
            commit_message="Update checkpoint manifests and repository index",
        )

    if args.write_manifest:
        _write_local_manifest(manifest_path, fieldnames, rows)
    _write_publication_report(publish_rows, revisions, manifest_path.parent)


if __name__ == "__main__":
    main()
