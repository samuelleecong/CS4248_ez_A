"""Upload trained models from a two-phase KD run directory to the HuggingFace Hub.

Usage:
    mbpp-kd-upload --run-dir artifacts/two_phase_kd/20260325_014013 --hf-user myusername
    mbpp-kd-upload --run-dir artifacts/two_phase_kd/20260325_014013 --hf-org cs4248-team --prefix cs4248
    mbpp-kd-upload --run-dir artifacts/two_phase_kd/20260325_014013 --hf-user myusername --dry-run

Naming convention:
    {prefix}-{role}-{dataset_slug}-{timestamp}

    Examples:
        cs4248-phase1-student-taco-20260325-014013
        cs4248-phase2-score-distill-taco-20260325-014013
        cs4248-phase1-teacher-code-search-net-20260325-014013
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ── Role name helpers ──────────────────────────────────────────────────────────

_ROLE_MAP: dict[str, str] = {
    "ft_student_phase1": "phase1-student",
    "ft_teacher_phase1": "phase1-teacher",
}


def _dir_to_role(dir_name: str) -> str:
    """Convert a model subdirectory name to a human-readable role slug."""
    if dir_name in _ROLE_MAP:
        return _ROLE_MAP[dir_name]
    # phase2_score_distill → phase2-score-distill
    slug = dir_name.replace("_", "-")
    return slug


def _dataset_slug(dataset_name: str) -> str:
    """Convert a HuggingFace dataset name to a URL-safe slug.

    'BAAI/TACO'           → 'taco'
    'code_search_net'     → 'code-search-net'
    'google-research-datasets/mbpp' → 'mbpp'
    """
    name = dataset_name.split("/")[-1]
    name = name.lower().replace("_", "-")
    return name


def _timestamp_slug(run_dir_name: str) -> str:
    """Normalise the run directory name to a URL-safe timestamp.

    '20260325_014013' → '20260325-014013'
    'full_pretrained_phase_1' → 'full-pretrained-phase-1'
    """
    return run_dir_name.replace("_", "-")


def _make_repo_name(prefix: str, role: str, dataset: str, timestamp: str) -> str:
    parts = [prefix, role, dataset, timestamp]
    repo = "-".join(p for p in parts if p)
    # Sanitise: only allow alphanumerics and hyphens
    repo = re.sub(r"[^a-z0-9-]", "-", repo.lower())
    repo = re.sub(r"-{2,}", "-", repo).strip("-")
    return repo


# ── Discovery ─────────────────────────────────────────────────────────────────

def _find_model_dirs(run_dir: Path) -> list[tuple[str, Path]]:
    """Walk run_dir and return (role, backbone_path) for every saved model.

    A model is recognised when a directory named 'backbone' contains
    'model.safetensors' (HuggingFace format saved by save_pretrained).
    """
    found: list[tuple[str, Path]] = []
    for backbone_dir in sorted(run_dir.rglob("backbone")):
        if not (backbone_dir / "model.safetensors").exists():
            continue
        # Parent chain:  run_dir / {phase1|phase2} / {role} / model / backbone
        role_dir = backbone_dir.parent.parent  # the {role} directory
        role = _dir_to_role(role_dir.name)
        found.append((role, backbone_dir))
    return found


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


# ── Card generation ────────────────────────────────────────────────────────────

def _make_model_card(
    role: str,
    repo_id: str,
    run_config: dict,
    run_dir_name: str,
) -> str:
    """Generate a minimal model card (README.md) for the HuggingFace repo."""
    dataset = run_config.get("dataset_name", "unknown")
    teacher = run_config.get("teacher_model", "unknown")
    student = run_config.get("student_model", "unknown")
    phase = "Phase 1" if "phase1" in role else "Phase 2"
    method = role.replace("phase2-", "").replace("phase1-", "")

    lines = [
        f"# {repo_id}",
        "",
        "Code-search embedding model trained with the CS4248 two-phase KD pipeline.",
        "",
        "## Model details",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Role | `{role}` |",
        f"| Phase | {phase} |",
        f"| Method | `{method}` |",
        f"| Dataset | `{dataset}` |",
        f"| Teacher | `{teacher}` |",
        f"| Student base | `{student}` |",
        f"| Run timestamp | `{run_dir_name}` |",
        "",
        "## Usage",
        "",
        "```python",
        "from transformers import AutoModel, AutoTokenizer",
        "",
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")',
        f'model = AutoModel.from_pretrained("{repo_id}")',
        "```",
        "",
        "Mean-pool the last hidden state to get a fixed-size embedding:",
        "",
        "```python",
        "import torch",
        "",
        "def mean_pool(model_output, attention_mask):",
        "    token_embeddings = model_output.last_hidden_state",
        "    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()",
        "    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)",
        "",
        'inputs = tokenizer("your query here", return_tensors="pt", truncation=True, max_length=160)',
        "with torch.no_grad():",
        "    outputs = model(**inputs)",
        "embedding = mean_pool(outputs, inputs['attention_mask'])",
        "```",
    ]
    return "\n".join(lines)


# ── Upload ─────────────────────────────────────────────────────────────────────

def _upload_model(
    backbone_dir: Path,
    repo_id: str,
    role: str,
    run_config: dict,
    run_dir_name: str,
    private: bool,
    dry_run: bool,
) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    # Collect files to upload: backbone/* + tokenizer/* (sibling of backbone)
    model_root = backbone_dir.parent  # .../model/
    files_to_upload: list[tuple[Path, str]] = []
    for p in sorted(model_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(model_root)
            files_to_upload.append((p, str(rel)))

    if not files_to_upload:
        print(f"  WARNING: no files found under {model_root}, skipping.")
        return

    card_content = _make_model_card(role, repo_id, run_config, run_dir_name)

    if dry_run:
        print(f"  [dry-run] Would create repo: {repo_id} (private={private})")
        for local, remote in files_to_upload:
            print(f"  [dry-run]   {remote}  ← {local.name} ({local.stat().st_size // 1024} KB)")
        print("  [dry-run]   README.md  ← (generated model card)")
        return

    api = HfApi()
    print(f"  Creating repo '{repo_id}' …")
    create_repo(repo_id=repo_id, exist_ok=True, private=private)

    for local_path, path_in_repo in files_to_upload:
        print(f"  Uploading {path_in_repo} …")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
        )

    print("  Uploading README.md …")
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print(f"  Done: https://huggingface.co/{repo_id}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload two-phase KD models to the HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    dest = parser.add_mutually_exclusive_group(required=True)
    dest.add_argument(
        "--hf-user",
        metavar="USERNAME",
        help="Your HuggingFace username (models uploaded as username/repo-name)",
    )
    dest.add_argument(
        "--hf-org",
        metavar="ORG",
        help="HuggingFace organisation name (models uploaded as org/repo-name)",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        metavar="PATH",
        help=(
            "Path to the two-phase KD run directory, e.g. "
            "artifacts/two_phase_kd/20260325_014013"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="cs4248",
        metavar="PREFIX",
        help="Repo name prefix (default: cs4248)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private HuggingFace repositories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading anything",
    )
    parser.add_argument(
        "--roles",
        metavar="ROLE1,ROLE2",
        default="",
        help=(
            "Comma-separated list of roles to upload (default: all found). "
            "Examples: phase1-student,phase2-score-distill"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        # Try relative to cwd / artifacts
        from .constants import ARTIFACT_ROOT
        candidate = ARTIFACT_ROOT / args.run_dir
        if candidate.exists():
            run_dir = candidate
        else:
            print(f"ERROR: run directory not found: {args.run_dir}")
            sys.exit(1)

    run_dir = run_dir.resolve()
    run_dir_name = run_dir.name
    namespace = args.hf_user or args.hf_org

    run_config = _load_run_config(run_dir)
    if not run_config:
        print(f"WARNING: no config.json found in {run_dir}; dataset slug will be 'unknown'.")

    dataset = _dataset_slug(run_config.get("dataset_name", "unknown"))
    timestamp = _timestamp_slug(run_dir_name)

    model_dirs = _find_model_dirs(run_dir)
    if not model_dirs:
        print(f"No saved models found in {run_dir}")
        print("  (Models are only saved when --save-models is passed to mbpp-kd-two-phase)")
        sys.exit(1)

    # Filter by role if requested
    filter_roles: set[str] = set()
    if args.roles.strip():
        filter_roles = {r.strip() for r in args.roles.split(",") if r.strip()}

    selected = [(role, path) for role, path in model_dirs if not filter_roles or role in filter_roles]
    if not selected:
        print(f"No models matched the requested roles: {filter_roles}")
        print(f"Available roles: {[r for r, _ in model_dirs]}")
        sys.exit(1)

    print(f"Run directory : {run_dir}")
    print(f"Dataset slug  : {dataset}")
    print(f"Timestamp slug: {timestamp}")
    print(f"Namespace     : {namespace}")
    print(f"Models found  : {len(selected)}")
    print()

    for role, backbone_dir in selected:
        repo_name = _make_repo_name(args.prefix, role, dataset, timestamp)
        repo_id = f"{namespace}/{repo_name}"
        print(f"[{role}] → {repo_id}")
        _upload_model(
            backbone_dir=backbone_dir,
            repo_id=repo_id,
            role=role,
            run_config=run_config,
            run_dir_name=run_dir_name,
            private=args.private,
            dry_run=args.dry_run,
        )
        print()

    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to upload.")
    else:
        print("All uploads complete.")


if __name__ == "__main__":
    main()
