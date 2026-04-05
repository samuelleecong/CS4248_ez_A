from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.metrics import paired_ranks
from mbpp_kd_suite.modeling import StudentQueryEncoder, encode_student_texts, encode_texts_backbone
from mbpp_kd_suite.runtime import pick_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submission" / "experiments" / "analysis_outputs" / "person_e"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "submission" / "experiments" / "hf_cache"
DEFAULT_METHOD_REPOS: dict[str, str] = {
    "score_distill": "cs4248-nlp/paper-s1-score-dw100-tinybert-general-4l-312d-taco-hf-20260402-015143",
    "embed_distill": "cs4248-nlp/paper-s1-embed-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
    "hard_neg_pair": "cs4248-nlp/paper-s1-hnp-dw100-pw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
    "bimga": "cs4248-nlp/paper-s1-bimga-dw50-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
}
DEFAULT_TEACHER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATASET_NAME = "BEE-spoke-data/TACO-hf"


@dataclass(frozen=True)
class LoadedStudent:
    name: str
    repo_id: str
    snapshot_dir: Path
    model: StudentQueryEncoder
    tokenizer: AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Person E analyses: error overlap matrix and margin-stratified MRR.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--teacher-repo", default=DEFAULT_TEACHER_REPO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for Hugging Face snapshots and dataset cache.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-query-length", type=int, default=160)
    parser.add_argument("--max-code-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return pick_device()
    return torch.device(requested)


def ensure_snapshot(repo_id: str, cache_dir: Path) -> Path:
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
    )
    return Path(path)


def load_student(repo_id: str, name: str, cache_dir: Path, device: torch.device) -> LoadedStudent:
    snapshot_dir = ensure_snapshot(repo_id, cache_dir)
    projection_path = snapshot_dir / "projection.pt"
    target_hidden_size = None
    projection_state = None
    if projection_path.exists():
        projection_state = torch.load(projection_path, map_location="cpu")
        weight = projection_state.get("weight")
        if weight is None:
            raise ValueError(f"{projection_path} does not contain a projection weight")
        target_hidden_size = int(weight.shape[0])

    model = StudentQueryEncoder(str(snapshot_dir), target_hidden_size=target_hidden_size).to(device)
    if projection_state is not None:
        model.proj.load_state_dict(projection_state)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
    return LoadedStudent(
        name=name,
        repo_id=repo_id,
        snapshot_dir=snapshot_dir,
        model=model,
        tokenizer=tokenizer,
    )


def load_teacher(repo_id: str, cache_dir: Path, device: torch.device) -> tuple[Path, AutoModel, AutoTokenizer]:
    snapshot_dir = ensure_snapshot(repo_id, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
    model = AutoModel.from_pretrained(str(snapshot_dir)).to(device)
    model.eval()
    return snapshot_dir, model, tokenizer


def encode_student_split(
    student: LoadedStudent,
    queries: list[str],
    codes: list[str],
    max_query_length: int,
    max_code_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_embs = encode_student_texts(
        student_model=student.model,
        tokenizer=student.tokenizer,
        texts=queries,
        text_role="query",
        max_length=max_query_length,
        batch_size=batch_size,
        device=device,
        desc=f"{student.name}_queries",
    )
    code_embs = encode_student_texts(
        student_model=student.model,
        tokenizer=student.tokenizer,
        texts=codes,
        text_role="document",
        max_length=max_code_length,
        batch_size=batch_size,
        device=device,
        desc=f"{student.name}_docs",
    )
    return query_embs, code_embs


def encode_teacher_split(
    teacher_model: AutoModel,
    teacher_tokenizer: AutoTokenizer,
    queries: list[str],
    codes: list[str],
    max_query_length: int,
    max_code_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoding_spec = getattr(teacher_model, "_codex_encoding_spec", None)
    if encoding_spec is None:
        from mbpp_kd_suite.modeling import infer_model_encoding_spec

        encoding_spec = infer_model_encoding_spec(
            getattr(teacher_model.config, "_name_or_path", None),
            getattr(teacher_tokenizer, "name_or_path", None),
        )
        teacher_model._codex_encoding_spec = encoding_spec  # type: ignore[attr-defined]

    query_embs = encode_texts_backbone(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        texts=queries,
        text_role="query",
        encoding_spec=encoding_spec,
        max_length=max_query_length,
        batch_size=batch_size,
        device=device,
        desc="teacher_queries",
    )
    code_embs = encode_texts_backbone(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        texts=codes,
        text_role="document",
        encoding_spec=encoding_spec,
        max_length=max_code_length,
        batch_size=batch_size,
        device=device,
        desc="teacher_docs",
    )
    return query_embs, code_embs


def build_margin_bins(margins: np.ndarray) -> np.ndarray:
    order = np.argsort(margins)
    bins = np.empty(len(margins), dtype=object)
    splits = np.array_split(order, 3)
    labels = ["low", "medium", "high"]
    for label, indices in zip(labels, splits, strict=True):
        bins[indices] = label
    return bins


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard overlap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_margin_mrr(summary_df: pd.DataFrame, output_path: Path) -> None:
    methods = list(summary_df["method"].unique())
    bins = ["low", "medium", "high"]
    x = np.arange(len(bins))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for idx, method in enumerate(methods):
        subset = summary_df[summary_df["method"] == method].set_index("margin_bin").reindex(bins)
        ax.bar(
            x + (idx - (len(methods) - 1) / 2) * width,
            subset["mrr"].values,
            width=width,
            label=method,
        )

    ax.set_xticks(x, labels=[label.title() for label in bins])
    ax.set_ylabel("MRR")
    ax.set_xlabel("Teacher margin tercile")
    ax.set_title("Margin-Stratified MRR")
    ax.legend(frameon=False)
    ax.set_ylim(0.0, max(0.35, float(summary_df["mrr"].max()) * 1.15))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rank_cdf(per_query_df: pd.DataFrame, output_path: Path, max_rank: int = 50) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    methods = list(per_query_df["method"].unique())
    ks = np.arange(1, max_rank + 1)
    for method in methods:
        ranks = per_query_df.loc[per_query_df["method"] == method, "rank"].to_numpy()
        cdf = np.array([(ranks <= k).mean() for k in ks])
        ax.plot(ks, cdf, label=method, linewidth=2)

    for guide in (1, 5, 10):
        ax.axvline(guide, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Rank k")
    ax.set_ylabel("P(correct rank <= k)")
    ax.set_title("Rank Distribution CDF")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(1, max_rank)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_retrieval_dataset(
        dataset_name=args.dataset_name,
        taco_val_size=1000,
        seed=args.seed,
    )
    splits = dataset_dict_to_splits(dataset)
    queries = splits.test.queries
    codes = splits.test.codes

    teacher_snapshot, teacher_model, teacher_tokenizer = load_teacher(
        repo_id=args.teacher_repo,
        cache_dir=args.cache_dir,
        device=device,
    )
    teacher_query_embs, teacher_doc_embs = encode_teacher_split(
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        queries=queries,
        codes=codes,
        max_query_length=args.max_query_length,
        max_code_length=args.max_code_length,
        batch_size=args.batch_size,
        device=device,
    )
    teacher_scores = (teacher_query_embs @ teacher_doc_embs.T).numpy()
    diagonal = np.diag(teacher_scores)
    masked = teacher_scores.copy()
    np.fill_diagonal(masked, -np.inf)
    hardest_negative = masked.max(axis=1)
    teacher_margin = diagonal - hardest_negative
    margin_bin = build_margin_bins(teacher_margin)

    base_df = pd.DataFrame(
        {
            "query_idx": np.arange(len(queries)),
            "query": queries,
            "code": codes,
            "teacher_positive_score": diagonal,
            "teacher_hardest_negative_score": hardest_negative,
            "teacher_margin": teacher_margin,
            "margin_bin": margin_bin,
        }
    )

    per_query_frames: list[pd.DataFrame] = []
    method_results: dict[str, dict[str, object]] = {}

    for method_name, repo_id in DEFAULT_METHOD_REPOS.items():
        student = load_student(
            repo_id=repo_id,
            name=method_name,
            cache_dir=args.cache_dir,
            device=device,
        )
        query_embs, doc_embs = encode_student_split(
            student=student,
            queries=queries,
            codes=codes,
            max_query_length=args.max_query_length,
            max_code_length=args.max_code_length,
            batch_size=args.batch_size,
            device=device,
        )
        scores = (query_embs @ doc_embs.T).numpy()
        ranks = paired_ranks(scores)
        reciprocal_rank = 1.0 / ranks.astype(np.float64)
        top1_hit = ranks == 1

        method_df = base_df.copy()
        method_df["method"] = method_name
        method_df["repo_id"] = repo_id
        method_df["rank"] = ranks
        method_df["reciprocal_rank"] = reciprocal_rank
        method_df["top1_hit"] = top1_hit
        per_query_frames.append(method_df)

        method_results[method_name] = {
            "repo_id": repo_id,
            "snapshot_dir": str(student.snapshot_dir),
            "test_mrr": float(reciprocal_rank.mean()),
            "recall_at_1": float(top1_hit.mean()),
        }

    per_query_df = pd.concat(per_query_frames, ignore_index=True)
    per_query_csv = args.output_dir / "per_query_results.csv"
    per_query_df.to_csv(per_query_csv, index=False)

    overlap_methods = list(DEFAULT_METHOD_REPOS.keys())
    success_matrix = pd.DataFrame(index=overlap_methods, columns=overlap_methods, dtype=float)
    failure_matrix = pd.DataFrame(index=overlap_methods, columns=overlap_methods, dtype=float)
    top1_map = {
        method: per_query_df.loc[per_query_df["method"] == method, "top1_hit"].to_numpy(dtype=bool)
        for method in overlap_methods
    }
    for left in overlap_methods:
        for right in overlap_methods:
            success_matrix.loc[left, right] = jaccard(top1_map[left], top1_map[right])
            failure_matrix.loc[left, right] = jaccard(~top1_map[left], ~top1_map[right])

    success_matrix.to_csv(args.output_dir / "error_overlap_success_jaccard.csv")
    failure_matrix.to_csv(args.output_dir / "error_overlap_failure_jaccard.csv")
    plot_heatmap(
        success_matrix,
        title="Analysis 3: Top-1 Success Overlap (Jaccard)",
        output_path=args.output_dir / "analysis3_success_overlap_heatmap.png",
    )
    plot_heatmap(
        failure_matrix,
        title="Analysis 3: Failure Overlap (Jaccard)",
        output_path=args.output_dir / "analysis3_failure_overlap_heatmap.png",
    )

    margin_summary = (
        per_query_df.groupby(["method", "margin_bin"], sort=False)
        .agg(
            queries=("query_idx", "count"),
            mrr=("reciprocal_rank", "mean"),
            recall_at_1=("top1_hit", "mean"),
            median_rank=("rank", "median"),
            mean_teacher_margin=("teacher_margin", "mean"),
        )
        .reset_index()
    )
    margin_summary["margin_bin"] = pd.Categorical(
        margin_summary["margin_bin"],
        categories=["low", "medium", "high"],
        ordered=True,
    )
    margin_summary = margin_summary.sort_values(["margin_bin", "method"]).reset_index(drop=True)
    margin_summary.to_csv(args.output_dir / "analysis4_margin_stratified_mrr.csv", index=False)
    plot_margin_mrr(
        margin_summary,
        output_path=args.output_dir / "analysis4_margin_stratified_mrr.png",
    )

    plot_rank_cdf(
        per_query_df=per_query_df,
        output_path=args.output_dir / "rank_distribution_cdf.png",
    )

    summary_payload = {
        "dataset_name": args.dataset_name,
        "teacher_repo": args.teacher_repo,
        "teacher_snapshot_dir": str(teacher_snapshot),
        "methods": method_results,
        "margin_bin_counts": {
            label: int((margin_bin == label).sum())
            for label in ("low", "medium", "high")
        },
        "best_high_margin_method": (
            margin_summary[margin_summary["margin_bin"] == "high"]
            .sort_values("mrr", ascending=False)
            .iloc[0][["method", "mrr"]]
            .to_dict()
        ),
        "best_low_margin_method": (
            margin_summary[margin_summary["margin_bin"] == "low"]
            .sort_values("mrr", ascending=False)
            .iloc[0][["method", "mrr"]]
            .to_dict()
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Wrote per-query results to {per_query_csv}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
