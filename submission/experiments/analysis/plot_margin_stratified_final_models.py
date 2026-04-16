from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = PROJECT_ROOT / "mbpp_kd_suite"
SRC_ROOT = SUITE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset  # noqa: E402
from mbpp_kd_suite.metrics import paired_ranks  # noqa: E402
from mbpp_kd_suite.modeling import (  # noqa: E402
    StudentQueryEncoder,
    encode_student_texts,
    encode_texts_backbone,
    infer_model_encoding_spec,
)
from mbpp_kd_suite.runtime import pick_device, set_seed  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submission" / "experiments" / "analysis" / "final_margin_plots"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "submission" / "experiments" / "hf_cache"
DEFAULT_DATASET_NAME = "BEE-spoke-data/TACO-hf"
DEFAULT_TEACHER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
FINAL_MODEL_REPOS: dict[str, str] = {
    "control": "cs4248-nlp/paper-s7-control-bs32-tinybert-general-4l-312d-taco-hf-20260402-015143",
    "embed_distill": "cs4248-nlp/paper-s7-embed-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
    "bimga_uniform": "cs4248-nlp/paper-s8-a2-bimga-uniform-tinybert-general-4l-312d-taco-hf-20260402-015143",
    "hard_neg_pair": "cs4248-nlp/paper-s8-hnp-dw100-pw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
    "score_distill": "cs4248-nlp/paper-s9-score-dw100-tinybert-general-4l-312d-taco-hf-20260402-015143",
    "bimga": "cs4248-nlp/paper-s10-bimga-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
}
METHOD_ORDER = list(FINAL_MODEL_REPOS.keys())
METHOD_COLORS = {
    "control": "#718096",
    "score_distill": "#2B6CB0",
    "embed_distill": "#2F855A",
    "hard_neg_pair": "#D69E2E",
    "bimga_uniform": "#805AD5",
    "bimga": "#C53030",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate margin-stratified plots for the 6 final saturated models.",
    )
    parser.add_argument("--per-query-csv", type=Path, default=None, help="Reuse an existing per-query CSV instead of replaying models.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--teacher-repo", default=DEFAULT_TEACHER_REPO)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
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
    return Path(snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir)))


def load_student_from_snapshot(snapshot_dir: Path, device: torch.device) -> tuple[StudentQueryEncoder, AutoTokenizer]:
    projection_state = None
    target_hidden_size = None
    projection_path = snapshot_dir / "projection.pt"
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
    return model, tokenizer


def resolve_teacher_paths(snapshot_dir: Path) -> tuple[Path, Path]:
    model_path = snapshot_dir
    tokenizer_path = snapshot_dir
    if not ((snapshot_dir / "model.safetensors").exists() or (snapshot_dir / "pytorch_model.bin").exists()):
        backbone_dir = snapshot_dir / "backbone"
        tokenizer_dir = snapshot_dir / "tokenizer"
        if (backbone_dir / "model.safetensors").exists() or (backbone_dir / "pytorch_model.bin").exists():
            model_path = backbone_dir
        if tokenizer_dir.exists():
            tokenizer_path = tokenizer_dir
    return model_path, tokenizer_path


def load_teacher(repo_id: str, cache_dir: Path, device: torch.device) -> tuple[Path, AutoModel, AutoTokenizer, Any]:
    snapshot_dir = ensure_snapshot(repo_id, cache_dir)
    model_path, tokenizer_path = resolve_teacher_paths(snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = AutoModel.from_pretrained(str(model_path)).to(device)
    model.eval()
    encoding_spec = infer_model_encoding_spec(
        getattr(model.config, "_name_or_path", None),
        getattr(tokenizer, "name_or_path", None),
    )
    return snapshot_dir, model, tokenizer, encoding_spec


def build_margin_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    order = np.argsort(values)
    labels = np.empty(len(values), dtype=np.int64)
    for idx, indices in enumerate(np.array_split(order, num_bins), start=1):
        labels[indices] = idx
    return labels


def replay_models(args: argparse.Namespace) -> pd.DataFrame:
    device = resolve_device(args.device)
    set_seed(args.seed)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_retrieval_dataset(dataset_name=args.dataset_name, taco_val_size=1000, seed=args.seed)
    splits = dataset_dict_to_splits(dataset)
    queries = splits.test.queries
    codes = splits.test.codes

    teacher_snapshot, teacher_model, teacher_tokenizer, teacher_encoding_spec = load_teacher(
        repo_id=args.teacher_repo,
        cache_dir=args.cache_dir,
        device=device,
    )
    teacher_q = encode_texts_backbone(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        texts=queries,
        text_role="query",
        encoding_spec=teacher_encoding_spec,
        max_length=args.max_query_length,
        batch_size=args.batch_size,
        device=device,
        desc="teacher_queries",
    )
    teacher_d = encode_texts_backbone(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        texts=codes,
        text_role="document",
        encoding_spec=teacher_encoding_spec,
        max_length=args.max_code_length,
        batch_size=args.batch_size,
        device=device,
        desc="teacher_docs",
    )
    teacher_scores = (teacher_q @ teacher_d.T).numpy()
    teacher_positive = np.diag(teacher_scores)
    masked = teacher_scores.copy()
    np.fill_diagonal(masked, -np.inf)
    teacher_hardest_negative = masked.max(axis=1)
    teacher_margin = teacher_positive - teacher_hardest_negative

    rows: list[dict[str, Any]] = []
    for method, repo_id in FINAL_MODEL_REPOS.items():
        snapshot_dir = ensure_snapshot(repo_id, args.cache_dir)
        model, tokenizer = load_student_from_snapshot(snapshot_dir, device=device)
        query_embs = encode_student_texts(
            student_model=model,
            tokenizer=tokenizer,
            texts=queries,
            text_role="query",
            max_length=args.max_query_length,
            batch_size=args.batch_size,
            device=device,
            desc=f"{method}_queries",
        )
        doc_embs = encode_student_texts(
            student_model=model,
            tokenizer=tokenizer,
            texts=codes,
            text_role="document",
            max_length=args.max_code_length,
            batch_size=args.batch_size,
            device=device,
            desc=f"{method}_docs",
        )
        scores = (query_embs @ doc_embs.T).numpy()
        ranks = paired_ranks(scores)
        reciprocal_rank = 1.0 / ranks.astype(np.float64)
        top1 = ranks == 1
        positives = np.diag(scores)
        masked_scores = scores.copy()
        np.fill_diagonal(masked_scores, -np.inf)
        hardest_negative = masked_scores.max(axis=1)
        student_margin = positives - hardest_negative

        for idx in range(len(queries)):
            rows.append(
                {
                    "run_name": next(name for name, rid in FINAL_MODEL_REPOS.items() if rid == repo_id),
                    "method": method,
                    "repo_id": repo_id,
                    "query_idx": idx,
                    "rank": int(ranks[idx]),
                    "reciprocal_rank": float(reciprocal_rank[idx]),
                    "top1_hit": bool(top1[idx]),
                    "teacher_margin": float(teacher_margin[idx]),
                    "student_margin": float(student_margin[idx]),
                }
            )

    df = pd.DataFrame(rows)
    df["teacher_repo"] = args.teacher_repo
    df.attrs["teacher_snapshot_dir"] = str(teacher_snapshot)
    return df


def summarize_bins(df: pd.DataFrame, column: str, label_name: str) -> pd.DataFrame:
    summary = (
        df.groupby(["method", column], sort=False)
        .agg(
            queries=("query_idx", "count"),
            mrr=("reciprocal_rank", "mean"),
            recall_at_1=("top1_hit", "mean"),
            median_rank=("rank", "median"),
            mean_teacher_margin=("teacher_margin", "mean"),
            min_teacher_margin=("teacher_margin", "min"),
            max_teacher_margin=("teacher_margin", "max"),
        )
        .reset_index()
        .rename(columns={column: label_name})
    )
    return summary


def plot_decile_lines(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.75))
    for method in METHOD_ORDER:
        subset = summary[summary["method"] == method].sort_values("margin_decile")
        linewidth = 3 if method == "bimga" else 2
        alpha = 1.0 if method == "bimga" else 0.85
        ax.plot(
            subset["margin_decile"],
            subset["mrr"],
            marker="o",
            linewidth=linewidth,
            alpha=alpha,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax.set_xlabel("Teacher Margin Decile (low confidence -> high confidence)")
    ax.set_ylabel("MRR")
    ax.set_title("Margin-Stratified MRR by Teacher-Margin Decile")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.0, max(0.35, float(summary["mrr"].max()) * 1.08))
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_decile_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        summary.pivot(index="method", columns="margin_decile", values="mrr")
        .reindex(index=METHOD_ORDER, columns=list(range(1, 11)))
    )
    fig, ax = plt.subplots(figsize=(11, 4.75))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("Teacher Margin Decile (low -> high)")
    ax.set_ylabel("Method")
    ax.set_title("MRR Heatmap Across Teacher-Margin Deciles")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MRR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tercile_bars(summary: pd.DataFrame, output_path: Path) -> None:
    bins = [1, 2, 3]
    labels = ["Low", "Medium", "High"]
    x = np.arange(len(bins))
    width = 0.12
    fig, ax = plt.subplots(figsize=(10, 5.75))
    for idx, method in enumerate(METHOD_ORDER):
        subset = summary[summary["method"] == method].sort_values("margin_tercile")
        ax.bar(
            x + (idx - (len(METHOD_ORDER) - 1) / 2) * width,
            subset["mrr"],
            width=width,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax.set_xticks(x, labels=labels)
    ax.set_xlabel("Teacher Margin Tercile")
    ax.set_ylabel("MRR")
    ax.set_title("Margin-Stratified MRR (Terciles)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.per_query_csv is not None:
        df = pd.read_csv(args.per_query_csv)
    else:
        df = replay_models(args)

    df = df[df["method"].isin(METHOD_ORDER)].copy()
    if df.empty:
        raise ValueError("No rows found for the 6 final saturated models.")

    query_margin = (
        df[["query_idx", "teacher_margin"]]
        .drop_duplicates()
        .sort_values("query_idx")
        .reset_index(drop=True)
    )
    query_margin["margin_decile"] = build_margin_bins(query_margin["teacher_margin"].to_numpy(), 10)
    query_margin["margin_tercile"] = build_margin_bins(query_margin["teacher_margin"].to_numpy(), 3)

    df = df.drop(columns=[c for c in ("margin_decile", "margin_tercile") if c in df.columns], errors="ignore")
    df = df.merge(query_margin, on=["query_idx", "teacher_margin"], how="left")

    per_query_path = args.output_dir / "per_query_results_final6.csv"
    df.to_csv(per_query_path, index=False)

    tercile_summary = summarize_bins(df, "margin_tercile", "margin_tercile").sort_values(["margin_tercile", "method"])
    decile_summary = summarize_bins(df, "margin_decile", "margin_decile").sort_values(["margin_decile", "method"])
    tercile_summary.to_csv(args.output_dir / "margin_terciles_final6.csv", index=False)
    decile_summary.to_csv(args.output_dir / "margin_deciles_final6.csv", index=False)

    plot_tercile_bars(tercile_summary, args.output_dir / "margin_terciles_final6.png")
    plot_decile_lines(decile_summary, args.output_dir / "margin_decile_lines_final6.png")
    plot_decile_heatmap(decile_summary, args.output_dir / "margin_decile_heatmap_final6.png")

    summary = {
        "teacher_repo": args.teacher_repo,
        "dataset_name": args.dataset_name,
        "num_queries": int(df["query_idx"].nunique()),
        "methods": METHOD_ORDER,
        "output_dir": str(args.output_dir),
        "used_existing_per_query_csv": args.per_query_csv is not None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
