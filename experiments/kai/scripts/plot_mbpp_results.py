#!/usr/bin/env python3
"""Plot MBPP retrieval experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_METRICS = ["mrr", "recall@10", "map@10", "ndcg@10"]
RECALL_KS = [1, 5, 10, 20]
METHOD_ORDER = [
    "Random",
    "TF-IDF",
    "Best Pretrained",
    "Finetuned (MNR)",
    "Finetuned (Hard-Neg)",
]
COLORS = {
    "Random": "#8f8f8f",
    "TF-IDF": "#d81b60",
    "Best Pretrained": "#1e88e5",
    "Finetuned (MNR)": "#43a047",
    "Finetuned (Hard-Neg)": "#fb8c00",
}
STAGE_ORDER = ["Zero-shot", "Finetuned (MNR)", "Finetuned (Hard-Neg)"]
STAGE_COLORS = {
    "Zero-shot": "#1e88e5",
    "Finetuned (MNR)": "#43a047",
    "Finetuned (Hard-Neg)": "#fb8c00",
}


def load_table(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column: {col}")
    return df


def resolve_run_file(run_dir: Path, subdir: str, filename: str) -> Path:
    preferred = run_dir / subdir / filename
    if preferred.exists():
        return preferred
    legacy = run_dir / filename
    if legacy.exists():
        return legacy
    return preferred


def shorten_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def pick_pretrained_rows(metrics: pd.DataFrame, protocol: str) -> pd.DataFrame:
    data = metrics[
        (metrics["status"] == "success")
        & (metrics["protocol"] == protocol)
        & (metrics["method"] == "pretrained")
        & (metrics["stage"] == "pretrained")
    ].copy()
    if data.empty:
        raise ValueError(f"No pretrained rows for protocol={protocol}")
    data = data.drop_duplicates(subset=["model_name"], keep="last")
    data = data.sort_values(by=["mrr", "recall@10"], ascending=[False, False]).reset_index(drop=True)
    data["model_short"] = data["model_name"].map(shorten_model_name)
    return data


def build_before_after_table(metrics: pd.DataFrame, protocol: str) -> pd.DataFrame:
    pretrained = pick_pretrained_rows(metrics, protocol)
    final_standard = metrics[
        (metrics["status"] == "success")
        & (metrics["protocol"] == protocol)
        & (metrics["stage"] == "final_standard")
    ].copy()
    final_hardneg = metrics[
        (metrics["status"] == "success")
        & (metrics["protocol"] == protocol)
        & (metrics["stage"] == "final_hardneg")
    ].copy()

    if not final_standard.empty:
        final_standard = (
            final_standard.sort_values("timestamp")
            .drop_duplicates(subset=["model_name"], keep="last")
            .set_index("model_name")
        )
    else:
        final_standard = pd.DataFrame()

    if not final_hardneg.empty:
        final_hardneg = (
            final_hardneg.sort_values("timestamp")
            .drop_duplicates(subset=["model_name"], keep="last")
            .set_index("model_name")
        )
    else:
        final_hardneg = pd.DataFrame()

    rows = []
    for _, pre in pretrained.iterrows():
        model_name = str(pre["model_name"])
        row = {
            "model_name": model_name,
            "model_short": shorten_model_name(model_name),
        }
        for metric in PRIMARY_METRICS:
            row[f"zero_{metric}"] = float(pre[metric])
            row[f"mnr_{metric}"] = np.nan
            row[f"hardneg_{metric}"] = np.nan

        if not final_standard.empty and model_name in final_standard.index:
            for metric in PRIMARY_METRICS:
                row[f"mnr_{metric}"] = float(final_standard.loc[model_name, metric])
        if not final_hardneg.empty and model_name in final_hardneg.index:
            for metric in PRIMARY_METRICS:
                row[f"hardneg_{metric}"] = float(final_hardneg.loc[model_name, metric])

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["zero_mrr", "zero_recall@10"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return out


def pick_method_rows(metrics: pd.DataFrame, protocol: str) -> pd.DataFrame:
    data = metrics[(metrics["status"] == "success") & (metrics["protocol"] == protocol)].copy()
    if data.empty:
        raise ValueError(f"No success rows for protocol={protocol}")

    random_row = data[(data["method"] == "baseline") & (data["technique"] == "random")].iloc[0]
    tfidf_row = data[(data["method"] == "baseline") & (data["technique"] == "tfidf")].iloc[0]
    best_pretrained_row = data[data["method"] == "pretrained"].sort_values(
        by=["mrr", "recall@10"], ascending=[False, False]
    ).iloc[0]
    mnr_rows = data[data["stage"] == "final_standard"].copy()
    mnr_rows = mnr_rows.sort_values(by=["mrr", "recall@10"], ascending=[False, False])
    finetune_mnr_row = mnr_rows.iloc[0]
    hard_rows = data[data["stage"] == "final_hardneg"].copy()
    hard_rows = hard_rows.sort_values(by=["mrr", "recall@10"], ascending=[False, False])
    finetune_hardneg_row = hard_rows.iloc[0]

    rows = [
        ("Random", random_row),
        ("TF-IDF", tfidf_row),
        ("Best Pretrained", best_pretrained_row),
        ("Finetuned (MNR)", finetune_mnr_row),
        ("Finetuned (Hard-Neg)", finetune_hardneg_row),
    ]
    out = pd.DataFrame([{"label": label, **row.to_dict()} for label, row in rows])
    out["label"] = pd.Categorical(out["label"], categories=METHOD_ORDER, ordered=True)
    out = out.sort_values("label")
    return out


def plot_pretrained_model_bars(pretrained: pd.DataFrame, title_prefix: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    labels = pretrained["model_short"].tolist()
    x = np.arange(len(labels))
    palette = plt.get_cmap("tab10")

    for idx, metric in enumerate(PRIMARY_METRICS):
        ax = axes[idx]
        y = pretrained[metric].to_numpy(dtype=float)
        colors = [palette(i % 10) for i in range(len(labels))]
        ax.bar(x, y, color=colors, alpha=0.9)
        ax.set_title(metric.upper())
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for j, val in enumerate(y):
            ax.text(j, min(0.995, val + 0.015), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{title_prefix}: Pretrained Model Comparison", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(rows: pd.DataFrame, title_prefix: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for idx, metric in enumerate(PRIMARY_METRICS):
        ax = axes[idx]
        x = np.arange(len(rows))
        y = rows[metric].to_numpy(dtype=float)
        colors = [COLORS[label] for label in rows["label"].astype(str)]
        ax.bar(x, y, color=colors, alpha=0.9)
        ax.set_title(metric.upper())
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(rows["label"], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for j, val in enumerate(y):
            ax.text(j, min(0.995, val + 0.015), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{title_prefix}: Primary Metrics", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pretrained_before_after_values(before_after: pd.DataFrame, title_prefix: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    labels = before_after["model_short"].tolist()
    x = np.arange(len(labels))
    width = 0.22

    for idx, metric in enumerate(PRIMARY_METRICS):
        ax = axes[idx]
        zero_vals = before_after[f"zero_{metric}"].to_numpy(dtype=float)
        mnr_vals = before_after[f"mnr_{metric}"].to_numpy(dtype=float)
        hardneg_vals = before_after[f"hardneg_{metric}"].to_numpy(dtype=float)

        ax.bar(x - width, zero_vals, width=width, label="Zero-shot", color=STAGE_COLORS["Zero-shot"], alpha=0.9)
        ax.bar(x, mnr_vals, width=width, label="Finetuned (MNR)", color=STAGE_COLORS["Finetuned (MNR)"], alpha=0.9)
        ax.bar(
            x + width,
            hardneg_vals,
            width=width,
            label="Finetuned (Hard-Neg)",
            color=STAGE_COLORS["Finetuned (Hard-Neg)"],
            alpha=0.9,
        )

        for j in range(len(labels)):
            if np.isnan(mnr_vals[j]):
                ax.text(x[j], 0.02, "N/A", ha="center", va="bottom", fontsize=7, color="#666666", rotation=90)
            if np.isnan(hardneg_vals[j]):
                ax.text(
                    x[j] + width,
                    0.02,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#666666",
                    rotation=90,
                )

        ax.set_title(metric.upper())
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{title_prefix}: Before/After Finetuning by Pretrained Model", fontsize=14, y=1.08)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pretrained_before_after_deltas(before_after: pd.DataFrame, title_prefix: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    labels = before_after["model_short"].tolist()
    x = np.arange(len(labels))
    width = 0.28

    for idx, metric in enumerate(PRIMARY_METRICS):
        ax = axes[idx]
        zero_vals = before_after[f"zero_{metric}"].to_numpy(dtype=float)
        mnr_vals = before_after[f"mnr_{metric}"].to_numpy(dtype=float)
        hardneg_vals = before_after[f"hardneg_{metric}"].to_numpy(dtype=float)

        delta_mnr = mnr_vals - zero_vals
        delta_hardneg = hardneg_vals - zero_vals

        ax.bar(x - width / 2, delta_mnr, width=width, label="MNR - Zero-shot", color="#2e7d32", alpha=0.9)
        ax.bar(
            x + width / 2,
            delta_hardneg,
            width=width,
            label="Hard-Neg - Zero-shot",
            color="#ef6c00",
            alpha=0.9,
        )

        for j in range(len(labels)):
            if np.isnan(delta_mnr[j]):
                ax.text(x[j] - width / 2, 0.001, "N/A", ha="center", va="bottom", fontsize=7, color="#666666", rotation=90)
            if np.isnan(delta_hardneg[j]):
                ax.text(
                    x[j] + width / 2,
                    0.001,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#666666",
                    rotation=90,
                )

        lim = np.nanmax(np.abs(np.concatenate([delta_mnr, delta_hardneg])))
        lim = max(0.02, float(lim) * 1.35)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"Delta {metric.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{title_prefix}: Metric Deltas vs Zero-shot by Pretrained Model", fontsize=14, y=1.08)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_recall_curves(rows: pd.DataFrame, title_prefix: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for _, row in rows.iterrows():
        y = [float(row[f"recall@{k}"]) for k in RECALL_KS]
        label = str(row["label"])
        ax.plot(RECALL_KS, y, marker="o", linewidth=2.2, label=label, color=COLORS[label])
    ax.set_title(f"{title_prefix}: Recall@k Curves")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(RECALL_KS)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_deltas(comparisons: pd.DataFrame, output_path: Path) -> None:
    data = comparisons[
        (comparisons["status"] == "success")
        & (comparisons["protocol"] == "heldout_test")
        & (comparisons["metric"].isin(["mrr", "recall@10"]))
    ].copy()
    if data.empty:
        return

    display_map = {
        "best_pretrained_vs_tfidf": "Best Pretrained - TF-IDF",
        "final_standard_vs_best_pretrained": "Standard FT - Best Pretrained",
        "final_hardneg_vs_final_standard": "HardNeg FT - Standard FT",
    }
    data["comparison_label"] = data["comparison"].map(display_map).fillna(data["comparison"])
    data["label"] = data["comparison_label"] + " (" + data["metric"].str.upper() + ")"
    data = data.sort_values(by=["metric", "delta"], ascending=[True, False])

    y = np.arange(len(data))
    delta = data["delta"].to_numpy(dtype=float)
    err_low = (delta - data["ci_low"].to_numpy(dtype=float)).clip(min=0.0)
    err_high = (data["ci_high"].to_numpy(dtype=float) - delta).clip(min=0.0)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    bars = ax.barh(y, delta, color="#3949ab", alpha=0.9)
    ax.errorbar(delta, y, xerr=np.vstack([err_low, err_high]), fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_xlabel("Delta")
    ax.set_title("Held-out Improvements with 95% Bootstrap CI")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, delta):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_validation(metrics: pd.DataFrame, output_path: Path) -> None:
    sweep = metrics[
        (metrics["status"] == "success")
        & (metrics["stage"] == "sweep_mnr")
        & (metrics["protocol"] == "tune_validation")
    ].copy()
    if sweep.empty:
        return

    sweep = sweep.sort_values(by=["mrr", "recall@10"], ascending=[False, False])
    x = np.arange(len(sweep))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - width, sweep["mrr"], width=width, label="MRR", color="#00897b")
    ax.bar(x, sweep["map@10"], width=width, label="MAP@10", color="#6d4c41")
    ax.bar(x + width, sweep["ndcg@10"], width=width, label="nDCG@10", color="#f4511e")
    ax.set_xticks(x)
    ax.set_xticklabels(sweep["config_id"], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Sweep (Validation): Config Comparison")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot MBPP retrieval experiment charts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("experiments/kai/results/mbpp_full_matrix"),
        help="Run directory containing metrics/ and reports/ artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <run-dir>/plots)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "plots")
    methodology_dir = out_dir / "aggregation_methodology"
    pretrained_dir = out_dir / "aggregation_pretrained_models"
    before_after_dir = out_dir / "aggregation_pretrained_before_after"
    for d in [out_dir, methodology_dir, pretrained_dir, before_after_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metrics_path = resolve_run_file(run_dir, "metrics", "metrics_all.csv")
    comparisons_path = resolve_run_file(run_dir, "metrics", "comparisons.csv")
    metrics = load_table(metrics_path, required=["status", "protocol", "mrr", "recall@10"])
    comparisons = load_table(
        comparisons_path,
        required=["status", "protocol", "comparison", "metric", "delta", "ci_low", "ci_high"],
    )

    heldout_rows = pick_method_rows(metrics, "heldout_test")
    full_rows = pick_method_rows(metrics, "full_corpus")
    heldout_pretrained = pick_pretrained_rows(metrics, "heldout_test")
    full_pretrained = pick_pretrained_rows(metrics, "full_corpus")
    heldout_before_after = build_before_after_table(metrics, "heldout_test")
    full_before_after = build_before_after_table(metrics, "full_corpus")

    # Aggregation: methodology
    plot_metric_bars(heldout_rows, "Held-out Test", methodology_dir / "heldout_primary_metrics.png")
    plot_metric_bars(full_rows, "Full-corpus Diagnostic", methodology_dir / "full_corpus_primary_metrics.png")
    plot_recall_curves(heldout_rows, "Held-out Test", methodology_dir / "heldout_recall_curves.png")
    plot_recall_curves(full_rows, "Full-corpus Diagnostic", methodology_dir / "full_corpus_recall_curves.png")
    plot_comparison_deltas(comparisons, methodology_dir / "heldout_delta_with_ci.png")
    plot_sweep_validation(metrics, methodology_dir / "sweep_validation_metrics.png")

    # Aggregation: pretrained model matrix
    plot_pretrained_model_bars(
        heldout_pretrained,
        "Held-out Test",
        pretrained_dir / "heldout_pretrained_model_metrics.png",
    )
    plot_pretrained_model_bars(
        full_pretrained,
        "Full-corpus Diagnostic",
        pretrained_dir / "full_corpus_pretrained_model_metrics.png",
    )

    # Aggregation: before/after per pretrained model
    plot_pretrained_before_after_values(
        heldout_before_after,
        "Held-out Test",
        before_after_dir / "heldout_before_after_values.png",
    )
    plot_pretrained_before_after_deltas(
        heldout_before_after,
        "Held-out Test",
        before_after_dir / "heldout_before_after_deltas.png",
    )
    plot_pretrained_before_after_values(
        full_before_after,
        "Full-corpus Diagnostic",
        before_after_dir / "full_corpus_before_after_values.png",
    )
    plot_pretrained_before_after_deltas(
        full_before_after,
        "Full-corpus Diagnostic",
        before_after_dir / "full_corpus_before_after_deltas.png",
    )

    manifest = out_dir / "README.txt"
    manifest.write_text(
        "\n".join(
            [
                "Generated plots by aggregation:",
                "",
                "[aggregation_methodology]",
                "- heldout_primary_metrics.png",
                "- full_corpus_primary_metrics.png",
                "- heldout_recall_curves.png",
                "- full_corpus_recall_curves.png",
                "- heldout_delta_with_ci.png",
                "- sweep_validation_metrics.png",
                "",
                "[aggregation_pretrained_models]",
                "- heldout_pretrained_model_metrics.png",
                "- full_corpus_pretrained_model_metrics.png",
                "",
                "[aggregation_pretrained_before_after]",
                "- heldout_before_after_values.png",
                "- heldout_before_after_deltas.png",
                "- full_corpus_before_after_values.png",
                "- full_corpus_before_after_deltas.png",
                "",
                "Note:",
                "- Before/after charts include every pretrained model with available finetune rows.",
                "- If a stage is missing for a model, it is shown as N/A.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
