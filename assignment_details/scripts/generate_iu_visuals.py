#!/usr/bin/env python3
"""Generate a complete visual pack for CS4248 IU slides."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.dates as mdates

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional at runtime
    load_dataset = None


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "visuals" / "iu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared project stats from current run
PRETRAINED = {"MRR": 0.587, "Recall@1": 0.458, "Recall@5": 0.742, "Recall@10": 0.814}
FINETUNED = {"MRR": 0.731, "Recall@1": 0.620, "Recall@5": 0.866, "Recall@10": 0.912}


def _style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )


def _save(fig: plt.Figure, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_slide3_quality_efficiency():
    """Slide 3: MRR/Recall@1 versus estimated latency index."""
    labels = [
        "Teacher (no FT)",
        "Teacher (FT)",
        "Student target (90%)",
        "Student stretch (95%)",
    ]
    latency_idx = np.array([1.00, 1.00, 22.0 / 125.0, 33.0 / 125.0], dtype=float)
    mrr = np.array([
        PRETRAINED["MRR"],
        FINETUNED["MRR"],
        0.90 * FINETUNED["MRR"],
        0.95 * FINETUNED["MRR"],
    ])
    recall1 = np.array([
        PRETRAINED["Recall@1"],
        FINETUNED["Recall@1"],
        0.90 * FINETUNED["Recall@1"],
        0.95 * FINETUNED["Recall@1"],
    ])
    colors = ["#8da0cb", "#1b9e77", "#fc8d62", "#e78ac3"]

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    for x, y, c, label in zip(latency_idx, mrr, colors, labels):
        ax.scatter(x, y, s=170, color=c, marker="o", edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(label, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9)

    for x, y, c in zip(latency_idx, recall1, colors):
        ax.scatter(x, y, s=170, color=c, marker="s", edgecolor="black", linewidth=0.7, alpha=0.92, zorder=3)

    # Connect each model's MRR and Recall@1 for easier comparison
    for x, y1, y2, c in zip(latency_idx, mrr, recall1, colors):
        ax.plot([x, x], [min(y1, y2), max(y1, y2)], color=c, linewidth=1.2, alpha=0.9, zorder=2)

    ax.set_title("MRR and Recall@1 vs Latency (Teacher vs Student Targets)")
    ax.set_xlabel("Estimated latency index (teacher=1.0, lower is faster)")
    ax.set_ylabel("Retrieval quality")
    ax.set_xlim(0.12, 1.08)
    ax.set_ylim(0.42, 0.78)
    ax.grid(alpha=0.25)

    # Minimal legend entries without repeating model labels
    ax.scatter([], [], s=100, marker="o", color="#666666", edgecolor="black", label="MRR")
    ax.scatter([], [], s=100, marker="s", color="#666666", edgecolor="black", label="Recall@1")
    ax.legend(loc="lower right", frameon=True)

    foot = "Teacher scores from current run; student quality values are retention targets (90%/95%). Latency index uses parameter-ratio proxy."
    fig.text(0.01, -0.02, foot, fontsize=8.7)
    _save(fig, "s03_quality_efficiency_tradeoff.png")


def fig_slide5_pipeline_overview():
    """Slide 5: system block diagram."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")

    boxes = [
        (0.02, 0.25, 0.16, 0.5, "MBPP\n(text, code)"),
        (0.23, 0.25, 0.17, 0.5, "Teacher\nUniXcoder FT"),
        (0.45, 0.25, 0.20, 0.5, "KD Training\nAlign + Rank + Contrastive"),
        (0.70, 0.25, 0.14, 0.5, "Student\nMiniLM / GTE"),
        (0.88, 0.25, 0.10, 0.5, "Retrieve\nTop-k"),
    ]
    colors = ["#fddbc7", "#d1e5f0", "#c7eae5", "#f6e8c3", "#d9f0a3"]

    for (x, y, w, h, text), color in zip(boxes, colors):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, transform=ax.transAxes)

    arrows = [(0.18, 0.50, 0.23, 0.50), (0.40, 0.50, 0.45, 0.50), (0.65, 0.50, 0.70, 0.50), (0.84, 0.50, 0.88, 0.50)]
    for x1, y1, x2, y2 in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.5,
                color="#444444",
                transform=ax.transAxes,
            )
        )

    ax.set_title("Distilled Retrieval Pipeline Overview", pad=12)
    _save(fig, "s05_pipeline_overview.png")


def fig_slide7_dataset_split():
    counts = np.array([374, 90, 500])
    labels = ["Train", "Validation", "Test"]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("MBPP Split Sizes")
    ax.set_ylabel("Number of Examples")
    total = counts.sum()

    for bar, c in zip(bars, counts):
        pct = 100 * c / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, f"{c}\n({pct:.1f}%)", ha="center", va="bottom")

    ax.set_ylim(0, max(counts) * 1.22)
    _save(fig, "s07_dataset_split_counts.png")


def _load_mbpp_stats():
    if load_dataset is None:
        return None
    try:
        ds = load_dataset("google-research-datasets/mbpp")
    except Exception:
        return None

    test = ds["test"]
    desc_words = np.array([len(ex["text"].split()) for ex in test], dtype=int)
    code_lines = np.array([len(ex["code"].split("\n")) for ex in test], dtype=int)
    return {
        "desc_words": desc_words,
        "code_lines": code_lines,
        "n_test": len(test),
    }


def fig_slide7_length_hist(stats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    if stats is None:
        for ax in axes:
            ax.axis("off")
        fig.text(0.5, 0.5, "Could not load MBPP dataset to compute length distributions.", ha="center", va="center", fontsize=12)
        _save(fig, "s07_dataset_length_distributions.png")
        return

    desc = stats["desc_words"]
    code = stats["code_lines"]

    axes[0].hist(desc, bins=20, color="#80b1d3", edgecolor="black", alpha=0.9)
    axes[0].set_title("Description Length (Words, Test Split)")
    axes[0].set_xlabel("Words")
    axes[0].set_ylabel("Count")

    axes[1].hist(code, bins=20, color="#fdb462", edgecolor="black", alpha=0.9)
    axes[1].set_title("Code Length (Lines, Test Split)")
    axes[1].set_xlabel("Lines")
    axes[1].set_ylabel("Count")

    fig.suptitle(f"MBPP Length Distributions (n={stats['n_test']})")
    _save(fig, "s07_dataset_length_distributions.png")


def fig_slide7_failure_case_card():
    query = "Write a python function to remove first and last occurrence of a given character from the string."
    top1 = "import re\ndef remove_lowercase(str1):"
    top2 = "import re\ndef remove_whitespaces(text1):"
    top3 = "import re\ndef remove_splchar(text):"
    correct = "def remove_char(s, ch):\n    # remove first and last occurrence of ch"

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")

    # Panels
    q_box = FancyBboxPatch((0.02, 0.62), 0.96, 0.32, boxstyle="round,pad=0.02", facecolor="#e8f1fa", edgecolor="#4c78a8", linewidth=1.2)
    r_box = FancyBboxPatch((0.02, 0.08), 0.64, 0.48, boxstyle="round,pad=0.02", facecolor="#fff4e6", edgecolor="#f58518", linewidth=1.2)
    c_box = FancyBboxPatch((0.70, 0.08), 0.28, 0.48, boxstyle="round,pad=0.02", facecolor="#eaf7ea", edgecolor="#54a24b", linewidth=1.2)
    for b in [q_box, r_box, c_box]:
        ax.add_patch(b)

    ax.text(0.04, 0.89, "Query (example hard case)", fontsize=11, weight="bold")
    ax.text(0.04, 0.74, textwrap.fill(query, 90), fontsize=10)

    ax.text(0.04, 0.52, "Top retrieved (wrong but semantically related)", fontsize=11, weight="bold")
    ax.text(0.05, 0.43, f"Rank 1: {top1}", family="monospace", fontsize=9)
    ax.text(0.05, 0.33, f"Rank 2: {top2}", family="monospace", fontsize=9)
    ax.text(0.05, 0.23, f"Rank 3: {top3}", family="monospace", fontsize=9)
    ax.text(0.05, 0.13, "Correct rank in this case: 19", fontsize=10, color="#b22222")

    ax.text(0.72, 0.52, "Correct target", fontsize=11, weight="bold")
    ax.text(0.72, 0.34, correct, family="monospace", fontsize=9)

    ax.set_title("Retrieval Failure Case Card (for Analysis Slide)", pad=8)
    _save(fig, "s07_retrieval_failure_case_card.png")


def fig_slide8_results_bar():
    metrics = list(PRETRAINED.keys())
    before = np.array([PRETRAINED[m] for m in metrics])
    after = np.array([FINETUNED[m] for m in metrics])

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars1 = ax.bar(x - width / 2, before, width, label="Pretrained teacher", color="#8da0cb", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, after, width, label="Fine-tuned teacher", color="#66c2a5", edgecolor="black", linewidth=0.8)

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Teacher Retrieval Metrics: Before vs After Fine-tuning")
    ax.set_ylabel("Score")
    ax.set_xticks(x, metrics)
    ax.set_ylim(0.35, 1.0)
    ax.legend(loc="upper left")
    _save(fig, "s08_teacher_before_after_metrics.png")


def fig_slide13_timeline_gantt():
    tasks = [
        ("Baseline + teacher FT + initial eval", "2026-02-24", "2026-03-04", "#a6cee3"),
        ("Lane 1: baseline KD", "2026-03-05", "2026-03-12", "#1f78b4"),
        ("Lane 2: ranking-aware KD", "2026-03-05", "2026-03-12", "#33a02c"),
        ("Lane 3: hard-negative mining", "2026-03-05", "2026-03-12", "#fb9a99"),
        ("Lane 4: benchmark + ablations", "2026-03-05", "2026-03-12", "#ff7f00"),
        ("Joint ablations + val selection", "2026-03-13", "2026-03-19", "#cab2d6"),
        ("Final test + report + slides", "2026-03-20", "2026-04-06", "#b2df8a"),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    y = np.arange(len(tasks))

    for i, (name, s, e, color) in enumerate(tasks):
        s_dt = datetime.fromisoformat(s)
        e_dt = datetime.fromisoformat(e)
        ax.barh(i, (e_dt - s_dt).days + 1, left=s_dt, color=color, edgecolor="black", height=0.62)

    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in tasks])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_title("IU to Final: 4-Lane Parallel Timeline")
    ax.set_xlabel("2026 Timeline")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _save(fig, "s13_parallel_timeline_gantt.png")


def fig_slide16_kd_choices_matrix():
    rows = ["A: DistilCSE-style", "B: EmbedDistill-style", "C: Hybrid (A+B)"]
    cols = ["Implementation\nSimplicity", "Ranking\nAlignment", "Tuning\nLoad", "Expected\nQuality"]
    # Higher is better except tuning load where higher means heavier
    vals = np.array(
        [
            [4, 2, 2, 3],
            [2, 4, 3, 4],
            [3, 5, 5, 5],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    im = ax.imshow(vals, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(rows)), labels=rows)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, int(vals[i, j]), ha="center", va="center", color="black", fontsize=11, weight="bold")

    ax.set_title("Teacher-Student Distillation Design Choice Matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative score (1-5)")
    _save(fig, "s16_kd_design_choice_matrix.png")


def write_index():
    mapping = [
        ("Slide 3", "s03_quality_efficiency_tradeoff.png"),
        ("Slide 5", "s05_pipeline_overview.png"),
        ("Slide 7", "s07_dataset_split_counts.png"),
        ("Slide 7", "s07_dataset_length_distributions.png"),
        ("Slide 7", "s07_retrieval_failure_case_card.png"),
        ("Slide 8", "s08_teacher_before_after_metrics.png"),
        ("Slide 13", "s13_parallel_timeline_gantt.png"),
        ("Slide 16", "s16_kd_design_choice_matrix.png"),
    ]

    lines = ["# IU Visual Pack", "", "Generated by `assignment_details/scripts/generate_iu_visuals.py`.", "", "## Slide Mapping"]
    for slide, filename in mapping:
        lines.append(f"- {slide}: `{filename}`")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main():
    _style()
    fig_slide3_quality_efficiency()
    fig_slide5_pipeline_overview()
    fig_slide7_dataset_split()
    mbpp_stats = _load_mbpp_stats()
    fig_slide7_length_hist(mbpp_stats)
    fig_slide7_failure_case_card()
    fig_slide8_results_bar()
    fig_slide13_timeline_gantt()
    fig_slide16_kd_choices_matrix()
    write_index()

    print(f"Generated visuals in: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
