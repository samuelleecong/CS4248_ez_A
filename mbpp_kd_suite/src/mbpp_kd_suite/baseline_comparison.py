"""Zero-shot baseline comparison across all teacher/student candidates."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.metrics import evaluate_symmetric_backbone
from mbpp_kd_suite.runtime import pick_device, set_seed

MODELS = [
    # student
    "sentence-transformers/all-MiniLM-L6-v2",
    # default teacher
    "sentence-transformers/all-MiniLM-L12-v2",
    # kai candidates
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    # large teachers
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
]

SEED = 42
MAX_QUERY_LENGTH = 160
MAX_CODE_LENGTH = 256
EVAL_BATCH_SIZE = 64


def short_name(model: str) -> str:
    return model.split("/")[-1]


def main() -> None:
    set_seed(SEED)
    device = pick_device()
    print(f"Device: {device}")

    print("Loading MBPP dataset...")
    dataset = load_retrieval_dataset(
        dataset_name="google-research-datasets/mbpp",
        taco_val_size=1000,
        seed=SEED,
    )
    data = dataset_dict_to_splits(dataset)
    print(
        f"Splits -> train: {len(data.train.queries)}, "
        f"val: {len(data.validation.queries)}, test: {len(data.test.queries)}"
    )

    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in MODELS:
        slug = short_name(model_name)
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        t0 = time.time()
        metrics = evaluate_symmetric_backbone(
            model_name=model_name,
            val_queries=data.validation.queries,
            val_codes=data.validation.codes,
            test_queries=data.test.queries,
            test_codes=data.test.codes,
            max_query_length=MAX_QUERY_LENGTH,
            max_code_length=MAX_CODE_LENGTH,
            eval_batch_size=EVAL_BATCH_SIZE,
            device=device,
        )
        elapsed = time.time() - t0
        results[slug] = metrics
        test = metrics["test"]
        print(
            f"  test | MRR={test['MRR']:.4f} | R@1={test['Recall@1']:.4f} | "
            f"R@5={test['Recall@5']:.4f} | R@10={test['Recall@10']:.4f} | "
            f"MedianRank={test['MedianRank']:.1f} | {elapsed:.1f}s"
        )

    # Save raw results
    out_dir = Path("artifacts/baseline_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("ZERO-SHOT BASELINE COMPARISON (test split)")
    print("=" * 80)
    header = f"{'Model':<35} {'MRR':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MedRk':>6}"
    print(header)
    print("-" * len(header))
    for slug, metrics in results.items():
        t = metrics["test"]
        print(
            f"{slug:<35} {t['MRR']:>6.4f} {t['Recall@1']:>6.4f} "
            f"{t['Recall@5']:>6.4f} {t['Recall@10']:>6.4f} {t['MedianRank']:>6.1f}"
        )

    # ── Plot ──────────────────────────────────────────────────────
    model_names = list(results.keys())
    test_metrics = {slug: results[slug]["test"] for slug in model_names}

    metric_keys = ["MRR", "Recall@1", "Recall@5", "Recall@10"]
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6), sharey=True)

    # Color models by category
    colors = []
    for name in model_names:
        if "MiniLM-L6" in name:
            colors.append("#66b3ff")     # student - light blue
        elif "MiniLM-L12" in name:
            colors.append("#3399ff")     # default teacher - blue
        elif "large" in name:
            colors.append("#ff6b35")     # large models - orange
        else:
            colors.append("#2ecc71")     # base candidates - green

    for ax_idx, metric in enumerate(metric_keys):
        ax = axes[ax_idx]
        values = [test_metrics[slug][metric] for slug in model_names]
        bars = ax.barh(range(n_models), values, color=colors, edgecolor="white", linewidth=0.5)

        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=8)

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(values) * 1.15)
        ax.set_yticks(range(n_models))
        if ax_idx == 0:
            ax.set_yticklabels(model_names, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#66b3ff", label="Student"),
        Patch(facecolor="#3399ff", label="Default teacher"),
        Patch(facecolor="#2ecc71", label="Base candidates"),
        Patch(facecolor="#ff6b35", label="Large models"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9, frameon=False)

    fig.suptitle("Zero-Shot Baseline Comparison on MBPP (test split)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    chart_path = out_dir / "baseline_comparison.png"
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved to: {chart_path}")
    print(f"Results saved to: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
