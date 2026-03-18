"""Fine-tune all baseline models and compare zero-shot vs fine-tuned performance."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

# Disable MPS high watermark to allow using all available memory
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from mbpp_kd_suite.config import DistillTargets, TrainConfig
from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.metrics import score_metrics_from_embeddings
from mbpp_kd_suite.modeling import encode_texts_backbone, infer_model_encoding_spec
from mbpp_kd_suite.runtime import pick_device, set_seed
from mbpp_kd_suite.training import train_student
from transformers import AutoModel, AutoTokenizer

MODELS = [
    # Large models first (most likely to OOM — fail fast)
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
    # Base models
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

SEED = 42
CFG = TrainConfig(
    epochs=8,
    batch_size=32,
    eval_batch_size=64,
    lr=2e-5,
    seed=SEED,
    eval_mode="symmetric",
    run_diagnostics=False,
    save_models=False,
)

# Reduced batch config for large models that OOM on MPS
CFG_LARGE = TrainConfig(
    epochs=8,
    batch_size=4,
    eval_batch_size=8,
    lr=2e-5,
    seed=SEED,
    eval_mode="symmetric",
    run_diagnostics=False,
    save_models=False,
)

LARGE_MODELS = {"BAAI/bge-large-en-v1.5", "intfloat/e5-large-v2"}


def short_name(model: str) -> str:
    return model.split("/")[-1]


def encode_dummy_targets(
    model_name: str, data: Any, cfg: TrainConfig, device: torch.device,
) -> DistillTargets:
    """Encode targets from a reference model. In symmetric eval mode these
    are not used for metric computation but are needed by the function signatures."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    encoding_spec = infer_model_encoding_spec(
        model_name,
        getattr(model.config, "_name_or_path", None),
        getattr(tokenizer, "name_or_path", None),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    encoded: dict[str, torch.Tensor] = {}
    split_prefixes = {"train": "train", "validation": "val", "test": "test"}
    for split_name, split in data.items():
        prefix = split_prefixes[split_name]
        encoded[f"{prefix}_query"] = encode_texts_backbone(
            model=model, tokenizer=tokenizer, texts=split.queries,
            text_role="query", encoding_spec=encoding_spec,
            max_length=cfg.max_query_length, batch_size=cfg.eval_batch_size,
            device=device, desc=f"ref_{prefix}_q",
        )
        encoded[f"{prefix}_doc"] = encode_texts_backbone(
            model=model, tokenizer=tokenizer, texts=split.codes,
            text_role="document", encoding_spec=encoding_spec,
            max_length=cfg.max_code_length, batch_size=cfg.eval_batch_size,
            device=device, desc=f"ref_{prefix}_d",
        )
    del model
    torch.mps.empty_cache() if device.type == "mps" else None
    return DistillTargets(name="ref_targets", **encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and compare all models")
    parser.add_argument("--dataset-name", default="google-research-datasets/mbpp")
    parser.add_argument("--taco-val-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training set size")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: 32)")
    args = parser.parse_args()

    set_seed(SEED)
    device = pick_device()
    print(f"Device: {device}")

    dataset_name = args.dataset_name
    dataset_slug = dataset_name.split("/")[-1]
    print(f"Loading dataset: {dataset_name}")
    dataset = load_retrieval_dataset(
        dataset_name=dataset_name, taco_val_size=args.taco_val_size, seed=SEED,
    )
    data = dataset_dict_to_splits(dataset)

    # Truncate training set if requested
    if args.max_train_samples and len(data.train.queries) > args.max_train_samples:
        from mbpp_kd_suite.config import RetrievalSplit, RetrievalSplits
        n = args.max_train_samples
        data = RetrievalSplits(
            train=RetrievalSplit(queries=data.train.queries[:n], codes=data.train.codes[:n]),
            validation=data.validation,
            test=data.test,
        )

    print(
        f"Splits -> train: {len(data.train.queries)}, "
        f"val: {len(data.validation.queries)}, test: {len(data.test.queries)}"
    )

    # Apply overrides to configs
    if args.epochs:
        CFG.epochs = args.epochs
        CFG_LARGE.epochs = args.epochs
    if args.batch_size:
        CFG.batch_size = args.batch_size
        CFG.eval_batch_size = args.batch_size * 2
        CFG_LARGE.batch_size = args.batch_size
        CFG_LARGE.eval_batch_size = args.batch_size * 2

    # Load zero-shot results from previous run
    zs_path = Path(f"artifacts/baseline_comparison_{dataset_slug}/results.json")
    if zs_path.exists():
        with zs_path.open() as f:
            zs_results = json.load(f)
        print(f"Loaded zero-shot results from {zs_path}")
    else:
        print(f"WARNING: No zero-shot results found. Run: mbpp-kd-baselines --dataset-name {dataset_name}")
        zs_results = {}

    # We need DistillTargets for the train_student signatures.
    # In symmetric eval mode these are unused for metrics, but
    # the validation loop still passes them through.
    # Encode from the smallest model to save time.
    print("Encoding reference targets (MiniLM-L6, for function signatures only)...")
    ref_targets = encode_dummy_targets(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        data=data, cfg=CFG, device=device,
    )

    out_dir = Path(f"artifacts/finetune_comparison_{dataset_slug}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Check for already-completed results from prior runs
    ft_results: dict[str, dict[str, Any]] = {}
    existing_runs = sorted(out_dir.iterdir())
    for prior_run in existing_runs:
        if not prior_run.is_dir() or prior_run == run_dir:
            continue
        for subdir in prior_run.iterdir():
            metrics_file = subdir / "metrics.json"
            if metrics_file.exists() and subdir.name.startswith("finetuned_"):
                slug = subdir.name.removeprefix("finetuned_")
                with metrics_file.open() as f:
                    ft_results[slug] = json.load(f)
                print(f"Loaded cached result for {slug} from {metrics_file}")

    # On CUDA (Colab), large models fit fine with full batch size
    use_large_cfg = device.type == "mps"

    for model_name in MODELS:
        slug = short_name(model_name)
        if slug in ft_results:
            print(f"Skipping {model_name} (already have results)")
            continue

        model_cfg = CFG_LARGE if (use_large_cfg and model_name in LARGE_MODELS) else CFG
        print(f"\n{'='*60}")
        print(f"Fine-tuning: {model_name} (batch_size={model_cfg.batch_size})")
        print(f"{'='*60}")
        t0 = time.time()

        metrics, _, _ = train_student(
            name=f"finetuned_{slug}",
            cfg=model_cfg,
            run_dir=run_dir,
            device=device,
            data=data,
            targets=ref_targets,
            full_teacher_targets=ref_targets,
            model_name=model_name,
            supervised=True,
        )
        elapsed = time.time() - t0
        ft_results[slug] = metrics
        test = metrics["test"]
        print(
            f"  test | MRR={test['MRR']:.4f} | R@1={test['Recall@1']:.4f} | "
            f"R@5={test['Recall@5']:.4f} | R@10={test['Recall@10']:.4f} | "
            f"MedianRank={test['MedianRank']:.1f} | {elapsed:.1f}s"
        )

        # Aggressive cleanup to free MPS memory before next model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Save fine-tuned results
    with (out_dir / "finetuned_results.json").open("w") as f:
        json.dump(ft_results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("ZERO-SHOT vs FINE-TUNED COMPARISON (test split)")
    print("=" * 95)
    header = (
        f"{'Model':<30} {'ZS MRR':>7} {'FT MRR':>7} {'Delta':>7} "
        f"{'ZS R@1':>7} {'FT R@1':>7} {'ZS R@10':>7} {'FT R@10':>7}"
    )
    print(header)
    print("-" * len(header))
    for slug in [short_name(m) for m in MODELS]:
        zs = zs_results.get(slug, {}).get("test", {})
        ft = ft_results.get(slug, {}).get("test", {})
        zs_mrr = zs.get("MRR", 0)
        ft_mrr = ft.get("MRR", 0)
        delta = ft_mrr - zs_mrr
        print(
            f"{slug:<30} {zs_mrr:>7.4f} {ft_mrr:>7.4f} {delta:>+7.4f} "
            f"{zs.get('Recall@1', 0):>7.4f} {ft.get('Recall@1', 0):>7.4f} "
            f"{zs.get('Recall@10', 0):>7.4f} {ft.get('Recall@10', 0):>7.4f}"
        )

    # ── Chart: grouped bars (zero-shot vs fine-tuned) ─────────────
    slugs = [short_name(m) for m in MODELS]
    metric_keys = ["MRR", "Recall@1", "Recall@5", "Recall@10"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    bar_width = 0.35
    x = np.arange(len(slugs))

    for ax_idx, metric in enumerate(metric_keys):
        ax = axes[ax_idx]
        zs_vals = [zs_results.get(s, {}).get("test", {}).get(metric, 0) for s in slugs]
        ft_vals = [ft_results.get(s, {}).get("test", {}).get(metric, 0) for s in slugs]

        bars_zs = ax.bar(x - bar_width / 2, zs_vals, bar_width, label="Zero-shot",
                         color="#a8d8ea", edgecolor="white", linewidth=0.5)
        bars_ft = ax.bar(x + bar_width / 2, ft_vals, bar_width, label="Fine-tuned",
                         color="#ff6b6b", edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars_zs, zs_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="#555")
        for bar, val in zip(bars_ft, ft_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="#c0392b")

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, rotation=35, ha="right", fontsize=8)
        all_vals = zs_vals + ft_vals
        ax.set_ylim(min(all_vals) * 0.9, min(max(all_vals) * 1.08, 1.0))
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        f"Zero-Shot vs Fine-Tuned (8 epochs, supervised contrastive) on {dataset_slug.upper()} test",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = out_dir / "finetune_comparison.png"
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ── Chart 2: delta (improvement from fine-tuning) ─────────────
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    deltas_mrr = [
        ft_results.get(s, {}).get("test", {}).get("MRR", 0)
        - zs_results.get(s, {}).get("test", {}).get("MRR", 0)
        for s in slugs
    ]
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas_mrr]
    bars = ax2.bar(x, deltas_mrr, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, deltas_mrr):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.002 if val >= 0 else -0.008),
                 f"{val:+.4f}", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(slugs, rotation=35, ha="right", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("MRR Delta (fine-tuned - zero-shot)")
    ax2.set_title("MRR Improvement from Fine-Tuning", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    delta_path = out_dir / "finetune_delta.png"
    fig2.savefig(delta_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nCharts saved to:")
    print(f"  {chart_path}")
    print(f"  {delta_path}")
    print(f"Per-model artifacts saved to: {run_dir}")
    print(f"Results JSON: {out_dir / 'finetuned_results.json'}")


if __name__ == "__main__":
    main()
