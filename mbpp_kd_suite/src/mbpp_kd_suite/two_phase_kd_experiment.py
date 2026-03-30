"""Two-phase KD experiment.

Phase 1: Finetune both teacher (all-mpnet-base-v2) and student (all-MiniLM-L6-v2)
         for N epochs using supervised contrastive loss.

Phase 2: Starting from the finetuned student weights, run:
         - Control: supervised finetuning (no KD)
         - All KD methods: knowledge distillation from the finetuned teacher

This lets us isolate whether KD adds value on top of a pre-finetuned student.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer

from .config import DistillTargets, TrainConfig, resolve_output_root
from .constants import KD_METHOD_ORDER
from .data import dataset_dict_to_splits, load_retrieval_dataset
from .metrics import evaluate_symmetric_backbone
from .modeling import encode_texts_backbone, infer_model_encoding_spec
from .runtime import apply_device_runtime_optimizations, maybe_empty_device_cache, pick_device, set_seed
from .training import make_method_targets, train_student

TWO_PHASE_OUTPUT_DIR = "two_phase_kd"


def _encode_as_distill_targets(
    name: str,
    backbone: Any,
    tokenizer: AutoTokenizer,
    encoding_spec: Any,
    data: Any,
    cfg: TrainConfig,
    device: Any,
) -> DistillTargets:
    split_prefixes = {"train": "train", "validation": "val", "test": "test"}
    encoded: dict[str, Any] = {}
    for split_name, split in data.items():
        prefix = split_prefixes[split_name]
        encoded[f"{prefix}_query"] = encode_texts_backbone(
            model=backbone,
            tokenizer=tokenizer,
            texts=split.queries,
            text_role="query",
            encoding_spec=encoding_spec,
            max_length=cfg.max_query_length,
            batch_size=cfg.eval_batch_size,
            device=device,
            desc=f"{name}_{prefix}_q",
        )
        encoded[f"{prefix}_doc"] = encode_texts_backbone(
            model=backbone,
            tokenizer=tokenizer,
            texts=split.codes,
            text_role="document",
            encoding_spec=encoding_spec,
            max_length=cfg.max_code_length,
            batch_size=cfg.eval_batch_size,
            device=device,
            desc=f"{name}_{prefix}_d",
        )
        maybe_empty_device_cache(device)
    return DistillTargets(name=name, **encoded)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _plot_phase1_training(run_dir: Path) -> None:
    """Plot train loss and val MRR per epoch for teacher and student phase 1 runs."""
    runs = {
        "teacher": run_dir / "phase1" / "ft_teacher" / "history.json",
        "student": run_dir / "phase1" / "ft_student" / "history.json",
    }
    histories: dict[str, list[dict]] = {}
    for label, path in runs.items():
        if path.exists():
            with path.open() as f:
                histories[label] = json.load(f)

    if not histories:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"teacher": "#8e44ad", "student": "#95a5a6"}

    for label, history in histories.items():
        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        val_mrrs = [h["val_MRR"] for h in history]
        color = colors[label]
        ax1.plot(epochs, losses, marker="o", label=label, color=color)
        ax2.plot(epochs, val_mrrs, marker="o", label=label, color=color)
        # Mark best val MRR
        best_idx = int(np.argmax(val_mrrs))
        ax2.scatter([epochs[best_idx]], [val_mrrs[best_idx]], color=color, s=120, zorder=5, marker="*")

    ax1.set_title("Phase 1: Train Loss per Epoch", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Phase 1: Val MRR per Epoch", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val MRR")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    chart_path = run_dir / "phase1_training_curves.png"
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Phase 1 training curves saved to: {chart_path}")


def _plot_results(results: dict[str, Any], run_dir: Path, dataset_name: str) -> None:
    dataset_slug = dataset_name.split("/")[-1]
    metric_keys = ["MRR", "Recall@1", "Recall@5", "Recall@10"]

    # Build ordered list of runs to display
    display_order = [
        "zeroshot_teacher", "zeroshot_student",
        "ft_teacher", "ft_student", "control_supervised",
    ] + list(KD_METHOD_ORDER)
    short_map = {
        "zeroshot_teacher": "zs_teacher",
        "zeroshot_student": "zs_student",
        "control_supervised": "ft_student_nodistill",
    }
    labels = []
    for key in display_order:
        if key not in results:
            continue
        short = short_map.get(key, key)
        labels.append((key, short))

    keys = [k for k, _ in labels]
    short_labels = [s for _, s in labels]
    x = np.arange(len(keys))

    # Colour palette
    colors = []
    for k in keys:
        if k in ("zeroshot_teacher", "zeroshot_student"):
            colors.append("#bdc3c7")
        elif k == "ft_teacher":
            colors.append("#8e44ad")
        elif k == "ft_student":
            colors.append("#95a5a6")
        elif k == "control_supervised":
            colors.append("#3498db")
        else:
            colors.append("#e67e22")

    # ── Chart 1: 2×2 metric comparison ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(max(14, len(keys) * 1.1), 10))
    axes = axes.flatten()

    for ax_idx, metric in enumerate(metric_keys):
        ax = axes[ax_idx]
        vals = [results[k]["test"].get(metric, 0) for k in keys]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="#333",
            )
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(min(vals) * 0.92, min(max(vals) * 1.10, 1.0))
        ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#bdc3c7", label="Zero-shot baseline"),
        Patch(color="#8e44ad", label="Phase 1 finetuned teacher"),
        Patch(color="#95a5a6", label="Phase 1 finetuned student"),
        Patch(color="#3498db", label="Phase 2 control (supervised)"),
        Patch(color="#e67e22", label="Phase 2 KD methods"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Two-Phase KD: Phase 2 results on {dataset_slug.upper()} test split\n"
        f"(all Phase 2 runs init from finetuned student)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    chart_path = run_dir / "two_phase_kd_comparison.png"
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ── Chart 2: MRR delta vs phase2 control ──────────────────────
    control_mrr = results.get("control_supervised", {}).get("test", {}).get("MRR", 0)
    kd_keys = [k for k in keys if k in set(KD_METHOD_ORDER)]
    kd_labels = [s for k, s in labels if k in kd_keys]
    kd_deltas = [results[k]["test"].get("MRR", 0) - control_mrr for k in kd_keys]

    if kd_keys:
        fig2, ax2 = plt.subplots(figsize=(max(10, len(kd_keys) * 1.2), 5))
        bar_colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in kd_deltas]
        bars2 = ax2.bar(np.arange(len(kd_keys)), kd_deltas, color=bar_colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars2, kd_deltas):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.001 if val >= 0 else -0.007),
                f"{val:+.4f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold",
            )
        ax2.set_xticks(np.arange(len(kd_keys)))
        ax2.set_xticklabels(kd_labels, rotation=35, ha="right", fontsize=9)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("MRR delta vs Phase 2 control (supervised)")
        ax2.set_title("KD Methods: MRR Improvement over Supervised Control", fontsize=13, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        delta_path = run_dir / "two_phase_kd_delta.png"
        fig2.savefig(delta_path, dpi=180, bbox_inches="tight")
        plt.close(fig2)
        print(f"  {delta_path}")

    print(f"Charts saved to:")
    print(f"  {chart_path}")


def _print_results(results: dict[str, Any]) -> None:
    print("\n=== Zero-Shot Baselines ===")
    for key in ["zeroshot_teacher", "zeroshot_student"]:
        if key not in results:
            continue
        m = results[key]["test"]
        print(
            f"{key:>35} | MRR={m['MRR']:.4f} | R@1={m['Recall@1']:.4f} | "
            f"R@5={m['Recall@5']:.4f} | R@10={m['Recall@10']:.4f}"
        )

    print("\n=== Phase 1 Results ===")
    for key in ["ft_teacher", "ft_student"]:
        if key not in results:
            continue
        m = results[key]["test"]
        print(
            f"{key:>35} | MRR={m['MRR']:.4f} | R@1={m['Recall@1']:.4f} | "
            f"R@5={m['Recall@5']:.4f} | R@10={m['Recall@10']:.4f}"
        )

    print("\n=== Phase 2 Results (initialized from finetuned student) ===")
    phase2_keys = ["control_supervised"] + [k for k in KD_METHOD_ORDER if k in results]
    for key in phase2_keys:
        m = results[key]["test"]
        print(
            f"{key:>35} | MRR={m['MRR']:.4f} | R@1={m['Recall@1']:.4f} | "
            f"R@5={m['Recall@5']:.4f} | R@10={m['Recall@10']:.4f}"
        )


def run(
    teacher_model: str = "sentence-transformers/all-mpnet-base-v2",
    student_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    dataset_name: str = "code_search_net",
    phase1_epochs: int = 3,
    phase2_epochs: int = 2,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    lr: float = 2e-5,
    seed: int = 42,
    temperature: float = 0.05,
    distill_temperature: float = 4.0,
    distill_weight: float = 1.0,
    align_weight: float = 1.0,
    pair_weight: float = 1.0,
    relation_weight: float = 1.0,
    output_dir: str = TWO_PHASE_OUTPUT_DIR,
    skip_diagnostics: bool = False,
    methods: tuple[str, ...] | None = None,
    taco_val_size: int = 1000,
    resume_from_phase1: str | None = None,
    phase1_patience: int = 3,
    phase2_patience: int = 3,
) -> dict[str, Any]:
    if methods is None:
        methods = tuple(KD_METHOD_ORDER)

    set_seed(seed)
    device = pick_device()
    print(f"Device: {device}")

    output_root = resolve_output_root(output_dir)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = run_dir / "tensorboard"
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"  Run: tensorboard --logdir {tb_log_dir}")

    print(f"Loading dataset: {dataset_name}")
    dataset = load_retrieval_dataset(dataset_name=dataset_name, taco_val_size=taco_val_size, seed=seed)
    data = dataset_dict_to_splits(dataset)
    print(
        f"Splits -> train: {len(data.train.queries)}, "
        f"val: {len(data.validation.queries)}, test: {len(data.test.queries)}"
    )

    phase1_cfg = TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset_name=dataset_name,
        epochs=phase1_epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        lr=lr,
        seed=seed,
        temperature=temperature,
        distill_temperature=distill_temperature,
        distill_weight=distill_weight,
        align_weight=align_weight,
        pair_weight=pair_weight,
        relation_weight=relation_weight,
        run_diagnostics=not skip_diagnostics,
        output_dir=output_dir,
        early_stopping_patience=phase1_patience,
        save_models=True,
    )
    phase2_cfg = TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset_name=dataset_name,
        epochs=phase2_epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        lr=lr,
        seed=seed,
        temperature=temperature,
        distill_temperature=distill_temperature,
        distill_weight=distill_weight,
        align_weight=align_weight,
        pair_weight=pair_weight,
        relation_weight=relation_weight,
        run_diagnostics=not skip_diagnostics,
        output_dir=output_dir,
        save_models=True,
        early_stopping_patience=phase2_patience,
    )
    apply_device_runtime_optimizations(cfg=phase1_cfg, device=device)
    apply_device_runtime_optimizations(cfg=phase2_cfg, device=device)

    # --- Resume from a saved phase 1 checkpoint if requested ---
    if resume_from_phase1 is not None:
        ckpt_path = Path(resume_from_phase1)
        print(f"\nLoading phase 1 checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        phase1_student_backbone_state: dict[str, torch.Tensor] = ckpt["student_backbone_state"]
        ft_teacher_targets = DistillTargets(**ckpt["ft_teacher_targets"])
        zs_teacher = ckpt.get("zeroshot_teacher", {})
        zs_student = ckpt.get("zeroshot_student", {})
        ft_teacher_metrics = ckpt.get("ft_teacher_metrics", {})
        ft_student_metrics = ckpt.get("ft_student_metrics", {})
    else:
        phase1_student_backbone_state = None
        ft_teacher_targets = None

    if resume_from_phase1 is None:
        # Zero-shot baselines (before any training)
        print("\nEvaluating zero-shot baselines...")
        zs_teacher = evaluate_symmetric_backbone(
            model_name=teacher_model,
            val_queries=data.validation.queries, val_codes=data.validation.codes,
            test_queries=data.test.queries, test_codes=data.test.codes,
            max_query_length=128, max_code_length=256,
            eval_batch_size=eval_batch_size, device=device,
        )
        zs_student = evaluate_symmetric_backbone(
            model_name=student_model,
            val_queries=data.validation.queries, val_codes=data.validation.codes,
            test_queries=data.test.queries, test_codes=data.test.codes,
            max_query_length=128, max_code_length=256,
            eval_batch_size=eval_batch_size, device=device,
        )
        maybe_empty_device_cache(device)

        # Encode raw (zero-shot) teacher targets for use during phase 1 teacher training
        print(f"\nEncoding raw teacher targets: {teacher_model}")
        raw_teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
        raw_teacher_backbone = AutoModel.from_pretrained(teacher_model).to(device)
        raw_teacher_encoding_spec = infer_model_encoding_spec(teacher_model)
        raw_teacher_backbone.eval()
        for p in raw_teacher_backbone.parameters():
            p.requires_grad_(False)

        raw_teacher_targets = _encode_as_distill_targets(
            name="raw_teacher",
            backbone=raw_teacher_backbone,
            tokenizer=raw_teacher_tokenizer,
            encoding_spec=raw_teacher_encoding_spec,
            data=data,
            cfg=phase1_cfg,
            device=device,
        )
        del raw_teacher_backbone
        maybe_empty_device_cache(device)

        # --- Phase 1: Finetune teacher (supervised contrastive, 3 epochs) ---
        print(f"\n=== Phase 1: Finetuning teacher ({teacher_model}, {phase1_epochs} epochs) ===")
        ft_teacher_metrics, ft_teacher_model, ft_teacher_tokenizer = train_student(
            name="ft_teacher",
            cfg=phase1_cfg,
            run_dir=run_dir / "phase1",
            device=device,
            data=data,
            targets=raw_teacher_targets,
            full_teacher_targets=raw_teacher_targets,
            model_name=teacher_model,
            supervised=True,
            tb_writer=tb_writer,
        )

        # Re-encode targets using the finetuned teacher (these become the KD distillation targets)
        print("\nEncoding KD targets from finetuned teacher...")
        ft_teacher_backbone = ft_teacher_model.backbone
        ft_teacher_backbone.eval()
        for p in ft_teacher_backbone.parameters():
            p.requires_grad_(False)

        ft_teacher_targets = _encode_as_distill_targets(
            name="finetuned_teacher",
            backbone=ft_teacher_backbone,
            tokenizer=ft_teacher_tokenizer,
            encoding_spec=ft_teacher_model.encoding_spec,
            data=data,
            cfg=phase1_cfg,
            device=device,
        )
        del ft_teacher_backbone, ft_teacher_model
        maybe_empty_device_cache(device)

        # --- Phase 1: Finetune student (supervised contrastive, 3 epochs) ---
        print(f"\n=== Phase 1: Finetuning student ({student_model}, {phase1_epochs} epochs) ===")
        # Eval uses symmetric mode so fixed_doc_embs are not used; ft_teacher_targets supplies shape info for diagnostics
        ft_student_metrics, ft_student_model, _ = train_student(
            name="ft_student",
            cfg=phase1_cfg,
            run_dir=run_dir / "phase1",
            device=device,
            data=data,
            targets=ft_teacher_targets,
            full_teacher_targets=ft_teacher_targets,
            model_name=student_model,
            supervised=True,
            tb_writer=tb_writer,
        )

        # Capture finetuned student backbone weights for phase 2 initialization
        phase1_student_backbone_state: dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone() for k, v in ft_student_model.backbone.state_dict().items()
        }
        del ft_student_model
        maybe_empty_device_cache(device)

        # Plot phase 1 training curves (loss + val MRR per epoch)
        _plot_phase1_training(run_dir)

        # Save phase 1 checkpoint so phase 2 can be resumed without re-running phase 1
        phase1_ckpt_path = run_dir / "phase1" / "checkpoint.pt"
        phase1_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student_backbone_state": phase1_student_backbone_state,
                "ft_teacher_targets": {
                    "name": ft_teacher_targets.name,
                    "train_query": ft_teacher_targets.train_query,
                    "train_doc": ft_teacher_targets.train_doc,
                    "val_query": ft_teacher_targets.val_query,
                    "val_doc": ft_teacher_targets.val_doc,
                    "test_query": ft_teacher_targets.test_query,
                    "test_doc": ft_teacher_targets.test_doc,
                },
                "zeroshot_teacher": zs_teacher,
                "zeroshot_student": zs_student,
                "ft_teacher_metrics": ft_teacher_metrics,
                "ft_student_metrics": ft_student_metrics,
            },
            phase1_ckpt_path,
        )
        print(f"Phase 1 checkpoint saved to: {phase1_ckpt_path}")

        # Write config.json early so resume_phase2.py can read model names
        _write_json(run_dir / "config.json", {
            "teacher_model": teacher_model,
            "student_model": student_model,
            "dataset_name": dataset_name,
            "phase1_epochs": phase1_epochs,
            "phase1_patience": phase1_patience,
            "phase2_epochs": phase2_epochs,
            "phase2_patience": phase2_patience,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "lr": lr,
            "seed": seed,
            "temperature": temperature,
            "distill_temperature": distill_temperature,
            "distill_weight": distill_weight,
            "align_weight": align_weight,
            "pair_weight": pair_weight,
            "relation_weight": relation_weight,
            "methods": list(methods),
        })

    # Build method-specific distillation targets (e.g., HPD applies PCA compression)
    method_targets = make_method_targets(cfg=phase2_cfg, full_teacher_targets=ft_teacher_targets)

    results: dict[str, Any] = {
        "zeroshot_teacher": zs_teacher,
        "zeroshot_student": zs_student,
        "ft_teacher": ft_teacher_metrics,
        "ft_student": ft_student_metrics,
    }

    # --- Phase 2: Control — supervised finetuning (no KD) from phase 1 student ---
    print(f"\n=== Phase 2: Control - supervised ({phase2_epochs} epochs, init from phase1 student) ===")
    control_metrics, _, _ = train_student(
        name="control_supervised",
        cfg=phase2_cfg,
        run_dir=run_dir / "phase2",
        device=device,
        data=data,
        targets=ft_teacher_targets,
        full_teacher_targets=ft_teacher_targets,
        model_name=student_model,
        supervised=True,
        initial_backbone_state_dict=phase1_student_backbone_state,
        tb_writer=tb_writer,
    )
    results["control_supervised"] = control_metrics

    # --- Phase 2: KD methods from phase 1 student ---
    for method_name in methods:
        print(f"\n=== Phase 2: KD '{method_name}' ({phase2_epochs} epochs, init from phase1 student) ===")
        kd_metrics, _, _ = train_student(
            name=method_name,
            cfg=phase2_cfg,
            run_dir=run_dir / "phase2",
            device=device,
            data=data,
            targets=method_targets[method_name],
            full_teacher_targets=ft_teacher_targets,
            initial_backbone_state_dict=phase1_student_backbone_state,
            tb_writer=tb_writer,
        )
        results[method_name] = kd_metrics

    config_payload = {
        "teacher_model": teacher_model,
        "student_model": student_model,
        "dataset_name": dataset_name,
        "phase1_epochs": phase1_epochs,
        "phase1_patience": phase1_patience,
        "phase2_epochs": phase2_epochs,
        "phase2_patience": phase2_patience,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "lr": lr,
        "seed": seed,
        "temperature": temperature,
        "distill_temperature": distill_temperature,
        "distill_weight": distill_weight,
        "align_weight": align_weight,
        "pair_weight": pair_weight,
        "relation_weight": relation_weight,
        "methods": list(methods),
    }
    _write_json(run_dir / "config.json", config_payload)
    _write_json(run_dir / "results_summary.json", results)

    _plot_results(results, run_dir, dataset_name)
    tb_writer.close()
    print(f"\nArtifacts saved to: {run_dir}")
    print(f"TensorBoard logs: {tb_log_dir}")
    _print_results(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase KD: finetune teacher+student, then distill from finetuned teacher"
    )
    parser.add_argument(
        "--teacher-model", default="sentence-transformers/all-mpnet-base-v2",
        help="Teacher model (default: all-mpnet-base-v2)",
    )
    parser.add_argument(
        "--student-model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="Student model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument("--dataset-name", default="code_search_net")
    parser.add_argument("--phase1-epochs", type=int, default=20, help="Max epochs for phase 1 (early stopping will terminate sooner)")
    parser.add_argument("--phase1-patience", type=int, default=3, help="Early stopping patience for phase 1 (0 = disabled)")
    parser.add_argument("--phase2-epochs", type=int, default=10, help="Max epochs for phase 2 (early stopping will terminate sooner)")
    parser.add_argument("--phase2-patience", type=int, default=3, help="Early stopping patience for phase 2 (0 = disabled)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Contrastive (one_hot) temperature (default: 0.05)")
    parser.add_argument("--distill-temperature", type=float, default=4.0,
                        help="Temperature for KD softmax (higher = softer teacher distributions, default: 4.0)")
    parser.add_argument("--distill-weight", type=float, default=1.0,
                        help="Weight for distill_kl and dark_kl losses (default: 1.0)")
    parser.add_argument("--align-weight", type=float, default=1.0,
                        help="Weight for embedding alignment loss (default: 1.0)")
    parser.add_argument("--pair-weight", type=float, default=1.0,
                        help="Weight for pairwise preference loss (default: 1.0)")
    parser.add_argument("--relation-weight", type=float, default=1.0,
                        help="Weight for relation/contrastive KD loss (default: 1.0)")
    parser.add_argument("--output-dir", default=TWO_PHASE_OUTPUT_DIR)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--taco-val-size", type=int, default=1000)
    parser.add_argument(
        "--methods",
        default=",".join(KD_METHOD_ORDER),
        help="Comma-separated list of KD methods to run in phase 2",
    )
    parser.add_argument(
        "--resume-from-phase1",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a phase1/checkpoint.pt from a previous run; skips phase 1 entirely",
    )
    args = parser.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    run(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        dataset_name=args.dataset_name,
        phase1_epochs=args.phase1_epochs,
        phase1_patience=args.phase1_patience,
        phase2_epochs=args.phase2_epochs,
        phase2_patience=args.phase2_patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        seed=args.seed,
        temperature=args.temperature,
        distill_temperature=args.distill_temperature,
        distill_weight=args.distill_weight,
        align_weight=args.align_weight,
        pair_weight=args.pair_weight,
        relation_weight=args.relation_weight,
        output_dir=args.output_dir,
        skip_diagnostics=args.skip_diagnostics,
        methods=methods,
        taco_val_size=args.taco_val_size,
        resume_from_phase1=args.resume_from_phase1,
    )
