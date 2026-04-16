"""Run bimga vs bimga_uniform at same hyperparameters (dw=100, aw=10) to saturation.

This directly compares:
- bimga: margin-weighted bidirectional alignment
- bimga_uniform: uniform-weighted bidirectional alignment

Both at dw=100, aw=10, 200 epochs, patience=15

Usage:
    uv run python run_bimga_comparison.py
    uv run python run_bimga_comparison.py --resume path/to/dir
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from mbpp_kd_suite.config import DistillTargets, TrainConfig, resolve_output_root
from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.runtime import (
    apply_device_runtime_optimizations,
    maybe_empty_device_cache,
    pick_device,
    set_seed,
)
from mbpp_kd_suite.training import make_method_targets, train_student


# ── Phase 1 checkpoint ────────────────────────────────────────────────────────
CHECKPOINT = "artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt"

# ── Saturation constants ──────────────────────────────────────────────────────
SATURATION_EPOCHS = 200
SATURATION_PATIENCE = 15
EVAL_BATCH_SIZE = 64
BASE_SEED = 42


@dataclass
class RunConfig:
    """A single training run."""
    name: str
    method: str
    distill_weight: float = 100.0
    align_weight: float = 10.0
    distill_temperature: float = 0.2


def get_existing_mapping(source_dir: Path) -> dict[str, str]:
    """Check if we have existing runs to copy.

    Returns: {new_run_name: existing_run_path}
    """
    mapping = {}

    # Look for bimga_uniform from Set 8 (already saturated)
    bimga_uniform_path = source_dir / "s8_A2_bimga_uniform"
    if bimga_uniform_path.exists():
        # Check if it has metrics
        for method_dir in bimga_uniform_path.iterdir():
            if method_dir.is_dir() and (method_dir / "metrics.json").exists():
                mapping["s14_bimga_uniform_dw100_aw10"] = bimga_uniform_path
                break

    return mapping


def copy_existing_run(source_dir: Path, target_dir: Path, run_name: str) -> bool:
    """Copy an already-completed run."""
    if not source_dir.exists():
        return False

    # Check if source has metrics.json
    metrics_file = None
    for method_dir in source_dir.iterdir():
        if method_dir.is_dir():
            potential_metrics = method_dir / "metrics.json"
            if potential_metrics.exists():
                metrics_file = potential_metrics
                break

    if not metrics_file:
        return False

    print(f"  Copying existing run: {source_dir.name} -> {run_name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    return True


def run_single(
    rc: RunConfig,
    *,
    run_dir: Path,
    device: torch.device,
    data,
    ft_teacher_targets: DistillTargets,
    student_backbone_state: dict,
    teacher_model: str,
    student_model: str,
    dataset_name: str,
    tb_writer: SummaryWriter,
) -> dict:
    """Run a single training experiment."""
    cfg = TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset_name=dataset_name,
        epochs=SATURATION_EPOCHS,
        batch_size=32,
        eval_batch_size=EVAL_BATCH_SIZE,
        lr=2e-5,
        seed=BASE_SEED,
        distill_temperature=rc.distill_temperature,
        distill_weight=rc.distill_weight,
        align_weight=rc.align_weight,
        pair_weight=1.0,
        save_models=True,
        run_diagnostics=True,
        early_stopping_patience=SATURATION_PATIENCE,
    )
    apply_device_runtime_optimizations(cfg=cfg, device=device)

    exp_dir = run_dir / rc.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with (exp_dir / "run_config.json").open("w") as f:
        json.dump({
            "name": rc.name,
            "method": rc.method,
            "batch_size": 32,
            "seed": BASE_SEED,
            "distill_weight": rc.distill_weight,
            "align_weight": rc.align_weight,
            "pair_weight": 1.0,
            "distill_temperature": rc.distill_temperature,
        }, f, indent=2)

    method_targets = make_method_targets(cfg=cfg, full_teacher_targets=ft_teacher_targets)
    metrics, _, _ = train_student(
        name=rc.method,
        cfg=cfg,
        run_dir=exp_dir,
        device=device,
        data=data,
        targets=method_targets[rc.method],
        full_teacher_targets=ft_teacher_targets,
        initial_backbone_state_dict=student_backbone_state,
        tb_writer=tb_writer,
    )

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare bimga vs bimga_uniform at same hyperparameters"
    )
    parser.add_argument("--output-dir", default="paper_experiments_bimga_comp", help="Output directory")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Phase 1 checkpoint path")
    parser.add_argument("--resume", default=None, help="Path to existing run dir to resume into")
    parser.add_argument("--source-dir", default=None, help="Path to existing paper_experiments dir")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    # Define the two runs to compare
    runs = [
        RunConfig(name="s14_bimga_uniform_dw100_aw10", method="bimga_uniform"),
        RunConfig(name="s14_bimga_dw100_aw10", method="bimga"),
    ]

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    run_dir_original = ckpt_path.parent.parent
    config_path = run_dir_original / "config.json"
    original_cfg = {}
    if config_path.exists():
        with config_path.open() as f:
            original_cfg = json.load(f)

    teacher_model = original_cfg.get("teacher_model", "sentence-transformers/all-MiniLM-L6-v2")
    student_model = original_cfg.get("student_model", "huawei-noah/TinyBERT_General_4L_312D")
    dataset_name = original_cfg.get("dataset_name", "BEE-spoke-data/TACO-hf")

    print(f"Teacher: {teacher_model}")
    print(f"Student: {student_model}")
    print(f"Dataset: {dataset_name}")

    # Load checkpoint data
    print(f"\nLoading phase 1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student_backbone_state = ckpt["student_backbone_state"]
    ft_teacher_targets = DistillTargets(**ckpt["ft_teacher_targets"])

    # Load dataset
    set_seed(BASE_SEED)
    dataset = load_retrieval_dataset(dataset_name=dataset_name, taco_val_size=1000, seed=BASE_SEED)
    data = dataset_dict_to_splits(dataset)
    print(f"Splits -> train: {len(data.train.queries)}, val: {len(data.validation.queries)}, test: {len(data.test.queries)}")

    # Setup output directory
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            print(f"Resume dir not found: {run_dir}")
            return
        print(f"Resuming into: {run_dir}")
    else:
        output_root = resolve_output_root(args.output_dir)
        run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    # Get existing runs to copy
    source_dir = None
    if args.source_dir:
        source_dir = Path(args.source_dir)
    elif run_dir_original.name.startswith("202"):
        source_dir = run_dir_original

    existing_mapping = {}
    if source_dir:
        existing_mapping = get_existing_mapping(source_dir)
        if existing_mapping:
            print(f"\nFound {len(existing_mapping)} existing runs to copy:")
            for new_name, old_path in existing_mapping.items():
                print(f"  {new_name} <- {old_path.name}")

    # Load existing results if resuming
    all_results: dict[str, dict] = {}
    results_path = run_dir / "results_summary.json"
    if results_path.exists():
        with results_path.open() as f:
            all_results = json.load(f)
        all_results = {k: v for k, v in all_results.items() if "error" not in v}
        print(f"Loaded {len(all_results)} existing successful results")

    print(f"\n{'='*70}")
    print(f"BIMGa VS BIMGa_UNIFORM COMPARISON (dw=100, aw=10, 200 epochs)")
    print(f"{'='*70}")

    total_runs = len(runs)
    for i, rc in enumerate(runs, 1):
        # Check if we can copy an existing run
        if rc.name in existing_mapping:
            source_path = existing_mapping[rc.name]
            target_path = run_dir / rc.name

            if rc.name not in all_results:
                if copy_existing_run(source_path, target_path, rc.name):
                    # Load and store the metrics
                    metrics_file = None
                    for method_dir in target_path.iterdir():
                        if method_dir.is_dir():
                            mf = method_dir / "metrics.json"
                            if mf.exists():
                                metrics_file = mf
                                break

                    if metrics_file:
                        with metrics_file.open() as f:
                            metrics = json.load(f)
                        all_results[rc.name] = metrics
                        test_mrr = metrics["test"]["MRR"]
                        print(f"\n[{i}/{total_runs}] {rc.name} => COPIED (MRR={test_mrr:.4f})")

                # Save intermediate results
                with (run_dir / "results_summary.json").open("w") as f:
                    json.dump(all_results, f, indent=2)
            continue

        # Skip if already completed
        if rc.name in all_results:
            print(f"\n[{i}/{total_runs}] {rc.name} => SKIPPED (already done)")
            continue

        print(f"\n[{i}/{total_runs}] Training: {rc.name} ({rc.method})")

        try:
            metrics = run_single(
                rc,
                run_dir=run_dir,
                device=device,
                data=data,
                ft_teacher_targets=ft_teacher_targets,
                student_backbone_state=student_backbone_state,
                teacher_model=teacher_model,
                student_model=student_model,
                dataset_name=dataset_name,
                tb_writer=tb_writer,
            )
            all_results[rc.name] = metrics
            test_mrr = metrics["test"]["MRR"]
            print(f"  => MRR={test_mrr:.4f}")
        except Exception as e:
            print(f"  => FAILED: {e}")
            all_results[rc.name] = {"error": str(e)}

        maybe_empty_device_cache(device)

        # Save intermediate results
        with (run_dir / "results_summary.json").open("w") as f:
            json.dump(all_results, f, indent=2)

    tb_writer.close()

    # Print comparison summary
    print(f"\n{'='*80}")
    print(f"BIMGa VS BIMGa_UNIFORM COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Method':>30} | {'Test MRR':>10} | {'R@1':>6} | {'R@10':>6} | {'Asym MRR':>10}")
    print("-" * 80)

    for rc in runs:
        if rc.name not in all_results or "error" in all_results[rc.name]:
            print(f"{rc.method:>30} | FAILED")
            continue

        m = all_results[rc.name]
        t = m["test"]
        asym_mrr = m.get("diagnostics", {}).get("asymmetric_test", {}).get("MRR", "N/A")
        print(f"{rc.method:>30} | {t['MRR']:>10.4f} | {t['Recall@1']:>6.4f} | {t['Recall@10']:>6.4f} | {asym_mrr:>10}")

    # Calculate difference if both succeeded
    if all("error" not in all_results.get(rc.name, {}) for rc in runs):
        bimga_uniform_mrr = all_results[runs[0].name]["test"]["MRR"]
        bimga_mrr = all_results[runs[1].name]["test"]["MRR"]
        diff = bimga_mrr - bimga_uniform_mrr
        print(f"\nDifference (bimga - bimga_uniform): {diff:+.4f}")

    print(f"\nAll artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
