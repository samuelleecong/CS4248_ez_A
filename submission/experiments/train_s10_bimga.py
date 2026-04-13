"""Train bimga at dw=100, aw=10 to saturation (Set 10).

This runs bimga with the same hyperparameters as bimga_uniform (dw=100, aw=10)
to compare margin-weighted vs uniform-weighted bidirectional alignment.

Usage:
    uv run python train_s10_bimga.py
"""
from __future__ import annotations

import json
import time
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


# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT = "artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt"
EPOCHS = 200
PATIENCE = 15
EVAL_BATCH_SIZE = 64
BATCH_SIZE = 32
SEED = 42
DISTILL_WEIGHT = 100.0
ALIGN_WEIGHT = 10.0
DISTILL_TEMPERATURE = 0.2
LEARNING_RATE = 2e-5

# Run name
RUN_NAME = "s10_bimga_dw100_aw10"
METHOD = "bimga"


def main():
    device = pick_device()
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = Path(CHECKPOINT)
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
    print(f"\nTraining configuration:")
    print(f"  Method: {METHOD}")
    print(f"  distill_weight: {DISTILL_WEIGHT}")
    print(f"  align_weight: {ALIGN_WEIGHT}")
    print(f"  epochs: {EPOCHS}")
    print(f"  patience: {PATIENCE}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  seed: {SEED}")

    # Load checkpoint data
    print(f"\nLoading phase 1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student_backbone_state = ckpt["student_backbone_state"]
    ft_teacher_targets = DistillTargets(**ckpt["ft_teacher_targets"])

    # Load dataset
    set_seed(SEED)
    dataset = load_retrieval_dataset(dataset_name=dataset_name, taco_val_size=1000, seed=SEED)
    data = dataset_dict_to_splits(dataset)
    print(f"\nDataset splits:")
    print(f"  train: {len(data.train.queries)}")
    print(f"  val: {len(data.validation.queries)}")
    print(f"  test: {len(data.test.queries)}")

    # Setup output directory
    output_root = resolve_output_root("paper_experiments")
    run_dir = output_root / "20260402_015143"  # Use existing run dir
    run_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = run_dir / RUN_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    tb_log_dir = run_dir / "tensorboard"
    tb_log_dir.mkdir(exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    # Create training config
    cfg = TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset_name=dataset_name,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        lr=LEARNING_RATE,
        seed=SEED,
        distill_temperature=DISTILL_TEMPERATURE,
        distill_weight=DISTILL_WEIGHT,
        align_weight=ALIGN_WEIGHT,
        pair_weight=1.0,
        save_models=True,
        run_diagnostics=True,
        early_stopping_patience=PATIENCE,
    )
    apply_device_runtime_optimizations(cfg=cfg, device=device)

    # Save run config
    with (exp_dir / "run_config.json").open("w") as f:
        json.dump({
            "name": RUN_NAME,
            "method": METHOD,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "distill_weight": DISTILL_WEIGHT,
            "align_weight": ALIGN_WEIGHT,
            "pair_weight": 1.0,
            "distill_temperature": DISTILL_TEMPERATURE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LEARNING_RATE,
        }, f, indent=2)

    # Train
    print(f"\n{'='*70}")
    print(f"Starting training: {RUN_NAME}")
    print(f"{'='*70}\n")

    method_targets = make_method_targets(cfg=cfg, full_teacher_targets=ft_teacher_targets)

    try:
        metrics, _, _ = train_student(
            name=METHOD,
            cfg=cfg,
            run_dir=exp_dir,
            device=device,
            data=data,
            targets=method_targets[METHOD],
            full_teacher_targets=ft_teacher_targets,
            initial_backbone_state_dict=student_backbone_state,
            tb_writer=tb_writer,
        )

        # Print results
        test_mrr = metrics["test"]["MRR"]
        test_r1 = metrics["test"]["Recall@1"]
        test_r10 = metrics["test"]["Recall@10"]
        val_mrr = metrics["validation"]["MRR"]

        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"{'='*70}")
        print(f"Test MRR:  {test_mrr:.4f}")
        print(f"Test R@1:  {test_r1:.4f}")
        print(f"Test R@10: {test_r10:.4f}")
        print(f"Val MRR:   {val_mrr:.4f}")

        # Compare with bimga_uniform if available
        bimga_uniform_dir = run_dir / "s8_A2_bimga_uniform"
        if bimga_uniform_dir.exists():
            for method_dir in bimga_uniform_dir.iterdir():
                if method_dir.is_dir():
                    metrics_file = method_dir / "metrics.json"
                    if metrics_file.exists():
                        with metrics_file.open() as f:
                            uniform_metrics = json.load(f)

                        uniform_mrr = uniform_metrics["test"]["MRR"]
                        diff = test_mrr - uniform_mrr

                        print(f"\n{'='*70}")
                        print(f"Comparison with bimga_uniform (s8_A2_bimga_uniform):")
                        print(f"{'='*70}")
                        print(f"bimga (dw=100, aw=10):           {test_mrr:.4f}")
                        print(f"bimga_uniform (dw=100, aw=10):   {uniform_mrr:.4f}")
                        print(f"Difference:                      {diff:+.4f}")
                        break

        print(f"\nModel saved to: {exp_dir}")
        print(f"Tensorboard logs: {tb_log_dir}")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tb_writer.close()
        maybe_empty_device_cache(device)


if __name__ == "__main__":
    main()
