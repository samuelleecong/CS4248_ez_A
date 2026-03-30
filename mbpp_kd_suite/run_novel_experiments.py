"""Run novel KD approach experiments: progressive annealing, hard neg mining,
attention pooling, and semantic bridge — all built on BiMGA."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from mbpp_kd_suite.config import DistillTargets, TrainConfig, resolve_output_root
from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.runtime import (
    apply_device_runtime_optimizations,
    maybe_empty_device_cache,
    pick_device,
    set_seed,
)
from mbpp_kd_suite.training import precompute_hard_negatives, train_student


NOVEL_METHODS = (
    "bimga",             # baseline for comparison
    "bimga_progressive", # progressive margin annealing
    "bimga_hardneg",     # corpus-level hard negative mining
    "bimga_attn_pool",   # attention-weighted pooling
    "bimga_bridge",      # semantic bridge with orthogonal subspaces
)


def run_novel_experiments(
    checkpoint_path: str,
    methods: tuple[str, ...] = NOVEL_METHODS,
    distill_temperature: float = 0.2,
    distill_weight: float = 50.0,
    align_weight: float = 10.0,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    lr: float = 2e-5,
    seed: int = 42,
    output_dir: str = "novel_experiments",
    hard_neg_k: int = 32,
) -> None:
    set_seed(seed)
    device = pick_device()
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    run_dir_original = ckpt_path.parent.parent
    original_cfg = {}
    config_path = run_dir_original / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            original_cfg = json.load(f)

    teacher_model = original_cfg.get(
        "teacher_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    student_model = original_cfg.get(
        "student_model", "huawei-noah/TinyBERT_General_4L_312D"
    )
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
    print(f"Loading dataset: {dataset_name}")
    dataset = load_retrieval_dataset(
        dataset_name=dataset_name, taco_val_size=1000, seed=seed
    )
    data = dataset_dict_to_splits(dataset)
    print(
        f"Splits -> train: {len(data.train.queries)}, "
        f"val: {len(data.validation.queries)}, "
        f"test: {len(data.test.queries)}"
    )

    # Pre-compute hard negatives for hard neg mining method
    hard_neg_indices = None
    if "bimga_hardneg" in methods:
        print(f"\nPre-computing top-{hard_neg_k} hard negatives per query...")
        hard_neg_indices = precompute_hard_negatives(ft_teacher_targets, k=hard_neg_k)
        print(f"Hard negative index shape: {hard_neg_indices.shape}")

    output_root = resolve_output_root(output_dir)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    cfg = TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        lr=lr,
        seed=seed,
        distill_temperature=distill_temperature,
        distill_weight=distill_weight,
        align_weight=align_weight,
        save_models=False,
        run_diagnostics=True,
        early_stopping_patience=patience,
    )
    apply_device_runtime_optimizations(cfg=cfg, device=device)

    all_results: dict[str, dict] = {}

    # Control baseline
    print(f"\n{'='*60}")
    print("Running control_supervised")
    print(f"{'='*60}")
    ctrl_metrics, _, _ = train_student(
        name="control_supervised",
        cfg=cfg,
        run_dir=run_dir,
        device=device,
        data=data,
        targets=ft_teacher_targets,
        full_teacher_targets=ft_teacher_targets,
        model_name=student_model,
        supervised=True,
        initial_backbone_state_dict=student_backbone_state,
        tb_writer=tb_writer,
    )
    all_results["control_supervised"] = ctrl_metrics
    maybe_empty_device_cache(device)

    # Novel KD methods
    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"Running {method_name}")
        print(f"{'='*60}")
        kd_metrics, _, _ = train_student(
            name=method_name,
            cfg=cfg,
            run_dir=run_dir,
            device=device,
            data=data,
            targets=ft_teacher_targets,
            full_teacher_targets=ft_teacher_targets,
            initial_backbone_state_dict=student_backbone_state,
            tb_writer=tb_writer,
            hard_neg_indices=hard_neg_indices,
        )
        all_results[method_name] = kd_metrics
        maybe_empty_device_cache(device)

    # Save config and results
    config_payload = {
        "checkpoint": checkpoint_path,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "dataset_name": dataset_name,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "distill_temperature": distill_temperature,
        "distill_weight": distill_weight,
        "align_weight": align_weight,
        "hard_neg_k": hard_neg_k,
        "methods": list(methods),
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)
    with (run_dir / "results_summary.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    tb_writer.close()

    # Print comparison table
    print(f"\n{'='*100}")
    print("NOVEL EXPERIMENTS: Results")
    print(f"{'='*100}")
    print(
        f"{'Method':>25} | {'MRR':>8} | {'R@1':>8} | {'R@10':>8} | "
        f"{'Asym MRR':>9} | {'Asym R@1':>9} | {'Epochs':>6}"
    )
    print("-" * 90)

    for method_name, m in all_results.items():
        sym = m["test"]
        asym = m.get("diagnostics", {}).get("asymmetric_test", {})
        stopped = m.get("stopped_epoch", "?")
        asym_mrr = f"{asym['MRR']:.4f}" if asym else "N/A"
        asym_r1 = f"{asym['Recall@1']:.4f}" if asym else "N/A"
        print(
            f"{method_name:>25} | {sym['MRR']:>8.4f} | "
            f"{sym['Recall@1']:>8.4f} | {sym['Recall@10']:>8.4f} | "
            f"{asym_mrr:>9} | {asym_r1:>9} | {stopped:>6}"
        )

    print(f"\nResults saved to: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")


if __name__ == "__main__":
    CHECKPOINT = "artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt"

    run_novel_experiments(
        checkpoint_path=CHECKPOINT,
        methods=NOVEL_METHODS,
        epochs=10,
        patience=3,
    )
