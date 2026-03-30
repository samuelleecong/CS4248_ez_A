"""Final evaluation run: all KD methods with model saving and full diagnostics.

Runs both symmetric and asymmetric evaluation, saves models, and produces
a comparison table of both evaluation modes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from mbpp_kd_suite.config import DistillTargets, TrainConfig, resolve_output_root
from mbpp_kd_suite.constants import KD_METHOD_ORDER
from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.runtime import apply_device_runtime_optimizations, maybe_empty_device_cache, pick_device, set_seed
from mbpp_kd_suite.training import make_method_targets, train_student


def run_final(
    checkpoint_path: str,
    methods: tuple[str, ...],
    distill_temperature: float = 0.2,
    distill_weight: float = 50.0,
    align_weight: float = 10.0,
    pair_weight: float = 1.0,
    relation_weight: float = 1.0,
    phase2_epochs: int = 30,
    phase2_patience: int = 5,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    lr: float = 2e-5,
    seed: int = 42,
    output_dir: str = "final_eval",
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
    print(f"Loading dataset: {dataset_name}")
    dataset = load_retrieval_dataset(dataset_name=dataset_name, taco_val_size=1000, seed=seed)
    data = dataset_dict_to_splits(dataset)
    print(f"Splits -> train: {len(data.train.queries)}, val: {len(data.validation.queries)}, test: {len(data.test.queries)}")

    output_root = resolve_output_root(output_dir)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    cfg = TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset_name=dataset_name,
        epochs=phase2_epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        lr=lr,
        seed=seed,
        distill_temperature=distill_temperature,
        distill_weight=distill_weight,
        align_weight=align_weight,
        pair_weight=pair_weight,
        relation_weight=relation_weight,
        save_models=True,
        run_diagnostics=True,
        early_stopping_patience=phase2_patience,
    )
    apply_device_runtime_optimizations(cfg=cfg, device=device)
    method_targets = make_method_targets(cfg=cfg, full_teacher_targets=ft_teacher_targets)

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

    # KD methods
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
            targets=method_targets[method_name],
            full_teacher_targets=ft_teacher_targets,
            initial_backbone_state_dict=student_backbone_state,
            tb_writer=tb_writer,
        )
        all_results[method_name] = kd_metrics
        maybe_empty_device_cache(device)

    # Save config and results
    config_payload = {
        "checkpoint": checkpoint_path,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "dataset_name": dataset_name,
        "phase2_epochs": phase2_epochs,
        "phase2_patience": phase2_patience,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "lr": lr,
        "seed": seed,
        "distill_temperature": distill_temperature,
        "distill_weight": distill_weight,
        "align_weight": align_weight,
        "pair_weight": pair_weight,
        "relation_weight": relation_weight,
        "methods": list(methods),
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)
    with (run_dir / "results_summary.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    tb_writer.close()

    # Print comparison table
    ctrl_sym = all_results["control_supervised"]["test"]
    ctrl_asym = all_results["control_supervised"].get("diagnostics", {}).get("asymmetric_test", {})

    print(f"\n{'='*100}")
    print("FINAL RESULTS: Symmetric vs Asymmetric Evaluation")
    print(f"{'='*100}")
    print(f"{'Method':>30} | {'Sym MRR':>8} | {'Sym R@1':>8} | {'Sym R@10':>8} | {'Asym MRR':>9} | {'Asym R@1':>9} | {'Asym R@10':>9} | {'Epochs':>6}")
    print("-" * 100)

    for name, m in all_results.items():
        sym = m["test"]
        asym = m.get("diagnostics", {}).get("asymmetric_test", {})
        stopped = m.get("stopped_epoch", "?")
        asym_mrr = f"{asym['MRR']:.4f}" if asym else "N/A"
        asym_r1 = f"{asym['Recall@1']:.4f}" if asym else "N/A"
        asym_r10 = f"{asym['Recall@10']:.4f}" if asym else "N/A"
        print(f"{name:>30} | {sym['MRR']:>8.4f} | {sym['Recall@1']:>8.4f} | {sym['Recall@10']:>8.4f} | {asym_mrr:>9} | {asym_r1:>9} | {asym_r10:>9} | {stopped:>6}")

    print(f"\nModels saved to: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")


if __name__ == "__main__":
    CHECKPOINT = "artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt"

    METHODS = (
        "bimga",
        "embed_distill",
        "score_distill",
        "qed_align",
        "adam_lite",
        "hpd",
        "margin_mse",
        "distilcse_lite",
        "hard_negative_pair_distill",
        "all_pairs_distill",
        "pointwise",
    )

    run_final(
        checkpoint_path=CHECKPOINT,
        methods=METHODS,
        distill_temperature=0.2,
        distill_weight=50.0,
        align_weight=10.0,
        phase2_epochs=30,
        phase2_patience=5,
    )
