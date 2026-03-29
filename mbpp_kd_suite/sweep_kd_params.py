"""Sequential KD hyperparameter sweep, resuming from a phase 1 checkpoint.

Sweeps distill_weight, distill_temperature, batch_size, and lr
while reusing the same finetuned teacher targets and student backbone.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from mbpp_kd_suite.config import DistillTargets, TrainConfig, resolve_output_root
from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.runtime import apply_device_runtime_optimizations, maybe_empty_device_cache, pick_device, set_seed
from mbpp_kd_suite.training import make_method_targets, train_student


@dataclass
class SweepConfig:
    name: str
    distill_temperature: float = 0.2
    distill_weight: float = 1.0
    align_weight: float = 1.0
    pair_weight: float = 1.0
    relation_weight: float = 1.0
    batch_size: int = 32
    lr: float = 2e-5
    methods: tuple[str, ...] = ("embed_distill", "score_distill")


def run_sweep(
    checkpoint_path: str,
    configs: list[SweepConfig],
    output_dir: str = "sweep_kd_params",
    phase2_epochs: int = 10,
    phase2_patience: int = 3,
    eval_batch_size: int = 64,
    seed: int = 42,
) -> dict[str, dict]:
    set_seed(seed)
    device = pick_device()
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
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
    print(f"Loading dataset: {dataset_name}")
    dataset = load_retrieval_dataset(dataset_name=dataset_name, taco_val_size=1000, seed=seed)
    data = dataset_dict_to_splits(dataset)
    print(f"Splits -> train: {len(data.train.queries)}, val: {len(data.validation.queries)}, test: {len(data.test.queries)}")

    output_root = resolve_output_root(output_dir)
    sweep_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(sweep_dir / "tensorboard"))

    all_results: dict[str, dict] = {}

    # First run control_supervised as baseline (same for all configs, just use first batch_size)
    print(f"\n{'='*60}")
    print(f"Running control_supervised baseline")
    print(f"{'='*60}")
    ctrl_cfg = TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset_name=dataset_name,
        epochs=phase2_epochs,
        batch_size=configs[0].batch_size,
        eval_batch_size=eval_batch_size,
        lr=configs[0].lr,
        seed=seed,
        save_models=False,
        run_diagnostics=False,
        early_stopping_patience=phase2_patience,
    )
    apply_device_runtime_optimizations(cfg=ctrl_cfg, device=device)
    ctrl_metrics, _, _ = train_student(
        name="control_supervised",
        cfg=ctrl_cfg,
        run_dir=sweep_dir,
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
    control_mrr = ctrl_metrics["test"]["MRR"]
    print(f"  Control MRR: {control_mrr:.4f}")
    maybe_empty_device_cache(device)

    # Run each sweep config
    for sc in configs:
        print(f"\n{'='*60}")
        print(f"Config: {sc.name}")
        print(f"  dt={sc.distill_temperature}, dw={sc.distill_weight}, aw={sc.align_weight}, bs={sc.batch_size}, lr={sc.lr}")
        print(f"  Methods: {', '.join(sc.methods)}")
        print(f"{'='*60}")

        cfg = TrainConfig(
            teacher_model=teacher_model,
            student_model=student_model,
            dataset_name=dataset_name,
            epochs=phase2_epochs,
            batch_size=sc.batch_size,
            eval_batch_size=eval_batch_size,
            lr=sc.lr,
            seed=seed,
            distill_temperature=sc.distill_temperature,
            distill_weight=sc.distill_weight,
            align_weight=sc.align_weight,
            pair_weight=sc.pair_weight,
            relation_weight=sc.relation_weight,
            save_models=False,
            run_diagnostics=False,
            early_stopping_patience=phase2_patience,
        )
        apply_device_runtime_optimizations(cfg=cfg, device=device)
        method_targets = make_method_targets(cfg=cfg, full_teacher_targets=ft_teacher_targets)

        config_dir = sweep_dir / sc.name
        config_dir.mkdir(parents=True, exist_ok=True)
        # Save per-config hyperparameters
        with (config_dir / "sweep_config.json").open("w") as f:
            json.dump({
                "name": sc.name,
                "distill_temperature": sc.distill_temperature,
                "distill_weight": sc.distill_weight,
                "align_weight": sc.align_weight,
                "pair_weight": sc.pair_weight,
                "relation_weight": sc.relation_weight,
                "batch_size": sc.batch_size,
                "lr": sc.lr,
                "methods": list(sc.methods),
            }, f, indent=2)
        for method_name in sc.methods:
            tag = f"{sc.name}/{method_name}"
            print(f"\n  --- {tag} ---")
            kd_metrics, _, _ = train_student(
                name=method_name,
                cfg=cfg,
                run_dir=config_dir,
                device=device,
                data=data,
                targets=method_targets[method_name],
                full_teacher_targets=ft_teacher_targets,
                initial_backbone_state_dict=student_backbone_state,
                tb_writer=tb_writer,
            )
            all_results[tag] = kd_metrics
            test_mrr = kd_metrics["test"]["MRR"]
            delta = test_mrr - control_mrr
            print(f"  {tag}: MRR={test_mrr:.4f} (delta={delta:+.4f})")
            maybe_empty_device_cache(device)

    # Save results
    with (sweep_dir / "results_summary.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    # Save sweep config index (all configs with their params + results)
    sweep_index = {
        "checkpoint": checkpoint_path,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "dataset_name": dataset_name,
        "phase2_epochs": phase2_epochs,
        "phase2_patience": phase2_patience,
        "eval_batch_size": eval_batch_size,
        "seed": seed,
        "control_test_mrr": control_mrr,
        "configs": {},
    }
    for sc in configs:
        config_results = {}
        for method_name in sc.methods:
            tag = f"{sc.name}/{method_name}"
            if tag in all_results:
                config_results[method_name] = all_results[tag]["test"]
        sweep_index["configs"][sc.name] = {
            "distill_temperature": sc.distill_temperature,
            "distill_weight": sc.distill_weight,
            "align_weight": sc.align_weight,
            "pair_weight": sc.pair_weight,
            "relation_weight": sc.relation_weight,
            "batch_size": sc.batch_size,
            "lr": sc.lr,
            "results": config_results,
        }
    with (sweep_dir / "sweep_index.json").open("w") as f:
        json.dump(sweep_index, f, indent=2)

    tb_writer.close()

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SWEEP RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config/Method':>45} | {'MRR':>6} | {'R@1':>6} | {'R@10':>6} | {'vs ctrl':>8}")
    print("-" * 80)
    print(f"{'control_supervised':>45} | {control_mrr:>6.4f} | {ctrl_metrics['test']['Recall@1']:>6.4f} | {ctrl_metrics['test']['Recall@10']:>6.4f} |     —")
    for tag, m in all_results.items():
        if tag == "control_supervised":
            continue
        t = m["test"]
        delta = t["MRR"] - control_mrr
        print(f"{tag:>45} | {t['MRR']:>6.4f} | {t['Recall@1']:>6.4f} | {t['Recall@10']:>6.4f} | {delta:>+8.4f}")

    print(f"\nArtifacts saved to: {sweep_dir}")
    return all_results


if __name__ == "__main__":
    # ── Edit these two lines to point at your checkpoint and choose methods ──
    CHECKPOINT = "artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt"
    METHODS = ("embed_distill", "score_distill", "adam_lite")

    # ── Default sweep: parameters that matter most ──────────────────────────
    #
    # From our experiments, the two highest-impact params are:
    #   1. distill_weight (dw)  — scales the KD loss to match supervised loss
    #   2. align_weight  (aw)  — scales embedding alignment (embed_distill, qed_align, hpd)
    #
    # Lower impact:
    #   - batch_size: modest gains from more in-batch negatives
    #   - lr: default 2e-5 is fine, lower hurts
    #   - distill_temperature: 0.2 is the sweet spot for cosine similarities
    #
    # Results from initial sweep (TinyBERT-4L student, MiniLM-L6 teacher, TACO):
    #   control_supervised:           MRR=0.1983
    #   dw=50, dt=0.2:  score_distill MRR=0.2274 (+0.0290)  *** best overall
    #   dw=50, dt=0.2:  embed_distill MRR=0.2262 (+0.0278)
    #   dw=50, dt=0.2:  adam_lite     MRR=0.2225 (+0.0242)
    #   aw=5,  dw=10:   embed_distill MRR=0.2219 (+0.0235)
    #   dw=25, dt=0.2:  score_distill MRR=0.2106 (+0.0123)
    #   bs=128, dw=10:  score_distill MRR=0.2099 (+0.0116)

    configs = [
        # --- distill_weight sweep (highest impact) ---
        SweepConfig(name="dw25",   distill_temperature=0.2, distill_weight=25.0,  methods=METHODS),
        SweepConfig(name="dw50",   distill_temperature=0.2, distill_weight=50.0,  methods=METHODS),
        SweepConfig(name="dw100",  distill_temperature=0.2, distill_weight=100.0, methods=METHODS),

        # --- align_weight sweep (high impact for embed_distill/qed_align) ---
        SweepConfig(name="aw5_dw50",  distill_temperature=0.2, distill_weight=50.0, align_weight=5.0, methods=METHODS),
        SweepConfig(name="aw10_dw50", distill_temperature=0.2, distill_weight=50.0, align_weight=10.0, methods=METHODS),

        # --- best combo candidates ---
        SweepConfig(name="dw50_bs128",      distill_temperature=0.2, distill_weight=50.0, batch_size=128, methods=METHODS),
        SweepConfig(name="dw50_aw5_bs128",  distill_temperature=0.2, distill_weight=50.0, align_weight=5.0, batch_size=128, methods=METHODS),
    ]

    run_sweep(CHECKPOINT, configs)
