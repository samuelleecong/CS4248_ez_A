"""Saturation sweep for Sets 1-4 (renamed as Sets 10-14).

Runs all Set 1-4 experiments to full saturation (200 epochs, patience=15).
Automatically copies already-completed saturation runs from Sets 7-9.

Model pairs:
  Pair 1 (default): TinyBERT-4L (student) -> MiniLM-L6-v2 (teacher)
    Checkpoint: artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt

Usage:
    # Run all saturation sets:
    uv run python run_saturation_sweep.py

    # Run specific sets:
    uv run python run_saturation_sweep.py --sets 10,11

    # Resume into existing directory:
    uv run python run_saturation_sweep.py --sets 10,11,12,13 --resume
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
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
    batch_size: int = 32
    seed: int = 42
    distill_weight: float = 50.0
    align_weight: float = 1.0
    pair_weight: float = 1.0
    distill_temperature: float = 0.2
    supervised: bool = False


def build_all_runs() -> dict[str, list[RunConfig]]:
    """Build the saturation experiment plan. Returns {set_name: [runs]}."""

    sets: dict[str, list[RunConfig]] = {}

    # ── Set 10: Core Sweep Saturation (bs=32, seed=42, 200 epochs) ──────────
    set10 = []
    set10.append(RunConfig(name="s10_control_bs32", method="control", supervised=True))
    for dw in [25, 50, 100]:
        set10.append(RunConfig(name=f"s10_score_dw{dw}", method="score_distill", distill_weight=dw))
    for dw, aw in [(50, 1), (50, 5), (50, 10), (100, 10)]:
        set10.append(RunConfig(name=f"s10_embed_dw{dw}_aw{aw}", method="embed_distill", distill_weight=dw, align_weight=aw))
    for dw, pw in [(50, 1), (50, 5), (50, 10), (100, 10)]:
        set10.append(RunConfig(name=f"s10_hnp_dw{dw}_pw{pw}", method="hard_negative_pair_distill", distill_weight=dw, pair_weight=pw))
    for dw, aw in [(50, 1), (50, 5), (50, 10), (100, 10)]:
        set10.append(RunConfig(name=f"s10_bimga_dw{dw}_aw{aw}", method="bimga", distill_weight=dw, align_weight=aw))
    sets["set10_core_sweep_sat"] = set10

    # ── Set 11: Batch Size Saturation (bs=64, seed=42, 200 epochs) ───────────
    set11 = []
    set11.append(RunConfig(name="s11_control_bs64", method="control", batch_size=64, supervised=True))
    set11.append(RunConfig(name="s11_score_dw100_bs64", method="score_distill", batch_size=64, distill_weight=100))
    set11.append(RunConfig(name="s11_embed_dw100_aw10_bs64", method="embed_distill", batch_size=64, distill_weight=100, align_weight=10))
    set11.append(RunConfig(name="s11_hnp_dw100_pw10_bs64", method="hard_negative_pair_distill", batch_size=64, distill_weight=100, pair_weight=10))
    set11.append(RunConfig(name="s11_bimga_dw100_aw10_bs64", method="bimga", batch_size=64, distill_weight=100, align_weight=10))
    sets["set11_batch_size_sat"] = set11

    # ── Set 12: BiMGA Ablation Saturation (bs=32, seed=42, 200 epochs) ──────
    set12 = []
    set12.append(RunConfig(name="s12_A2_bimga_uniform", method="bimga_uniform", distill_weight=100, align_weight=10))
    set12.append(RunConfig(name="s12_A3_bimga_query_only", method="bimga_query_only", distill_weight=100, align_weight=10))
    sets["set12_ablation_sat"] = set12

    # ── Set 13: Multi-Seed Saturation (bs=32, seeds 123+456, 200 epochs) ────
    set13 = []
    for seed in [123, 456]:
        set13.append(RunConfig(name=f"s13_score_dw100_seed{seed}", method="score_distill", seed=seed, distill_weight=100))
        set13.append(RunConfig(name=f"s13_embed_dw100_aw10_seed{seed}", method="embed_distill", seed=seed, distill_weight=100, align_weight=10))
        set13.append(RunConfig(name=f"s13_hnp_dw100_pw10_seed{seed}", method="hard_negative_pair_distill", seed=seed, distill_weight=100, pair_weight=10))
        set13.append(RunConfig(name=f"s13_bimga_dw100_aw10_seed{seed}", method="bimga", seed=seed, distill_weight=100, align_weight=10))
    sets["set13_multi_seed_sat"] = set13

    return sets


def copy_existing_run(
    source_dir: Path,
    target_dir: Path,
    run_name: str,
) -> bool:
    """Copy an already-completed run from source to target.

    Returns True if successful, False if source doesn't exist or is incomplete.
    """
    if not source_dir.exists():
        return False

    # Check if source has metrics.json (indicates completion)
    metrics_file = None
    for method_dir in source_dir.iterdir():
        if method_dir.is_dir():
            potential_metrics = method_dir / "metrics.json"
            if potential_metrics.exists():
                metrics_file = potential_metrics
                break

    if not metrics_file:
        return False

    # Copy the entire run directory
    print(f"  Copying existing run: {source_dir.name} -> {run_name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    return True


def get_mapping_from_existing_runs(run_dir: Path) -> dict[str, str]:
    """Map new run names to existing run directories that can be copied.

    Returns: {new_run_name: existing_run_path}
    """
    mapping = {}

    # Check if we're in the paper_experiments directory
    if not run_dir.name.startswith("202"):
        return mapping

    # Map Set 7-9 runs that are already saturated to Set 10-14 names
    existing_runs = {
        # Set 7 -> Set 10 (core sweep)
        "s10_control_bs32": "s7_control_bs32",
        "s10_score_dw100": "s9_score_dw100",  # Use Set 9 (better saturation)
        "s10_embed_dw100_aw10": "s7_embed_dw100_aw10",
        "s10_hnp_dw100_pw10": "s8_hnp_dw100_pw10",
        "s10_bimga_dw50_aw10": "s9_bimga_dw50_aw10",  # Use Set 9 (best model)

        # Set 12 (ablation) - already saturated
        "s12_A2_bimga_uniform": "s8_A2_bimga_uniform",
    }

    # Verify which ones actually exist
    for new_name, old_name in existing_runs.items():
        old_path = run_dir / old_name
        if old_path.exists():
            # Check if it has metrics
            has_metrics = False
            for method_dir in old_path.iterdir():
                if method_dir.is_dir() and (method_dir / "metrics.json").exists():
                    has_metrics = True
                    break
            if has_metrics:
                mapping[new_name] = old_path

    return mapping


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
        batch_size=rc.batch_size,
        eval_batch_size=EVAL_BATCH_SIZE,
        lr=2e-5,
        seed=rc.seed,
        distill_temperature=rc.distill_temperature,
        distill_weight=rc.distill_weight,
        align_weight=rc.align_weight,
        pair_weight=rc.pair_weight,
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
            "batch_size": rc.batch_size,
            "seed": rc.seed,
            "distill_weight": rc.distill_weight,
            "align_weight": rc.align_weight,
            "pair_weight": rc.pair_weight,
            "distill_temperature": rc.distill_temperature,
            "supervised": rc.supervised,
        }, f, indent=2)

    if rc.supervised:
        metrics, _, _ = train_student(
            name="control_supervised",
            cfg=cfg,
            run_dir=exp_dir,
            device=device,
            data=data,
            targets=ft_teacher_targets,
            full_teacher_targets=ft_teacher_targets,
            model_name=student_model,
            supervised=True,
            initial_backbone_state_dict=student_backbone_state,
            tb_writer=tb_writer,
        )
    else:
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
    parser = argparse.ArgumentParser(description="Run saturation sweep for Sets 1-4")
    parser.add_argument("--sets", default="10,11,12,13", help="Comma-separated set numbers to run")
    parser.add_argument("--output-dir", default="paper_experiments_saturation", help="Output directory")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Phase 1 checkpoint path")
    parser.add_argument("--resume", default=None, help="Path to existing run dir to resume into")
    parser.add_argument("--source-dir", default=None, help="Path to existing paper_experiments dir to copy from")
    args = parser.parse_args()

    requested_sets = {int(s.strip()) for s in args.sets.split(",")}
    print(f"Running saturation sets: {sorted(requested_sets)}")

    device = pick_device()
    print(f"Device: {device}")

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

    # Get mapping of existing runs that can be copied
    source_dir = None
    if args.source_dir:
        source_dir = Path(args.source_dir)
    elif run_dir_original.name.startswith("202"):
        source_dir = run_dir_original

    existing_mapping = {}
    if source_dir:
        existing_mapping = get_mapping_from_existing_runs(source_dir)
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

    all_sets = build_all_runs()
    set_names = sorted(all_sets.keys())

    total_runs = sum(len(v) for v in all_sets.values())
    completed = 0

    for set_name in set_names:
        set_num = int(set_name[3])  # "set10_..." -> 10
        if set_num not in requested_sets:
            print(f"\nSkipping {set_name} (not in --sets {args.sets})")
            completed += len(all_sets[set_name])
            continue

        runs = all_sets[set_name]
        print(f"\n{'='*70}")
        print(f"  {set_name.upper()} ({len(runs)} runs)")
        print(f"{'='*70}")

        for rc in runs:
            completed += 1

            # Check if we can copy an existing run
            if rc.name in existing_mapping:
                source_path = existing_mapping[rc.name]
                target_path = run_dir / rc.name

                # Skip if already exists in results
                if rc.name in all_results:
                    print(f"\n--- [{completed}/{total_runs}] {rc.name} --- SKIPPED (already done)")
                    continue

                # Copy the existing run
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
                        print(f"  => COPIED (MRR={test_mrr:.4f})")
                else:
                    print(f"\n--- [{completed}/{total_runs}] {rc.name} --- COPY FAILED, will train")

                # Save intermediate results
                with (run_dir / "results_summary.json").open("w") as f:
                    json.dump(all_results, f, indent=2)
                continue

            # Skip if already completed successfully
            if rc.name in all_results:
                print(f"\n--- [{completed}/{total_runs}] {rc.name} --- SKIPPED (already done)")
                continue

            print(f"\n--- [{completed}/{total_runs}] {rc.name} ({rc.method}, bs={rc.batch_size}, seed={rc.seed}) ---")

            # Set seed for this run
            set_seed(rc.seed)

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

            # Save intermediate results after each run
            with (run_dir / "results_summary.json").open("w") as f:
                json.dump(all_results, f, indent=2)

    tb_writer.close()

    # Print summary
    print(f"\n{'='*80}")
    print(f"SATURATION SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'Run':>40} | {'MRR':>6} | {'R@1':>6} | {'R@10':>6}")
    print("-" * 70)
    for name, m in all_results.items():
        if "error" in m:
            print(f"{name:>40} | FAILED")
            continue
        t = m["test"]
        print(f"{name:>40} | {t['MRR']:>6.4f} | {t['Recall@1']:>6.4f} | {t['Recall@10']:>6.4f}")

    print(f"\nAll artifacts saved to: {run_dir}")
    print(f"Models saved in each run subdirectory")


if __name__ == "__main__":
    main()
