"""Full experiment sweep for the paper.

Runs experiments from a phase 1 checkpoint. All models saved locally
for later HuggingFace upload.

Model pairs:
  Pair 1 (default): TinyBERT-4L (student) -> MiniLM-L6-v2 (teacher)
    Checkpoint: artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt

  Pair 2: MiniLM-L6-v2 (student) -> all-mpnet-base-v2 (teacher)
    Needs new phase 1 first:
      uv run mbpp-kd-two-phase \
        --student-model sentence-transformers/all-MiniLM-L6-v2 \
        --teacher-model sentence-transformers/all-mpnet-base-v2 \
        --dataset-name BEE-spoke-data/TACO-hf \
        --phase1-epochs 20 --phase1-patience 3 \
        --phase2-epochs 1 --batch-size 32
    Then use the generated checkpoint.pt path.

Usage:
    # Run Sets 1-4 with pair 1 (default):
    uv run python run_full_sweep.py

    # Run a single set:
    uv run python run_full_sweep.py --sets 1

    # Run specific sets:
    uv run python run_full_sweep.py --sets 1,2

    # Run Set 5 with pair 2 (after creating phase 1):
    uv run python run_full_sweep.py --sets 5 \
      --checkpoint path/to/pair2/phase1/checkpoint.pt \
      --output-dir paper_experiments_pair2

    # Run all sets including Set 5 with pair 1:
    uv run python run_full_sweep.py --sets 1,2,3,4,5
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
from mbpp_kd_suite.runtime import (
    apply_device_runtime_optimizations,
    maybe_empty_device_cache,
    pick_device,
    set_seed,
)
from mbpp_kd_suite.training import make_method_targets, train_student


# ── Phase 1 checkpoint (edit this) ──────────────────────────────────────────
CHECKPOINT = "artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt"

# ── Constants ───────────────────────────────────────────────────────────────
PHASE2_EPOCHS = 30
PHASE2_PATIENCE = 5
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
    epochs: int = 0  # 0 = use PHASE2_EPOCHS default
    patience: int = 0  # 0 = use PHASE2_PATIENCE default


def build_all_runs() -> dict[str, list[RunConfig]]:
    """Build the full experiment plan. Returns {set_name: [runs]}."""

    sets: dict[str, list[RunConfig]] = {}

    # ── Set 1: Core Sweep (bs=32, seed=42) ──────────────────────────────
    set1 = []
    # Control
    set1.append(RunConfig(name="s1_control_bs32", method="control", supervised=True))
    # score_distill: sweep dw
    for dw in [25, 50, 100]:
        set1.append(RunConfig(name=f"s1_score_dw{dw}", method="score_distill", distill_weight=dw))
    # embed_distill: sweep dw × aw
    for dw, aw in [(50, 1), (50, 5), (50, 10), (100, 10)]:
        set1.append(RunConfig(name=f"s1_embed_dw{dw}_aw{aw}", method="embed_distill", distill_weight=dw, align_weight=aw))
    # hard_negative_pair_distill: sweep dw × pw
    for dw, pw in [(50, 1), (50, 5), (50, 10), (100, 10)]:
        set1.append(RunConfig(name=f"s1_hnp_dw{dw}_pw{pw}", method="hard_negative_pair_distill", distill_weight=dw, pair_weight=pw))
    # bimga: sweep dw × aw
    for dw, aw in [(50, 1), (50, 5), (50, 10), (100, 10)]:
        set1.append(RunConfig(name=f"s1_bimga_dw{dw}_aw{aw}", method="bimga", distill_weight=dw, align_weight=aw))
    sets["set1_core_sweep"] = set1

    # ── Set 2: Batch Size Ablation (bs=64, seed=42, best configs from Set 1) ─
    # Best configs based on Set 1 results:
    #   score_distill: dw=100 (MRR=0.2664)
    #   embed_distill: dw=100, aw=10 (MRR=0.2818)
    #   hard_neg_pair: dw=100, pw=10 (MRR=0.2683)
    #   bimga: TBD — using dw=100, aw=10 (same sweep as embed for comparison)
    set2 = []
    set2.append(RunConfig(name="s2_control_bs64", method="control", batch_size=64, supervised=True))
    set2.append(RunConfig(name="s2_score_dw100_bs64", method="score_distill", batch_size=64, distill_weight=100))
    set2.append(RunConfig(name="s2_embed_dw100_aw10_bs64", method="embed_distill", batch_size=64, distill_weight=100, align_weight=10))
    set2.append(RunConfig(name="s2_hnp_dw100_pw10_bs64", method="hard_negative_pair_distill", batch_size=64, distill_weight=100, pair_weight=10))
    set2.append(RunConfig(name="s2_bimga_dw100_aw10_bs64", method="bimga", batch_size=64, distill_weight=100, align_weight=10))
    sets["set2_batch_size"] = set2

    # ── Set 3: BiMGA Ablation A2+A3 (bs=32, seed=42, dw=100, aw=10) ────
    # A1 (embed_distill) and A4 (bimga) are already in Set 1 at dw=100/aw=10
    set3 = []
    set3.append(RunConfig(name="s3_A2_bimga_uniform", method="bimga_uniform", distill_weight=100, align_weight=10))
    set3.append(RunConfig(name="s3_A3_bimga_query_only", method="bimga_query_only", distill_weight=100, align_weight=10))
    sets["set3_ablation"] = set3

    # ── Set 4: Multi-Seed (bs=32, seeds 123+456, best configs) ──────────
    set4 = []
    for seed in [123, 456]:
        set4.append(RunConfig(name=f"s4_score_dw100_seed{seed}", method="score_distill", seed=seed, distill_weight=100))
        set4.append(RunConfig(name=f"s4_embed_dw100_aw10_seed{seed}", method="embed_distill", seed=seed, distill_weight=100, align_weight=10))
        set4.append(RunConfig(name=f"s4_hnp_dw100_pw10_seed{seed}", method="hard_negative_pair_distill", seed=seed, distill_weight=100, pair_weight=10))
        set4.append(RunConfig(name=f"s4_bimga_dw100_aw10_seed{seed}", method="bimga", seed=seed, distill_weight=100, align_weight=10))
    sets["set4_multi_seed"] = set4

    # ── Set 9: Deep saturation (200 epochs, patience=15) ─────────────────
    set9 = []
    set9.append(RunConfig(name="s9_score_dw100", method="score_distill", distill_weight=100, epochs=200, patience=15))
    set9.append(RunConfig(name="s9_bimga_dw50_aw10", method="bimga", distill_weight=50, align_weight=10, epochs=200, patience=15))
    sets["set9_deep_saturation"] = set9

    # ── Set 8: Extended saturation (120 epochs, patience=10) ──────────────
    set8 = []
    set8.append(RunConfig(name="s8_score_dw100", method="score_distill", distill_weight=100, epochs=120, patience=10))
    set8.append(RunConfig(name="s8_hnp_dw100_pw10", method="hard_negative_pair_distill", distill_weight=100, pair_weight=10, epochs=120, patience=10))
    set8.append(RunConfig(name="s8_bimga_dw50_aw10", method="bimga", distill_weight=50, align_weight=10, epochs=120, patience=10))
    set8.append(RunConfig(name="s8_A2_bimga_uniform", method="bimga_uniform", distill_weight=100, align_weight=10, epochs=120, patience=10))
    sets["set8_saturation_ext"] = set8

    # ── Set 7: Saturation runs (50 epochs, best configs) ─────────────────
    set7 = []
    set7.append(RunConfig(name="s7_control_bs32", method="control", supervised=True))
    set7.append(RunConfig(name="s7_score_dw100", method="score_distill", distill_weight=100))
    set7.append(RunConfig(name="s7_embed_dw100_aw10", method="embed_distill", distill_weight=100, align_weight=10))
    set7.append(RunConfig(name="s7_hnp_dw100_pw10", method="hard_negative_pair_distill", distill_weight=100, pair_weight=10))
    set7.append(RunConfig(name="s7_bimga_dw50_aw10", method="bimga", distill_weight=50, align_weight=10))
    set7.append(RunConfig(name="s7_A2_bimga_uniform", method="bimga_uniform", distill_weight=100, align_weight=10))
    sets["set7_saturation"] = set7

    # ── Set 6: Higher dw/aw exploration ──────────────���──────────────────
    set6 = []
    set6.append(RunConfig(name="s6_bimga_dw50_aw20", method="bimga", distill_weight=50, align_weight=20))
    set6.append(RunConfig(name="s6_bimga_dw200_aw10", method="bimga", distill_weight=200, align_weight=10))
    set6.append(RunConfig(name="s6_bimga_dw100_aw20", method="bimga", distill_weight=100, align_weight=20))
    set6.append(RunConfig(name="s6_embed_dw100_aw20", method="embed_distill", distill_weight=100, align_weight=20))
    set6.append(RunConfig(name="s6_score_dw200", method="score_distill", distill_weight=200))
    sets["set6_higher_params"] = set6

    # ── Set 5: Model pair validation (best configs only) ────────────────
    # Uses same configs as Set 1 best, but run with a DIFFERENT checkpoint
    # (different student-teacher pair). Pass a different --checkpoint.
    set5 = []
    set5.append(RunConfig(name="s5_control", method="control", supervised=True))
    set5.append(RunConfig(name="s5_score_dw100", method="score_distill", distill_weight=100))
    set5.append(RunConfig(name="s5_embed_dw100_aw10", method="embed_distill", distill_weight=100, align_weight=10))
    set5.append(RunConfig(name="s5_bimga_dw100_aw10", method="bimga", distill_weight=100, align_weight=10))
    sets["set5_model_pair"] = set5

    return sets


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
        epochs=rc.epochs if rc.epochs > 0 else PHASE2_EPOCHS,
        batch_size=rc.batch_size,
        eval_batch_size=EVAL_BATCH_SIZE,
        lr=2e-5,
        seed=rc.seed,
        distill_temperature=rc.distill_temperature,
        distill_weight=rc.distill_weight,
        align_weight=rc.align_weight,
        pair_weight=rc.pair_weight,
        save_models=True,  # Save all models for HuggingFace upload
        run_diagnostics=True,  # Include sym/asym diagnostics
        early_stopping_patience=rc.patience if rc.patience > 0 else PHASE2_PATIENCE,
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
    parser = argparse.ArgumentParser(description="Run full experiment sweep for paper")
    parser.add_argument("--sets", default="1,2,3,4", help="Comma-separated set numbers to run (e.g., '1,2,3,4' or '5')")
    parser.add_argument("--output-dir", default="paper_experiments", help="Output directory")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Phase 1 checkpoint path")
    parser.add_argument("--resume", default=None, help="Path to existing run dir to resume into (skips completed runs)")
    args = parser.parse_args()

    requested_sets = {int(s.strip()) for s in args.sets.split(",")}
    print(f"Running sets: {sorted(requested_sets)}")

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

    # Setup output — resume into existing dir or create new
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

    # Load existing results if resuming
    all_results: dict[str, dict] = {}
    results_path = run_dir / "results_summary.json"
    if results_path.exists():
        with results_path.open() as f:
            all_results = json.load(f)
        # Filter out failed runs so they get retried
        all_results = {k: v for k, v in all_results.items() if "error" not in v}
        print(f"Loaded {len(all_results)} existing successful results")
    all_sets = build_all_runs()
    set_names = sorted(all_sets.keys())

    total_runs = sum(len(v) for v in all_sets.values())
    completed = 0

    for set_name in set_names:
        set_num = int(set_name[3])  # "set1_..." -> 1
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
    print(f"FULL SWEEP RESULTS")
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
    print(f"Models saved in each run subdirectory (--save-models=True)")


if __name__ == "__main__":
    main()
