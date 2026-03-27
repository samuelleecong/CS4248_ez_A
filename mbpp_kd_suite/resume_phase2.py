"""Convenience script to resume phase 2 KD from an existing phase 1 checkpoint.

Lists available checkpoints and lets you re-run phase 2 with different
hyperparameters (e.g., distill_temperature, methods, epochs) without
repeating phase 1 teacher/student finetuning.

Usage:
    # List available checkpoints
    uv run python resume_phase2.py --list

    # Resume with new distill temperature
    uv run python resume_phase2.py \
        --checkpoint artifacts/two_phase_tinybert4l_taco_dt4/20260328_.../phase1/checkpoint.pt \
        --distill-temperature 4.0 \
        --output-dir two_phase_tinybert4l_taco_dt4_sweep

    # Resume with specific methods only
    uv run python resume_phase2.py \
        --checkpoint artifacts/two_phase_tinybert4l_taco_dt4/20260328_.../phase1/checkpoint.pt \
        --methods embed_distill,score_distill,margin_mse \
        --distill-temperature 8.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"


def list_checkpoints() -> list[Path]:
    checkpoints = sorted(ARTIFACTS.rglob("phase1/checkpoint.pt"))
    if not checkpoints:
        print("No phase 1 checkpoints found under artifacts/")
        return []

    print(f"Found {len(checkpoints)} checkpoint(s):\n")
    for i, ckpt in enumerate(checkpoints, 1):
        run_dir = ckpt.parent.parent
        config_path = run_dir / "config.json"
        info = ""
        if config_path.exists():
            with config_path.open() as f:
                cfg = json.load(f)
            info = (
                f"  teacher={cfg.get('teacher_model', '?').split('/')[-1]}"
                f"  student={cfg.get('student_model', '?').split('/')[-1]}"
                f"  dataset={cfg.get('dataset_name', '?').split('/')[-1]}"
                f"  dt={cfg.get('distill_temperature', '?')}"
            )
        print(f"  [{i}] {ckpt}")
        if info:
            print(f"      {info}")
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume phase 2 KD from an existing phase 1 checkpoint"
    )
    parser.add_argument("--list", action="store_true", help="List available checkpoints and exit")
    parser.add_argument("--checkpoint", type=str, help="Path to phase1/checkpoint.pt")
    parser.add_argument("--pick", type=int, help="Pick checkpoint by number from --list")
    parser.add_argument("--student-model", type=str, default=None,
                        help="Override student model (must match checkpoint architecture)")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Override teacher model (must match checkpoint architecture)")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Override dataset (default: same as phase 1)")
    parser.add_argument("--distill-temperature", type=float, default=4.0)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--phase2-patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated KD methods (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as original run's output-dir with _resume suffix)")
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_checkpoints()
        return

    # Resolve checkpoint path
    ckpt_path: str | None = args.checkpoint
    if ckpt_path is None and args.pick is not None:
        checkpoints = list_checkpoints()
        if not checkpoints:
            sys.exit(1)
        idx = args.pick - 1
        if idx < 0 or idx >= len(checkpoints):
            print(f"\nInvalid pick: {args.pick} (must be 1-{len(checkpoints)})")
            sys.exit(1)
        ckpt_path = str(checkpoints[idx])
        print(f"\nUsing checkpoint [{args.pick}]: {ckpt_path}")

    if ckpt_path is None:
        print("Error: provide --checkpoint or --pick (use --list to see available checkpoints)")
        sys.exit(1)

    # Load original config to get defaults
    run_dir = Path(ckpt_path).parent.parent
    config_path = run_dir / "config.json"
    original_cfg: dict = {}
    if config_path.exists():
        with config_path.open() as f:
            original_cfg = json.load(f)
        print(f"Loaded original config from: {config_path}")

    # Build the command
    teacher = args.teacher_model or original_cfg.get("teacher_model", "sentence-transformers/all-mpnet-base-v2")
    student = args.student_model or original_cfg.get("student_model", "sentence-transformers/all-MiniLM-L6-v2")
    dataset = args.dataset_name or original_cfg.get("dataset_name", "code_search_net")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        orig_output = original_cfg.get("output_dir", str(run_dir.parent.name))
        output_dir = f"{orig_output}_resume_dt{args.distill_temperature}"

    from mbpp_kd_suite.two_phase_kd_experiment import run
    from mbpp_kd_suite.constants import KD_METHOD_ORDER

    methods_str = args.methods
    if methods_str:
        methods = tuple(m.strip() for m in methods_str.split(",") if m.strip())
    else:
        methods = tuple(KD_METHOD_ORDER)

    print(f"\nResuming phase 2:")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Teacher: {teacher}")
    print(f"  Student: {student}")
    print(f"  Dataset: {dataset}")
    print(f"  Distill temperature: {args.distill_temperature}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Output: {output_dir}")
    print()

    run(
        teacher_model=teacher,
        student_model=student,
        dataset_name=dataset,
        phase1_epochs=20,  # ignored when resuming
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        seed=args.seed,
        distill_temperature=args.distill_temperature,
        output_dir=output_dir,
        skip_diagnostics=args.skip_diagnostics,
        methods=methods,
        resume_from_phase1=ckpt_path,
        phase2_patience=args.phase2_patience,
    )


if __name__ == "__main__":
    main()
