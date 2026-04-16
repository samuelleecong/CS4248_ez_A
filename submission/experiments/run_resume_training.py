"""Resume training from saved checkpoints until early stopping triggers.

Loads training_checkpoint.pt (model + optimizer state) and continues
training from where it left off.

Usage:
    .venv/Scripts/python.exe run_resume_training.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from mbpp_kd_suite.config import DistillTargets, TrainConfig
from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.metrics import evaluate_symmetric_student
from mbpp_kd_suite.modeling import StudentQueryEncoder
from mbpp_kd_suite.runtime import (
    apply_device_runtime_optimizations,
    maybe_empty_device_cache,
    pick_device,
    set_seed,
)
from mbpp_kd_suite.training import (
    _build_train_loader,
    _compute_kd_batch_losses,
    _build_diagnostics,
    make_method_targets,
)

PHASE1_CHECKPOINT = "artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt"
RESULTS_DIR = Path("artifacts/paper_experiments/20260402_015143")
STUDENT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
ADDITIONAL_EPOCHS = 50  # Train up to this many MORE epochs
PATIENCE = 10
EVAL_BATCH_SIZE = 64

# Runs to resume (not saturated at 70 epochs)
RUNS_TO_RESUME = {
    "s7_bimga_dw50_aw10": {"method": "bimga", "distill_weight": 50, "align_weight": 10},
    "s7_A2_bimga_uniform": {"method": "bimga_uniform", "distill_weight": 100, "align_weight": 10},
    "s7_hnp_dw100_pw10": {"method": "hard_negative_pair_distill", "distill_weight": 100, "pair_weight": 10},
    "s7_score_dw100": {"method": "score_distill", "distill_weight": 100},
}


def resume_and_train(
    run_name: str,
    method_cfg: dict,
    device: torch.device,
    data,
    ft_teacher_targets: DistillTargets,
):
    """Resume training from checkpoint and continue until early stopping."""
    run_dir = RESULTS_DIR / run_name
    method_name = method_cfg["method"]

    # Find checkpoint
    ckpt_dirs = list(run_dir.glob("*/training_checkpoint.pt"))
    if not ckpt_dirs:
        print(f"  No checkpoint found in {run_dir}, skipping")
        return None
    ckpt_path = ckpt_dirs[0]
    print(f"  Loading checkpoint: {ckpt_path}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    start_epoch = ckpt["epoch"]
    best_val_mrr = ckpt["best_val_mrr"]
    print(f"  Resuming from epoch {start_epoch}, best val MRR={best_val_mrr:.4f}")

    # Build config
    cfg = TrainConfig(
        teacher_model="sentence-transformers/all-MiniLM-L6-v2",
        student_model=STUDENT_MODEL,
        dataset_name="BEE-spoke-data/TACO-hf",
        epochs=start_epoch + ADDITIONAL_EPOCHS,
        batch_size=32,
        eval_batch_size=EVAL_BATCH_SIZE,
        lr=2e-5,
        seed=42,
        distill_temperature=0.2,
        distill_weight=method_cfg.get("distill_weight", 1.0),
        align_weight=method_cfg.get("align_weight", 1.0),
        pair_weight=method_cfg.get("pair_weight", 1.0),
        save_models=True,
        run_diagnostics=True,
        early_stopping_patience=PATIENCE,
    )
    apply_device_runtime_optimizations(cfg=cfg, device=device)

    # Rebuild model and load state
    target_hidden_size = ft_teacher_targets.train_query.shape[-1]
    use_attn_pool = "attn_pool" in method_name
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = StudentQueryEncoder(
        model_name=STUDENT_MODEL,
        target_hidden_size=target_hidden_size,
        use_attention_pool=use_attn_pool,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Rebuild optimizer and load state
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Move optimizer state to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Build data loader and targets
    method_targets = make_method_targets(cfg=cfg, full_teacher_targets=ft_teacher_targets)
    targets = method_targets[method_name]

    train_loader = _build_train_loader(
        name=method_name,
        split=data.train,
        tokenizer=tokenizer,
        student_model=model,
        cfg=cfg,
    )

    # Load existing history
    history_path = list(run_dir.glob("*/history.json"))
    history = []
    if history_path:
        with history_path[0].open() as f:
            history = json.load(f)

    # Continue training
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    stopped_epoch = cfg.epochs
    total_epochs = cfg.epochs

    for epoch in range(start_epoch + 1, total_epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            losses = _compute_kd_batch_losses(
                name=method_name,
                student_model=model,
                batch=batch,
                targets=targets,
                full_teacher_targets=ft_teacher_targets,
                cfg=cfg,
                device=device,
                epoch=epoch,
                total_epochs=total_epochs,
            )
            loss = losses.total(cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)

        # Evaluate
        model.eval()
        val_metrics = evaluate_symmetric_student(
            student_model=model, tokenizer=tokenizer,
            queries=data.validation.queries, codes=data.validation.codes,
            max_query_length=cfg.max_query_length, max_code_length=cfg.max_code_length,
            eval_batch_size=cfg.eval_batch_size, device=device,
        )
        val_mrr = val_metrics["MRR"]

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "validation_MRR": val_mrr,
        })

        improved = "NEW BEST" if val_mrr > best_val_mrr else ""
        print(f"  Epoch {epoch}/{total_epochs}: loss={avg_loss:.4f}, val_MRR={val_mrr:.4f} {improved}")

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                stopped_epoch = epoch
                break

    # Restore best model
    model.load_state_dict(best_state_dict)
    model.eval()

    # Final evaluation
    test_metrics = evaluate_symmetric_student(
        student_model=model, tokenizer=tokenizer,
        queries=data.test.queries, codes=data.test.codes,
        max_query_length=cfg.max_query_length, max_code_length=cfg.max_code_length,
        eval_batch_size=cfg.eval_batch_size, device=device,
    )

    diagnostics = _build_diagnostics(
        student_model=model, tokenizer=tokenizer,
        data=data, targets=targets, cfg=cfg, device=device,
    )

    print(f"  Final test MRR={test_metrics['MRR']:.4f} (stopped at epoch {stopped_epoch})")

    # Save updated results
    exp_dir = list(run_dir.glob("*/"))[0]  # method subdirectory
    with (exp_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    metrics = {
        "model_name": STUDENT_MODEL,
        "target_space": "finetuned_teacher",
        "evaluation_mode": "symmetric",
        "stopped_epoch": stopped_epoch,
        "train": {},
        "validation": val_metrics,
        "test": test_metrics,
        "diagnostics": diagnostics,
    }
    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Save new checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": stopped_epoch,
        "best_val_mrr": best_val_mrr,
    }, exp_dir / "training_checkpoint.pt")

    # Save model
    if cfg.save_models:
        model_dir = exp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.backbone.save_pretrained(model_dir / "backbone")
        tokenizer.save_pretrained(model_dir / "tokenizer")
        if not isinstance(model.proj, torch.nn.Identity):
            torch.save(model.proj.state_dict(), model_dir / "projection.pt")

    return metrics


def main():
    set_seed(42)
    device = pick_device()
    print(f"Device: {device}")

    # Load dataset
    dataset = load_retrieval_dataset(dataset_name="BEE-spoke-data/TACO-hf", taco_val_size=1000, seed=42)
    data = dataset_dict_to_splits(dataset)

    # Load teacher targets
    ckpt = torch.load(PHASE1_CHECKPOINT, map_location="cpu", weights_only=False)
    ft_teacher_targets = DistillTargets(**ckpt["ft_teacher_targets"])

    # Update results_summary.json after each run
    results_path = RESULTS_DIR / "results_summary.json"
    with results_path.open() as f:
        all_results = json.load(f)

    for run_name, method_cfg in RUNS_TO_RESUME.items():
        print(f"\n{'='*60}")
        print(f"  Resuming: {run_name} ({method_cfg['method']})")
        print(f"{'='*60}")

        try:
            metrics = resume_and_train(
                run_name=run_name,
                method_cfg=method_cfg,
                device=device,
                data=data,
                ft_teacher_targets=ft_teacher_targets,
            )
            if metrics:
                all_results[run_name] = metrics
                with results_path.open("w") as f:
                    json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        maybe_empty_device_cache(device)

    print("\nDone!")


if __name__ == "__main__":
    main()
