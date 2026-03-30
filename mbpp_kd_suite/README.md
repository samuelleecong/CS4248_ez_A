# MBPP KD Suite

`mbpp_kd_suite` is a standalone `uv` project for comparing embedding-focused knowledge distillation methods on MBPP-style code retrieval.

Core training and modeling code lives under `src/mbpp_kd_suite/`. The standalone decoupled evaluator and its local run history now live entirely under `eval/`.

## What This Repo Covers

- Retrieval datasets:
  - `google-research-datasets/mbpp` by default
  - `BEE-spoke-data/TACO-hf` via `--dataset-name`
- Reported metrics:
  - `MRR`
  - `Recall@1/5/10`
  - `nDCG@k`
  - `MAP@k`
  - `MedianRank`
- Baselines:
  - `direct_big_teacher`
  - `direct_small_student`
  - `supervised_student`
- Distillation methods:
  - `score_distill`
  - `embed_distill`
  - `qed_align`
  - `distilcse_lite`
  - `hard_negative_pair_distill`
  - `all_pairs_distill`
  - `adam_lite`
  - `hpd`

## Start Here

If you are new to the repo, use this reading order:

1. `README.md`
2. [DISTILL_METHOD_QUICKSTART.md](DISTILL_METHOD_QUICKSTART.md) if you only care about `embed_distill` or `hard_negative_pair_distill`
3. `docs/PAPER_IMPLEMENTATIONS.md`
4. `docs/PROJECT_STATUS.md`
5. `docs/EXPERIMENT_LOG.md`
6. `src/mbpp_kd_suite/experiment.py`
7. `src/mbpp_kd_suite/training.py`
8. `src/mbpp_kd_suite/modeling.py`
9. `src/mbpp_kd_suite/data.py`
10. `eval/run.py`
11. `eval/README.md`

## Quick Start

```bash
cd mbpp_kd_suite
uv sync
uv run mbpp-kd-suite
```

If MBPP is already cached locally, the suite reads the Arrow files directly and avoids write-lock issues in the global Hugging Face cache.

If you only want the `embed_distill` or `hard_negative_pair_distill` path, start with [DISTILL_METHOD_QUICKSTART.md](DISTILL_METHOD_QUICKSTART.md).

## Common Commands

Run the default benchmark:

```bash
uv run mbpp-kd-suite
```

Run a short smoke check:

```bash
uv run mbpp-kd-suite --epochs 1 --batch-size 16 --eval-batch-size 32 --output-dir mbpp/smoke
```

Compare a subset of methods:

```bash
uv run mbpp-kd-suite --methods score_distill,embed_distill,hard_negative_pair_distill,all_pairs_distill
```

Switch to TACO:

```bash
uv run mbpp-kd-suite --dataset-name BEE-spoke-data/TACO-hf --output-dir taco/fair_compare
```

Use a stronger teacher and cross-space projection:

```bash
uv run mbpp-kd-suite --teacher-model sentence-transformers/all-mpnet-base-v2 --methods embed_distill --projection-init least_squares_both
```

Inspect saved runs:

```bash
uv run mbpp-kd-inventory
uv run mbpp-kd-inventory --json
```

Run the standalone evaluator against a saved checkpoint or a Hugging Face model:

```bash
uv run mbpp-kd-eval \
  --dataset-name mbpp \
  --model-source hf \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --split test
```

Detailed evaluator usage, flag behavior, input path rules, and output layout live in [eval/README.md](eval/README.md). Keep that file as the source of truth for evaluator operations.

## Two-Phase KD Experiments

Run a full two-phase experiment (phase 1: finetune teacher + student, phase 2: KD methods):

```bash
uv run mbpp-kd-two-phase \
  --student-model sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-model sentence-transformers/all-mpnet-base-v2 \
  --dataset-name BEE-spoke-data/TACO-hf \
  --distill-temperature 0.2 \
  --distill-weight 50 \
  --batch-size 32 \
  --eval-batch-size 64
```

### Key Hyperparameters

| Parameter | Default | What it controls |
|-----------|--------:|------------------|
| `--temperature` | 0.05 | Contrastive (supervised) loss sharpness |
| `--distill-temperature` | 4.0 | KD softmax temperature (use 0.2 for cosine similarities) |
| `--distill-weight` | 1.0 | Weight for `distill_kl` and `dark_kl` losses (use 50 for meaningful KD) |
| `--align-weight` | 1.0 | Weight for embedding alignment loss (`embed_distill`, `qed_align`, `hpd`) |
| `--pair-weight` | 1.0 | Weight for pairwise preference loss (`hard_negative_pair_distill`, `margin_mse`) |
| `--relation-weight` | 1.0 | Weight for relation/contrastive loss (`distilcse_lite`) |

**Why `distill-weight=50`?** The KD loss (`distill_kl`) is scaled by T² internally. With cosine similarities in [-1, 1] and `distill-temperature=0.2`, T²=0.04 shrinks the raw KL from ~0.19 to ~0.007 — negligible vs the supervised loss (~0.3). Setting `distill-weight=50` compensates, making KD contribute ~0.35 to match the supervised signal.

### Resuming Phase 2 from a Checkpoint

Phase 1 saves a checkpoint at `<run_dir>/phase1/checkpoint.pt`. You can skip phase 1 entirely and re-run phase 2 with different hyperparameters (e.g., temperature, methods, epochs) without retraining teacher or student. Previous run directories are never overwritten — each resume creates a new timestamped directory.

List available checkpoints:

```bash
uv run python resume_phase2.py --list
```

Resume with a different distill temperature:

```bash
uv run python resume_phase2.py \
  --pick 1 \
  --distill-temperature 8.0
```

Resume with specific methods only:

```bash
uv run python resume_phase2.py \
  --checkpoint artifacts/.../phase1/checkpoint.pt \
  --methods embed_distill,margin_mse,adam_lite \
  --distill-temperature 2.0
```

Sweep temperatures from the same checkpoint:

```bash
for dt in 1.0 2.0 4.0 8.0; do
  uv run python resume_phase2.py --pick 1 --distill-temperature $dt --output-dir sweep_dt${dt}
done
```

You can also use the built-in flag directly:

```bash
uv run mbpp-kd-two-phase \
  --resume-from-phase1 artifacts/.../phase1/checkpoint.pt \
  --distill-temperature 4.0 \
  --methods embed_distill,score_distill
```

### Hyperparameter Sweep

`sweep_kd_params.py` runs a systematic sweep over KD hyperparameters from a phase 1 checkpoint. Edit `CHECKPOINT`, `METHODS`, and the `configs` list at the bottom of the file, then run:

```bash
uv run python sweep_kd_params.py
```

All configs run sequentially on one GPU for maximum throughput.

**Which parameters matter most** (ranked by impact from our experiments):

| Rank | Parameter | Impact | Notes |
|:----:|-----------|--------|-------|
| 1 | `distill_weight` | +0.029 MRR at dw=50 | Scales KD loss to match supervised loss. Default 1.0 is far too low. |
| 2 | `align_weight` | +0.024 MRR at aw=5 | Boosts embedding alignment for `embed_distill`, `qed_align`, `hpd`. |
| 3 | `batch_size` | +0.012 MRR at bs=128 | More in-batch negatives = richer teacher signal. |
| 4 | `distill_temperature` | dt=0.2 is optimal | For cosine similarities in [-1,1]. Higher or lower hurts. |
| 5 | `lr` | No improvement | Default 2e-5 is fine. Lower (5e-6) hurts. |

The default configs in the script sweep `distill_weight` (25/50/100) and `align_weight` (5/10), plus a best-combo candidate (dw=100, aw=5).

**Output structure:**

```text
artifacts/sweep_kd_params/<timestamp>/
  sweep_index.json          # all configs + params + results in one file
  results_summary.json      # flat map of config/method -> metrics
  control_supervised/       # baseline (no KD)
  dw50/                     # one dir per sweep config
    sweep_config.json        # hyperparameters used
    embed_distill/
      metrics.json           # final test metrics
      history.json           # per-epoch training curves
  tensorboard/              # loss curves for all configs
```

**To customise**, edit the `configs` list:

```python
configs = [
    SweepConfig(name="dw100",       distill_temperature=0.2, distill_weight=100.0, methods=METHODS),
    SweepConfig(name="aw10_dw50",   distill_temperature=0.2, distill_weight=50.0, align_weight=10.0, methods=METHODS),
    SweepConfig(name="dw100_aw5",   distill_temperature=0.2, distill_weight=100.0, align_weight=5.0, methods=METHODS),
]
```

### Monitoring

TensorBoard logs are saved automatically for every run. To view live loss curves:

```bash
uv run tensorboard --logdir artifacts/
```

Or use the lightweight CLI monitor:

```bash
uv run python monitor.py                          # auto-detect latest run
uv run python monitor.py artifacts/<run_dir>/      # specific run
```

## Papers

Download or refresh the local paper copies with:

```bash
bash papers/download_papers.sh
```

Existing PDFs under `assignment_details/papers/` are reused when available; missing papers are downloaded from primary sources.

## Output Layout

Relative `--output-dir` values for training runs are written under `artifacts/`.

Examples:

```text
uv run mbpp-kd-suite
-> artifacts/runs/<timestamp>/

uv run mbpp-kd-suite --output-dir mbpp/fair_compare
-> artifacts/mbpp/fair_compare/<timestamp>/

uv run mbpp-kd-suite --output-dir ./scratch/manual_check
-> ./scratch/manual_check/<timestamp>/
```

Each training run directory contains:

- `config.json`
- `paper_registry.json`
- `results_summary.json`
- `diagnostics_summary.json`
- `<method>/history.json`
- `<method>/metrics.json`
- `<method>/model/` when `--save-models` is enabled

Historical runs from the earlier flat layout now live under `artifacts/legacy/`.

## Repo Layout

- `src/mbpp_kd_suite/experiment.py`: top-level run orchestration and CLI entrypoint
- `src/mbpp_kd_suite/training.py`: KD objectives, target-space preparation, and the student training loop
- `src/mbpp_kd_suite/metrics.py`: retrieval metrics, evaluation helpers, and summary analysis
- `src/mbpp_kd_suite/modeling.py`: encoder wrappers, pooling, and embedding helpers
- `src/mbpp_kd_suite/data.py`: dataset loading and dataloader construction
- `src/mbpp_kd_suite/config.py`: `TrainConfig`, output-dir resolution, and CLI parsing
- `src/mbpp_kd_suite/runtime.py`: device selection, cache clearing, and MPS-safe runtime tuning
- `src/mbpp_kd_suite/experiment_inventory.py`: CLI for locating saved experiment runs
- `eval/`: standalone decoupled evaluation package
- `eval/README.md`: evaluator-focused layout and usage
- `eval/tests/`: unit tests for eval adapters and retrieval metrics
- `eval/runs/`: repo-local evaluator run history; generated outputs remain git-ignored
- `papers/`: paper registry and download script
- `DISTILL_METHOD_QUICKSTART.md`: focused quickstart for the `embed_distill` / pairwise KD workflows
- `docs/PAPER_IMPLEMENTATIONS.md`: method notes and faithfulness/gap summary
- `docs/ORGANIZATION.md`: suite layout and experiment storage conventions
- `docs/EMBEDDISTILL_MERMAID.md`: focused Mermaid diagram for the `embed_distill` method
- `docs/embeddistill_loss_explainer.html`: HTML explainer for the `embed_distill` losses, tensors, and KL intuition
- `docs/STUDENT_TEACHER_MERMAID.md`: Mermaid diagrams for the current student-teacher setup
- `docs/EXPERIMENT_LOG.md`: run history
- `docs/PROJECT_STATUS.md`: current scope and next steps
