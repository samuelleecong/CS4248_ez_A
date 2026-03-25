# MBPP KD Suite

`mbpp_kd_suite` is a standalone `uv` project for comparing embedding-focused knowledge distillation methods on MBPP-style code retrieval.

The supported code lives under `src/mbpp_kd_suite/`. There are no legacy compatibility entrypoints in this repo; import from the concrete modules directly.

## What This Repo Covers

- Retrieval datasets:
  - `google-research-datasets/mbpp` by default
  - `BEE-spoke-data/TACO-hf` via `--dataset-name`
- Reported metrics:
  - `MRR`
  - `Recall@1/5/10`
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
  - `pair_distill`
  - `adam_lite`
  - `hpd`

## Start Here

If you are new to the repo, use this reading order:

1. `README.md`
2. [DISTILL_METHOD_QUICKSTART.md](DISTILL_METHOD_QUICKSTART.md) if you only care about `embed_distill` or `pair_distill`
3. `docs/PAPER_IMPLEMENTATIONS.md`
4. `docs/PROJECT_STATUS.md`
5. `docs/EXPERIMENT_LOG.md`
6. `src/mbpp_kd_suite/experiment.py`
7. `src/mbpp_kd_suite/training.py`
8. `src/mbpp_kd_suite/modeling.py`
9. `src/mbpp_kd_suite/data.py`

## Quick Start

```bash
cd mbpp_kd_suite
uv sync
uv run mbpp-kd-suite
```

If MBPP is already cached locally, the suite reads the Arrow files directly and avoids write-lock issues in the global Hugging Face cache.

If you only want the `embed_distill` or `pair_distill` path, start with [DISTILL_METHOD_QUICKSTART.md](DISTILL_METHOD_QUICKSTART.md).

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
uv run mbpp-kd-suite --methods score_distill,embed_distill,pair_distill
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

Train and evaluate a cross-encoder reranker teacher first:

```bash
uv run mbpp-kd-cross-teacher \
  --model-name microsoft/codebert-base \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --negative-strategy mixed \
  --negative-miner-model sentence-transformers/all-mpnet-base-v2 \
  --train-objective combined \
  --baseline-bi-encoder-model sentence-transformers/all-mpnet-base-v2 \
  --output-dir teachers/cross_encoder_smoke
```

If `--model-name` points to a generic encoder checkpoint instead of an existing reranker,
the command loads it as a single-score sequence-classification model and fine-tunes that head
for retrieval reranking.

The cross-encoder teacher now supports stronger training measures:

- hard negative mining from a configurable bi-encoder miner
- mixed hard+random negative groups
- per-query grouped softmax ranking loss, optionally combined with pairwise BCE
- side-by-side comparison against a chosen bi-encoder baseline teacher

See `docs/CROSS_ENCODER_TEACHER.md` for the detailed workflow and rationale.

Compare a cross-encoder against a bi-encoder fairly on the same candidate pool:

```bash
uv run mbpp-kd-compare-cross-vs-bi \
  --cross-encoder-model mixedbread-ai/mxbai-rerank-base-v1 \
  --bi-encoder-model sentence-transformers/all-mpnet-base-v2 \
  --protocol heldout_test \
  --rerank-top-k 10,25,50 \
  --output-dir ../experiments/immanuel_tim/cross_vs_mpnet_pipeline
```

This command now reports three comparable views on the same candidate pool:

- standalone cross-encoder full-pool ranking
- standalone bi-encoder full-pool ranking
- bi-encoder retrieval followed by cross-encoder reranking of the bi-encoder top-k candidates

`--projection-init least_squares_queries` and `--projection-init least_squares_both` are meant for cross-family KD runs where the student must learn a new embedding dimension, such as `MiniLM -> MPNet`.
The default evaluation mode is `symmetric`, which makes checkpoint selection and reported metrics use student-query x student-code retrieval for fair trained-student comparisons.
`--eval-mode asymmetric` keeps the teacher-document evaluation path for explicit ablation runs.
`--optimize-for-mps` keeps MPS selected as the device, forces high matmul precision, and clamps the run to more conservative batch sizes (`8` train, `16` eval) to avoid the worst memory-fragmentation and thrashing cases on Apple Silicon.
For `TACO`, the suite converts each problem into one retrieval pair by using the problem statement plus starter code as the query and the first non-empty verified solution as the code target, then derives a validation split from train via `--taco-val-size`.

## Papers

Download or refresh the local paper copies with:

```bash
bash papers/download_papers.sh
```

Existing PDFs under `assignment_details/papers/` are reused when available; missing papers are downloaded from primary sources.

## Output Layout

Relative `--output-dir` values are written under `artifacts/`.

Examples:

```text
uv run mbpp-kd-suite
-> artifacts/runs/<timestamp>/

uv run mbpp-kd-suite --output-dir mbpp/fair_compare
-> artifacts/mbpp/fair_compare/<timestamp>/

uv run mbpp-kd-suite --output-dir ./scratch/manual_check
-> ./scratch/manual_check/<timestamp>/
```

Each run directory contains:

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
- `src/mbpp_kd_suite/cross_encoder_teacher.py`: teacher-first training and evaluation for cross-encoder rerankers
- `src/mbpp_kd_suite/compare_cross_vs_biencoder.py`: fair full-pool comparison between a cross-encoder and a bi-encoder
- `papers/`: paper registry and download script
- `DISTILL_METHOD_QUICKSTART.md`: focused quickstart for the `embed_distill` / `pair_distill` workflows
- `docs/PAPER_IMPLEMENTATIONS.md`: method notes and faithfulness/gap summary
- `docs/CROSS_ENCODER_TEACHER.md`: cross-encoder teacher training recipe and comparison workflow
- `docs/ORGANIZATION.md`: suite layout and experiment storage conventions
- `docs/EMBEDDISTILL_MERMAID.md`: focused Mermaid diagram for the `embed_distill` method
- `docs/embeddistill_loss_explainer.html`: HTML explainer for the `embed_distill` losses, tensors, and KL intuition
- `docs/STUDENT_TEACHER_MERMAID.md`: Mermaid diagrams for the current student-teacher setup
- `docs/EXPERIMENT_LOG.md`: run history
- `docs/PROJECT_STATUS.md`: current scope and next steps
