# Distill Method Quickstart

This quickstart is for contributors who only want to run one of the two strongest KD baselines in this repo:

- `embed_distill`
- `pair_distill`

## Which One To Pick

- `embed_distill`: retrieval loss + score KL + direct query-embedding alignment
- `pair_distill`: retrieval loss + score KL + pairwise preference loss over teacher-chosen hard negatives

If you want the simpler geometric baseline, start with `embed_distill`.
If you want the ranking-oriented variant, start with `pair_distill`.

## Fastest Way To Run Either Method

```bash
cd mbpp_kd_suite
uv sync
uv run mbpp-kd-suite --methods embed_distill
uv run mbpp-kd-suite --methods pair_distill
```

Each run writes artifacts to:

```text
artifacts/runs/<timestamp>/
```

## Recommended Smoke Runs

Use one of these first:

```bash
uv run mbpp-kd-suite \
  --methods embed_distill \
  --epochs 1 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --output-dir mbpp/embed_distill_smoke
```

```bash
uv run mbpp-kd-suite \
  --methods pair_distill \
  --epochs 1 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --output-dir mbpp/pair_distill_smoke
```

## How To Compare Them Fairly

Do not compare either method only against `direct_small_student`.

The fair trained baseline is `supervised_student`, because all three methods update a student model. Use:

```bash
uv run mbpp-kd-suite \
  --methods supervised_student,embed_distill,pair_distill \
  --epochs 1 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --eval-mode symmetric \
  --output-dir mbpp/distill_vs_supervised
```

## Method-Specific Knobs

For `embed_distill`, a stronger teacher plus projection initialization is the first thing worth trying:

```bash
uv run mbpp-kd-suite \
  --teacher-model sentence-transformers/all-mpnet-base-v2 \
  --methods embed_distill \
  --projection-init least_squares_both
```

For `pair_distill`, the main extra control is the number of teacher-selected hard negatives:

```bash
uv run mbpp-kd-suite \
  --methods pair_distill \
  --pair-hard-negatives 4
```

## TACO Variants

```bash
uv run mbpp-kd-suite \
  --dataset-name BEE-spoke-data/TACO-hf \
  --methods embed_distill \
  --epochs 1 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --output-dir taco/embed_distill_smoke
```

```bash
uv run mbpp-kd-suite \
  --dataset-name BEE-spoke-data/TACO-hf \
  --methods pair_distill \
  --epochs 1 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --output-dir taco/pair_distill_smoke
```

## What To Inspect After A Run

Inside the run directory, start with:

- `results_summary.json`
- `diagnostics_summary.json`
- `embed_distill/history.json` and `embed_distill/metrics.json`
- `pair_distill/history.json` and `pair_distill/metrics.json`

If you enabled `--save-models`, the saved checkpoint lives under `<method>/model/`.

## Where The Logic Lives

- `src/mbpp_kd_suite/experiment.py`: teacher loading, teacher precompute, run orchestration
- `src/mbpp_kd_suite/training.py`: `embed_distill` and `pair_distill` loss branches
- `src/mbpp_kd_suite/modeling.py`: student encoder and optional projection layer
- `src/mbpp_kd_suite/metrics.py`: symmetric vs asymmetric evaluation

## Read Next

- `docs/PAPER_IMPLEMENTATIONS.md`
- `docs/EMBEDDISTILL_MERMAID.md`
- `docs/embeddistill_loss_explainer.html`
