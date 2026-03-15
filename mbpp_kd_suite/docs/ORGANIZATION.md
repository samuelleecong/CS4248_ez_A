# Organization

This note describes where code, papers, caches, and experiment artifacts live in `mbpp_kd_suite`.

## Top-Level Layout

- `src/mbpp_kd_suite/`: package code split by concern
- `docs/`: run history, implementation notes, status notes, and diagrams
- `papers/`: local PDF copies and the paper registry
- `artifacts/`: recommended home for all new experiment outputs
- `.hf_cache/`: local Hugging Face cache kept inside the suite
- `.venv/`: local virtual environment

## Package Layout

- `experiment.py`: top-level run orchestration and CLI entrypoint
- `training.py`: KD losses, target-space preparation, and the student training loop
- `metrics.py`: evaluation helpers and summary analysis
- `modeling.py`: encoder wrappers and embedding helpers
- `data.py`: retrieval dataset loading and dataloaders
- `config.py`: config dataclasses and CLI parsing
- `runtime.py`: device/runtime helpers

Import from `mbpp_kd_suite` modules directly; there is no compatibility shim package in this repo.

## Experiment Output Layout

The training pipeline writes each run to:

```text
<resolved-output-dir>/<timestamp>/
```

The timestamp format is `YYYYMMDD_HHMMSS`.

For normal usage, `<resolved-output-dir>` is derived like this:

- `uv run mbpp-kd-suite`
  saves to `artifacts/runs/<timestamp>/`
- `uv run mbpp-kd-suite --output-dir mbpp/smoke`
  saves to `artifacts/mbpp/smoke/<timestamp>/`
- `uv run mbpp-kd-suite --output-dir ./scratch/check`
  saves to `./scratch/check/<timestamp>/`
- `uv run mbpp-kd-suite --output-dir /tmp/mbpp-check`
  saves to `/tmp/mbpp-check/<timestamp>/`

Each run directory contains:

- `config.json`
- `paper_registry.json`
- `results_summary.json`
- `diagnostics_summary.json`
- `<method>/history.json`
- `<method>/metrics.json`
- `<method>/model/` when `--save-models` is enabled

## Recommended Naming

Use dataset-first buckets so runs stay easy to scan:

- `artifacts/mbpp/smoke/`
- `artifacts/mbpp/fair_compare/`
- `artifacts/mbpp/baselines/`
- `artifacts/taco/smoke/`
- `artifacts/taco/fair_compare/`
- `artifacts/taco/mpnet/`

This works with the existing CLI because `--output-dir` accepts nested relative paths.

## Historical Runs

Older runs from the earlier flat layout were moved under `artifacts/legacy/`.
Examples include:

- `artifacts/legacy/runs/`
- `artifacts/legacy/smoke_runs/`
- `artifacts/legacy/quick_compare/`
- `artifacts/legacy/longer_compare/`
- `artifacts/legacy/fair_compare_mbpp/`
- `artifacts/legacy/taco_smoke_runs/`

Those are still valid experiment records; only their parent location changed.

## Inventory Command

Use the built-in inventory helper to find saved runs and their bucket paths:

```bash
uv run mbpp-kd-inventory
uv run mbpp-kd-inventory --json
```
