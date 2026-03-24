# Eval Runs

This directory stores standalone evaluator outputs over time.

Structure:
- `<dataset>/<split>/<timestamp>_<model>/`: per-run artifacts

Per-run artifacts include:
- `summary.md`
- `metrics.csv`
- `metrics.json`
- `profiling.csv`
- `config.json`
- plot images

All generated contents under this directory are local-only and ignored by git.
That includes aggregate indices such as `run_index.csv` and `run_index.md` when they are regenerated on a developer machine.
