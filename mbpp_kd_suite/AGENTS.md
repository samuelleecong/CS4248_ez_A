# AGENTS.md (mbpp_kd_suite)

This `uv` subproject is the MBPP knowledge-distillation benchmark suite.

You are a 10x ml engineer. Organise your code clearly and reduce code duplication.

## Reading Order

1. `README.md`
2. `DISTILL_METHOD_QUICKSTART.md` if the task is specifically about `embed_distill` or `hard_negative_pair_distill`
3. `docs/PAPER_IMPLEMENTATIONS.md`
4. `docs/PROJECT_STATUS.md`
5. `docs/EXPERIMENT_LOG.md`
6. `src/mbpp_kd_suite/experiment.py`
7. `src/mbpp_kd_suite/training.py`
8. `src/mbpp_kd_suite/modeling.py`
9. `src/mbpp_kd_suite/data.py`

## Output Convention

- All experiment artifacts are written to `<output-dir>/<timestamp>/`.
- Relative `--output-dir` values are rooted under `artifacts/`, so the default CLI run now lands in `artifacts/runs/<timestamp>/`.
- Historical run buckets from the earlier flat layout now live under `artifacts/legacy/`, for example `artifacts/legacy/runs/` and `artifacts/legacy/fair_compare_mbpp/`.
- Shared outputs:
  - `config.json`
  - `results_summary.json`
  - `diagnostics_summary.json`
- Per-method outputs:
  - `<method>/history.json`
  - `<method>/metrics.json`

## Code Layout

- `src/mbpp_kd_suite/experiment.py`: top-level run orchestration and CLI entrypoint
- `src/mbpp_kd_suite/training.py`: KD objectives, target-space preparation, and student training loop
- `src/mbpp_kd_suite/metrics.py`: retrieval metrics, evaluation helpers, and run-summary analysis
- `src/mbpp_kd_suite/modeling.py`: encoder wrappers and text embedding helpers
- `src/mbpp_kd_suite/data.py`: dataset loading and PyTorch dataloaders
- `src/mbpp_kd_suite/config.py`: config dataclasses and CLI argument parsing
- `src/mbpp_kd_suite/runtime.py`: device selection, seeding, and MPS runtime tuning
