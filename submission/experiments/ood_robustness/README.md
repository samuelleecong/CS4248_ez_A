# OOD + Perturbation Robustness

Out-of-distribution (MBPP) and perturbation (TACO) evaluation for the 7-model student set distilled from the fine-tuned mpnet teacher.

## Files

- `ood_taco_robustness_colab.ipynb` — runs the eval. Clones this repo into Colab, installs `mbpp_kd_suite`, downloads each student model from HF, and produces per-run CSVs + JSONs (`metrics.csv`, `per_query_results.csv`, `lexical_probe_results.csv`, `summary.{json,md}`, `analysis_summary.md`, `example_cases.csv`).
- `analysis.ipynb` — consumes the per-run outputs and produces the 4 writeup panels: MBPP OOD MRR, overall TACO perturbation robustness, lexical perturbation probe, and qualitative example cases.

## Underlying library

All the heavy lifting lives in `mbpp_kd_suite/eval/ood_analysis/`:

| Entry point | What it does |
|---|---|
| `ood_analysis.run_workflow(WorkflowConfig(...))` / `python -m mbpp_kd_suite.eval.ood_analysis.ood_robustness …` | Runs both MBPP OOD and TACO perturbation eval for one or more model IDs. |
| `ood_analysis.perturb_queries(...)` + `PERTURBATION_TIERS` | Apply a tier of query perturbations (typos, paraphrase, lexical swaps, etc.). |
| `ood_analysis.LEXICAL_PROBE_TIERS` + `lexical_replacements.json` | Curated synonym / paraphrase substitutions for the lexical probe. |
| `ood_analysis.load_mbpp_ood_corpus(...)` / `load_taco_retrieval_corpus(...)` | Build the OOD / TACO corpora consumed by the workflow. |

A reference smoke run is checked in at `mbpp_kd_suite/eval/scratch_eval_smoke/runs/20260409_134818/` — useful as a schema example for what the notebook produces per model.

## How to run

1. Open `ood_taco_robustness_colab.ipynb` in Colab.
2. **Set `GIT_BRANCH = "main"`** in the second cell (the notebook still defaults to `"ood_perturb"` — that branch was merged into `main`).
3. Optionally set `AUTO_DISCOVER_MODELS = True` and provide an HF token to discover all `cs4248-nlp/paper-*` checkpoints; otherwise it uses the default 7-model `MODEL_IDS` list (s7 / s8 / s9 / s10 saturated set).
4. Run all cells. Outputs land in `mbpp_kd_suite/eval/runs/<timestamp>/<model_id>/`.
5. Open `analysis.ipynb` and point `RUNS_DIR` at the same `runs/<timestamp>/` directory to regenerate the figures.

## Local (non-Colab) run

```bash
cd mbpp_kd_suite
.venv/bin/python -m mbpp_kd_suite.eval.ood_analysis.ood_robustness \
    --model cs4248-nlp/paper-s10-bimga-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143 \
    --task all \
    --split test \
    --perturbation-tier all \
    --output-dir eval/runs/local
```

Add `--model` repeatedly (or `--models-file`) to evaluate multiple checkpoints. Pass `--task mbpp_ood` or `--task taco_robustness` to run only one half.
