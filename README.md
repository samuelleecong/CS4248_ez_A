# CS4248_ez_A

NLP project on **knowledge distillation for text-to-code retrieval**, evaluated on the [TACO](https://huggingface.co/datasets/BEE-spoke-data/TACO-hf) competitive-programming dataset.

We distil a small TinyBERT-4L (14M params, 312d) student from a fine-tuned `all-mpnet-base-v2` teacher, comparing four distillation families (score distill, embed distill, hard-negative-pair distill, BiMGA) and probing where each method's gains come from.

## Repo Layout

```
.
├── mbpp_kd_suite/          # ★ main package: training, modeling, eval library
│   ├── src/mbpp_kd_suite/      training/modeling/data/metrics
│   ├── eval/                   decoupled evaluator + ood_analysis/ library
│   ├── README.md               package-level docs (read this first for code)
│   └── pyproject.toml          uv project, exposes the `mbpp-kd-suite` CLI
│
├── submission/             # ★ everything that ships with the paper / poster
│   ├── README.md               start here for the report artifacts
│   └── experiments/
│       ├── analysis/           core quantitative analysis (figures, csv, write-ups)
│       ├── analysis/new_teacher/   failure-overlap + margin-stratified MRR
│       ├── attention_analysis/  CKA + teacher-KL probing of BiMGA
│       ├── ood_robustness/      MBPP OOD + TACO perturbation notebooks
│       ├── artifacts/           per-run history.json / metrics.json / configs
│       └── run_*.py, train_*.py training entry points used to produce the runs
│
├── experiments/            # legacy per-teammate workspaces (kai/, sam/)
├── assignment_details/     # PROJ_reqs.md, report template, slide notes
├── ML_PIPELINE.md          # 7-stage pipeline walkthrough
├── MBPP_TUTORIAL.md        # task-specific deep dive (gen + search)
└── QUICKSTART.md           # 15-minute conceptual overview
```

The two folders that matter:

- **`mbpp_kd_suite/`** is the package — every model, loss, training loop, and evaluation lives here. All scripts in `submission/experiments/` import from it.
- **`submission/`** is the report bundle — figures, CSVs, per-run artifacts, and the analysis scripts that produced them. This is what gets referenced in the paper and on the poster.

## Setup

```bash
cd mbpp_kd_suite
uv sync                       # installs the package + all deps into .venv/
```

Run any command below from this `mbpp_kd_suite/` directory (or activate the venv) so `mbpp_kd_suite` is importable.

## Main Things to Run

### Train / sweep (produces the run artifacts)

```bash
# Default sweep over all distillation methods (writes to artifacts/)
uv run mbpp-kd-suite

# Or use one of the curated sweep entry points used for the paper:
.venv/bin/python ../submission/experiments/run_full_sweep.py
.venv/bin/python ../submission/experiments/run_saturation_sweep.py
.venv/bin/python ../submission/experiments/train_s10_bimga.py
```

Each run writes `history.json`, `metrics.json`, and `run_config.json` under `submission/experiments/artifacts/`.

### Evaluate a checkpoint

```bash
# Evaluator CLI for a single HF model:
.venv/bin/python -m mbpp_kd_suite.eval.run --model <hf_repo_id> --split test
```

### Reproduce the paper figures

```bash
# Replays all HF-uploaded checkpoints on the fixed TACO test split,
# rebuilds csv/, figures/, and figures_md/ used in the report.
.venv/bin/python ../submission/experiments/analysis/build_quant_package.py
```

### The other analyses (each has its own how-to in `submission/README.md`)

| Analysis | Run with |
|---|---|
| New-teacher diagnostics (failure overlap, margin-stratified MRR, rank CDF) | `python submission/experiments/person_e_analysis.py` |
| Attention / CKA probing (BiMGA mechanism) | `attention_probing.py` → `attention_skeptical.py` → `attention_teacher_kl.py` → `attention_poster_final.py` in `submission/experiments/attention_analysis/` |
| MBPP OOD + TACO perturbation robustness | open `submission/experiments/ood_robustness/ood_taco_robustness_colab.ipynb` in Colab |

## Models and Dataset

| Role | HuggingFace |
|------|-------------|
| Student backbone | `huawei-noah/TinyBERT_General_4L_312D` |
| Fine-tuned teacher | `cs4248-nlp/ft-teacher-all-mpnet-base-v2-taco-20260326-110507` |
| Per-run student checkpoints | `cs4248-nlp/paper-{run_name}-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| Dataset | `BEE-spoke-data/TACO-hf` (18K train / 1K val / 1K test) |

Full per-run table: `submission/experiments/README.md`.

## Reading Order

1. **`QUICKSTART.md`** — 15-minute conceptual overview.
2. **`mbpp_kd_suite/README.md`** — the code: package layout, distillation methods, eval CLI.
3. **`submission/README.md`** — the results: how each figure / CSV in the report was produced.
4. **`ML_PIPELINE.md`** + **`MBPP_TUTORIAL.md`** — long-form walkthroughs.
5. **`assignment_details/PROJ_reqs.md`** — the original assignment brief.

## Evaluation Metrics

- **MRR** — Mean Reciprocal Rank of the correct code snippet (primary metric)
- **Recall@k** — fraction of queries with the correct snippet in the top-k
- **nDCG@k**, **MAP@k**, **MedianRank** — additional retrieval metrics reported in the eval suite
