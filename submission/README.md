# CS4248 Submission — Knowledge Distillation for Text-to-Code Retrieval

This folder is the single home for everything that ships with the report: experiment artifacts, analyses, figures, and the scripts that produced them.

All checkpoints referenced below live on the [`cs4248-nlp` HuggingFace organization](https://huggingface.co/cs4248-nlp). The dataset is [`BEE-spoke-data/TACO-hf`](https://huggingface.co/datasets/BEE-spoke-data/TACO-hf).

## Layout

```
submission/
├── README.md                       ← you are here
└── experiments/
    ├── README.md                   ← detailed run table, training setup, HF model IDs
    ├── SESSION_CHANGELOG.md
    ├── analysis/                   ← core quantitative analysis (Sets 1-10 sweep)
    │   ├── README.md
    │   ├── csv/                    ← run registry, replayed metrics, per-query margins
    │   ├── figures/                ← paper figures (fig01…fig08)
    │   ├── figures_md/             ← one Markdown explainer per figure
    │   ├── new_teacher/            ← failure-overlap + margin-stratified MRR vs new teacher
    │   ├── poster/, significance/
    │   ├── build_quant_package.py
    │   ├── plot_margin_stratified_final_models.py
    │   └── run_saturated_significance.py
    ├── artifacts/                  ← per-run history.json / metrics.json / run_config.json
    ├── attention_analysis/         ← BiMGA CKA + teacher-KL attention probing
    │   ├── ANALYSIS_WRITEUP.md
    │   ├── POSTER_CAPTIONS.md
    │   ├── attention_probing.py, attention_skeptical.py
    │   ├── attention_teacher_kl.py, attention_poster_final.py
    │   └── poster_fig_A_cka.{png,pdf}, poster_fig_B_teacher_kl.{png,pdf}
    ├── ood_robustness/             ← MBPP OOD + TACO perturbation evaluation
    │   ├── README.md
    │   ├── ood_taco_robustness_colab.ipynb
    │   └── analysis.ipynb
    ├── person_e_analysis.py        ← driver for analysis/new_teacher/
    ├── attention_analysis.py, attention_final_figures.py, attention_figures/
    ├── run_*.py                    ← training/sweep entry points (sweep, resume, saturation, …)
    ├── train_s10_bimga.py
    └── (poster + routing diagram assets: kd_poster.*, fig_routing_*, formula_screenshots/, …)
```

The KD library that all of this is built on lives at `mbpp_kd_suite/` (one level up from `submission/`). The OOD robustness package lives at `mbpp_kd_suite/eval/ood_analysis/`.

## How to reproduce, by section

Run everything from the repo root unless noted. Activate the project venv (`mbpp_kd_suite/.venv/bin/activate` or equivalent) so that `mbpp_kd_suite` is importable.

### 1. Core quantitative analysis — `experiments/analysis/`

Replay the HF-hosted checkpoints on the fixed TACO test split and rebuild every CSV, figure, and figure explainer used in the paper.

```bash
cd mbpp_kd_suite
.venv/bin/python ../submission/experiments/analysis/build_quant_package.py
```

For just the 6 final saturated models (tercile bar, decile line, decile heatmap, fresh per-query CSV):

```bash
.venv/bin/python ../submission/experiments/analysis/plot_margin_stratified_final_models.py
```

Detailed options and outputs: see `experiments/analysis/README.md`.

### 2. New-teacher diagnostics — `experiments/analysis/new_teacher/`

Failure / success overlap heatmaps (Jaccard between methods), margin-stratified MRR (low / medium / high), and rank-distribution CDF — all evaluated against the fine-tuned mpnet teacher (`cs4248-nlp/ft-teacher-all-mpnet-base-v2-taco-20260326-110507`).

```bash
python submission/experiments/person_e_analysis.py
```

Outputs land in `submission/experiments/analysis/new_teacher/` (PNGs, CSVs, `summary.json`, and a 375k-row `per_query_results.csv`).

### 3. Attention / CKA probing — `experiments/attention_analysis/`

Probes whether BiMGA's advantage comes from middle-layer representational reshaping (CKA) or just output-layer alignment, and how teacher-like each method's attention distributions are (KL).

Run in this order:

```bash
python submission/experiments/attention_analysis/attention_probing.py    # CKA heatmaps + per-layer gain
python submission/experiments/attention_analysis/attention_skeptical.py  # bootstrap CIs + random-baseline checks
python submission/experiments/attention_analysis/attention_teacher_kl.py # per-layer teacher↔student KL
python submission/experiments/attention_analysis/attention_poster_final.py  # composes poster_fig_A and poster_fig_B
```

Reading order for the results: `ANALYSIS_WRITEUP.md` → `POSTER_CAPTIONS.md` → `poster_fig_A_cka.png` / `poster_fig_B_teacher_kl.png`.

### 4. OOD + perturbation robustness — `experiments/ood_robustness/`

Two notebooks designed for Colab; backed by the `mbpp_kd_suite/eval/ood_analysis/` library. Full instructions: `experiments/ood_robustness/README.md`.

- `ood_taco_robustness_colab.ipynb` — clones the repo into Colab, runs MBPP OOD eval and TACO perturbation eval across the 7-model set, writes per-run CSVs/JSONs.
- `analysis.ipynb` — consumes those CSVs and produces the 4-panel writeup figures (MBPP OOD MRR, overall TACO perturbation robustness, lexical perturbation probe, qualitative cases).

## Models and dataset

| Role | HuggingFace Repo |
|---|---|
| Fine-tuned teacher | `cs4248-nlp/ft-teacher-all-mpnet-base-v2-taco-20260326-110507` |
| Student backbone | `huawei-noah/TinyBERT_General_4L_312D` (4L, 312d, 14M params) |
| Dataset | `BEE-spoke-data/TACO-hf` (18K train / 1K val / 1K test) |

Per-run student checkpoints follow the pattern  
`cs4248-nlp/paper-{run_name}-tinybert-general-4l-312d-taco-hf-20260402-015143`  
(underscores replaced by hyphens in `{run_name}`). The full table — Sets 1-3 (initial sweep) and Sets 7-10 (final saturated models) — lives in `experiments/README.md`.

## Where to start reading

1. **`experiments/README.md`** — full setup, results table for every run.
2. **`experiments/analysis/QUANT_FINDINGS_REPORT.md`** + **`PAPER_QUANT_SECTION.md`** — the quantitative story.
3. **`experiments/attention_analysis/ANALYSIS_WRITEUP.md`** — the mechanistic story (why BiMGA wins).
4. **`experiments/analysis/new_teacher/summary.json`** — headline numbers for the new-teacher diagnostics.
5. **`experiments/ood_robustness/README.md`** — robustness story + how to re-run.
