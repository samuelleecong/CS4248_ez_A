# TACO HF Quant Analysis Package

This package replays the HF-uploaded submission experiment checkpoints on the fixed TACO test split, writes CSV summaries, builds paper-oriented figures, and adds one Markdown explainer per figure.

## Outputs

- `csv/`: run registry, replayed metrics, diagnostics, per-query margins, and summaries
- `figures/`: PNG figures for the paper and appendix
- `figures_md/`: one Markdown explainer per figure

## Build command

```bash
cd Project/CS4248_ez_A/mbpp_kd_suite
.venv/bin/python ../submission/experiments/analysis/build_quant_package.py
```

## Margin-Stratified Plots For Final 6 Models

Use [plot_margin_stratified_final_models.py](/Users/Immanuel/NUS/Modules/CS4248%20Natural%20Language%20Processing/Project/CS4248_ez_A/submission/experiments/analysis/plot_margin_stratified_final_models.py) when you want only the margin-stratified visualisations for the 6 final saturated models from the experiment README:

- `s7_control_bs32`
- `s7_embed_dw100_aw10`
- `s8_A2_bimga_uniform`
- `s8_hnp_dw100_pw10`
- `s9_score_dw100`
- `s10_bimga_dw100_aw10`

This script focuses only on:

- a tercile bar chart
- a decile line chart
- a decile heatmap
- a fresh per-query CSV for those 6 models

### Default behavior

By default, the script:

- loads the TACO test split from `BEE-spoke-data/TACO-hf`
- uses the teacher repo `sentence-transformers/all-MiniLM-L6-v2`
- downloads the 6 final saturated student checkpoints from Hugging Face
- writes outputs to `submission/experiments/analysis/final_margin_plots`

### Run by replaying models

From the suite directory:

```bash
cd /Users/Immanuel/NUS/Modules/CS4248\ Natural\ Language\ Processing/Project/CS4248_ez_A/mbpp_kd_suite
.venv/bin/python ../submission/experiments/analysis/plot_margin_stratified_final_models.py --device cuda
```

CPU-only:

```bash
cd /Users/Immanuel/NUS/Modules/CS4248\ Natural\ Language\ Processing/Project/CS4248_ez_A/mbpp_kd_suite
.venv/bin/python ../submission/experiments/analysis/plot_margin_stratified_final_models.py --device cpu
```

### Use the fine-tuned teacher

If you want the fine-tuned teacher instead of the default README teacher:

```bash
cd /Users/Immanuel/NUS/Modules/CS4248\ Natural\ Language\ Processing/Project/CS4248_ez_A/mbpp_kd_suite
.venv/bin/python ../submission/experiments/analysis/plot_margin_stratified_final_models.py \
  --device cuda \
  --teacher-repo cs4248-nlp/ft-teacher-all-mpnet-base-v2-taco-20260326-110507
```

### Reuse an existing per-query CSV

If you already generated the per-query file for these 6 models, you can skip replay and only rebuild the plots:

```bash
cd /Users/Immanuel/NUS/Modules/CS4248\ Natural\ Language\ Processing/Project/CS4248_ez_A/mbpp_kd_suite
.venv/bin/python ../submission/experiments/analysis/plot_margin_stratified_final_models.py \
  --per-query-csv ../submission/experiments/analysis/final_margin_plots/per_query_results_final6.csv
```

### Important flags

- `--per-query-csv`: reuse an existing per-query CSV
- `--output-dir`: change where outputs are written
- `--cache-dir`: change the Hugging Face cache directory
- `--dataset-name`: override the retrieval dataset
- `--teacher-repo`: swap teacher model used for teacher margins
- `--device`: one of `auto`, `cpu`, `cuda`, `mps`

### Output files

The script writes these files under the output directory:

- `per_query_results_final6.csv`
- `margin_terciles_final6.csv`
- `margin_deciles_final6.csv`
- `margin_terciles_final6.png`
- `margin_decile_lines_final6.png`
- `margin_decile_heatmap_final6.png`
- `summary.json`
