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
