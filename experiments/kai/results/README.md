# Results Layout

Each run is stored under `experiments/kai/results/<run_id>/` with a stable structure:

```
experiments/kai/results/<run_id>/
├── checkpoints/                # fine-tuned model checkpoints
│   ├── final_standard_mnr/<model_slug>/
│   ├── final_hardneg_triplet/<model_slug>/
│   ├── sweep/<config_id>/
│   └── legacy_selected_model/  # optional migrated legacy one-off checkpoints
├── logs/
│   └── failures.log            # step-level failures for resume/debug
├── metadata/
│   └── run_metadata.json       # environment + config snapshot
├── metrics/
│   ├── metrics_all.csv         # all metric rows
│   ├── training_stats.csv      # per-epoch training stats
│   └── comparisons.csv         # before/after deltas + CIs
├── plots/                      # generated visualization files
├── ranks/                      # raw rank arrays for bootstrap/diagnostics
└── reports/
    ├── summary.md
    └── summary.txt
```
