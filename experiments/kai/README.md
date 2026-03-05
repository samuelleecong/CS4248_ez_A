# MBPP Retrieval Experiments (Kai)

This folder contains the reproducible experiment pipeline and notebook for MBPP text-to-code retrieval.

## Structure

```
experiments/kai/
├── notebooks/
│   └── mbpp.ipynb               # interactive development notebook
├── mbpp.ipynb                   # compatibility symlink -> notebooks/mbpp.ipynb
├── scripts/
│   ├── run_mbpp_experiments.py  # main experiment runner (source of truth)
│   ├── plot_mbpp_results.py     # plot generator from run artifacts
│   └── run_all.sh               # end-to-end helper script
└── README.md
```

## Run Full Matrix

From project root:

```bash
python experiments/kai/scripts/run_mbpp_experiments.py \
  --output-dir experiments/kai/results \
  --run-id mbpp_full_matrix \
  --device auto \
  --seed 42 \
  --full-matrix \
  --finetune-all-pretrained
```

Resume missing steps only:

```bash
python experiments/kai/scripts/run_mbpp_experiments.py \
  --output-dir experiments/kai/results \
  --run-id mbpp_full_matrix \
  --device auto \
  --seed 42 \
  --full-matrix \
  --finetune-all-pretrained \
  --resume
```

Generate plots:

```bash
python experiments/kai/scripts/plot_mbpp_results.py --run-dir experiments/kai/results/mbpp_full_matrix
```
