# meso-eval

Per-skill and per-difficulty MRR breakdown for trained KD models.
Produces figures comparing distillation methods across TACO problem categories.

## Run from project root

All commands must be run from `mbpp_kd_suite/` (where `pyproject.toml` lives).

### Step 1 — Evaluate models

Encodes all saved models against the TACO test set and computes MRR broken down
by `skill_types` and `difficulty`. Also evaluates the teacher as an upper bound.

```bash
uv run python meso-eval/eval_tag_breakdown.py \
  --run-dir /path/to/paper_experiments_sX/TIMESTAMP \
  --teacher-model sentence-transformers/all-MiniLM-L6-v2 \
  --output meso-eval/eval_skill_results.json
```

| Flag | Description |
|------|-------------|
| `--run-dir` | Sweep timestamp directory (contains `s1_*` subdirs with saved models) |
| `--teacher-model` | HuggingFace model name for upper-bound eval (optional) |
| `--output` | Output JSON path |
| `--batch-size` | Encoding batch size (default: 64) |
| `--dataset` | HuggingFace dataset name (default: `BEE-spoke-data/TACO-hf`) |
| `--runs` | Comma-separated run names to eval — omit to eval all |
| `--dry-run` | Print discovered models and exit without running |

**Example (Set 1):**
```bash
uv run python meso-eval/eval_tag_breakdown.py \
  --run-dir /Users/samuel/dev/CS4248_ez_A/submission/experiments/artifacts/12april/paper_experiments_s1/20260410_234932 \
  --teacher-model sentence-transformers/all-MiniLM-L6-v2 \
  --output meso-eval/eval_skill_results.json
```

### Step 2 — Generate figures

Reads the JSON from Step 1 and writes figures to `--output-dir`.

```bash
uv run python meso-eval/plot_tag_results.py \
  --input meso-eval/eval_skill_results.json \
  --output-dir meso-eval/figures/
```

| Flag | Description |
|------|-------------|
| `--input` | JSON produced by Step 1 |
| `--output-dir` | Directory for output figures (created if missing) |
| `--top-n-tags` | Max skills shown in line chart and heatmap (default: 20) |

## Output figures

| File | What it shows |
|------|--------------|
| `overall_mrr.png` | MRR / R@1 / R@10 per method (best config per family) |
| `by_difficulty.png` | MRR per difficulty level (EASY → VERY_HARD) grouped by method |
| `tag_gain_lines.png` | MRR gain over control per method, one line per method, skills on X-axis |
| `tag_gap_heatmap.png` | Heatmap: MRR gain over control per (method × skill) |
| `margin_correlation.png` | Teacher confidence (avg margin) vs BiMGA − embed_distill delta, by skill and difficulty — tests whether low teacher confidence predicts where BiMGA underperforms |

## Output JSON structure

```json
{
  "s1_bimga_dw100_aw10": {
    "overall": { "MRR": 0.315, "Recall@1": 0.23, "Recall@5": ..., "Recall@10": 0.48, "n": 1000 },
    "by_skill": {
      "Dynamic programming": { "MRR": ..., "Recall@1": ..., "n": 72 },
      ...
    },
    "by_difficulty": {
      "EASY": { "MRR": ..., "n": 86 },
      ...
    }
  },
  "__teacher__": {
    "overall": { ... },
    "by_skill": { ... },
    "by_difficulty": { ... },
    "margin_by_skill": { "Dynamic programming": { "mean": -0.181, "n": 72 }, ... },
    "margin_by_difficulty": { "EASY": { "mean": -0.101, "n": 86 }, ... },
    "margin_overall": { "mean": -0.153, "n": 1000 }
  }
}
```

The `margin_*` fields are only present in the `__teacher__` entry.

## Notes

- One problem can belong to multiple skill buckets — a problem tagged `["Sorting", "Greedy algorithms"]` contributes to both groups independently.
- `skill_types` is parsed with `ast.literal_eval` since TACO stores it as a Python list string.
- The plot script picks the **best MRR config per method family** automatically — no need to specify which run to use.
- See `docs/SKILL_DIFFICULTY_ANALYSIS.md` for interpretation of results.
