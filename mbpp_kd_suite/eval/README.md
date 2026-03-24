# Eval Package

The standalone evaluator is kept entirely under `eval/`. This README is the main usage guide for peers who want to run retrieval evaluation without touching the training pipeline.

## Usage

```bash
uv run mbpp-kd-eval \
  --dataset-name {mbpp|codesearchnet} \
  --model-source {hf|local} \
  --model-name-or-path MODEL \
  [--dataset-path PATH] \
  [--checkpoint-format {auto|hf_dir|suite_student_dir}] \
  [--split {validation|test}] \
  [--ks 1,5,10] \
  [--max-query-length 160] \
  [--max-code-length 256] \
  [--batch-size 64] \
  [--device {auto|cpu|cuda|mps}] \
  [--output-dir runs] \
  [--seed 42]
```

Required flags:
- `--dataset-name`
- `--model-source`
- `--model-name-or-path`

Everything else is optional and has a default.

For built-in CLI help, run:

```bash
uv run mbpp-kd-eval --help
```

A `run` means one evaluator execution over exactly one dataset split with one model. Each run writes one timestamped output folder plus local aggregate indices under `eval/runs/`.

## Examples

Evaluate a Hugging Face bi-encoder on MBPP test using the default dataset resolution path:

```bash
uv run mbpp-kd-eval \
  --dataset-name mbpp \
  --model-source hf \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --split test
```

Use this when you want a zero-shot baseline from a public Hugging Face model and are fine with the adapter's normal local-first dataset lookup.

Evaluate a saved suite student checkpoint on MBPP test:

```bash
uv run mbpp-kd-eval \
  --dataset-name mbpp \
  --model-source local \
  --model-name-or-path artifacts/runs/20260320_101500/embed_distill/model \
  --checkpoint-format suite_student_dir \
  --split test
```

Use this for checkpoints saved by this suite with `backbone/`, `tokenizer/`, and optional `projection.pt`.

Evaluate a local Hugging Face directory instead of downloading from the Hub:

```bash
uv run mbpp-kd-eval \
  --dataset-name mbpp \
  --model-source local \
  --model-name-or-path /absolute/path/to/save_pretrained_dir \
  --checkpoint-format hf_dir \
  --split test
```

Use this when you already exported a model with `save_pretrained`.

Evaluate CodeSearchNet Python using a local dataset clone:

```bash
uv run mbpp-kd-eval \
  --dataset-name codesearchnet \
  --dataset-path /absolute/path/to/CodeSearchNet/resources/data/python/final/jsonl \
  --model-source hf \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --split test
```

Use this when you have the CodeSearchNet Python JSONL tree locally and want to avoid remote dataset loading.

Evaluate on the validation split instead of the test split:

```bash
uv run mbpp-kd-eval \
  --dataset-name mbpp \
  --model-source hf \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --split validation
```

Use this when tuning models or checking intermediate checkpoints.

Write outputs somewhere other than the default `eval/runs/`:

```bash
uv run mbpp-kd-eval \
  --dataset-name mbpp \
  --model-source hf \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --split test \
  --output-dir ./scratch/manual_eval
```

Use this when you want temporary outputs outside the repo-local evaluator run tree.

Force CPU evaluation:

```bash
uv run mbpp-kd-eval \
  --dataset-name mbpp \
  --model-source hf \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --split test \
  --device cpu
```

Use this when you need reproducible CPU-only behavior or want to avoid CUDA/MPS device issues.

## Default Behavior

The evaluator uses exact paired retrieval. For a given split, query `i` is treated as relevant only to code `i` in that same split.

If you do not pass `--dataset-path`, the evaluator tries local data first and then falls back to remote loading where supported:
- `mbpp`: repo-local `datasets/mbpp/mbpp.jsonl` first, then official Hugging Face MBPP
- `codesearchnet`: repo-local Python JSONL tree first, then remote dataset fallbacks

If you use `--model-source local` with `--checkpoint-format auto`, the evaluator inspects the directory shape to decide whether it is:
- a suite student checkpoint directory
- or a Hugging Face `save_pretrained` directory

The default `--output-dir runs` is resolved under `eval/`, so the default output root becomes:

```text
eval/runs/
```

## What It Does

For each run, the evaluator:
- loads a supported dataset
- normalizes each example into `(id, query, code, split)`
- loads a supported encoder model
- encodes queries and code separately
- computes exact paired retrieval scores with a dense similarity matrix
- reports ranking metrics such as `MRR`, `Recall@k`, `MAP@k`, and `nDCG@k`
- profiles runtime and peak memory by stage
- writes per-run artifacts under `eval/runs/` unless you override the output path

This is an offline retrieval evaluator. It does not train models, judge semantic relevance manually, or run code execution tests.

## Flags

| Flag | Required | Default | What it means | Important behavior / caveat |
| --- | --- | --- | --- | --- |
| `--dataset-name` | Yes | None | Chooses the dataset adapter. | Supported values are only `mbpp` and `codesearchnet`. |
| `-h`, `--help` | No | None | Prints the CLI help text and exits. | Use this when you want the raw argparse view instead of the richer explanations in this README. |
| `--dataset-path` | No | `None` | Local filesystem override for dataset loading. | This must be a real local path. It is not a Hugging Face dataset ID. |
| `--model-source` | Yes | None | Chooses whether the model comes from Hugging Face or a local directory. | Supported values are only `hf` and `local`. |
| `--model-name-or-path` | Yes | None | Hugging Face model ID or local directory path. | For `hf`, pass a Hub ID or local HF directory. For `local`, pass a real local checkpoint path. |
| `--checkpoint-format` | No | `auto` | Tells the loader how to interpret a local model directory. | `auto` resolves by directory shape, not by user intent. |
| `--split` | No | `test` | Dataset split to evaluate. | Only `validation` and `test` are exposed by the CLI. |
| `--ks` | No | `1,5,10` | Retrieval cutoffs used for `Recall@k`, `MAP@k`, and `nDCG@k`. | Parsed as comma-separated integers, then deduplicated and sorted internally. |
| `--max-query-length` | No | `160` | Max tokenizer length for queries. | Longer queries are truncated before encoding. |
| `--max-code-length` | No | `256` | Max tokenizer length for code documents. | Longer code snippets are truncated before encoding. |
| `--batch-size` | No | `64` | Encoding batch size. | Applies to both query and code encoding. Larger values may hit device memory limits. |
| `--device` | No | `auto` | Device selection policy. | `auto` uses suite runtime selection. Explicit `cuda` or `mps` will fail if that backend is unavailable. |
| `--output-dir` | No | `runs` | Output root for generated evaluator artifacts. | Relative values like `runs` resolve under `eval/`. Relative paths starting with `./` or `../` stay relative to the current working directory. |
| `--seed` | No | `42` | Random seed for reproducibility. | The evaluator itself is deterministic, but the seed is still logged with each run. |

## Inputs And Expected Paths

### Dataset inputs

`mbpp` local inputs:
- a split directory with `train.jsonl` / `validation.jsonl` / `test.jsonl`
- or a single file such as `mbpp.jsonl` or `sanitized-mbpp.json`

If you point `--dataset-path` at a single unsplit MBPP file, the adapter will create synthetic train/validation/test partitions by slicing that file. That is useful for ad hoc checks, but it is not the same as the official MBPP split.

`codesearchnet` local inputs:
- ideally `CodeSearchNet/resources/data/python/final/jsonl`
- or a parent directory that contains `python/final/jsonl`

The loader expects Python split directories named `train`, `valid`, and `test`.

### Model inputs

For `--model-source hf`:
- pass a Hugging Face model ID such as `sentence-transformers/all-MiniLM-L6-v2`
- or a local Hugging Face directory that contains `config.json`

For `--model-source local`:
- `--checkpoint-format hf_dir` expects a Hugging Face-style directory with `config.json`
- `--checkpoint-format suite_student_dir` expects either:
  - `backbone/` and `tokenizer/` directly in the path
  - or `model/backbone/` and `model/tokenizer/` under the path

If `projection.pt` exists for a suite student checkpoint, it is loaded automatically.

## Outputs

Each run writes one timestamped folder:

```text
eval/runs/<dataset>/<split>/<timestamp>_<model>/
```

That folder contains:
- `summary.md`: human-readable run summary
- `metrics.csv`: flat machine-readable metrics row
- `metrics.json`: nested full result payload
- `profiling.csv`: per-stage runtime and peak memory
- `config.json`: resolved evaluator config
- `metrics_overview.png`: summary metric plot
- `runtime_memory.png`: runtime and memory plot

The evaluator also rebuilds aggregate indices such as `run_index.csv` and `run_index.md` locally when runs are generated.

Important:
- `eval/runs/` is local generated state
- it is intentionally git-ignored
- your peers will generate their own local run history
- they should not expect your `run_index.md` or run folders to appear when they pull the repo

## Common Gotchas

- `--dataset-path google-research-datasets/mbpp` will fail. That flag expects a local filesystem path, not a Hugging Face dataset name.
- If you run `mbpp` with no `--dataset-path`, the evaluator may prefer the repo-local `datasets/mbpp/mbpp.jsonl` before remote loading. If you need the official split specifically, make sure your local data layout is what you expect.
- Full exhaustive evaluation is intended for bi-encoder retrieval. Cross-encoders should usually be used as rerankers, not full-corpus scorers.
- Metrics are exact paired retrieval metrics, not judged relevance or functional correctness.
- Large datasets can be slow even when they fit in memory, because the dense score matrix still has to be computed.
- Local model directories must look like either a Hugging Face save directory or a suite student checkpoint directory, or `--checkpoint-format auto` will fail to resolve them.
- `--output-dir runs` writes inside `eval/`, not the current working directory.

## Current Limitations

- Supported datasets are only `mbpp` and `codesearchnet`.
- The retrieval protocol is exact paired retrieval only.
- The evaluator assumes separate query and code encoders with a dense similarity matrix.
- There is no built-in cross-encoder reranking mode.
- There is no human-judged relevance, execution-based evaluation, or multi-positive relevance handling.
- The CLI only exposes `validation` and `test`, not `train`.
