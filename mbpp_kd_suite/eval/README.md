# Eval Package

The standalone evaluator is kept entirely under `eval/`.

Layout:
- `data_adapters.py`: dataset normalization for MBPP and CodeSearchNet
- `model_adapters.py`: Hugging Face and local checkpoint loading
- `engine.py`: retrieval evaluation orchestration
- `profiler.py`: per-stage timing and memory tracking
- `reporting.py`: per-run artifacts plus local aggregate index generation
- `run.py`: CLI entrypoint
- `tests/`: eval-specific unit tests
- `runs/`: local run history; generated outputs stay git-ignored

Run the evaluator with:

```bash
uv run mbpp-kd-eval --dataset-name mbpp --model-source hf --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 --split test
```

Default output root:

```text
eval/runs/
```
