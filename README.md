# CS4248_ez_A

NLP project for text-to-code generation and code search on the [MBPP dataset](https://huggingface.co/datasets/google-research-datasets/mbpp) (974 Python programming problems).

## Project Tasks

**Task 1 — Text-to-Code Generation**: Fine-tune a code LLM (e.g. CodeLlama-7B with QLoRA) to generate Python solutions from natural language descriptions. Evaluated with pass@k.

**Task 2 — Text-to-Code Search**: Fine-tune an embedding model (e.g. UniXcoder) to retrieve relevant code snippets given a text query. Evaluated with MRR and Recall@k.

## Repo Structure

```
.
├── ML_PIPELINE.md              # Start here — 7-stage pipeline walkthrough
├── MBPP_TUTORIAL.md            # Full tutorial for both tasks + advanced techniques
├── experiments/
│   └── kai/
│       ├── notebooks/          # Notebook workspace (mbpp.ipynb)
│       ├── artifacts/          # Kai-local generated artifacts (ignored by git)
│       ├── results/            # Kai-local run outputs (ignored by git)
│       ├── README.md           # Experiment-specific usage
│       └── scripts/            # Reproducible runners/plotters
├── assignment_details/
│   ├── PROJ_reqs.md            # Original project requirements
│   └── cs4248-iu-template.pdf  # Report template
├── CLAUDE.md                   # AI assistant context
└── README.md
```

## Getting Started

1. **Read `ML_PIPELINE.md` first** — covers the end-to-end pipeline (data → preprocess → model → train → evaluate → save) with runnable code snippets.

2. **Then read `MBPP_TUTORIAL.md`** — task-specific guides for generation (Part 1) and search (Part 2), plus model recommendations (Part 3) and advanced techniques like DPO, RLEF, and RAG (Part 4).

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Run reproducible MBPP retrieval matrix**:
   ```bash
   python experiments/kai/scripts/run_mbpp_experiments.py \
     --output-dir experiments/kai/results \
     --run-id mbpp_full_matrix \
     --device auto \
     --seed 42 \
     --full-matrix \
     --finetune-all-pretrained
   python experiments/kai/scripts/plot_mbpp_results.py --run-dir experiments/kai/results/mbpp_full_matrix
   ```

## Key Models

| Task | Recommended Model | Size | Link |
|------|-------------------|------|------|
| Code Generation | CodeLlama-7b-Instruct + QLoRA | 7B | [HuggingFace](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) |
| Code Search | UniXcoder-base | 125M | [HuggingFace](https://huggingface.co/microsoft/unixcoder-base) |
| Zero-shot Baseline | Qwen2.5-Coder-7B-Instruct | 7B | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |

## Evaluation Metrics

- **pass@k** (Generation): Generate k samples per problem, pass if any sample passes all test cases
- **MRR** (Search): Mean Reciprocal Rank of the correct code snippet
- **Recall@k** (Search): Fraction of correct results in top-k
