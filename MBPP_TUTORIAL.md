# MBPP Project Tutorial: Text-to-Code Generation & Code Search

> **New here? Start with [QUICKSTART.md](./QUICKSTART.md)** — a 15-minute conceptual overview. This document is the full reference with runnable code, advanced techniques, and paper lists.

**Dataset**: [MBPP (Mostly Basic Python Problems)](https://huggingface.co/datasets/google-research-datasets/mbpp) — 974 Python programming tasks with natural language descriptions, code solutions, and test cases.

---

## Table of Contents

- [Part 0: Shared Setup](#part-0-shared-setup)
- [Part 1: Text-to-Code Generation](#part-1-text-to-code-generation)
- [Part 2: Text-to-Code Search](#part-2-text-to-code-search)
- [Part 3: Frontier Models to Test](#part-3-frontier-models-to-test-20242026)
- [Part 4: Extras — Advanced Techniques](#part-4-extras--advanced-techniques)
- [Appendix A: Glossary](#appendix-a-glossary)
- [Appendix B: Paper Reading List](#appendix-b-paper-reading-list)

---

## Part 0: Shared Setup

### 0.1 MBPP Dataset

Each MBPP problem contains:
- **text**: Natural language task description (e.g., "Write a function to find the sum of a list")
- **code**: Python solution
- **test_list**: 3 test cases (e.g., `assert sum_list([1,2,3]) == 6`)

| Split | IDs | Count | Purpose |
|-------|-----|-------|---------|
| Prompt | 1–10 | 10 | Few-shot prompts |
| Test | 11–510 | 500 | Evaluation |
| Validation | 511–600 | 90 | Hyperparameter tuning |
| Train | 601–974 | 374 | Fine-tuning |

```python
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/mbpp")
print(dataset['train'][0])
# {'task_id': 601, 'text': '...', 'code': '...', 'test_list': [...]}
```

### 0.2 Environment

```bash
pip install torch transformers datasets peft bitsandbytes
pip install huggingface_hub trl accelerate
pip install sentence-transformers  # for Task 2
```

### 0.3 Key Concepts (30-min primer)

| Concept | What it means | Why it matters |
|---------|--------------|----------------|
| **Fine-tuning** | Continue training a pre-trained model on your data | Adapts general knowledge to your specific task |
| **LoRA** | Freeze model, add tiny trainable matrices | 60-80% less memory, trains in hours not days |
| **Tokenizer** | Converts text/code to numbers the model understands | Each model has its own tokenizer |
| **Embeddings** | Fixed-size vector representations of text/code | Enable similarity search |
| **Contrastive learning** | Train by pushing similar pairs together, dissimilar apart | Core technique for search/retrieval |
| **pass@k** | Generate k code samples, pass if any works | Primary metric for code generation |
| **MRR** | Mean Reciprocal Rank — average 1/rank of correct answer | Primary metric for code search |

---

> **Start here**: Read [ML_PIPELINE.md](./ML_PIPELINE.md) first for the full 7-stage pipeline walkthrough (data loading, preprocessing, model loading, training, evaluation, saving).

---

## Part 1: Text-to-Code Generation

**Task**: Given a natural language description, generate Python code that satisfies the test cases.

### 1.1 How It Works

```
Input:  "Write a function to find the sum of a list"
Output: def sum_list(lst): return sum(lst)
```

A pre-trained code LLM already knows Python. We fine-tune it on MBPP so it learns to follow our specific instruction format and produce clean, test-passing solutions.

### 1.2 Model Selection

Pick based on your GPU:

| GPU VRAM | Model | Why |
|----------|-------|-----|
| < 8 GB | [Salesforce/codet5-large](https://huggingface.co/Salesforce/codet5-large) (770M) | Encoder-decoder, fits on CPU/small GPU |
| 8–16 GB | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) with QLoRA | Best balance for students |
| 16–24 GB | [bigcode/starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b) with LoRA | Strong open-source alternative |
| No GPU | [Salesforce/codet5-small](https://huggingface.co/Salesforce/codet5-small) (60M) | CPU-friendly, lower quality |

**Recommendation**: Start with **CodeLlama-7b-Instruct + QLoRA**. It's the most documented path.

### 1.3 Fine-Tuning with LoRA (Step by Step)

**What is LoRA?** Instead of updating all 7 billion parameters, LoRA freezes the model and adds small trainable matrices (~0.1% of params). You get 95-98% of full fine-tuning quality with a fraction of the compute.

**Key tutorial**: [HuggingFace Cookbook — Fine-tuning Code LLM on Single GPU](https://huggingface.co/learn/cookbook/fine_tuning_code_llm_on_single_gpu)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 1. Load model + tokenizer
model_id = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_4bit=True,          # QLoRA: 4-bit quantization
    device_map="auto",
)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=8,                        # rank (8-16 is good)
    lora_alpha=16,              # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~4M / 7B total = 0.06%

# 3. Prepare MBPP data
dataset = load_dataset("google-research-datasets/mbpp")

def format_prompt(example):
    """Format as instruction-following prompt."""
    return {
        "text": f"""### Instruction:
{example['text']}

### Test Cases:
{chr(10).join(example['test_list'][:2])}

### Solution:
{example['code']}"""
    }

train_data = dataset["train"].map(format_prompt)

# 4. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=TrainingArguments(
        output_dir="./codegen-lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    ),
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()
trainer.save_model("./codegen-lora-final")
```

### 1.4 Inference (Generating Code)

```python
from peft import PeftModel

# Load base + LoRA adapter
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "./codegen-lora-final")

def generate_code(description, num_samples=1):
    prompt = f"### Instruction:\n{description}\n\n### Solution:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,        # lower = more deterministic
        do_sample=True,
        num_return_sequences=num_samples,
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# Example
code = generate_code("Write a function to find the sum of a list")
print(code[0])
```

### 1.5 Evaluation: pass@k

**pass@k** = generate k code samples per problem, count it as solved if any sample passes all test cases.

```python
import subprocess
import numpy as np

def execute_code(code_str, test_cases):
    """Run generated code against test cases. Returns True if all pass."""
    full_code = code_str + "\n" + "\n".join(test_cases)
    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True, timeout=5, text=True
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def pass_at_k(problems, k=1, n=10):
    """
    For each problem, generate n samples.
    pass@k = expected probability that at least 1 of k samples passes.
    """
    scores = []
    for problem in problems:
        samples = generate_code(problem["text"], num_samples=n)
        c = sum(execute_code(s, problem["test_list"]) for s in samples)
        # Unbiased estimator (from Codex paper)
        if n - c < k:
            scores.append(1.0)
        else:
            scores.append(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
    return np.mean(scores)
```

**What to report**: pass@1, pass@5, pass@10

**Baselines to compare against** (from literature):
| Model | MBPP pass@1 |
|-------|-------------|
| CodeT5-large (770M) | ~40% |
| CodeLlama-7B | ~48% |
| CodeLlama-13B | ~50% |
| StarCoder2-15B | ~50-55% |

### 1.6 Supplementary Metrics

- **CodeBLEU**: Combines token match + AST match + data-flow match. Good for analysis but doesn't guarantee correctness. ([Paper](https://arxiv.org/abs/2009.10297))
- **ChrF**: Character-level F-score. Recent research shows better correlation with human judgment than BLEU. ([Guide](https://towardsdatascience.com/a-gentle-introduction-to-code-generation-evaluation-c8dff8c3d19a/))
- **BLEU**: Do NOT use for code. It ignores syntax structure and penalizes equivalent code.

### 1.7 Key Resources for Task 1

| Resource | Type | Link |
|----------|------|------|
| HF Cookbook: Fine-tune Code LLM | Tutorial | [huggingface.co/learn/cookbook/fine_tuning_code_llm_on_single_gpu](https://huggingface.co/learn/cookbook/fine_tuning_code_llm_on_single_gpu) |
| MBPP Paper | Paper | [arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732) |
| LoRA Paper | Paper | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| QLoRA Paper | Paper | [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314) |
| PEFT Docs | Docs | [huggingface.co/docs/peft](https://huggingface.co/docs/peft) |
| CodeLlama Blog | Blog | [huggingface.co/blog/codellama](https://huggingface.co/blog/codellama) |
| CodeT5+ Paper | Paper | [arxiv.org/abs/2305.07922](https://arxiv.org/abs/2305.07922) |
| BigCode Eval Harness | Code | [github.com/bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) |
| PEFT for Code Gen (2024) | Paper | [arxiv.org/abs/2308.10462](https://arxiv.org/abs/2308.10462) |

---

## Part 2: Text-to-Code Search

**Task**: Given a natural language query, find the most relevant code snippet from a corpus using learned embeddings.

### 2.1 How It Works

```
Query:  "Write a function to find the sum of a list"
          ↓ [Encoder A]
     query_embedding (768-dim vector)
                                          ↕ cosine similarity
     code_embedding  (768-dim vector)
          ↑ [Encoder B]
Code:   "def sum_list(lst): return sum(lst)"
```

A **dual-encoder** architecture encodes queries and code independently into the same vector space. At search time, you compute cosine similarity between the query vector and all code vectors, then return the closest matches.

### 2.2 Model Selection

| Model | Size | Description | Link |
|-------|------|-------------|------|
| microsoft/unixcoder-base | 125M | Best balance for code search (recommended) | [HuggingFace](https://huggingface.co/microsoft/unixcoder-base) |
| Salesforce/codet5p-110m-embedding | 110M | Purpose-built for code embeddings | [HuggingFace](https://huggingface.co/Salesforce/codet5p-110m-embedding) |
| codesage/codesage-small-v2 | 130M | Amazon's code embedding model | [HuggingFace](https://huggingface.co/codesage/codesage-small-v2) |
| microsoft/codebert-base | 125M | Foundational, well-documented | [HuggingFace](https://huggingface.co/microsoft/codebert-base) |
| thenlper/gte-small | 33M | Smallest option, works well without fine-tuning | [HuggingFace](https://huggingface.co/thenlper/gte-small) |
| sentence-transformers/all-MiniLM-L6-v2 | 22M | General-purpose, very small | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

**Recommendation**: Start with **UniXcoder-base** or **CodeBERT-base** and fine-tune with Sentence-Transformers.

### 2.3 Preparing MBPP for Code Search

Convert MBPP into (query, code) pairs:

```python
from datasets import load_dataset
import json

dataset = load_dataset("google-research-datasets/mbpp")

# Create query-code pairs
pairs = []
for split in ['train', 'validation', 'test']:
    for example in dataset[split]:
        pairs.append({
            'query': example['text'],
            'code': example['code'],
            'task_id': example['task_id']
        })

print(f"Total pairs: {len(pairs)}")
# ~974 pairs

# Split for search task
train_pairs = [p for p in pairs if p['task_id'] >= 601]        # 374
val_pairs   = [p for p in pairs if 511 <= p['task_id'] <= 600]  # 90
test_pairs  = [p for p in pairs if 11 <= p['task_id'] <= 510]   # 500
```

### 2.4 Training with Sentence-Transformers (Step by Step)

**Why Sentence-Transformers?** It's a high-level framework that handles contrastive learning, batching, and evaluation in a few lines. Perfect for teams new to ML.

```python
from sentence_transformers import (
    SentenceTransformer, InputExample, losses, evaluation
)
from torch.utils.data import DataLoader

# 1. Load pre-trained model
model = SentenceTransformer('microsoft/unixcoder-base')

# 2. Create training examples
#    MultipleNegativesRankingLoss uses in-batch negatives automatically:
#    each (query_i, code_i) pair is positive; all other codes in the
#    batch are treated as negatives for query_i.
train_examples = [
    InputExample(texts=[p['query'], p['code']])
    for p in train_pairs
]

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=32  # adjust to GPU memory
)

# 3. Loss function: contrastive with in-batch negatives
train_loss = losses.MultipleNegativesRankingLoss(model)

# 4. (Optional) Evaluation during training
#    Uses Information Retrieval metrics
queries = {str(i): p['query'] for i, p in enumerate(val_pairs)}
corpus  = {str(i): p['code']  for i, p in enumerate(val_pairs)}
relevant = {str(i): {str(i)}  for i in range(len(val_pairs))}

ir_evaluator = evaluation.InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant,
    name="mbpp-val",
)

# 5. Train!
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
    evaluator=ir_evaluator,
    evaluation_steps=50,
    output_path="./code-search-model",
    save_best_model=True,
)
```

**What's happening under the hood**:
- Each batch has 32 (query, code) pairs
- For query_0, code_0 is the positive match; code_1..code_31 are negatives
- The loss pushes query_0 closer to code_0 and farther from code_1..code_31
- This is called **contrastive learning with in-batch negatives**

### 2.5 Adding LoRA (If Memory Is Tight)

```python
from peft import LoraConfig, get_peft_model, TaskType

# Wrap the underlying model with LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["query", "value"],  # attention layers
    lora_dropout=0.1,
    task_type=TaskType.FEATURE_EXTRACTION,
)
model._first_module().auto_model = get_peft_model(
    model._first_module().auto_model, lora_config
)
# Then train as above
```

### 2.6 Inference (Searching for Code)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("./code-search-model")

# Build code index (encode all code snippets once)
all_codes = [p['code'] for p in test_pairs]
code_embeddings = model.encode(all_codes, convert_to_numpy=True)

def search(query, top_k=5):
    query_emb = model.encode(query, convert_to_numpy=True)
    # Cosine similarity
    similarities = np.dot(code_embeddings, query_emb) / (
        np.linalg.norm(code_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    top_indices = np.argsort(-similarities)[:top_k]
    return [(all_codes[i], similarities[i]) for i in top_indices]

# Example
results = search("Write a function to find the sum of a list")
for code, score in results:
    print(f"Score: {score:.3f}")
    print(code[:100])
    print("---")
```

### 2.7 Evaluation: MRR and Recall@k

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_code_search(model, test_pairs):
    queries = [p['query'] for p in test_pairs]
    codes   = [p['code']  for p in test_pairs]

    query_embs = model.encode(queries, convert_to_numpy=True)
    code_embs  = model.encode(codes, convert_to_numpy=True)

    sims = cosine_similarity(query_embs, code_embs)  # (N, N) matrix

    mrr_scores = []
    recall_at = {1: [], 5: [], 10: []}

    for i in range(len(queries)):
        # Ground truth: query i matches code i
        rankings = np.argsort(-sims[i])
        rank = np.where(rankings == i)[0][0] + 1  # 1-indexed

        mrr_scores.append(1.0 / rank)
        for k in [1, 5, 10]:
            recall_at[k].append(1.0 if i in rankings[:k] else 0.0)

    return {
        'MRR':       np.mean(mrr_scores),
        'Recall@1':  np.mean(recall_at[1]),
        'Recall@5':  np.mean(recall_at[5]),
        'Recall@10': np.mean(recall_at[10]),
    }

results = evaluate_code_search(model, test_pairs)
print(results)
# Example: {'MRR': 0.72, 'Recall@1': 0.65, 'Recall@5': 0.82, 'Recall@10': 0.89}
```

**What to report**: MRR, Recall@1, Recall@5, Recall@10

**Expected performance**:
| Approach | MRR |
|----------|-----|
| TF-IDF baseline | 0.10–0.20 |
| Pre-trained general model (no fine-tuning) | 0.40–0.50 |
| Pre-trained code model (no fine-tuning) | 0.60–0.70 |
| Fine-tuned code model | 0.75+ |

### 2.8 Key Resources for Task 2

| Resource | Type | Link |
|----------|------|------|
| CodeBERT Paper | Paper | [arxiv.org/abs/2002.08155](https://arxiv.org/abs/2002.08155) |
| UniXcoder Code Search | Code | [github.com/microsoft/CodeBERT/tree/master/UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) |
| Sentence-Transformers Docs | Docs | [sbert.net](https://www.sbert.net/) |
| CodeSearchNet Challenge | Paper | [arxiv.org/abs/1909.09436](https://arxiv.org/abs/1909.09436) |
| Contrastive Pre-Training for Code | Paper | [arxiv.org/abs/2201.10005](https://arxiv.org/abs/2201.10005) |
| CodeT5+ | Paper | [arxiv.org/abs/2305.07922](https://arxiv.org/abs/2305.07922) |
| CodeSage (ICLR 2024) | Paper+Code | [github.com/amazon-science/CodeSage](https://github.com/amazon-science/CodeSage) |
| GNN-Coder (2025) | Paper | [arxiv.org/abs/2502.15202](https://arxiv.org/abs/2502.15202) |
| GraphCodeBERT | Paper | [arxiv.org/abs/2009.08366](https://arxiv.org/abs/2009.08366) |

---

## Part 3: Frontier Models to Test (2024–2026)

You don't have to fine-tune everything from scratch. Many recent models already score well on MBPP out of the box. Use these as **baselines**, or fine-tune the smaller ones with LoRA.

### 3.1 Model Landscape

| Model | Sizes | HumanEval | MBPP | License | Local? |
|-------|-------|-----------|------|---------|--------|
| [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 0.5B–32B | 84% (7B) | — | Apache 2.0 | Yes |
| [DeepSeek-Coder](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) | 1B–33B | 50% (6.7B) | 76% (33B) | MIT | Yes |
| [Yi-Coder](https://huggingface.co/01-ai/Yi-Coder-9B-Chat) | 1.5B, 9B | 85% (9B) | 74% (9B) | Apache 2.0 | Yes |
| [Codestral 25.01](https://huggingface.co/mistralai/Codestral-25.01-2501) | 22B | 87% | 80% | MNPL* | Yes |
| [Llama 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | 70B | — | 88% | Llama 3 | API |
| [Phi-4](https://huggingface.co/microsoft/phi-4) | 14B | 83% | — | MIT | Yes |
| [StarCoder2](https://huggingface.co/bigcode/starcoder2-15b) | 3B–15B | — | 75% (15B) | OpenRAIL | Yes |
| [CodeGemma](https://huggingface.co/google/codegemma-7b) | 2B, 7B | — | — | Gemma | Yes |

*MNPL = Mistral Non-Production License (commercial use restricted)

### 3.2 Recommended Picks

**Best overall (start here)**: **Qwen2.5-Coder-7B-Instruct**
- Apache 2.0, GGUF quantized versions available, state-of-the-art for its size
- Run locally: `ollama pull qwen2.5-coder:7b-instruct`

**Best tiny model (<8GB VRAM)**: **Yi-Coder-9B** or **StarCoder2-3B**
- Yi-Coder-9B hits 85% HumanEval at just 9B params

**Best via free API**: **Llama 3.3-70B** on [Groq](https://console.groq.com/) (free tier: 1000 req/day) or [Together.ai](https://www.together.ai/) ($100 free credits at signup)

### 3.3 Quantization (Running Big Models Locally)

GGUF quantization lets you run models at reduced precision with minimal quality loss:

| Model Size | Quant | VRAM Needed | Quality vs FP16 |
|-----------|-------|-------------|-----------------|
| 7B | Q4_K_M | ~5–6 GB | ~92% |
| 14B | Q4_K_M | ~10–12 GB | ~92% |
| 32B | Q4_K_M | ~18–20 GB | ~92% |

```bash
# Easiest way: Ollama (one command)
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# Or use llama.cpp with GGUF files from HuggingFace
# e.g. Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
```

### 3.4 Quick Zero-Shot Baseline

Before fine-tuning anything, measure how well a frontier model does on MBPP out of the box:

```python
from openai import OpenAI

# Works with Ollama, Together.ai, Groq — all OpenAI-compatible
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama
    api_key="ollama",
)

def zero_shot_generate(description):
    response = client.chat.completions.create(
        model="qwen2.5-coder:7b-instruct",
        messages=[
            {"role": "system", "content": "Write Python code. Output only the function, no explanation."},
            {"role": "user", "content": description},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content

# Then evaluate with pass@k from Section 1.5
```

### 3.5 Leaderboard (Feb 2026)

Check live standings:
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) — HumanEval+ and MBPP+
- [LLM-Stats MBPP](https://llm-stats.com/benchmarks/mbpp) — MBPP-specific rankings

---

## Part 4: Extras — Advanced Techniques

These are follow-up approaches to explore after you have a working baseline. Each builds on SFT (Part 1) and can significantly improve results. Ordered from easiest to hardest.

### 4.1 Best-of-N Sampling (Easiest Win)

**Idea**: Generate N code samples, execute all against test cases, pick the one that passes.

No retraining needed. Just inference-time scaling.

```python
def best_of_n(description, test_cases, n=10):
    samples = [generate_code(description) for _ in range(n)]
    for sample in samples:
        if execute_code(sample, test_cases):
            return sample  # Found a passing solution
    return samples[0]  # Fallback to first if none pass
```

| N | Typical Improvement |
|---|-------------------|
| 5 | +15–20% pass@1 |
| 10 | +25–30% pass@1 |
| 100 | +35–40% pass@1 |

This is how AlphaCode works at scale: generate ~1M samples, filter by tests, cluster by execution behavior, submit diverse solutions. [Paper](https://arxiv.org/abs/2203.07814)

**Effort**: 2–3 hours. **GPU**: Same as inference.

---

### 4.2 RAG for Code (Connects Task 1 + Task 2)

**Idea**: Use your code search model (Task 2) to retrieve similar solved problems, then include them as context for code generation (Task 1).

This directly links both MBPP tasks together.

```python
# 1. Use your trained code search model to find similar problems
similar = search("Write a function to find the sum of a list", top_k=3)

# 2. Build a few-shot prompt with retrieved examples
prompt = "Here are similar solved problems:\n\n"
for code, score in similar:
    prompt += f"```python\n{code}\n```\n\n"
prompt += f"Now solve this:\n{new_description}\n"

# 3. Generate with the augmented prompt
solution = generate_code(prompt)
```

**Why it works**: The model sees solved examples that are structurally similar to the target problem. 5–15% pass@1 improvement, no retraining.

**Effort**: 3–4 hours. **GPU**: Same as inference.

Papers:
- [RAG Survey (2506.00054)](https://arxiv.org/abs/2506.00054)
- [Retrieval-Augmented Code Generation](https://www.edenai.co/post/the-2025-guide-to-retrieval-augmented-generation-rag)

---

### 4.3 Synthetic Data Generation

**Problem**: MBPP only has 374 training examples. More data = better models.

**Three approaches to generate more training data**:

#### Evol-Instruct (WizardCoder)
Evolve simple MBPP problems into harder variants:
```
Original:  "Write a function to find the sum of a list"
Evolved:   "Write a function to find the sum of all even numbers
            in a nested list, handling empty sublists gracefully"
```
Use an LLM to systematically add constraints, edge cases, and complexity. WizardCoder generated ~78K problems this way and hit 79.9% pass@1.

[Paper](https://arxiv.org/abs/2306.08568)

#### OSS-Instruct (Magicoder)
Extract real code snippets from GitHub → ask an LLM to generate a description for each → create new (description, code) pairs. Produces more realistic, production-grade problems. MagicoderS-7B hit 76.8% HumanEval, beating GPT-3.5-turbo.

[Paper](https://arxiv.org/abs/2312.02120)

#### Self-Instruct
Your model generates problems → you filter by execution → retrain on passing ones → repeat. No external data needed.

**Effort**: 6–16 hours depending on approach. **GPU**: 8GB+ for generation.

---

### 4.4 DPO (Direct Preference Optimization)

**Idea**: After SFT, teach the model to prefer correct code over incorrect code using pairs of (preferred, rejected) samples.

**Why DPO over RLHF?** DPO skips the reward model entirely. It's simpler, 2–3x faster, and works on a single 16GB GPU.

#### Creating preference data from MBPP automatically:

```python
# Generate multiple samples per problem
for problem in mbpp_train:
    samples = generate_code(problem["text"], num_samples=10, temperature=0.8)
    for sample in samples:
        passed = execute_code(sample, problem["test_list"])
        if passed:
            preferred.append(sample)
        else:
            rejected.append(sample)

# Create preference pairs
dpo_data = [
    {"prompt": problem["text"], "chosen": good_code, "rejected": bad_code}
    for good_code, bad_code in zip(preferred, rejected)
]
```

#### Training with HuggingFace TRL:

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=sft_model,                    # your SFT model from Part 1
    ref_model=None,                     # uses implicit reference
    args=DPOConfig(
        output_dir="./dpo-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-7,             # much lower than SFT
        beta=0.1,                       # KL divergence penalty
    ),
    train_dataset=dpo_dataset,
)
trainer.train()
```

**Expected boost**: +5–10% pass@1 over SFT alone.

**Effort**: 12–20 hours total. **GPU**: 16GB.

Papers:
- [DPO](https://arxiv.org/abs/2305.18290)
- [CodeDPO (self-verified)](https://arxiv.org/abs/2410.05605)
- [HuggingFace TRL DPOTrainer Docs](https://huggingface.co/docs/trl/en/dpo_trainer)

---

### 4.5 RL from Execution Feedback (RLEF)

**Idea**: Use test case pass/fail as a reward signal to train the model with reinforcement learning. No human labels needed — the tests are the judge.

```
Model generates code → Execute against tests → Pass = reward +1, Fail = reward -1 → Update model with PPO
```

This is how Google trained Gemma's code capabilities. The key insight: code has a built-in verifier (the test suite), so you don't need human feedback at all.

**Pipeline**:
1. Start with your SFT model
2. (Optional) Train a reward model that predicts pass/fail
3. Run PPO optimization using execution reward

**Expected boost**: +10–15% pass@1 over SFT alone (best results, but hardest to implement).

**Effort**: 20–30 hours. **GPU**: 24GB+. Hyperparameter tuning is tricky.

Papers:
- [RLEF](https://arxiv.org/abs/2410.02089) — Grounding Code LLMs in Execution Feedback
- [CodeRL+](https://arxiv.org/abs/2510.18471) — Execution Semantics Alignment
- [RLHF Implementation Details](https://iclr-blogposts.github.io/2024/blog/the-n-implementation-details-of-rlhf-with-ppo/)

---

### 4.6 Self-Play & Self-Improvement

**SPIN (Self-Play Fine-Tuning)**: The model iteratively improves by learning from its own outputs.

```
Iteration 1: Generate samples → keep passing ones → fine-tune
Iteration 2: Generate again (now better) → keep passing → fine-tune
Iteration 3: Repeat (usually 2-3 iterations is enough)
```

**STaR (Self-Taught Reasoner)**: When code fails, re-generate with the correct solution as a hint, then train on the successful reasoning trace.

**V-STaR** (Feb 2024): Train a verifier model alongside the generator — 4–17% accuracy improvement.

**Effort**: 12–15 hours (3 iterations). **GPU**: 16GB.

Papers:
- [SPIN](https://arxiv.org/abs/2401.01335)
- [STaR](https://arxiv.org/abs/2203.14465)
- [Self-Debugging with Self-Generated Tests](https://arxiv.org/abs/2501.12793)

---

### 4.7 Reward Modeling for Code

Train a model to score code quality, then use it to guide generation.

**Outcome Reward Model (ORM)**: Binary — will this code pass all tests?
```python
# Simple: fine-tune a classifier
# Input: (description, code) → Output: 0 or 1
```

**Process Reward Model (PRM)**: Score every line/step of generation. Provides denser learning signal.

**CodePRM** (ACL 2025): Combines process-level + execution feedback. Uses a Generate-Verify-Refine pipeline for +5–10% pass@1.

Papers:
- [CodePRM](https://aclanthology.org/2025.findings-acl.428/)
- [FunPRM — Function-as-Step PRM](https://arxiv.org/abs/2601.22249)

---

### 4.8 Constitutional AI / RLAIF for Code

**Idea**: Use a strong LLM (Claude, GPT-4) as a judge instead of human feedback.

Define a "constitution" — principles your code must follow:
1. Correctness: passes all test cases
2. Efficiency: O(n) preferred over O(n²) where possible
3. Readability: clear variable names, no unnecessary complexity
4. Error handling: graceful handling of edge cases

The AI critic evaluates generated code against these principles, and you use the scores as reward signal.

**Effort**: 7–11 hours + API costs (~$10–50). **GPU**: Not needed (API-based).

Papers:
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [C3AI — Graph-based Principle Selection](https://dl.acm.org/doi/10.1145/3696410.3714705)

---

### 4.9 Knowledge Distillation (Code Generation)

**Idea**: Train a small student model to mimic a large teacher model's outputs. Get 70B-quality code from a 7B model.

```
Teacher (70B) generates solutions → Student (7B) learns to reproduce them
```

**Why distillation over just using the small model?** A 7B model fine-tuned on MBPP with SFT hits ~48% pass@1. The same 7B model *distilled* from a 70B teacher can hit ~57% — because it learns the teacher's reasoning patterns, not just the training data.

#### Three approaches (easiest to hardest):

**1. Response-based distillation (simplest)**
Generate solutions with a strong teacher, then SFT the student on them:
```python
# Use a strong model (API or local) as teacher
for problem in mbpp_train:
    teacher_solution = teacher_generate(problem["text"])  # GPT-4, Llama-70B, etc.
    if execute_code(teacher_solution, problem["test_list"]):
        distill_data.append({"text": problem["text"], "code": teacher_solution})

# Then SFT the student on teacher outputs (same as Part 1)
```
This is what Magicoder and WizardCoder do at scale. Simple and effective.

**2. Reasoning distillation (CodePLAN)**
Teacher generates *solution plans* (pseudocode/reasoning steps) alongside code. Student learns both:
```
Teacher output:  Plan: "1. iterate list, 2. filter evens, 3. sum" + Code: "def sum_evens(lst)..."
Student learns:  Plan generation + Code generation (multi-task)
```
Backward reasoning (deduce plans from correct code) produces higher quality signals than forward reasoning. CodePLAN showed 130%+ improvement on APPS dataset using CodeT5-770M.

**3. Structural alignment distillation**
Add a semantic similarity loss (via CodeBERT embeddings) so the student's code matches the teacher's code *structurally*, not just token-by-token:
```python
# Dual loss with curriculum
loss = code_ce_loss + alpha(epoch) * (1 - cosine_sim(
    codebert.encode(student_code), codebert.encode(teacher_code)
))
# alpha grows from 0→1: token accuracy first, semantic alignment later
```
Llama 3.1 8B distilled from 70B: MBPP 48.2% → 56.9% with this approach.

#### Expected performance:

| Compression | Method | Typical Accuracy Drop |
|-------------|--------|----------------------|
| 70B → 7B | Response-based only | 15–25% relative |
| 70B → 7B | + Reasoning distillation | 10–15% relative |
| 70B → 7B | + Structural alignment | 3–8% relative |

**Effort**: 8–16 hours (response-based is fast; structural alignment adds complexity). **GPU**: 16GB for student training. Teacher can be API-based.

**Tooling**:
- [HuggingFace TRL GKDTrainer](https://huggingface.co/docs/trl/en/gkd_trainer) — Generalized Knowledge Distillation
- [DistillKit (Arcee AI)](https://github.com/arcee-ai/DistillKit) — production-ready logit distillation
- [EasyDistill (Modelscope)](https://github.com/modelscope/easydistill) — multi-strategy toolkit
- [DistiLLM](https://github.com/jongwooko/distillm) — optimized for generative models (ICML 2024)

Papers:
- [CodePLAN — Reasoning Distillation for Code](https://arxiv.org/abs/2403.13271)
- [Reasoning Distillation + Structural Alignment](https://arxiv.org/html/2510.17598)
- [Personalised Distillation for Code Gen](https://arxiv.org/abs/2310.18628) (EMNLP 2023)
- [AMR-Evol — Adaptive Modular Response Evolution](https://arxiv.org/abs/2410.00558) (EMNLP 2024)
- [MiniLLM — Reverse KLD for LLM Distillation](https://arxiv.org/abs/2306.08543) (ICLR 2024)
- [DistiLLM-2 — Contrastive Distillation](https://arxiv.org/abs/2503.07067) (ICML 2025)

---

### 4.10 Knowledge Distillation (Embedding Models / Code Search)

**Idea**: Compress your code search model (Task 2) from 125M → 33M or smaller while keeping 95%+ retrieval quality.

For embedding models, distillation means: the student's embeddings should land in the same place as the teacher's.

```
Teacher (UniXcoder-125M) encodes "find sum of list" → vec_teacher
Student (MiniLM-22M)     encodes "find sum of list" → vec_student
Loss = MSE(vec_student, vec_teacher)  or  1 - cosine_sim(vec_student, vec_teacher)
```

#### Three approaches:

**1. Embedding alignment (simplest)**
Train student to match teacher embeddings directly. Sentence-Transformers has built-in support:
```python
from sentence_transformers import SentenceTransformer, losses

teacher = SentenceTransformer("microsoft/unixcoder-base")       # 125M
student = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 22M

# Distillation loss: match student embeddings to teacher
train_loss = losses.MSELoss(model=student)

# Generate training pairs: (text, teacher_embedding)
train_examples = []
for pair in train_pairs:
    teacher_emb = teacher.encode(pair['query'])
    train_examples.append(InputExample(texts=[pair['query']], label=teacher_emb))
```

**2. Layer reduction**
Remove layers from the teacher (e.g., 12 → 6 layers). Retains teacher weights for remaining layers:
- 12 → 6 layers: 99.4% performance, 2.3x speedup
- 12 → 4 layers: ~97% performance, 3x speedup

**3. Contrastive Knowledge Distillation (CKD)**
Use the same InfoNCE contrastive loss for distillation as you use for training. This keeps the objective consistent across all stages:
```python
# Stage 1: Distill on unlabeled code-text pairs (InfoNCE with teacher embeddings)
# Stage 2: Fine-tune student with contrastive learning on labeled pairs
# Same loss function throughout → no objective mismatch
```
DistilCSE showed a 110M student outperforming Sentence-T5 (11B) with 1% of parameters.

#### Expected performance:

| Compression | Method | MRR Retention | Speedup |
|-------------|--------|---------------|---------|
| 125M → 66M (layer reduction) | Layer removal | ~99% | 2x |
| 125M → 33M | Embedding alignment | ~95% | 3–4x |
| 125M → 22M | CKD + fine-tune | ~92% | 5–6x |

**Effort**: 4–8 hours. **GPU**: 8GB (embedding models are small).

**Key insight**: For code search, you can also use **asymmetric deployment** — keep the large teacher for encoding the code corpus (done once) and use the small student only for encoding queries (done at search time). This gives you teacher-quality code embeddings with student-speed queries.

Papers:
- [SPENCER — Self-Adaptive Distillation for Code Retrieval](https://arxiv.org/abs/2508.00546) (ACM TOSEM 2024)
- [DistilCSE — Contrastive KD for Sentence Embeddings](https://arxiv.org/abs/2112.05638) (EMNLP 2023)
- [EmbedDistill — Geometric KD for Retrieval](https://arxiv.org/abs/2301.12005)
- [LEAF — Teacher-Aligned Representations](https://arxiv.org/abs/2509.12539)
- [Sentence-Transformers Distillation Docs](https://sbert.net/examples/sentence_transformer/training/distillation/README.html)

---

### 4.11 Technique Comparison

| Technique | Effort | GPU | pass@1 Boost | Complexity |
|-----------|--------|-----|-------------|------------|
| Best-of-N sampling | 2–3h | Same | +15–30% | Trivial |
| RAG (search→generate) | 3–4h | Same | +5–15% | Low |
| Synthetic data | 6–16h | 8GB | +3–8% | Medium |
| Distillation (code gen) | 8–16h | 16GB | +5–15% | Medium |
| Distillation (embeddings) | 4–8h | 8GB | N/A (search) | Medium |
| DPO | 12–20h | 16GB | +5–10% | Medium |
| Self-Play (SPIN) | 12–15h | 16GB | +5–10% | Medium-High |
| RLEF (PPO + execution) | 20–30h | 24GB+ | +10–15% | High |
| Reward modeling | 6–10h | 16GB | +5–15% | High |
| Constitutional AI | 7–11h | API | +5–10% | Medium |

**Suggested progression**:
1. SFT baseline (Part 1) → measure pass@k
2. Best-of-N + RAG → immediate boost, no retraining
3. Distillation → get strong-teacher quality in a small student model
4. Synthetic data → more training data, retrain SFT
5. DPO → preference optimization on top of SFT
6. RLEF / Self-Play → if you want the best possible results

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **MBPP** | Mostly Basic Python Problems — 974 coding problems with NL descriptions |
| **Fine-tuning** | Continuing to train a pre-trained model on new data |
| **LoRA** | Low-Rank Adaptation — adds small trainable matrices to frozen model |
| **QLoRA** | LoRA + 4-bit quantization — fits large models in small GPU memory |
| **Tokenizer** | Converts text/code into numerical tokens the model understands |
| **Embedding** | Fixed-size vector representation of a text/code snippet |
| **Dual-encoder** | Architecture with separate encoders for queries and documents |
| **Contrastive learning** | Training by pulling similar pairs together, pushing dissimilar apart |
| **In-batch negatives** | Using other batch items as negative examples (free negatives) |
| **Cosine similarity** | Measures angle between two vectors; 1 = identical, 0 = orthogonal |
| **pass@k** | % of problems solved when generating k code samples |
| **MRR** | Mean Reciprocal Rank — average of 1/rank of first correct result |
| **Recall@k** | Fraction of correct results appearing in top-k |
| **NDCG** | Normalized Discounted Cumulative Gain — order-aware relevance metric |
| **CodeBLEU** | Code-specific metric combining token match, AST match, data-flow match |
| **SFTTrainer** | Supervised Fine-Tuning Trainer from HuggingFace TRL library |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella term for LoRA, adapters, etc. |
| **DPO** | Direct Preference Optimization — learns from preferred vs rejected pairs, no reward model |
| **RLHF** | Reinforcement Learning from Human Feedback — classic alignment pipeline |
| **RLEF** | RL from Execution Feedback — uses test pass/fail as reward signal |
| **PPO** | Proximal Policy Optimization — RL algorithm used in RLHF |
| **Best-of-N** | Generate N samples, pick the one that passes tests |
| **RAG** | Retrieval-Augmented Generation — retrieve relevant context before generating |
| **Synthetic data** | Training data generated by LLMs rather than humans |
| **Self-Play** | Model iteratively improves by learning from its own outputs |
| **Reward model** | Model trained to score quality of generated outputs |
| **GGUF** | Quantization format for running LLMs locally with llama.cpp/Ollama |
| **Constitutional AI** | Using AI-defined principles to evaluate and improve model outputs |
| **Knowledge distillation** | Training a small student model to mimic a large teacher model |
| **Teacher-student** | Distillation setup: large model (teacher) transfers knowledge to small model (student) |
| **Reasoning distillation** | Distilling intermediate reasoning steps (plans, CoT) alongside final outputs |
| **CKD** | Contrastive Knowledge Distillation — uses InfoNCE loss for embedding distillation |
| **Layer reduction** | Removing transformer layers from teacher to create student (e.g., 12→6) |
| **Asymmetric deployment** | Using large model for corpus encoding, small model for query encoding |

---

## Appendix B: Paper Reading List

### Must-Read (Start Here)

1. **MBPP Paper** — Austin et al., 2021. "Program Synthesis with Large Language Models"
   [arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732)

2. **LoRA** — Hu et al., 2021. "Low-Rank Adaptation of Large Language Models"
   [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

3. **CodeBERT** — Feng et al., 2020. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages"
   [arxiv.org/abs/2002.08155](https://arxiv.org/abs/2002.08155)

### Recommended (Skim abstracts, read what's relevant)

4. **QLoRA** — Dettmers et al., 2023
   [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

5. **CodeT5+** — Wang et al., 2023
   [arxiv.org/abs/2305.07922](https://arxiv.org/abs/2305.07922)

6. **UniXcoder** — Guo et al., 2022
   [github.com/microsoft/CodeBERT/tree/master/UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder)

7. **CodeSearchNet** — Husain et al., 2019
   [arxiv.org/abs/1909.09436](https://arxiv.org/abs/1909.09436)

8. **Text and Code Embeddings by Contrastive Pre-Training** — Neelakantan et al., 2022
   [arxiv.org/abs/2201.10005](https://arxiv.org/abs/2201.10005)

### Advanced (For going deeper)

9. **PEFT for Code Generation** — Weyssow et al., 2024
   [arxiv.org/abs/2308.10462](https://arxiv.org/abs/2308.10462)

10. **GraphCodeBERT** — Guo et al., 2021
    [arxiv.org/abs/2009.08366](https://arxiv.org/abs/2009.08366)

11. **CodeSage** — Zhang et al., 2024 (ICLR)
    [github.com/amazon-science/CodeSage](https://github.com/amazon-science/CodeSage)

12. **StarCoder2** — Lozhkov et al., 2024
    [arxiv.org/abs/2402.19173](https://arxiv.org/abs/2402.19173)

13. **GNN-Coder** — 2025
    [arxiv.org/abs/2502.15202](https://arxiv.org/abs/2502.15202)

14. **CodeBLEU** — Ren et al., 2020
    [arxiv.org/abs/2009.10297](https://arxiv.org/abs/2009.10297)

### Post-Training & Alignment (Part 4 references)

15. **DPO** — Rafailov et al., 2023. "Direct Preference Optimization"
    [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

16. **RLEF** — 2024. "Grounding Code LLMs in Execution Feedback"
    [arxiv.org/abs/2410.02089](https://arxiv.org/abs/2410.02089)

17. **WizardCoder (Evol-Instruct)** — Luo et al., 2023
    [arxiv.org/abs/2306.08568](https://arxiv.org/abs/2306.08568)

18. **Magicoder (OSS-Instruct)** — Wei et al., 2023
    [arxiv.org/abs/2312.02120](https://arxiv.org/abs/2312.02120)

19. **SPIN (Self-Play Fine-Tuning)** — Chen et al., 2024
    [arxiv.org/abs/2401.01335](https://arxiv.org/abs/2401.01335)

20. **AlphaCode** — Li et al., 2022
    [arxiv.org/abs/2203.07814](https://arxiv.org/abs/2203.07814)

21. **CodePRM** — ACL 2025 Findings
    [aclanthology.org/2025.findings-acl.428](https://aclanthology.org/2025.findings-acl.428/)

22. **Constitutional AI** — Bai et al., 2022
    [arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)

### Knowledge Distillation (Part 4 references)

23. **CodePLAN** — Reasoning Distillation for Code Generation, 2024
    [arxiv.org/abs/2403.13271](https://arxiv.org/abs/2403.13271)

24. **MiniLLM** — Reverse KLD for LLM Distillation (ICLR 2024)
    [arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543)

25. **DistiLLM-2** — Contrastive LLM Distillation (ICML 2025)
    [arxiv.org/abs/2503.07067](https://arxiv.org/abs/2503.07067)

26. **Personalised Distillation** — Adaptive Learning for Code Generation (EMNLP 2023)
    [arxiv.org/abs/2310.18628](https://arxiv.org/abs/2310.18628)

27. **SPENCER** — Self-Adaptive Distillation for Code Retrieval (ACM TOSEM 2024)
    [arxiv.org/abs/2508.00546](https://arxiv.org/abs/2508.00546)

28. **DistilCSE** — Contrastive KD for Sentence Embeddings (EMNLP 2023)
    [arxiv.org/abs/2112.05638](https://arxiv.org/abs/2112.05638)

29. **EmbedDistill** — Geometric KD for Information Retrieval, 2023
    [arxiv.org/abs/2301.12005](https://arxiv.org/abs/2301.12005)

---

## Quick Reference: What To Do First

### If your team is doing Task 1 (Code Generation):
1. Read MBPP paper abstract (10 min)
2. Skim LoRA paper sections 1-3 (15 min)
3. Run the code in Section 1.3 above
4. Evaluate with pass@k (Section 1.5)

### If your team is doing Task 2 (Code Search):
1. Read CodeBERT paper abstract (10 min)
2. Understand dual-encoder diagram (Section 2.1)
3. Run the code in Section 2.4 above
4. Evaluate with MRR/Recall (Section 2.7)

### If your team is doing both:
1. Start with Task 2 (faster to prototype, simpler pipeline)
2. Then Task 1 (more compute-intensive, builds on Task 2 understanding)
