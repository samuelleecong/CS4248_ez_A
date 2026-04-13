---
language:
- en
license: mit
tags:
- code-search
- embeddings
- knowledge-distillation
- sentence-transformers
---

# cs4248-nlp/paper-s5-bimga-all-minilm-l6-v2-taco-20260402-143306

Code-search embedding model trained with the CS4248 two-phase KD pipeline.

## Model details

| Field | Value |
|-------|-------|
| Role | `bimga` |
| Phase | Phase 2 |
| Method | `bimga` |
| Dataset | `unknown` |
| Teacher | `unknown` |
| Student base | `unknown` |
| Phase 1 epochs | `unknown` |
| Phase 1 patience | `unknown` |
| Phase 2 epochs | `unknown` |
| Phase 2 patience | `unknown` |
| Batch size | `unknown` |
| Eval batch size | `unknown` |
| Learning rate | `unknown` |
| Seed | `unknown` |
| Run timestamp | `20260402_143306` |

## Usage

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("cs4248-nlp/paper-s5-bimga-all-minilm-l6-v2-taco-20260402-143306")
model = AutoModel.from_pretrained("cs4248-nlp/paper-s5-bimga-all-minilm-l6-v2-taco-20260402-143306")
```

Mean-pool the last hidden state to get a fixed-size embedding:

```python
import torch

def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

inputs = tokenizer("your query here", return_tensors="pt", truncation=True, max_length=160)
with torch.no_grad():
    outputs = model(**inputs)
embedding = mean_pool(outputs, inputs['attention_mask'])
```