# The ML Pipeline (End to End)

> Extracted from the MBPP Tutorial. Read this first to understand the full picture before diving into the task-specific guides in [MBPP_TUTORIAL.md](./MBPP_TUTORIAL.md).

Every fine-tuning project follows the same 7-stage pipeline. Understanding this structure makes everything else click.

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  1. Load     │───▶│ 2. Preprocess│───▶│ 3. Load     │───▶│ 4. Configure │
│     Data     │    │    & Tokenize│    │    Model     │    │    Training  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘
                                                                  │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐           │
│  7. Save &   │◀──│ 6. Evaluate  │◀──│ 5. Train     │◀──────────┘
│     Share    │    │              │    │              │
└─────────────┘    └──────────────┘    └─────────────┘
```

---

## Stage 1: Load Data

The `datasets` library loads data from HuggingFace Hub in one line. Everything is an Arrow table — fast, memory-mapped, column-oriented.

```python
from datasets import load_dataset

# Load MBPP
dataset = load_dataset("google-research-datasets/mbpp")

print(dataset)
# DatasetDict({
#     train:      Dataset(374 rows)
#     test:       Dataset(500 rows)
#     validation: Dataset(90 rows)
#     prompt:     Dataset(10 rows)
# })

# Inspect one example
print(dataset["train"][0])
# {
#   'task_id': 601,
#   'text': 'Write a function to ...',        ← natural language description
#   'code': 'def func(...):\n    ...',         ← Python solution
#   'test_list': ['assert func(...)==...'],     ← 3 test cases
#   'test_setup_code': '',                      ← optional setup
#   'challenge_test_list': [...]                ← harder tests
# }

# Also available: "sanitized" config (427 rows, cleaner descriptions)
dataset_clean = load_dataset("google-research-datasets/mbpp", "sanitized")
```

**Key `datasets` operations**:
```python
# Filter
small = dataset["train"].filter(lambda x: len(x["code"]) < 200)

# Select specific rows
subset = dataset["train"].select(range(100))

# Shuffle
shuffled = dataset["train"].shuffle(seed=42)

# Custom train/val split (if you need different proportions)
split = dataset["train"].train_test_split(test_size=0.15, seed=42)
train_data, val_data = split["train"], split["test"]
```

---

## Stage 2: Preprocess & Tokenize

Models don't understand text — they understand numbers. The **tokenizer** converts text into token IDs that the model can process.

### 2a. Load the tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# Critical: set pad token (many code models don't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# For causal LMs (GPT-style), pad on the LEFT so the last token is always real
tokenizer.padding_side = "left"
```

### 2b. Format your data

Turn raw MBPP examples into the text format the model expects:

```python
def format_for_training(example):
    """Convert MBPP example → single training string."""
    return {
        "text": (
            f"### Instruction:\n{example['text']}\n\n"
            f"### Tests:\n{example['test_list'][0]}\n\n"
            f"### Solution:\n{example['code']}"
        )
    }

# Apply to entire dataset with .map()
train_formatted = dataset["train"].map(
    format_for_training,
    remove_columns=dataset["train"].column_names,  # drop original columns
)

print(train_formatted[0]["text"])
# ### Instruction:
# Write a function to ...
#
# ### Tests:
# assert func(...) == ...
#
# ### Solution:
# def func(...):
#     ...
```

### 2c. Understanding `dataset.map()`

```python
# Non-batched: function receives one example
dataset.map(lambda x: {"text": x["text"].lower()})

# Batched: function receives dict of lists (10-100x faster)
dataset.map(lambda batch: {"text": [t.lower() for t in batch["text"]]}, batched=True)

# With multiprocessing
dataset.map(my_func, batched=True, num_proc=4)

# Results are cached automatically — re-running is instant
```

### 2d. How tokenization actually works

```python
# What the tokenizer does under the hood:
text = "def sum_list(lst):"
tokens = tokenizer(text)

print(tokens)
# {
#   'input_ids':      [1, 822, 2533, 29918, 1761, 29898, 29880, 303, 1125, ],
#   'attention_mask':  [1,   1,    1,     1,    1,    1,    1,    1,   1,   ]
# }
#
# input_ids       = the actual token numbers
# attention_mask  = 1 for real tokens, 0 for padding

# Decode back to text
tokenizer.decode(tokens["input_ids"])
# "def sum_list(lst):"

# With padding and truncation (for batches)
batch = tokenizer(
    ["def foo():", "def bar(x, y): return x + y"],
    padding=True,           # pad shorter sequences to match longest
    truncation=True,        # cut sequences longer than max_length
    max_length=512,         # max sequence length
    return_tensors="pt",    # return PyTorch tensors
)
```

> **Note**: When using `SFTTrainer` (see Stage 5), you usually don't need to tokenize manually — it handles tokenization internally. Just pass formatted text strings.

---

## Stage 3: Load the Model

### 3a. Basic loading

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,   # half precision (saves 50% memory)
    device_map="auto",           # auto-place on available GPU(s)
)
```

### 3b. Loading with 4-bit quantization (QLoRA)

This is how you fit a 7B model into 8GB of VRAM:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 4-bit weights
    bnb_4bit_use_double_quant=True,        # quantize the quantization constants
    bnb_4bit_quant_type="nf4",             # NormalFloat4 (best for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16, # compute in bf16
)

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**Memory comparison** (CodeLlama-7B):
| Precision | VRAM |
|-----------|------|
| float32 | ~28 GB |
| float16 | ~14 GB |
| 4-bit (QLoRA) | ~5 GB |

### 3c. Attaching LoRA adapters

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                                    # rank (8-16 typical)
    lora_alpha=16,                          # scaling = alpha/r
    target_modules=["q_proj", "v_proj"],    # which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.062
```

---

## Stage 4: Configure Training

Two config objects control everything:

```python
from trl import SFTConfig

training_config = SFTConfig(
    # === Output ===
    output_dir="./checkpoints",

    # === Data ===
    max_seq_length=512,             # max token length (512 is enough for MBPP)
    dataset_text_field="text",      # column name with formatted text
    packing=False,                  # True = pack short examples together (faster)

    # === Batch size ===
    per_device_train_batch_size=4,  # samples per GPU per step
    gradient_accumulation_steps=4,  # effective batch = 4 × 4 = 16
    # ↑ If you get OOM, decrease batch_size and increase accumulation

    # === Learning rate ===
    learning_rate=2e-4,             # higher for LoRA than full fine-tuning
    lr_scheduler_type="cosine",     # cosine decay (good default)
    warmup_steps=100,               # linear warmup before decay kicks in
    weight_decay=0.01,              # regularization

    # === Duration ===
    num_train_epochs=3,             # how many passes over the data

    # === Precision ===
    bf16=True,                      # use bfloat16 (A100/H100/M-series Mac)
    # fp16=True,                    # use this instead for older NVIDIA GPUs

    # === Checkpoints ===
    save_strategy="epoch",          # save after each epoch
    save_total_limit=3,             # keep only last 3 checkpoints

    # === Evaluation ===
    eval_strategy="epoch",          # evaluate after each epoch
    load_best_model_at_end=True,    # keep the best checkpoint

    # === Logging ===
    logging_steps=10,               # log loss every 10 steps
    report_to="none",               # or "wandb" / "tensorboard"
)
```

**What gradient accumulation does**:
```
Step 1: Forward + backward on batch of 4   → accumulate gradients
Step 2: Forward + backward on batch of 4   → accumulate gradients
Step 3: Forward + backward on batch of 4   → accumulate gradients
Step 4: Forward + backward on batch of 4   → accumulate gradients
        ↓
        Optimizer step: update weights using all 16 samples' gradients
```
This gives you an effective batch of 16 while only using memory for 4.

---

## Stage 5: Train

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_formatted,
    eval_dataset=dataset["validation"].map(format_for_training, remove_columns=dataset["validation"].column_names),
    processing_class=tokenizer,     # handles tokenization internally
    peft_config=lora_config,        # pass LoRA config here
)

# Go!
result = trainer.train()
print(f"Training loss: {result.training_loss:.4f}")
```

**What happens inside `trainer.train()`**:
1. Tokenizes each batch on-the-fly (using `processing_class`)
2. Forward pass → compute cross-entropy loss on next-token prediction
3. Backward pass → compute gradients
4. Accumulates gradients for `gradient_accumulation_steps` mini-batches
5. Optimizer step (AdamW) → update LoRA weights
6. Repeat until `num_train_epochs` is done
7. Runs evaluation at each `eval_strategy` checkpoint

**Monitoring**: Watch for `training_loss` decreasing over time. If `eval_loss` starts increasing while `training_loss` keeps decreasing → overfitting. Stop earlier or add regularization.

```python
# Monitor with TensorBoard
# In training_config: report_to="tensorboard", logging_dir="./logs"
# Then run: tensorboard --logdir ./logs

# Monitor with Weights & Biases
# In training_config: report_to="wandb"
# pip install wandb && wandb login
```

---

## Stage 6: Evaluate

### 6a. Loss-based evaluation (automatic)

```python
metrics = trainer.evaluate()
print(metrics)
# {'eval_loss': 1.23, 'eval_runtime': 12.4, ...}
```

### 6b. Generate code and compute pass@k (the real metric)

Loss tells you if the model is learning, but **pass@k** tells you if the code actually works.

```python
import subprocess, math
import numpy as np

def generate_solution(model, tokenizer, description, temperature=0.8):
    """Generate one code solution."""
    prompt = f"### Instruction:\n{description}\n\n### Solution:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,         # sampling (not greedy)
            temperature=temperature, # 0.2=focused, 0.8=diverse, 1.0=random
            top_p=0.95,             # nucleus sampling
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract just the code after "### Solution:"
    code = full_text.split("### Solution:\n")[-1]
    return code

def run_tests(code, test_cases, timeout=5):
    """Execute code + tests in a subprocess. Returns True if all pass."""
    full_code = code + "\n" + "\n".join(test_cases)
    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True, timeout=timeout, text=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False

def pass_at_k(n, c, k):
    """Unbiased pass@k estimator. n=total samples, c=correct, k=budget."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

# Full evaluation
def evaluate_pass_at_k(model, tokenizer, test_data, n=10, k_values=[1, 5, 10]):
    """Evaluate model on MBPP test set."""
    all_results = []

    for example in test_data:
        n_correct = 0
        for _ in range(n):
            code = generate_solution(model, tokenizer, example["text"])
            if run_tests(code, example["test_list"]):
                n_correct += 1

        all_results.append({
            "task_id": example["task_id"],
            "n": n, "c": n_correct,
            **{f"pass@{k}": pass_at_k(n, n_correct, k) for k in k_values},
        })

    # Aggregate
    return {
        f"pass@{k}": np.mean([r[f"pass@{k}"] for r in all_results])
        for k in k_values
    }

results = evaluate_pass_at_k(model, tokenizer, dataset["test"])
print(results)
# {'pass@1': 0.42, 'pass@5': 0.61, 'pass@10': 0.68}
```

**`model.generate()` key parameters**:

| Parameter | What it does | Typical value |
|-----------|-------------|---------------|
| `max_new_tokens` | Max tokens to generate | 256 for MBPP |
| `temperature` | Randomness (0=greedy, 1=diverse) | 0.2 for best single shot, 0.8 for diversity |
| `top_p` | Nucleus sampling cutoff | 0.95 |
| `top_k` | Keep top-k tokens | 50 |
| `do_sample` | Enable sampling (vs greedy) | True for pass@k |
| `num_return_sequences` | Generate multiple at once | 1 (loop externally for pass@k) |
| `num_beams` | Beam search width | 1 (don't use with sampling) |

---

## Stage 7: Save & Share

### 7a. Save locally

```python
# Save LoRA adapter (small — just the adapter weights)
trainer.save_model("./my-mbpp-lora")
tokenizer.save_pretrained("./my-mbpp-lora")
# Creates: adapter_model.safetensors (~17 MB), adapter_config.json, tokenizer files
```

### 7b. Load later

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reload base model
base = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)
# Attach your LoRA adapter
model = PeftModel.from_pretrained(base, "./my-mbpp-lora")
tokenizer = AutoTokenizer.from_pretrained("./my-mbpp-lora")
```

### 7c. Merge LoRA into base model (optional — for deployment)

```python
# Merge adapter weights into base model (no more adapter overhead)
merged = model.merge_and_unload()
merged.save_pretrained("./my-mbpp-merged")
tokenizer.save_pretrained("./my-mbpp-merged")
# Now it's a standalone model — no need for peft at inference time
```

### 7d. Push to HuggingFace Hub

```python
# Login first: huggingface-cli login
model.push_to_hub("your-username/mbpp-codellama-lora")
tokenizer.push_to_hub("your-username/mbpp-codellama-lora")
```

---

## Pipeline Summary

| Stage | What | Key Classes | Output |
|-------|------|------------|--------|
| 1. Load data | Get MBPP from Hub | `load_dataset()` | `DatasetDict` |
| 2. Preprocess | Format + tokenize | `AutoTokenizer`, `.map()` | Formatted dataset |
| 3. Load model | Load + quantize + LoRA | `AutoModelForCausalLM`, `BitsAndBytesConfig`, `LoraConfig` | Ready-to-train model |
| 4. Configure | Set hyperparameters | `SFTConfig` | Training config |
| 5. Train | Run optimization | `SFTTrainer.train()` | Trained weights |
| 6. Evaluate | pass@k on test set | `model.generate()`, `subprocess` | Metrics |
| 7. Save | Export adapter/model | `.save_pretrained()`, `.push_to_hub()` | Shareable model |
