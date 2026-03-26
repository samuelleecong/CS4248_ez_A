# Uploading Models to HuggingFace Hub

This guide explains how to upload trained models from a two-phase KD run to
the [HuggingFace Hub](https://huggingface.co/) so teammates and the broader
community can use them.

---

## 1. Connect to HuggingFace

You need a HuggingFace account and a write-access token.

### Get a token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → select **Write** scope → copy the token.

### Log in (one-time setup)

```bash
huggingface-cli login
# Paste your token when prompted. It is stored in ~/.cache/huggingface/token
```

Or export it as an environment variable (useful in scripts / CI):

```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 2. Run the upload command

```bash
mbpp-kd-upload --run-dir artifacts/two_phase_kd/<timestamp> --hf-org cs4248-nlp
```

### Examples

Upload everything from a timestamped run to the shared org:

```bash
mbpp-kd-upload \
  --run-dir artifacts/two_phase_kd/20260325_014013 \
  --hf-org cs4248-nlp
```

Preview what would be uploaded without actually uploading anything:

```bash
mbpp-kd-upload \
  --run-dir artifacts/two_phase_kd/20260325_014013 \
  --hf-org cs4248-nlp \
  --dry-run
```

Upload only specific models (e.g. just the phase 1 student and one KD method):

```bash
mbpp-kd-upload \
  --run-dir artifacts/two_phase_kd/20260325_014013 \
  --hf-org cs4248-nlp \
  --roles ft-student,score-distill
```

Make all uploaded repos private:

```bash
mbpp-kd-upload \
  --run-dir artifacts/two_phase_kd/20260325_014013 \
  --hf-org cs4248-nlp \
  --private
```

---

## 3. What gets uploaded

The tool scans the run directory for every model saved in HuggingFace format
(i.e. directories that contain `backbone/model.safetensors`).  For each one it:

1. Creates a HuggingFace repository (if it does not already exist).
2. Uploads the backbone weights (`config.json` + `model.safetensors`).
3. Uploads the tokenizer files.
4. Generates and uploads a `README.md` model card automatically.

Only runs where `--save-models` was passed to `mbpp-kd-two-phase` will contain
saved models.  Runs without saved models cannot be uploaded.

---

## 4. Naming convention

Repository names follow this pattern:

```
{prefix}-{role}-{dataset-slug}-{timestamp}
```

| Part | Example | Meaning |
|------|---------|---------|
| `prefix` | `cs4248` | Configurable via `--prefix` (default `cs4248`) |
| `role` | `phase1-student` | Which model in the pipeline |
| `dataset-slug` | `taco` | Lowercased last segment of the dataset name |
| `timestamp` | `20260325-014013` | Run directory name (underscores → hyphens) |

**Full example:**
`cs4248-nlp/cs4248-score-distill-taco-20260325-014013`

Available roles from a full two-phase run (role slug = directory name with `_` → `-`):

| Phase | Directory name | Role slug |
|-------|---------------|-----------|
| 1 | `phase1/ft_teacher/` | `ft-teacher` |
| 1 | `phase1/ft_student/` | `ft-student` |
| 2 | `phase2/control_supervised/` | `control-supervised` |
| 2 | `phase2/score_distill/` | `score-distill` |
| 2 | `phase2/embed_distill/` | `embed-distill` |
| 2 | `phase2/pair_distill/` | `pair-distill` |
| 2 | `phase2/qed_align/` | `qed-align` |
| 2 | `phase2/distilcse_lite/` | `distilcse-lite` |
| 2 | `phase2/adam_lite/` | `adam-lite` |
| 2 | `phase2/hpd/` | `hpd` |
| 2 | `phase2/margin_mse/` | `margin-mse` |
| 2 | `phase2/pointwise/` | `pointwise` |

---

## 5. Using an uploaded model

```python
from transformers import AutoModel, AutoTokenizer
import torch

repo_id = "cs4248-nlp/cs4248-ft-student-taco-20260325-014013"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModel.from_pretrained(repo_id)
model.eval()

def embed(texts: list[str], max_length: int = 160) -> torch.Tensor:
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       padding=True, max_length=max_length)
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(emb, dim=-1)

query_emb = embed(["find all prime numbers in a list"])
code_emb  = embed(["def get_primes(lst): return [x for x in lst if all(x%i for i in range(2,x))]"])

score = (query_emb @ code_emb.T).item()
print(f"Similarity: {score:.4f}")
```

---

## 6. All CLI options

```
mbpp-kd-upload --help

  --run-dir PATH        Path to the two-phase KD run directory
  --hf-user USERNAME    Your HuggingFace username (mutually exclusive with --hf-org)
  --hf-org ORG          HuggingFace organisation name (mutually exclusive with --hf-user)
  --prefix PREFIX       Repo name prefix (default: cs4248)
  --private             Create private repositories
  --dry-run             Preview uploads without actually uploading
  --roles ROLE1,ROLE2   Only upload the listed roles (default: all)
```
