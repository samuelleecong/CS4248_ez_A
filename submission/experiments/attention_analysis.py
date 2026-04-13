"""
Attention internals analysis across 6 trained student models on TACO.

Memory-efficient: computes statistics on-the-fly per batch instead of
storing all raw attention matrices. Only keeps a few cherry-picked
examples for rollout visualization.

Analyses:
  1. Attention entropy per head (layer x head heatmaps)
  2. CLS token attention distribution (what CLS attends to)
  3. Attention rollout (effective attention from input to CLS)
  4. Teacher-student CKA alignment
  5. Per-tag and per-difficulty breakdowns
  6. KL divergence between model attention distributions

Outputs figures to: mbpp_kd_suite/attention_figures/
"""

from __future__ import annotations

import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARTIFACT_BASE = Path("mbpp_kd_suite/artifacts/paper_experiments/20260402_015143")
FIGURE_DIR = Path("mbpp_kd_suite/attention_figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
TACO_DATASET = "BEE-spoke-data/TACO-hf"

EXPERIMENTS = {
    "s7_control":    ("s7_control_bs32",     "control_supervised"),
    "s7_embed":      ("s7_embed_dw100_aw10", "embed_distill"),
    "s8_bimga_uni":  ("s8_A2_bimga_uniform", "bimga_uniform"),
    "s8_hnp":        ("s8_hnp_dw100_pw10",   "hard_negative_pair_distill"),
    "s9_score":      ("s9_score_dw100",       "score_distill"),
    "s10_bimga":     ("s10_bimga_dw100_aw10", "bimga"),
}

DISPLAY_NAMES = {
    "s7_control":   "Control (Supervised)",
    "s7_embed":     "Embed Distill",
    "s8_bimga_uni": "BiMGA Uniform",
    "s8_hnp":       "Hard Neg Pairwise",
    "s9_score":     "Score Distill",
    "s10_bimga":    "BiMGA (Full)",
}

MRR_SCORES = {
    "s7_control": 0.205,
    "s7_embed": 0.303,
    "s8_bimga_uni": 0.313,
    "s8_hnp": 0.302,
    "s9_score": 0.301,
    "s10_bimga": 0.325,
}

PYTHON_KEYWORDS = {
    "def", "return", "if", "else", "elif", "for", "while", "in", "not",
    "and", "or", "class", "import", "from", "try", "except", "with",
    "as", "yield", "lambda", "pass", "break", "continue", "true", "false",
    "none", "is", "raise", "finally", "global", "assert",
}

MAX_QUERY_LEN = 160
BATCH_SIZE = 8  # Smaller batches to avoid OOM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cherry-pick indices for rollout visualization
ROLLOUT_INDICES = {0, 50, 200}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_tag_field(raw) -> list[str]:
    if not raw:
        return []
    try:
        return json.loads(raw.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return []


def load_taco_test():
    ds = load_dataset(TACO_DATASET, split="test")
    examples = []
    for row in ds:
        question = (row.get("question") or "").strip()
        solutions_raw = row.get("solutions") or "[]"
        if isinstance(solutions_raw, str):
            try:
                solutions = json.loads(solutions_raw)
            except json.JSONDecodeError:
                solutions = []
        else:
            solutions = solutions_raw
        solutions = [s.strip() for s in solutions if isinstance(s, str) and s.strip()]
        if not question or not solutions:
            continue

        starter = (row.get("starter_code") or "").strip()
        query = f"{question}\n\nStarter code:\n{starter}" if starter else question

        examples.append({
            "query": query,
            "code": solutions[0],
            "difficulty": row.get("difficulty", "UNKNOWN"),
            "tags": parse_tag_field(row.get("tags")),
            "skill_types": parse_tag_field(row.get("skill_types")),
            "source": row.get("source", "unknown"),
        })
    return examples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_student_model(exp_key: str) -> tuple[AutoModel, AutoTokenizer]:
    exp_dir, method = EXPERIMENTS[exp_key]
    model_path = ARTIFACT_BASE / exp_dir / method / "model"
    model = AutoModel.from_pretrained(
        str(model_path / "backbone"),
        attn_implementation="eager",  # Required for output_attentions
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path / "tokenizer"))
    model.eval().to(DEVICE)
    return model, tokenizer


def load_teacher_model() -> tuple[AutoModel, AutoTokenizer]:
    model = AutoModel.from_pretrained(
        TEACHER_MODEL,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model.eval().to(DEVICE)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Streaming statistics accumulators
# ---------------------------------------------------------------------------

class EntropyAccumulator:
    """Accumulates per-head entropy statistics without storing raw attentions."""

    def __init__(self):
        self.sum = None
        self.count = 0
        # Per-group accumulators: group_key -> (sum, count)
        self.group_sums = defaultdict(lambda: None)
        self.group_counts = defaultdict(int)

    def update(self, attn: np.ndarray, group_key: str | None = None):
        """attn: (num_layers, num_heads, seq_len, seq_len)"""
        eps = 1e-10
        log_attn = np.log(attn + eps)
        ent = -np.sum(attn * log_attn, axis=-1)  # (L, H, S)
        head_ent = ent.mean(axis=-1)  # (L, H)

        if self.sum is None:
            self.sum = np.zeros_like(head_ent)
        self.sum += head_ent
        self.count += 1

        if group_key is not None:
            if self.group_sums[group_key] is None:
                self.group_sums[group_key] = np.zeros_like(head_ent)
            self.group_sums[group_key] += head_ent
            self.group_counts[group_key] += 1

    def result(self) -> np.ndarray:
        return self.sum / max(self.count, 1)

    def group_results(self) -> dict[str, np.ndarray]:
        return {
            k: v / max(self.group_counts[k], 1)
            for k, v in self.group_sums.items()
            if v is not None
        }


class CLSAttentionAccumulator:
    """Tracks what fraction of CLS attention goes to special/keyword/content tokens."""

    def __init__(self):
        self.special = []
        self.keyword = []
        self.content = []

    def update(self, attn: np.ndarray, tokens: list[str]):
        """attn: (num_layers, num_heads, seq_len, seq_len)"""
        # Final layer, average across heads, CLS row
        cls_attn = attn[-1, :, 0, :].mean(axis=0)  # (seq_len,)
        sep, kw, cont = 0.0, 0.0, 0.0
        for j, tok in enumerate(tokens):
            tok_clean = tok.lower().replace("##", "")
            if tok in ("[SEP]", "[CLS]", "[PAD]"):
                sep += cls_attn[j]
            elif tok_clean in PYTHON_KEYWORDS:
                kw += cls_attn[j]
            else:
                cont += cls_attn[j]
        total = sep + kw + cont + 1e-10
        self.special.append(sep / total)
        self.keyword.append(kw / total)
        self.content.append(cont / total)

    def result(self) -> dict[str, float]:
        return {
            "Special ([CLS]/[SEP])": float(np.mean(self.special)),
            "Python Keywords": float(np.mean(self.keyword)),
            "Content Tokens": float(np.mean(self.content)),
        }


def attention_rollout(attn: np.ndarray) -> np.ndarray:
    """Compute attention rollout. attn: (L, H, S, S). Returns (S,)."""
    num_layers, _, seq_len, _ = attn.shape
    rollout = np.eye(seq_len)
    for layer in range(num_layers):
        attn_layer = attn[layer].mean(axis=0)
        attn_layer = 0.5 * attn_layer + 0.5 * np.eye(seq_len)
        attn_layer = attn_layer / attn_layer.sum(axis=-1, keepdims=True)
        rollout = attn_layer @ rollout
    return rollout[0]


# ---------------------------------------------------------------------------
# Single-pass extraction: computes all per-example stats in one forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_stats(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    queries: list[str],
    difficulties: list[str],
    primary_tags: list[str],
) -> dict:
    """
    Single pass through data. Returns dict with:
      entropy: (L, H) mean entropy
      entropy_by_difficulty: {diff: (L, H)}
      entropy_by_tag: {tag: (L, H)}
      cls_stats: {category: float}
      rollout_examples: {idx: (rollout_array, tokens)}
      cls_reps: list of (n_examples, hidden) per layer (for CKA)
    """
    entropy_acc = EntropyAccumulator()
    diff_entropy_acc = {d: EntropyAccumulator() for d in set(difficulties)}
    tag_entropy_acc = {t: EntropyAccumulator() for t in set(primary_tags)}
    cls_acc = CLSAttentionAccumulator()
    rollout_examples = {}
    all_cls_reps = None  # list of lists, one per layer

    global_idx = 0
    for start in tqdm(range(0, len(queries), BATCH_SIZE), desc="processing", leave=False):
        batch_texts = queries[start:start + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            max_length=MAX_QUERY_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = model(
            **encoded,
            output_attentions=True,
            output_hidden_states=True,
        )

        # Attentions: tuple of (batch, heads, seq, seq) per layer
        attn_tuple = outputs.attentions
        stacked = torch.stack(attn_tuple, dim=0).cpu().numpy()  # (L, B, H, S, S)
        stacked = stacked.transpose(1, 0, 2, 3, 4)  # (B, L, H, S, S)

        masks = encoded["attention_mask"].cpu().numpy().astype(bool)

        # Hidden states for CKA: CLS token at each layer
        hidden_states = outputs.hidden_states  # tuple of (B, S, D)
        cls_hidden = [h[:, 0, :].cpu().numpy() for h in hidden_states]  # list of (B, D)
        if all_cls_reps is None:
            all_cls_reps = [[] for _ in range(len(cls_hidden))]
        for li, h in enumerate(cls_hidden):
            all_cls_reps[li].append(h)

        for i in range(len(batch_texts)):
            idx = global_idx + i
            seq_len = int(masks[i].sum())
            attn_i = stacked[i, :, :, :seq_len, :seq_len]  # (L, H, S, S)
            tokens_i = tokenizer.convert_ids_to_tokens(
                encoded["input_ids"][i, :seq_len].cpu().tolist()
            )

            diff = difficulties[idx]
            tag = primary_tags[idx]

            # Entropy
            entropy_acc.update(attn_i, group_key=None)
            diff_entropy_acc[diff].update(attn_i)
            tag_entropy_acc[tag].update(attn_i)

            # CLS attention
            cls_acc.update(attn_i, tokens_i)

            # Rollout for cherry-picked examples
            if idx in ROLLOUT_INDICES:
                rollout_examples[idx] = (attention_rollout(attn_i), tokens_i)

        global_idx += len(batch_texts)

        # Free GPU memory
        del outputs, attn_tuple, stacked, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "entropy": entropy_acc.result(),
        "entropy_by_difficulty": {d: acc.result() for d, acc in diff_entropy_acc.items() if acc.count > 0},
        "entropy_by_tag": {t: acc.result() for t, acc in tag_entropy_acc.items() if acc.count > 0},
        "cls_stats": cls_acc.result(),
        "rollout_examples": rollout_examples,
        "cls_reps": [np.concatenate(layer_list, axis=0) for layer_list in all_cls_reps],
    }


# ---------------------------------------------------------------------------
# KL divergence: streaming pairwise comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_kl_between_models(
    model_a: AutoModel, tok_a: AutoTokenizer,
    model_b: AutoModel, tok_b: AutoTokenizer,
    queries: list[str],
) -> np.ndarray:
    """Compute mean KL(A || B) per (layer, head) in a streaming fashion."""
    kl_sum = None
    count = 0

    for start in tqdm(range(0, len(queries), BATCH_SIZE), desc="KL", leave=False):
        batch = queries[start:start + BATCH_SIZE]

        enc_a = tok_a(batch, max_length=MAX_QUERY_LEN, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        enc_b = tok_b(batch, max_length=MAX_QUERY_LEN, truncation=True, padding=True, return_tensors="pt").to(DEVICE)

        out_a = model_a(**enc_a, output_attentions=True)
        out_b = model_b(**enc_b, output_attentions=True)

        attn_a = torch.stack(out_a.attentions, dim=0).cpu().numpy().transpose(1, 0, 2, 3, 4)
        attn_b = torch.stack(out_b.attentions, dim=0).cpu().numpy().transpose(1, 0, 2, 3, 4)

        masks_a = enc_a["attention_mask"].cpu().numpy().astype(bool)
        masks_b = enc_b["attention_mask"].cpu().numpy().astype(bool)

        for i in range(len(batch)):
            sl_a = int(masks_a[i].sum())
            sl_b = int(masks_b[i].sum())
            sl = min(sl_a, sl_b)
            a = attn_a[i, :, :, :sl, :sl]
            b = attn_b[i, :, :, :sl, :sl]
            eps = 1e-10
            kl = np.sum(a * np.log((a + eps) / (b + eps)), axis=-1).mean(axis=-1)  # (L, H)
            if kl_sum is None:
                kl_sum = np.zeros_like(kl)
            kl_sum += kl
            count += 1

        del out_a, out_b, enc_a, enc_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return kl_sum / max(count, 1)


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def compute_cka_matrix(teacher_reps, student_reps) -> np.ndarray:
    n_t, n_s = len(teacher_reps), len(student_reps)
    cka = np.zeros((n_t, n_s))
    for i in range(n_t):
        for j in range(n_s):
            cka[i, j] = linear_cka(teacher_reps[i], student_reps[j])
    return cka


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_entropy_heatmaps(all_entropies: dict[str, np.ndarray]):
    n = len(all_entropies)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    vmin = min(e.min() for e in all_entropies.values())
    vmax = max(e.max() for e in all_entropies.values())
    for ax, (key, ent) in zip(axes, all_entropies.items()):
        im = ax.imshow(ent, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Head")
        ax.set_title(f"{DISPLAY_NAMES[key]}\n(MRR={MRR_SCORES[key]:.3f})", fontsize=9)
        ax.set_xticks(range(ent.shape[1]))
        ax.set_xticklabels(range(ent.shape[1]), fontsize=6)
        if ax == axes[0]:
            ax.set_ylabel("Layer")
            ax.set_yticks(range(ent.shape[0]))
    fig.colorbar(im, ax=axes, label="Mean Entropy (nats)", shrink=0.8)
    fig.suptitle("Attention Entropy per Head", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_attention_entropy.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_attention_entropy.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_attention_entropy")


def plot_cls_attention_bars(all_cls: dict[str, dict[str, float]]):
    models = list(all_cls.keys())
    categories = ["Special ([CLS]/[SEP])", "Python Keywords", "Content Tokens"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    for cat, color in zip(categories, colors):
        vals = [all_cls[m].get(cat, 0) for m in models]
        ax.bar(x, vals, bottom=bottoms, label=cat, color=color, width=0.6)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{DISPLAY_NAMES[m]}\n({MRR_SCORES[m]:.3f})" for m in models],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Fraction of [CLS] Attention (Final Layer)")
    ax.set_title("[CLS] Token Attention Distribution")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_cls_attention.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_cls_attention.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_cls_attention")


def plot_rollout_examples(all_rollouts: dict[str, dict[int, tuple]]):
    indices = sorted(ROLLOUT_INDICES)
    model_keys = list(all_rollouts.keys())
    n_ex = len(indices)
    n_m = len(model_keys)
    fig, axes = plt.subplots(n_ex, n_m, figsize=(4 * n_m, 3 * n_ex), squeeze=False)
    for row, idx in enumerate(indices):
        for col, key in enumerate(model_keys):
            ax = axes[row][col]
            if idx in all_rollouts[key]:
                rollout, tokens = all_rollouts[key][idx]
                n_tok = min(len(tokens), 35)
                ax.barh(range(n_tok), rollout[:n_tok], color="steelblue")
                ax.set_yticks(range(n_tok))
                ax.set_yticklabels(tokens[:n_tok], fontsize=5)
                ax.invert_yaxis()
            if row == 0:
                ax.set_title(DISPLAY_NAMES[key], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Ex {idx}", fontsize=9)
    fig.suptitle("Attention Rollout: Effective [CLS] -> Input Tokens", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_attention_rollout.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_attention_rollout.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_attention_rollout")


def plot_cka_matrices(all_cka: dict[str, np.ndarray]):
    n = len(all_cka)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (key, cka) in zip(axes, all_cka.items()):
        im = ax.imshow(cka, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("Student Layer")
        ax.set_title(f"{DISPLAY_NAMES[key]}\n(MRR={MRR_SCORES[key]:.3f})", fontsize=9)
        ax.set_xticks(range(cka.shape[1]))
        if ax == axes[0]:
            ax.set_ylabel("Teacher Layer")
            ax.set_yticks(range(cka.shape[0]))
    fig.colorbar(im, ax=axes, label="Linear CKA", shrink=0.8)
    fig.suptitle("Teacher-Student Representational Alignment (CKA)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_cka_alignment.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_cka_alignment.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_cka_alignment")


def plot_entropy_by_difficulty(all_diff_ent: dict[str, dict[str, np.ndarray]], diffs: list[str]):
    model_keys = list(all_diff_ent.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(model_keys))
    width = 0.8 / len(diffs)
    colors = sns.color_palette("Set2", len(diffs))
    for i, diff in enumerate(diffs):
        vals = []
        for key in model_keys:
            e = all_diff_ent[key].get(diff)
            vals.append(e.mean() if e is not None else 0)
        offset = (i - len(diffs) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=diff, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[k] for k in model_keys], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title("Attention Entropy by Problem Difficulty")
    ax.legend(title="Difficulty")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_entropy_by_difficulty.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_entropy_by_difficulty.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_entropy_by_difficulty")


def plot_entropy_by_tags(all_tag_ent: dict[str, dict[str, np.ndarray]], top_tags: list[str]):
    model_keys = list(all_tag_ent.keys())
    matrix = np.full((len(top_tags), len(model_keys)), np.nan)
    for j, key in enumerate(model_keys):
        for i, tag in enumerate(top_tags):
            e = all_tag_ent[key].get(tag)
            if e is not None:
                matrix[i, j] = e.mean()
    fig, ax = plt.subplots(figsize=(10, max(5, len(top_tags) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels([DISPLAY_NAMES[k] for k in model_keys], fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(top_tags)))
    ax.set_yticklabels(top_tags, fontsize=9)
    ax.set_title("Mean Attention Entropy by Algorithm Tag")
    fig.colorbar(im, label="Mean Entropy")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_entropy_by_tags.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_entropy_by_tags.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_entropy_by_tags")


def plot_kl_from_bimga(all_kl: dict[str, np.ndarray]):
    n = len(all_kl)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    vmin = min(v.min() for v in all_kl.values())
    vmax = max(v.max() for v in all_kl.values())
    for ax, (key, kl) in zip(axes, all_kl.items()):
        im = ax.imshow(kl, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Head")
        ax.set_title(f"KL({DISPLAY_NAMES[key]} || BiMGA)", fontsize=9)
        ax.set_xticks(range(kl.shape[1]))
        ax.set_xticklabels(range(kl.shape[1]), fontsize=6)
        if ax == axes[0]:
            ax.set_ylabel("Layer")
            ax.set_yticks(range(kl.shape[0]))
    fig.colorbar(im, ax=axes, label="KL Divergence", shrink=0.8)
    fig.suptitle("Attention Divergence from BiMGA (Full)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_kl_from_bimga.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_kl_from_bimga.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_kl_from_bimga")


def plot_cka_by_difficulty(all_diff_cka: dict[str, dict[str, float]], diffs: list[str]):
    model_keys = list(all_diff_cka.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(model_keys))
    width = 0.8 / len(diffs)
    colors = sns.color_palette("Set2", len(diffs))
    for i, diff in enumerate(diffs):
        vals = [all_diff_cka[key].get(diff, 0) for key in model_keys]
        offset = (i - len(diffs) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=diff, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[k] for k in model_keys], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Mean CKA (Teacher-Student)")
    ax.set_title("Teacher-Student Alignment by Problem Difficulty")
    ax.legend(title="Difficulty")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_cka_by_difficulty.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_cka_by_difficulty.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_cka_by_difficulty")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Attention Internals Analysis (Memory-Efficient)")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading TACO test set...")
    examples = load_taco_test()
    queries = [e["query"] for e in examples]
    difficulties = [e["difficulty"] for e in examples]
    all_tags = [e["tags"] for e in examples]
    primary_tags = [tags[0] if tags else "untagged" for tags in all_tags]

    tag_counter = Counter(tag for tags in all_tags for tag in tags)
    top_tags = [t for t, _ in tag_counter.most_common(12)]
    unique_diffs = sorted(set(difficulties))
    print(f"  {len(examples)} examples | Diffs: {unique_diffs}")
    print(f"  Top tags: {top_tags}")

    # 2. Teacher representations (for CKA)
    print("\n[2] Teacher model: extracting layer representations...")
    teacher_model, teacher_tok = load_teacher_model()
    teacher_stats = extract_all_stats(teacher_model, teacher_tok, queries, difficulties, primary_tags)
    teacher_reps = teacher_stats["cls_reps"]
    print(f"  Teacher: {len(teacher_reps)} layers, {teacher_reps[0].shape[1]}d")
    del teacher_stats  # Don't need teacher attention stats
    # Keep teacher model loaded for KL computation later? No — students have 4 layers, teacher 12.
    # KL only makes sense between same-architecture models.
    del teacher_model, teacher_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Process each student
    print("\n[3] Processing student models...")
    all_entropies = {}
    all_cls_stats = {}
    all_rollouts = {}
    all_cka = {}
    all_diff_entropy = {}
    all_tag_entropy = {}
    all_diff_cka = {}
    student_reps_cache = {}  # Only keep CLS reps for CKA, not attentions

    for key in EXPERIMENTS:
        name = DISPLAY_NAMES[key]
        print(f"\n  --- {name} ---")
        model, tok = load_student_model(key)
        stats = extract_all_stats(model, tok, queries, difficulties, primary_tags)

        all_entropies[key] = stats["entropy"]
        all_cls_stats[key] = stats["cls_stats"]
        all_rollouts[key] = stats["rollout_examples"]
        all_diff_entropy[key] = stats["entropy_by_difficulty"]
        all_tag_entropy[key] = stats["entropy_by_tag"]

        # CKA
        student_reps = stats["cls_reps"]
        cka_mat = compute_cka_matrix(teacher_reps, student_reps)
        all_cka[key] = cka_mat
        print(f"    Entropy={stats['entropy'].mean():.4f}  CKA={cka_mat.mean():.4f}")

        # Per-difficulty CKA
        diff_cka = {}
        for diff in unique_diffs:
            indices = [i for i, d in enumerate(difficulties) if d == diff]
            if len(indices) < 5:
                continue
            t_sub = [r[indices] for r in teacher_reps]
            s_sub = [r[indices] for r in student_reps]
            diff_cka[diff] = float(compute_cka_matrix(t_sub, s_sub).mean())
        all_diff_cka[key] = diff_cka

        del model, tok, stats, student_reps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. KL divergence: each model vs BiMGA
    print("\n[4] Computing KL divergence vs BiMGA...")
    bimga_model, bimga_tok = load_student_model("s10_bimga")
    all_kl = {}
    for key in EXPERIMENTS:
        if key == "s10_bimga":
            continue
        print(f"  KL({DISPLAY_NAMES[key]} || BiMGA)...")
        other_model, other_tok = load_student_model(key)
        all_kl[key] = compute_kl_between_models(other_model, other_tok, bimga_model, bimga_tok, queries)
        del other_model, other_tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del bimga_model, bimga_tok
    gc.collect()

    # 5. Generate all figures
    print("\n[5] Generating figures...")
    plot_entropy_heatmaps(all_entropies)
    plot_cls_attention_bars(all_cls_stats)
    plot_rollout_examples(all_rollouts)
    plot_cka_matrices(all_cka)
    plot_entropy_by_difficulty(all_diff_entropy, unique_diffs)
    plot_entropy_by_tags(all_tag_entropy, top_tags)
    plot_kl_from_bimga(all_kl)
    plot_cka_by_difficulty(all_diff_cka, unique_diffs)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    hdr = f"{'Model':<22} {'MRR':>6} {'Entropy':>8} {'CKA':>6} {'CLS->Content':>13}"
    print(hdr)
    print("-" * len(hdr))
    for key in EXPERIMENTS:
        print(
            f"{DISPLAY_NAMES[key]:<22} "
            f"{MRR_SCORES[key]:>6.3f} "
            f"{all_entropies[key].mean():>8.4f} "
            f"{all_cka[key].mean():>6.4f} "
            f"{all_cls_stats[key].get('Content Tokens', 0):>13.4f}"
        )

    print(f"\nAll figures in: {FIGURE_DIR.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()
