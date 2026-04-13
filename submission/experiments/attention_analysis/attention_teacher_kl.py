"""
KL divergence between each student model and the teacher.
Computes per (teacher_layer, student_layer) KL, reports best-matching teacher
layer per student layer, and an overall scalar per model.

Also bootstraps the overall scalar for significance testing.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

ARTIFACT_BASE = Path("mbpp_kd_suite/artifacts/paper_experiments/20260402_015143")
FIGURE_DIR = Path("mbpp_kd_suite/attention_figures/final")
CACHE_DIR = Path("mbpp_kd_suite/attention_figures/_cache")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
TACO_DATASET = "BEE-spoke-data/TACO-hf"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 300,
})

EXP_MAP = {
    "s7_control": ("s7_control_bs32", "control_supervised"),
    "s7_embed": ("s7_embed_dw100_aw10", "embed_distill"),
    "s8_bimga_uni": ("s8_A2_bimga_uniform", "bimga_uniform"),
    "s8_hnp": ("s8_hnp_dw100_pw10", "hard_negative_pair_distill"),
    "s9_score": ("s9_score_dw100", "score_distill"),
    "s10_bimga": ("s10_bimga_dw100_aw10", "bimga"),
}

EXPERIMENTS = ["s7_control", "s9_score", "s8_hnp", "s7_embed", "s8_bimga_uni", "s10_bimga"]

DISPLAY = {
    "s7_control": "Control (Supervised)", "s7_embed": "Embed Distill",
    "s8_bimga_uni": "BiMGA Uniform", "s8_hnp": "Hard Neg Pairwise",
    "s9_score": "Score Distill", "s10_bimga": "BiMGA (Full)",
}

MRR = {
    "s7_control": 0.205, "s7_embed": 0.303, "s8_bimga_uni": 0.313,
    "s8_hnp": 0.302, "s9_score": 0.301, "s10_bimga": 0.325,
}

COLORS = {
    "s7_control": "#d62728", "s7_embed": "#66c2a5",
    "s8_bimga_uni": "#4393c3", "s8_hnp": "#999999",
    "s9_score": "#999999", "s10_bimga": "#1a237e",
}

MARKERS = {
    "s7_control": "x", "s7_embed": "^",
    "s8_bimga_uni": "D", "s8_hnp": "s",
    "s9_score": "v", "s10_bimga": "o",
}

LW = {
    "s7_control": 2, "s7_embed": 1.5,
    "s8_bimga_uni": 2, "s8_hnp": 1.5,
    "s9_score": 1.5, "s10_bimga": 2.5,
}

LAYER_LABELS = ["L0\n(embed)", "L1\n(tf-0)", "L2\n(tf-1)", "L3\n(tf-2)", "L4\n(tf-3\noutput)"]

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_taco_queries():
    ds = load_dataset(TACO_DATASET, split="test")
    queries = []
    for row in ds:
        question = (row.get("question") or "").strip()
        solutions_raw = row.get("solutions") or "[]"
        if isinstance(solutions_raw, str):
            try:
                solutions = json.loads(solutions_raw)
            except:
                solutions = []
        else:
            solutions = solutions_raw
        solutions = [s.strip() for s in solutions if isinstance(s, str) and s.strip()]
        if not question or not solutions:
            continue
        starter = (row.get("starter_code") or "").strip()
        query = f"{question}\n\nStarter code:\n{starter}" if starter else question
        queries.append(query)
    return queries


def load_student(key):
    d, m = EXP_MAP[key]
    p = ARTIFACT_BASE / d / m / "model"
    model = AutoModel.from_pretrained(str(p / "backbone"), attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(str(p / "tokenizer"))
    model.eval().to(DEVICE)
    return model, tok


def load_teacher():
    model = AutoModel.from_pretrained(TEACHER_MODEL, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model.eval().to(DEVICE)
    return model, tok


@torch.no_grad()
def compute_teacher_student_kl(teacher_model, teacher_tok, student_model, student_tok, queries):
    """
    Compute per-example KL(student || teacher) for each (teacher_layer, student_layer) pair.

    Uses teacher tokenizer for both to ensure same sequence length.
    Student layers: 4, Teacher layers: 12. Both have 12 heads.

    Returns:
        per_example_kl: (n_examples, n_teacher_layers, n_student_layers) — mean KL per head per position
    """
    n_teacher_layers = teacher_model.config.num_hidden_layers
    n_student_layers = student_model.config.num_hidden_layers

    all_example_kl = []

    for start in tqdm(range(0, len(queries), BATCH_SIZE), desc="teacher-student KL", leave=False):
        batch = queries[start:start + BATCH_SIZE]

        # Use teacher tokenizer for both — same vocab (BERT WordPiece)
        enc = teacher_tok(batch, max_length=160, truncation=True, padding=True, return_tensors="pt").to(DEVICE)

        t_out = teacher_model(**enc, output_attentions=True)
        s_out = student_model(**enc, output_attentions=True)

        # t_attn: tuple of (batch, heads, seq, seq) x n_teacher_layers
        # s_attn: tuple of (batch, heads, seq, seq) x n_student_layers
        t_attn = [a.cpu().numpy() for a in t_out.attentions]  # list of (B, H, S, S)
        s_attn = [a.cpu().numpy() for a in s_out.attentions]

        masks = enc["attention_mask"].cpu().numpy().astype(bool)

        for i in range(len(batch)):
            sl = int(masks[i].sum())
            example_kl = np.zeros((n_teacher_layers, n_student_layers))

            for ti in range(n_teacher_layers):
                t_a = t_attn[ti][i, :, :sl, :sl]  # (H, sl, sl)
                for si in range(n_student_layers):
                    s_a = s_attn[si][i, :, :sl, :sl]  # (H, sl, sl)
                    eps = 1e-10
                    # KL(student || teacher) per head per row, averaged
                    kl = np.sum(s_a * np.log((s_a + eps) / (t_a + eps)), axis=-1).mean()
                    example_kl[ti, si] = kl

            all_example_kl.append(example_kl)

        del t_out, s_out, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.array(all_example_kl)  # (N, T_layers, S_layers)


def main():
    print("=" * 60)
    print("TEACHER-STUDENT KL DIVERGENCE")
    print("=" * 60)

    queries = load_taco_queries()
    print(f"Loaded {len(queries)} queries")

    teacher_model, teacher_tok = load_teacher()

    all_results = {}

    for key in EXPERIMENTS:
        print(f"\n{DISPLAY[key]}...")
        cache_file = CACHE_DIR / f"teacher_kl_{key}.npy"

        if cache_file.exists():
            per_ex_kl = np.load(str(cache_file))
            print(f"  Loaded from cache ({per_ex_kl.shape})")
        else:
            student_model, student_tok = load_student(key)
            per_ex_kl = compute_teacher_student_kl(teacher_model, teacher_tok,
                                                    student_model, student_tok, queries)
            np.save(str(cache_file), per_ex_kl)
            print(f"  Computed and cached ({per_ex_kl.shape})")
            del student_model, student_tok
            gc.collect()

        # Mean across examples: (T_layers, S_layers)
        mean_kl = per_ex_kl.mean(axis=0)
        # Best-matching teacher layer per student layer (min KL)
        best_kl = mean_kl.min(axis=0)  # (S_layers,)
        best_teacher = mean_kl.argmin(axis=0)

        all_results[key] = {
            "per_example": per_ex_kl,
            "mean_matrix": mean_kl,
            "best_kl": best_kl,
            "best_teacher": best_teacher,
            "overall_mean": float(best_kl[1:].mean()),  # skip L0 (embedding)
        }

        print(f"  Per student layer (best teacher match):")
        for si in range(len(best_kl)):
            print(f"    Student L{si} <- Teacher L{best_teacher[si]}: KL = {best_kl[si]:.4f}")
        print(f"  Overall mean (L1-L3): {all_results[key]['overall_mean']:.4f}")

    del teacher_model, teacher_tok
    gc.collect()

    # ---- Bootstrap overall mean ----
    print("\n" + "=" * 60)
    print("BOOTSTRAP (n=1000)")
    print("=" * 60)

    N_BOOT = 1000
    rng = np.random.RandomState(42)
    n = len(queries)

    boot_means = {k: [] for k in EXPERIMENTS}
    for _ in tqdm(range(N_BOOT), desc="bootstrap"):
        idx = rng.choice(n, size=n, replace=True)
        for key in EXPERIMENTS:
            per_ex = all_results[key]["per_example"][idx]  # (n, T, S)
            mean_mat = per_ex.mean(axis=0)  # (T, S)
            best = mean_mat.min(axis=0)  # (S,)
            boot_means[key].append(float(best[1:].mean()))  # L1-L3

    print(f"\n  {'Model':<25} {'Mean KL':>8} {'95% CI':>22}")
    print(f"  {'-'*57}")
    for key in EXPERIMENTS:
        vals = boot_means[key]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {DISPLAY[key]:<25} {np.mean(vals):>8.4f} [{lo:.4f}, {hi:.4f}]")

    # ---- Figures ----
    print("\nGenerating figures...")

    # Bar chart: overall mean KL from teacher
    fig, ax = plt.subplots(figsize=(8, 5))
    keys_sorted = sorted(EXPERIMENTS, key=lambda k: all_results[k]["overall_mean"])
    names = [DISPLAY[k] for k in keys_sorted]
    vals = [all_results[k]["overall_mean"] for k in keys_sorted]
    colors = [COLORS[k] for k in keys_sorted]

    bars = ax.barh(range(len(keys_sorted)), vals, color=colors,
                   edgecolor="black", linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(keys_sorted)))
    ax.set_yticklabels([f"{n}\n(MRR={MRR[k]:.3f})" for n, k in zip(names, keys_sorted)], fontsize=9)
    ax.set_xlabel("Mean KL Divergence from Teacher (best-matching layers, L1-L3)")
    ax.set_title("Attention Pattern Similarity to Teacher", fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_teacher_kl_bar.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_teacher_kl_bar.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_teacher_kl_bar")

    # Per-layer line plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in EXPERIMENTS:
        best_kl = all_results[key]["best_kl"][1:]  # skip embedding layer
        x = np.arange(len(best_kl))
        ax.plot(x, best_kl, marker=MARKERS[key],
                label=f"{DISPLAY[key]} (MRR={MRR[key]:.3f})",
                color=COLORS[key], linewidth=LW[key], markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_LABELS[1:1+len(x)])
    ax.set_ylabel("KL from Teacher (best-matching teacher layer)")
    ax.set_title("Per-layer Attention Divergence from Teacher", fontweight="bold")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_teacher_kl_perlayer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_teacher_kl_perlayer.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_teacher_kl_perlayer")

    print(f"\nAll figures in: {FIGURE_DIR.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()
