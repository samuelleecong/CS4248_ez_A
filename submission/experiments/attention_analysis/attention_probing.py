"""
Probing analysis: train linear classifiers on frozen layer representations
to measure what information is decodably present at each layer.

Uses cached CLS representations from the CKA computation.
Probes for: difficulty, algorithm tag, source.
"""

from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

CACHE_DIR = Path("mbpp_kd_suite/attention_figures/_cache")
FIGURE_DIR = Path("mbpp_kd_suite/attention_figures/final")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TACO_DATASET = "BEE-spoke-data/TACO-hf"

EXPERIMENTS = [
    "s7_control", "s7_embed", "s8_bimga_uni", "s8_hnp", "s9_score", "s10_bimga",
]

DISPLAY_NAMES = {
    "s7_control": "Control", "s7_embed": "Embed Distill",
    "s8_bimga_uni": "BiMGA Uniform", "s8_hnp": "Hard Neg Pairwise",
    "s9_score": "Score Distill", "s10_bimga": "BiMGA (Full)",
}

MRR = {
    "s7_control": 0.205, "s7_embed": 0.303, "s8_bimga_uni": 0.313,
    "s8_hnp": 0.302, "s9_score": 0.301, "s10_bimga": 0.325,
}

LAYER_LABELS = ["L0\n(embed)", "L1\n(tf-0)", "L2\n(tf-1)", "L3\n(tf-2)", "L4\n(tf-3)"]


def parse_tag_field(raw):
    if not raw:
        return []
    try:
        return json.loads(raw.replace("'", '"'))
    except:
        return []


def load_taco_labels():
    """Load TACO test set labels (difficulty, primary tag, source)."""
    ds = load_dataset(TACO_DATASET, split="test")
    difficulties = []
    primary_tags = []
    sources = []

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

        difficulties.append(row.get("difficulty", "UNKNOWN"))
        tags = parse_tag_field(row.get("tags"))
        primary_tags.append(tags[0] if tags else "untagged")
        sources.append(row.get("source", "unknown"))

    return difficulties, primary_tags, sources


def load_student_reps(key):
    """Load cached CLS representations for a student model."""
    cache_file = CACHE_DIR / f"{key}.npz"
    data = np.load(str(cache_file), allow_pickle=True)
    cka = data["cka"]  # (n_teacher, n_student) — just to get n_student
    n_layers = cka.shape[1]

    # The cls_reps were saved as part of extract_and_cache
    # They are stored in the npz under keys we need to reconstruct
    # Actually, we didn't cache per-layer reps directly...
    # We need to re-extract them. But wait — let me check if teacher_reps.npz has the format
    return None, n_layers


def load_teacher_reps():
    """Load cached teacher CLS representations."""
    cache_file = CACHE_DIR / "teacher_reps.npz"
    data = np.load(str(cache_file), allow_pickle=True)
    layer_keys = sorted([k for k in data.files if k.startswith("layer_")],
                        key=lambda x: int(x.split("_")[1]))
    return [data[k] for k in layer_keys]


def extract_student_reps_fresh(key):
    """Extract CLS reps per layer for a student model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    ARTIFACT_BASE = Path("mbpp_kd_suite/artifacts/paper_experiments/20260402_015143")
    exp_map = {
        "s7_control":    ("s7_control_bs32",     "control_supervised"),
        "s7_embed":      ("s7_embed_dw100_aw10", "embed_distill"),
        "s8_bimga_uni":  ("s8_A2_bimga_uniform", "bimga_uniform"),
        "s8_hnp":        ("s8_hnp_dw100_pw10",   "hard_negative_pair_distill"),
        "s9_score":      ("s9_score_dw100",       "score_distill"),
        "s10_bimga":     ("s10_bimga_dw100_aw10", "bimga"),
    }

    d, m = exp_map[key]
    p = ARTIFACT_BASE / d / m / "model"
    model = AutoModel.from_pretrained(str(p / "backbone"), attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(str(p / "tokenizer"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # Load queries
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

    all_reps = None
    batch_size = 16
    with torch.no_grad():
        for start in tqdm(range(0, len(queries), batch_size), desc=f"  {DISPLAY_NAMES[key]}", leave=False):
            batch = queries[start:start + batch_size]
            enc = tok(batch, max_length=160, truncation=True, padding=True, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True)
            cls = [h[:, 0, :].cpu().numpy() for h in out.hidden_states]
            if all_reps is None:
                all_reps = [[] for _ in range(len(cls))]
            for i, c in enumerate(cls):
                all_reps[i].append(c)
            del out, enc

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [np.concatenate(r, 0) for r in all_reps]


def probe_accuracy(X, y, n_splits=5):
    """5-fold cross-validated accuracy of a linear probe."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Filter classes with too few examples for stratified CV
    class_counts = Counter(y_enc)
    valid_mask = np.array([class_counts[yi] >= n_splits for yi in y_enc])
    if valid_mask.sum() < 20:
        return np.nan
    X_f = X[valid_mask]
    y_f = y_enc[valid_mask]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_f, y_f):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial", C=1.0)
        clf.fit(X_f[train_idx], y_f[train_idx])
        accs.append(clf.score(X_f[test_idx], y_f[test_idx]))
    return float(np.mean(accs))


def main():
    print("=" * 60)
    print("PROBING ANALYSIS")
    print("=" * 60)

    # Load labels
    print("\nLoading TACO labels...")
    difficulties, primary_tags, sources = load_taco_labels()
    n = len(difficulties)
    print(f"  {n} examples")
    print(f"  Difficulties: {Counter(difficulties)}")
    print(f"  Top tags: {Counter(primary_tags).most_common(8)}")
    print(f"  Sources: {Counter(sources)}")

    probe_tasks = {
        "Difficulty": np.array(difficulties),
        "Algorithm Tag": np.array(primary_tags),
        "Source": np.array(sources),
    }

    # --- Teacher probing ---
    print("\nProbing teacher...")
    teacher_reps = load_teacher_reps()
    teacher_results = {}
    for task_name, labels in probe_tasks.items():
        teacher_results[task_name] = {}
        for li, reps in enumerate(teacher_reps):
            acc = probe_accuracy(reps, labels)
            teacher_results[task_name][li] = acc
        print(f"  {task_name}: best layer = L{max(teacher_results[task_name], key=teacher_results[task_name].get)} "
              f"({max(teacher_results[task_name].values()):.3f})")

    # --- Student probing ---
    print("\nProbing students...")
    student_results = {}  # key -> task -> layer -> accuracy

    for key in EXPERIMENTS:
        print(f"\n  {DISPLAY_NAMES[key]}:")
        # Check if we have a dedicated probe cache
        probe_cache = CACHE_DIR / f"probe_reps_{key}.npz"
        if probe_cache.exists():
            data = np.load(str(probe_cache), allow_pickle=True)
            layer_keys = sorted([k for k in data.files if k.startswith("layer_")],
                                key=lambda x: int(x.split("_")[1]))
            reps = [data[k] for k in layer_keys]
        else:
            reps = extract_student_reps_fresh(key)
            # Cache for next time
            save_dict = {f"layer_{i}": r for i, r in enumerate(reps)}
            np.savez(str(probe_cache), **save_dict)
            print(f"    Cached to {probe_cache}")

        student_results[key] = {}
        for task_name, labels in probe_tasks.items():
            student_results[key][task_name] = {}
            for li, layer_reps in enumerate(reps):
                acc = probe_accuracy(layer_reps, labels)
                student_results[key][task_name][li] = acc
            best_layer = max(student_results[key][task_name],
                             key=student_results[key][task_name].get)
            best_acc = student_results[key][task_name][best_layer]
            print(f"    {task_name}: best L{best_layer} ({best_acc:.3f})")

        del reps
        gc.collect()

    # --- Print summary table ---
    print("\n" + "=" * 60)
    print("PROBING ACCURACY SUMMARY (best layer per model)")
    print("=" * 60)

    for task_name in probe_tasks:
        print(f"\n  {task_name}:")
        print(f"    {'Model':<20} {'Best Layer':>10} {'Accuracy':>10}")
        print(f"    {'-'*42}")
        # Teacher
        best_l = max(teacher_results[task_name], key=teacher_results[task_name].get)
        print(f"    {'Teacher (L12)':<20} {'L'+str(best_l):>10} {teacher_results[task_name][best_l]:>10.3f}")
        for key in EXPERIMENTS:
            best_l = max(student_results[key][task_name],
                         key=student_results[key][task_name].get)
            print(f"    {DISPLAY_NAMES[key]:<20} {'L'+str(best_l):>10} "
                  f"{student_results[key][task_name][best_l]:>10.3f}")

    # --- Generate figures ---
    print("\n\nGenerating figures...")

    for task_name in probe_tasks:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Teacher (dashed, across student x-axis using interpolation)
        teacher_accs = [teacher_results[task_name][li] for li in sorted(teacher_results[task_name])]
        # Show teacher best as horizontal line
        teacher_best = max(teacher_accs)
        ax.axhline(teacher_best, color="black", linestyle="--", linewidth=1.5,
                    label=f"Teacher best ({teacher_best:.3f})", alpha=0.7)

        # Students
        colors = {
            "s7_control": "#d62728", "s7_embed": "#66c2a5",
            "s8_bimga_uni": "#2166ac", "s8_hnp": "#aaaaaa",
            "s9_score": "#aaaaaa", "s10_bimga": "#d62728",
        }
        markers = {
            "s7_control": "x", "s7_embed": "^",
            "s8_bimga_uni": "o", "s8_hnp": "D",
            "s9_score": "s", "s10_bimga": "o",
        }
        linewidths = {
            "s7_control": 2, "s7_embed": 1.5,
            "s8_bimga_uni": 2.5, "s8_hnp": 1.5,
            "s9_score": 1.5, "s10_bimga": 3,
        }

        n_student_layers = len(student_results[EXPERIMENTS[0]][task_name])
        x = np.arange(n_student_layers)

        for key in EXPERIMENTS:
            accs = [student_results[key][task_name][li] for li in range(n_student_layers)]
            ax.plot(x, accs, marker=markers[key], label=f"{DISPLAY_NAMES[key]} (MRR={MRR[key]:.3f})",
                    color=colors[key], linewidth=linewidths[key], markersize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_LABELS[:n_student_layers], fontsize=9)
        ax.set_ylabel("Probe Accuracy (5-fold CV)", fontsize=11)
        ax.set_title(f"Linear Probing: {task_name} Prediction", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(-0.3, n_student_layers - 0.7)

        fig.tight_layout()
        safe_name = task_name.lower().replace(" ", "_")
        fig.savefig(FIGURE_DIR / f"fig_probe_{safe_name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"fig_probe_{safe_name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: fig_probe_{safe_name}")

    # --- Combined probe figure (all 3 tasks side by side) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, task_name in zip(axes, probe_tasks):
        teacher_best = max(teacher_results[task_name].values())
        ax.axhline(teacher_best, color="black", linestyle="--", linewidth=1.5,
                    label=f"Teacher best ({teacher_best:.3f})", alpha=0.7)

        n_sl = len(student_results[EXPERIMENTS[0]][task_name])
        x = np.arange(n_sl)

        for key in EXPERIMENTS:
            accs = [student_results[key][task_name][li] for li in range(n_sl)]
            ax.plot(x, accs, marker=markers[key],
                    label=f"{DISPLAY_NAMES[key]}",
                    color=colors[key], linewidth=linewidths[key], markersize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_LABELS[:n_sl], fontsize=8)
        ax.set_title(task_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Probe Accuracy" if ax == axes[0] else "", fontsize=10)

    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Linear Probing: What information is decodable at each layer?",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_probing_combined.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_probing_combined.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_probing_combined")

    print("\nDone!")


if __name__ == "__main__":
    main()
