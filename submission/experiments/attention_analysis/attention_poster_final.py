"""
Final condensed poster figures.

Figure A: CKA heatmaps + per-layer gain
Figure B: Teacher KL bar + Teacher KL per-layer
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

CACHE_DIR = Path("mbpp_kd_suite/attention_figures/_cache")
FIGURE_DIR = Path("mbpp_kd_suite/attention_figures/final")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 300,
})

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

LAYER_LABELS_SHORT = ["L0", "L1", "L2", "L3", "L4"]
LAYER_LABELS = ["L0\n(embed)", "L1\n(tf-0)", "L2\n(tf-1)", "L3\n(tf-2)", "L4\n(tf-3\noutput)"]
LAYER_LABELS_TF = ["L1\n(tf-0)", "L2\n(tf-1)", "L3\n(tf-2)", "L4\n(tf-3\noutput)"]


def load_cka(key):
    return np.load(str(CACHE_DIR / f"{key}.npz"), allow_pickle=True)["cka"]


def load_teacher_kl(key):
    return np.load(str(CACHE_DIR / f"teacher_kl_{key}.npy"))


# =========================================================================
# Figure A: CKA
# =========================================================================

def make_figure_a():
    cka_all = {k: load_cka(k) for k in EXPERIMENTS}
    ctrl_per_layer = cka_all["s7_control"].mean(axis=0)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 0.9], hspace=0.35, wspace=0.28,
                           width_ratios=[1, 1, 0.06])

    # Panel A: Control
    ax1 = fig.add_subplot(gs[0, 0])
    cka_ctrl = cka_all["s7_control"]
    im1 = ax1.imshow(cka_ctrl, aspect="auto", cmap="viridis", vmin=0, vmax=0.65)
    ax1.set_xlabel("Student Layer"); ax1.set_ylabel("Teacher Layer")
    ax1.set_title("(a) Control (Supervised)\nMRR = 0.205", fontweight="bold")
    ax1.set_xticks(range(5)); ax1.set_xticklabels(LAYER_LABELS_SHORT)
    ax1.set_yticks(range(cka_ctrl.shape[0]))
    for i in range(cka_ctrl.shape[0]):
        for j in range(cka_ctrl.shape[1]):
            v = cka_ctrl[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if v < 0.35 else "black")

    # Panel B: BiMGA
    ax2 = fig.add_subplot(gs[0, 1])
    cka_bimga = cka_all["s10_bimga"]
    im2 = ax2.imshow(cka_bimga, aspect="auto", cmap="viridis", vmin=0, vmax=0.65)
    ax2.set_xlabel("Student Layer")
    ax2.set_title("(b) BiMGA (Full)\nMRR = 0.325", fontweight="bold")
    ax2.set_xticks(range(5)); ax2.set_xticklabels(LAYER_LABELS_SHORT)
    ax2.set_yticks(range(cka_bimga.shape[0])); ax2.set_yticklabels([])
    for i in range(cka_bimga.shape[0]):
        for j in range(cka_bimga.shape[1]):
            v = cka_bimga[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if v < 0.35 else "black")

    cax = fig.add_subplot(gs[0, 2])
    fig.colorbar(im2, cax=cax, label="Linear CKA")

    # Panel C: Per-layer gain
    ax3 = fig.add_subplot(gs[1, :2])
    compare_keys = ["s9_score", "s8_hnp", "s7_embed", "s8_bimga_uni", "s10_bimga"]
    x = np.arange(5)
    for key in compare_keys:
        gain = cka_all[key].mean(axis=0) - ctrl_per_layer
        ax3.plot(x, gain, marker=MARKERS[key],
                 label=f"{DISPLAY[key]} (MRR={MRR[key]:.3f})",
                 color=COLORS[key], linewidth=LW[key], markersize=8)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax3.set_xticks(x); ax3.set_xticklabels(LAYER_LABELS)
    ax3.set_ylabel("CKA Gain over Control")
    ax3.set_title("(c) Per-layer CKA improvement over Control", fontweight="bold")
    ax3.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax3.set_xlim(-0.3, 4.3)
    ax3.annotate("BiMGA's gain peaks\nin middle layers",
                 xy=(2.5, 0.11), fontsize=9, fontweight="bold", color=COLORS["s10_bimga"],
                 ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                        edgecolor=COLORS["s10_bimga"], alpha=0.9))
    ax3.annotate("Score/HNP gain\nat output layer",
                 xy=(4.0, 0.033), fontsize=9, fontweight="bold", color="#666666",
                 ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                                        edgecolor="#aaaaaa", alpha=0.9))

    fig.savefig(FIGURE_DIR / "poster_fig_A_cka.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "poster_fig_A_cka.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: poster_fig_A_cka")


# =========================================================================
# Figure B: Teacher KL bar + per-layer
# =========================================================================

def make_figure_b():
    # Load per-example KL for all models
    # Attentions have 4 layers (transformer layers only, no embedding)
    # Label them L1-L4 to match CKA figure where L0=embedding
    all_kl = {}
    for key in EXPERIMENTS:
        per_ex = load_teacher_kl(key)  # (N, T_layers, S_layers)
        mean_mat = per_ex.mean(axis=0)  # (T, S)
        best_kl = mean_mat.min(axis=0)  # (S,) — best teacher match per student layer
        overall = float(best_kl.mean())  # mean over all transformer layers
        all_kl[key] = {"best_kl": best_kl, "overall": overall}

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 5),
                                           gridspec_kw={"width_ratios": [1, 1.1]})

    # Left: bar chart sorted by KL
    keys_sorted = sorted(EXPERIMENTS, key=lambda k: all_kl[k]["overall"])
    names = [DISPLAY[k] for k in keys_sorted]
    vals = [all_kl[k]["overall"] for k in keys_sorted]
    colors = [COLORS[k] for k in keys_sorted]

    bars = ax_bar.barh(range(len(keys_sorted)), vals, color=colors,
                       edgecolor="black", linewidth=0.5, height=0.6)
    ax_bar.set_yticks(range(len(keys_sorted)))
    ax_bar.set_yticklabels([f"{n}\n(MRR={MRR[k]:.3f})" for n, k in zip(names, keys_sorted)],
                           fontsize=9)
    ax_bar.set_xlabel("Mean KL from Teacher\n(best-matching teacher layer, L1\u2013L4)")
    ax_bar.set_title("(a) Overall Attention Similarity\nto Teacher", fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, max(vals) * 1.15)  # ensure bars fit
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    # Right: per-layer line plot (all 4 transformer layers = L1-L4 in CKA indexing)
    for key in EXPERIMENTS:
        best_kl = all_kl[key]["best_kl"]  # all transformer layers
        x = np.arange(len(best_kl))
        ax_line.plot(x, best_kl, marker=MARKERS[key],
                     label=f"{DISPLAY[key]}",
                     color=COLORS[key], linewidth=LW[key], markersize=8)

    ax_line.set_xticks(range(len(LAYER_LABELS_TF)))
    ax_line.set_xticklabels(LAYER_LABELS_TF)
    ax_line.set_ylabel("KL from Teacher\n(best-matching teacher layer)")
    ax_line.set_title("(b) Per-layer Attention Divergence\nfrom Teacher", fontweight="bold")
    ax_line.legend(fontsize=7, loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "poster_fig_B_teacher_kl.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "poster_fig_B_teacher_kl.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: poster_fig_B_teacher_kl")


if __name__ == "__main__":
    make_figure_a()
    make_figure_b()
    print(f"\nAll figures in: {FIGURE_DIR.resolve()}")
