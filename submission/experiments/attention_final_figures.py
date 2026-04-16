"""
Generate publication-ready figures for the attention analysis section.

Figure 1: CKA per-layer breakdown — showing middle-layer alignment advantage
Figure 2: KL divergence hierarchy (5 panels)
Figure 3: Compact summary bar (supplementary)
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

EXPERIMENTS = {
    "s7_control":    "Control (Supervised)",
    "s7_embed":      "Embed Distill",
    "s8_bimga_uni":  "BiMGA Uniform",
    "s8_hnp":        "Hard Neg Pairwise",
    "s9_score":      "Score Distill",
    "s10_bimga":     "BiMGA (Full)",
}

MRR = {
    "s7_control": 0.205, "s7_embed": 0.303, "s8_bimga_uni": 0.313,
    "s8_hnp": 0.302, "s9_score": 0.301, "s10_bimga": 0.325,
}

LAYER_LABELS = ["L0\n(embed)", "L1\n(tf-0)", "L2\n(tf-1)", "L3\n(tf-2)", "L4\n(tf-3\noutput)"]


def load_cka(key):
    data = np.load(str(CACHE_DIR / f"{key}.npz"), allow_pickle=True)
    return data["cka"]


def load_kl(key):
    return np.load(str(CACHE_DIR / f"kl_{key}.npy"))


# =========================================================================
# FIGURE 1: CKA heatmaps + per-layer gain chart
# =========================================================================

def make_figure_1():
    keys_all = ["s7_control", "s7_embed", "s8_bimga_uni", "s8_hnp", "s9_score", "s10_bimga"]
    names_short = ["Control", "Embed", "BiMGA-Uni", "HNP", "Score", "BiMGA"]

    cka_all = {k: load_cka(k) for k in keys_all}
    ctrl_per_layer = cka_all["s7_control"].mean(axis=0)

    # --- Layout: 2 CKA heatmaps on top, per-layer gain chart below ---
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 0.9], hspace=0.35, wspace=0.3,
                           width_ratios=[1, 1, 0.06])

    # Top left: Control CKA
    ax1 = fig.add_subplot(gs[0, 0])
    cka_ctrl = cka_all["s7_control"]
    im1 = ax1.imshow(cka_ctrl, aspect="auto", cmap="viridis", vmin=0, vmax=0.65)
    ax1.set_xlabel("Student Layer", fontsize=11)
    ax1.set_ylabel("Teacher Layer", fontsize=11)
    ax1.set_title("(a) Control (Supervised) — MRR = 0.205", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(cka_ctrl.shape[0]))
    for i in range(cka_ctrl.shape[0]):
        for j in range(cka_ctrl.shape[1]):
            v = cka_ctrl[i, j]
            c = "white" if v < 0.35 else "black"
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=c)

    # Top right: BiMGA CKA
    ax2 = fig.add_subplot(gs[0, 1])
    cka_bimga = cka_all["s10_bimga"]
    im2 = ax2.imshow(cka_bimga, aspect="auto", cmap="viridis", vmin=0, vmax=0.65)
    ax2.set_xlabel("Student Layer", fontsize=11)
    ax2.set_title("(b) BiMGA (Full) — MRR = 0.325", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(cka_bimga.shape[0]))
    ax2.set_yticklabels([])
    for i in range(cka_bimga.shape[0]):
        for j in range(cka_bimga.shape[1]):
            v = cka_bimga[i, j]
            c = "white" if v < 0.35 else "black"
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=c)

    # Colorbar
    cax = fig.add_subplot(gs[0, 2])
    fig.colorbar(im2, cax=cax, label="Linear CKA")

    # Bottom: Per-layer CKA gain over Control
    ax3 = fig.add_subplot(gs[1, :2])

    compare_keys = ["s9_score", "s8_hnp", "s7_embed", "s8_bimga_uni", "s10_bimga"]
    compare_names = ["Score Distill", "Hard Neg Pairwise", "Embed Distill", "BiMGA Uniform", "BiMGA (Full)"]
    colors = ["#aaaaaa", "#aaaaaa", "#66c2a5", "#2166ac", "#d62728"]
    markers = ["s", "D", "^", "o", "o"]

    x = np.arange(5)
    for key, name, color, marker in zip(compare_keys, compare_names, colors, markers):
        per_layer = cka_all[key].mean(axis=0)
        gain = per_layer - ctrl_per_layer
        mrr = MRR[key]
        lw = 3 if "bimga" in key.lower() or "BiMGA" in name else 1.5
        ax3.plot(x, gain, marker=marker, label=f"{name} (MRR={mrr:.3f})",
                 color=color, linewidth=lw, markersize=8, zorder=5 if lw > 2 else 3)

    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(LAYER_LABELS, fontsize=9)
    ax3.set_ylabel("CKA Gain over Control", fontsize=11)
    ax3.set_title("(c) Per-layer CKA improvement over Control (Supervised)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.set_xlim(-0.3, 4.3)

    # Annotate the key insight
    ax3.annotate(
        "BiMGA's gain is\nconcentrated in\nmiddle layers",
        xy=(2.5, 0.10), fontsize=9, fontweight="bold", color="#d62728",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#d62728", alpha=0.9),
    )
    ax3.annotate(
        "Score/HNP gain is\nat the output layer",
        xy=(4, 0.028), fontsize=9, fontweight="bold", color="#666666",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#aaaaaa", alpha=0.9),
    )

    fig.savefig(FIGURE_DIR / "fig1_cka_layerwise.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig1_cka_layerwise.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig1_cka_layerwise")


# =========================================================================
# FIGURE 2: KL divergence hierarchy
# =========================================================================

def make_figure_2():
    kl_keys = ["s8_bimga_uni", "s7_embed", "s9_score", "s8_hnp", "s7_control"]
    kl_labels = {
        "s8_bimga_uni": "BiMGA Uniform",
        "s7_embed":     "Embed Distill",
        "s9_score":     "Score Distill",
        "s8_hnp":       "Hard Neg Pairwise",
        "s7_control":   "Control (Supervised)",
    }

    kl_data = {k: load_kl(k) for k in kl_keys}
    mean_kls = {k: float(v.mean()) for k, v in kl_data.items()}
    vmax = max(v.max() for v in kl_data.values())

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

    for ax, key in zip(axes, kl_keys):
        kl = kl_data[key]
        im = ax.imshow(kl, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_xlabel("Head", fontsize=10)
        ax.set_title(
            f"{kl_labels[key]}\nMRR={MRR[key]:.3f} | Mean KL={mean_kls[key]:.3f}",
            fontsize=10, fontweight="bold",
        )
        ax.set_xticks(range(kl.shape[1]))
        ax.set_xticklabels(range(kl.shape[1]), fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_yticks(range(kl.shape[0]))
        for i in range(kl.shape[0]):
            for j in range(kl.shape[1]):
                v = kl[i, j]
                c = "white" if v > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=c)

    fig.colorbar(im, ax=axes, label="KL Divergence", shrink=0.8, pad=0.02)

    fig.suptitle(
        "Attention Divergence from BiMGA (Full), ordered by increasing divergence",
        fontsize=13, fontweight="bold", y=1.03,
    )

    fig.text(
        0.5, -0.04,
        "Models with explicit alignment objectives (BiMGA Uniform, Embed Distill) develop "
        "attention patterns closest to BiMGA.\n"
        "Output-matching methods (Score Distill, Hard Neg Pairwise) and supervised "
        "training diverge most, especially in layers 0 and 2\u20133.",
        ha="center", fontsize=10, style="italic",
    )

    fig.savefig(FIGURE_DIR / "fig2_kl_hierarchy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig2_kl_hierarchy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig2_kl_hierarchy")


# =========================================================================
# FIGURE 3 (bonus): L2/L4 ratio bar chart
# =========================================================================

def make_figure_3():
    """Show middle-layer vs output-layer CKA gain ratio for each method."""
    keys = ["s9_score", "s8_hnp", "s7_embed", "s8_bimga_uni", "s10_bimga"]
    names = ["Score\nDistill", "Hard Neg\nPairwise", "Embed\nDistill", "BiMGA\nUniform", "BiMGA\n(Full)"]
    mrrs = [MRR[k] for k in keys]

    ctrl_cka = load_cka("s7_control").mean(axis=0)

    l2_gains = []
    l3_gains = []
    l4_gains = []
    for k in keys:
        cka = load_cka(k).mean(axis=0)
        diff = cka - ctrl_cka
        l2_gains.append(diff[2])
        l3_gains.append(diff[3])
        l4_gains.append(diff[4])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(keys))
    w = 0.25

    bars_l2 = ax.bar(x - w, l2_gains, w, label="L2 (middle)", color="#2166ac", edgecolor="black", linewidth=0.5)
    bars_l3 = ax.bar(x, l3_gains, w, label="L3 (penultimate)", color="#66c2a5", edgecolor="black", linewidth=0.5)
    bars_l4 = ax.bar(x + w, l4_gains, w, label="L4 (output)", color="#d62728", edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nMRR={m:.3f}" for n, m in zip(names, mrrs)], fontsize=9)
    ax.set_ylabel("CKA Gain over Control", fontsize=11)
    ax.set_title("Where does each method improve teacher-student alignment?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_layer_gains.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig3_layer_gains.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig3_layer_gains")


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
    make_figure_3()
    print(f"\nAll final figures in: {FIGURE_DIR.resolve()}")
