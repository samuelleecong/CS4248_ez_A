#!/usr/bin/env python3
"""Single combined figure: Quality Retention + MRR + Recall@1 + Recall@10."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"

TEACHER_MRR = 0.8222111488995274

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cmp(run_dir: Path) -> dict:
    df = pd.read_csv(run_dir / "metrics" / "comparisons.csv", keep_default_na=False)
    df = df[df["status"] == "success"].copy()
    for col in ["base_value", "compare_value", "ci_low", "ci_high"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return {row["metric"]: row for _, row in df.iterrows()}


listwise_cmp   = load_cmp(RESULTS / "kd_minilm_listwise_sweep")
pairwise_cmp   = load_cmp(RESULTS / "kd_minilm_pairwise_sweep")
pairdistil_cmp = load_cmp(RESULTS / "kd_minilm_pairdistil_sweep")
pointwise_cmp  = load_cmp(RESULTS / "kd_minilm_pointwise_sweep")

ZS = {m: listwise_cmp[m]["base_value"] for m in ["mrr", "recall@1", "recall@10"]}

BARS = [
    ("Zero-shot",        ZS,             "#9E9E9E"),
    ("Pointwise (MSE)",  pointwise_cmp,  "#C44E52"),
    ("Listwise (KL)",    listwise_cmp,   "#4C72B0"),
    ("Pairwise (MMSE)",  pairwise_cmp,   "#DD8452"),
    ("PairDistil",       pairdistil_cmp, "#55A868"),
]

def get_val_err(data, metric):
    """Return (value, err_lo, err_hi). ZS has no CI."""
    if data is ZS:
        return float(ZS[metric]), 0.0, 0.0
    row  = data[metric]
    v    = float(row["compare_value"])
    base = float(row["base_value"])
    return v, v - (base + float(row["ci_low"])), (base + float(row["ci_high"])) - v


labels = [b[0] for b in BARS]
colors = [b[2] for b in BARS]
x      = np.arange(len(BARS))
W      = 0.55

# ---------------------------------------------------------------------------
# Shared helper to draw one subplot
# ---------------------------------------------------------------------------

def draw_panel(ax, metric, title, y_label=None,
               reference_line=None, ref_label=None,
               value_fmt=".4f", label_pad=0.001):

    vals, errs_lo, errs_hi = [], [], []
    for _, data, _ in BARS:
        v, elo, ehi = get_val_err(data, metric)
        vals.append(v)
        errs_lo.append(elo)
        errs_hi.append(ehi)

    bars = ax.bar(x, vals, color=colors, width=W, edgecolor="white", linewidth=0.8)

    for i in range(1, len(BARS)):
        ax.errorbar(x[i], vals[i],
                    yerr=[[errs_lo[i]], [errs_hi[i]]],
                    fmt="none", color="black", capsize=4, linewidth=1.4)

    if reference_line is not None:
        ax.axhline(reference_line, color="#333333", linestyle=":",
                   linewidth=1.4, label=ref_label)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + label_pad,
                f"{v:{value_fmt}}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    y_min = min(vals) - 0.015
    y_max = max(vals + ([reference_line] if reference_line else [])) + 0.025
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10)

# ---------------------------------------------------------------------------
# Build combined 2×2 figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
(ax_ret, ax_mrr), (ax_r1, ax_r10) = axes

# ── Quality Retention (top-left) ──────────────────────────────────────────
ret_vals, ret_lo, ret_hi = [], [], []
for _, data, _ in BARS:
    v, elo, ehi = get_val_err(data, "mrr")
    ret_vals.append(v / TEACHER_MRR * 100)
    ret_lo.append(elo / TEACHER_MRR * 100)
    ret_hi.append(ehi / TEACHER_MRR * 100)

bars_ret = ax_ret.bar(x, ret_vals, color=colors, width=W,
                      edgecolor="white", linewidth=0.8)
for i in range(1, len(BARS)):
    ax_ret.errorbar(x[i], ret_vals[i],
                    yerr=[[ret_lo[i]], [ret_hi[i]]],
                    fmt="none", color="black", capsize=4, linewidth=1.4)
ax_ret.axhline(100, color="#333333", linestyle=":", linewidth=1.4, label="Teacher (100%)")
for bar, v in zip(bars_ret, ret_vals):
    ax_ret.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
ax_ret.set_ylim(86, 104)
ax_ret.set_xticks(x)
ax_ret.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
ax_ret.set_title("Quality Retention", fontsize=12, fontweight="bold", pad=6)
ax_ret.set_ylabel("Student MRR / Teacher MRR  ×  100", fontsize=9.5)
ax_ret.yaxis.grid(True, linestyle="--", alpha=0.45)
ax_ret.set_axisbelow(True)
ax_ret.legend(fontsize=9)

# ── MRR (top-right) ───────────────────────────────────────────────────────
draw_panel(ax_mrr, "mrr", "MRR",
           reference_line=TEACHER_MRR, ref_label=f"Teacher MRR ({TEACHER_MRR:.4f})",
           value_fmt=".4f", label_pad=0.001)
ax_mrr.legend(fontsize=9)

# ── Recall@1 (bottom-left) ────────────────────────────────────────────────
draw_panel(ax_r1, "recall@1", "Recall@1",
           y_label="Score", value_fmt=".4f", label_pad=0.001)

# ── Recall@10 (bottom-right) ──────────────────────────────────────────────
draw_panel(ax_r10, "recall@10", "Recall@10",
           value_fmt=".4f", label_pad=0.0005)

# ── Shared legend ─────────────────────────────────────────────────────────
legend_handles = [mpatches.Patch(color=c, label=l) for l, _, c in BARS]
legend_handles += [
    plt.Line2D([0], [0], color="#333333", linestyle=":", linewidth=1.4,
               label="Teacher reference"),
    plt.Line2D([0], [0], color="black",   linestyle="-", linewidth=1.4,
               label="95% CI (bootstrap, n=2000)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=7,
           fontsize=9, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

fig.suptitle(
    "KD Method Comparison — Best Config per Method\n"
    "Student: MiniLM-L6-v2  ·  Teacher: all-mpnet-base-v2",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = RESULTS / "kd_comparison_all.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
