"""Generate styled TACO results table matching poster style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data from the 6 TACO runs (R@1, R@10 from README Final Saturated Models table)
rows = [
    # (Run, Variant, Config, Epochs, MRR, R@1, R@10)
    ("s7_control_bs32",       "control (no KD)",        "—",            39,  0.205, 0.143, 0.331),
    ("s7_embed_dw100_aw10",   "embed_distill",          "dw=100, aw=10", 69, 0.303, 0.218, 0.461),
    ("s8_A2_bimga_uniform",   "BiMGA-uniform (A2)",     "dw=100, aw=10", 78, 0.313, 0.232, 0.469),
    ("s8_hnp_dw100_pw10",     "Hard Negative Pair",     "dw=100, pw=10", 82, 0.302, 0.221, 0.461),
    ("s9_bimga_dw100_aw20",   "BiMGA (margin-gated)",   "dw=100, aw=20", 159, 0.325, 0.241, 0.486),
    ("s9_score_dw100",        "Score Distillation",     "dw=100",        132, 0.301, 0.215, 0.466),
]

BEST_IDX = 4  # s9_bimga_dw100_aw20

headers = ["Run", "Variant", "Config", "Epochs\n(stop)", "Test\nMRR", "R@1", "R@10"]

# Row colors (matching poster palette)
row_colors = [
    "#E8E8E8",  # gray - control
    "#FFD9B3",  # orange - embed
    "#B3D4F0",  # blue - bimga uniform
    "#E8B3B3",  # red - hnp
    "#D4E8B3",  # green - bimga best
    "#E0CCF5",  # purple - score
]

best_color = "#C8E6A0"  # highlight green for best row

def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

cell_text = []
for r in rows:
    cell_text.append([str(r[0]), r[1], r[2], str(r[3]), fmt(r[4]), fmt(r[5]), fmt(r[6])])

fig, ax = plt.subplots(figsize=(14, 6.5))
ax.axis("off")

n_rows = len(rows)
n_cols = len(headers)

# Build color array
cell_colours = []
for i, _ in enumerate(rows):
    base = best_color if i == BEST_IDX else row_colors[i]
    cell_colours.append([base] * n_cols)

header_color = "#2C2C2C"

table = ax.table(
    cellText=cell_text,
    colLabels=headers,
    cellColours=cell_colours,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

# Style header
for j in range(n_cols):
    cell = table[0, j]
    cell.set_facecolor(header_color)
    cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    cell.set_edgecolor("white")
    cell.set_linewidth(1.5)

# Style data cells
for i in range(1, n_rows + 1):
    for j in range(n_cols):
        cell = table[i, j]
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)
        cell.set_text_props(fontsize=11)
        # Bold the best row
        if i - 1 == BEST_IDX:
            cell.set_text_props(fontweight="bold", fontsize=11)
        # Left-align Run and Variant columns
        if j <= 1:
            cell.set_text_props(ha="left")
            cell.PAD = 0.05

# Adjust column widths
col_widths = [0.20, 0.22, 0.14, 0.08, 0.08, 0.07, 0.07]
for j, w in enumerate(col_widths):
    for i in range(n_rows + 1):
        table[i, j].set_width(w)

# Mark best row with arrow
ax.annotate(
    " BEST",
    xy=(0.98, 0.295),
    xycoords="axes fraction",
    fontsize=10,
    fontweight="bold",
    color="#2E7D32",
    ha="right",
)

plt.title("Table 2. TACO Out-of-Domain Transfer Results", fontsize=14, fontweight="bold", pad=20)

caption = (
    "Knowledge distillation results on TACO (code search), evaluating out-of-domain transfer from MBPP-trained models. "
    "Student: TinyBERT-General-4L-312D; Teacher: all-MiniLM-L6-v2. All models trained with early stopping (patience=15). "
    "BiMGA (margin-gated) achieves the highest MRR (0.325), a 59% relative improvement over the supervised control, "
    "demonstrating that bidirectional margin-gated alignment produces the most transferable representations. "
    "Bold = best result per column."
)
fig.text(
    0.5, 0.03, caption,
    ha="center", va="bottom", fontsize=10,
    wrap=True, style="italic",
    transform=fig.transFigure,
    multialignment="center",
    bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5", ec="none"),
)
plt.subplots_adjust(bottom=0.25)

out = "/Users/samuellee/BME/CS4248/proj_1/CS4248_ez_A/submission/experiments/taco_results_table.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
